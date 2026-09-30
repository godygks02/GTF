"""Training and evaluation helpers for SG-BPTT baselines."""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from gtf.snn import TwoMoonsSNN


@dataclass(frozen=True)
class EpochMetrics:
    """Classification and spike-activity metrics for one dataset pass."""

    loss: float
    accuracy: float
    positive_spikes_per_sample: float
    negative_spikes_per_sample: float
    active_spikes_per_sample: float


@dataclass(frozen=True)
class TrainingResult:
    """Outputs of one complete model fit."""

    model_name: str
    best_epoch: int
    epochs_completed: int
    fit_wall_seconds: float
    train_step_seconds: float
    evaluation_seconds: float
    best_validation: EpochMetrics
    test: EpochMetrics
    history: list[dict[str, float | int]]
    best_checkpoint: Path
    last_checkpoint: Path


def seed_everything(seed: int, *, deterministic: bool) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _loader(
    dataset: TensorDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[tuple[Tensor, Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def _finalize_metrics(
    *,
    total_loss: float,
    correct: int,
    positive_spikes: float,
    negative_spikes: float,
    num_samples: int,
) -> EpochMetrics:
    return EpochMetrics(
        loss=total_loss / num_samples,
        accuracy=correct / num_samples,
        positive_spikes_per_sample=positive_spikes / num_samples,
        negative_spikes_per_sample=negative_spikes / num_samples,
        active_spikes_per_sample=(positive_spikes + negative_spikes) / num_samples,
    )


def train_epoch(
    model: TwoMoonsSNN,
    loader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: Adam,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> EpochMetrics:
    """Run one SG-BPTT optimization epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    positive_spikes = 0.0
    negative_spikes = 0.0
    num_samples = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(features)
        loss = criterion(output.logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        correct += (output.logits.argmax(dim=1) == labels).sum().item()
        positive_spikes += output.positive_spikes.item()
        negative_spikes += output.negative_spikes.item()
        num_samples += batch_size

    return _finalize_metrics(
        total_loss=total_loss,
        correct=correct,
        positive_spikes=positive_spikes,
        negative_spikes=negative_spikes,
        num_samples=num_samples,
    )


@torch.no_grad()
def evaluate(
    model: TwoMoonsSNN,
    loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> EpochMetrics:
    """Evaluate classification quality and spike activity."""
    model.eval()
    total_loss = 0.0
    correct = 0
    positive_spikes = 0.0
    negative_spikes = 0.0
    num_samples = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        output = model(features)
        batch_size = labels.shape[0]
        total_loss += criterion(output.logits, labels).item() * batch_size
        correct += (output.logits.argmax(dim=1) == labels).sum().item()
        positive_spikes += output.positive_spikes.item()
        negative_spikes += output.negative_spikes.item()
        num_samples += batch_size

    return _finalize_metrics(
        total_loss=total_loss,
        correct=correct,
        positive_spikes=positive_spikes,
        negative_spikes=negative_spikes,
        num_samples=num_samples,
    )


def _checkpoint_payload(
    *,
    model: TwoMoonsSNN,
    optimizer: Adam,
    model_name: str,
    epoch: int,
    validation_metrics: EpochMetrics,
    data_metadata: dict[str, Any],
    training_config: dict[str, Any],
    timing: dict[str, float],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model_name": model_name,
        "epoch": epoch,
        "model_config": model.config(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_metrics": asdict(validation_metrics),
        "data_metadata": data_metadata,
        "training_config": training_config,
        "timing": timing,
    }


def fit_model(
    *,
    model: TwoMoonsSNN,
    model_name: str,
    train_data: TensorDataset,
    validation_data: TensorDataset,
    test_data: TensorDataset,
    data_metadata: dict[str, Any],
    training_config: dict[str, Any],
    checkpoint_directory: Path,
    device: torch.device,
    seed: int,
) -> TrainingResult:
    """Fit a model, save best/last checkpoints, and evaluate the best state."""
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    best_checkpoint = checkpoint_directory / "best.pt"
    last_checkpoint = checkpoint_directory / "last.pt"

    batch_size = int(training_config["batch_size"])
    epochs = int(training_config["epochs"])
    optimizer = Adam(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    train_loader = _loader(train_data, batch_size=batch_size, shuffle=True, seed=seed)
    validation_loader = _loader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
    test_loader = _loader(test_data, batch_size=batch_size, shuffle=False, seed=seed)

    history: list[dict[str, float | int]] = []
    best_epoch = 0
    best_validation: EpochMetrics | None = None
    train_step_seconds = 0.0
    evaluation_seconds = 0.0
    fit_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        _synchronize(device)
        train_start = time.perf_counter()
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        _synchronize(device)
        train_step_seconds += time.perf_counter() - train_start

        evaluation_start = time.perf_counter()
        validation_metrics = evaluate(model, validation_loader, criterion, device)
        _synchronize(device)
        evaluation_seconds += time.perf_counter() - evaluation_start

        history.append(
            {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in asdict(train_metrics).items()},
                **{
                    f"validation_{key}": value
                    for key, value in asdict(validation_metrics).items()
                },
            }
        )

        is_best = best_validation is None or (
            validation_metrics.accuracy > best_validation.accuracy
            or (
                validation_metrics.accuracy == best_validation.accuracy
                and validation_metrics.loss < best_validation.loss
            )
        )
        if is_best:
            best_epoch = epoch
            best_validation = validation_metrics
            torch.save(
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    model_name=model_name,
                    epoch=epoch,
                    validation_metrics=validation_metrics,
                    data_metadata=data_metadata,
                    training_config=training_config,
                    timing={
                        "train_step_seconds": train_step_seconds,
                        "evaluation_seconds": evaluation_seconds,
                    },
                ),
                best_checkpoint,
            )

    if best_validation is None:
        raise RuntimeError("Training completed without a validation result")

    fit_wall_seconds = time.perf_counter() - fit_start
    torch.save(
        _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            model_name=model_name,
            epoch=epochs,
            validation_metrics=validation_metrics,
            data_metadata=data_metadata,
            training_config=training_config,
            timing={
                "fit_wall_seconds": fit_wall_seconds,
                "train_step_seconds": train_step_seconds,
                "evaluation_seconds": evaluation_seconds,
            },
        ),
        last_checkpoint,
    )

    best_payload = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best_payload["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    return TrainingResult(
        model_name=model_name,
        best_epoch=best_epoch,
        epochs_completed=epochs,
        fit_wall_seconds=fit_wall_seconds,
        train_step_seconds=train_step_seconds,
        evaluation_seconds=evaluation_seconds,
        best_validation=best_validation,
        test=test_metrics,
        history=history,
        best_checkpoint=best_checkpoint,
        last_checkpoint=last_checkpoint,
    )
