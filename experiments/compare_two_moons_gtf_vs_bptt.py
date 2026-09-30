"""Compare binary/ternary GTF with binary/ternary SG-BPTT on two moons."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import TensorDataset

from gtf.data import make_two_moons_data
from gtf.gtf import GTFClassifier, GTFTrainingResult, fit_gtf_model, gtf_result_summary
from gtf.snn import TwoMoonsSNN
from gtf.training import TrainingResult, fit_model, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/two_moons_gtf_vs_bptt.yaml"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--bptt-epochs", type=int, default=None)
    parser.add_argument("--gtf-stages", type=int, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return config


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def numpy_split(dataset: TensorDataset) -> tuple[np.ndarray, np.ndarray]:
    features, labels = dataset.tensors
    return features.numpy(), labels.numpy()


def write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def bptt_summary(result: TrainingResult, model: TwoMoonsSNN) -> dict[str, Any]:
    return {
        "model": result.model_name,
        "training_method": "SG-BPTT-trainable-encoder",
        "best_epoch": result.best_epoch,
        "epochs_completed": result.epochs_completed,
        "fit_wall_seconds": result.fit_wall_seconds,
        "optimization_seconds": result.train_step_seconds,
        "seconds_per_epoch": result.train_step_seconds / result.epochs_completed,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "validation": asdict(result.best_validation),
        "test": asdict(result.test),
        "best_checkpoint": str(result.best_checkpoint),
        "last_checkpoint": str(result.last_checkpoint),
    }


def plot_comparison(
    *,
    bptt_results: list[TrainingResult],
    gtf_results: list[GTFTrainingResult],
    summaries: list[dict[str, Any]],
    output_path: Path,
) -> None:
    labels = [f"{row['model']}\n{row['training_method'].split('-')[0]}" for row in summaries]
    accuracies = [row["test"]["accuracy"] for row in summaries]
    fit_times = [row["fit_wall_seconds"] for row in summaries]
    colors = ["#4c78a8", "#f58518", "#72b7b2", "#e45756"]

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].bar(labels, accuracies, color=colors)
    axes[0, 0].set_ylim(max(0.5, min(accuracies) - 0.1), 1.0)
    axes[0, 0].set(title="Test accuracy", ylabel="Accuracy")
    axes[0, 1].bar(labels, fit_times, color=colors)
    axes[0, 1].set(title="Fit wall time", ylabel="Seconds")

    for result in bptt_results:
        axes[1, 0].plot(
            [row["epoch"] for row in result.history],
            [row["validation_accuracy"] for row in result.history],
            label=result.model_name,
        )
    axes[1, 0].set(
        title="SG-BPTT validation accuracy",
        xlabel="Epoch",
        ylabel="Accuracy",
    )
    axes[1, 0].legend()

    for result in gtf_results:
        axes[1, 1].plot(
            [row["stage"] for row in result.history],
            [row["validation_accuracy"] for row in result.history],
            marker="o",
            markersize=3,
            label=result.model_name,
        )
    axes[1, 1].set(
        title="GTF validation accuracy",
        xlabel="Selected temporal basis",
        ylabel="Accuracy",
    )
    axes[1, 1].legend()

    figure.suptitle("Two moons: SG-BPTT vs fixed-encoder GTF")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def print_comparison(summaries: list[dict[str, Any]]) -> None:
    print("\nTwo-moons training comparison")
    print("model         method       test_acc  test_loss  fit_s   spikes/sample")
    for summary in summaries:
        method = "BPTT" if summary["training_method"].startswith("SG") else "GTF"
        test = summary["test"]
        print(
            f"{summary['model']:<13} {method:<12} "
            f"{test['accuracy']:.4f}    {test['loss']:.4f}    "
            f"{summary['fit_wall_seconds']:.3f}   "
            f"{test['active_spikes_per_sample']:.1f}"
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["runtime"]["device"] = args.device
    if args.bptt_epochs is not None:
        config["bptt_training"]["epochs"] = args.bptt_epochs
    if args.gtf_stages is not None:
        config["gtf_model"]["max_stages"] = args.gtf_stages

    seed = int(config["reproducibility"]["seed"])
    deterministic = bool(config["reproducibility"]["deterministic"])
    device = resolve_device(str(config["runtime"]["device"]))
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M_two-moons_gtf-vs-bptt")
    run_directory = Path(config["project"]["run_root"]) / run_name
    checkpoint_root = Path(config["project"]["checkpoint_root"]) / run_name
    plot_path = Path(config["project"]["plot_root"]) / f"{run_name}.png"
    run_directory.mkdir(parents=True, exist_ok=False)

    data = make_two_moons_data(**config["data"], seed=seed)
    train_features, train_labels = numpy_split(data.train)
    validation_features, validation_labels = numpy_split(data.validation)
    test_features, test_labels = numpy_split(data.test)

    bptt_results: list[TrainingResult] = []
    bptt_models: list[TwoMoonsSNN] = []
    for spike_kind in ("binary", "ternary"):
        print(f"Training {spike_kind} SG-BPTT...")
        seed_everything(seed, deterministic=deterministic)
        model = TwoMoonsSNN(spike_kind=spike_kind, **config["bptt_model"])
        result = fit_model(
            model=model,
            model_name=f"{spike_kind}-bptt",
            train_data=data.train,
            validation_data=data.validation,
            test_data=data.test,
            data_metadata=data.metadata,
            training_config=config["bptt_training"],
            checkpoint_directory=checkpoint_root / f"{spike_kind}-bptt",
            device=device,
            seed=seed,
        )
        bptt_models.append(model)
        bptt_results.append(result)
        write_history(run_directory / f"{spike_kind}_bptt_history.csv", result.history)

    gtf_results: list[GTFTrainingResult] = []
    for spike_kind in ("binary", "ternary"):
        print(f"Training {spike_kind} GTF...")
        model = GTFClassifier(spike_kind=spike_kind, seed=seed, **config["gtf_model"])
        result = fit_gtf_model(
            model=model,
            model_name=f"{spike_kind}-gtf",
            train_features=train_features,
            train_labels=train_labels,
            validation_features=validation_features,
            validation_labels=validation_labels,
            test_features=test_features,
            test_labels=test_labels,
            data_metadata=data.metadata,
            checkpoint_path=checkpoint_root / f"{spike_kind}-gtf" / "best.pt",
        )
        gtf_results.append(result)
        write_history(run_directory / f"{spike_kind}_gtf_history.csv", result.history)

    summaries = [
        *[
            bptt_summary(result, model)
            for result, model in zip(bptt_results, bptt_models, strict=True)
        ],
        *[gtf_result_summary(result) for result in gtf_results],
    ]
    summary = {
        "run_name": run_name,
        "device": str(device),
        "seed": seed,
        "comparison_note": (
            "SG-BPTT trains the encoder; this first GTF experiment uses fixed random "
            "membrane features and trains only stagewise spike bases and decoders."
        ),
        "models": summaries,
    }
    with (run_directory / "summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)
    with (run_directory / "resolved_config.yaml").open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(config, output_file, sort_keys=False)

    plot_comparison(
        bptt_results=bptt_results,
        gtf_results=gtf_results,
        summaries=summaries,
        output_path=plot_path,
    )
    print_comparison(summaries)
    print(f"Summary: {run_directory / 'summary.json'}")
    print(f"Plot: {plot_path}")
    print(f"Checkpoints: {checkpoint_root}")


if __name__ == "__main__":
    main()
