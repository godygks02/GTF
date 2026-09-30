"""Compare recurrent Temporal GTF against SG-BPTT on two moons."""

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
from torch.utils.data import DataLoader, TensorDataset

from gtf.data import make_two_moons_data
from gtf.gtf import GTFTrainingResult, fit_gtf_model, gtf_result_summary
from gtf.snn import TwoMoonsSNN
from gtf.temporal_gtf import TemporalGTFClassifier
from gtf.training import TrainingResult, fit_model, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/two_moons_temporal_gtf.yaml"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--bptt-epochs", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
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


@torch.no_grad()
def bptt_accuracy_by_timestep(
    model: TwoMoonsSNN,
    dataset: TensorDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> list[float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = np.zeros(model.timesteps, dtype=np.int64)
    total = 0
    model.eval()
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        trace = model(features, return_timestep_logits=True).timestep_logits
        if trace is None:
            raise RuntimeError("SNN did not return timestep logits")
        predictions = trace.argmax(dim=2)
        correct += (predictions == labels.unsqueeze(0)).sum(dim=1).cpu().numpy()
        total += labels.shape[0]
    return (correct / total).tolist()


def bptt_summary(
    result: TrainingResult,
    model: TwoMoonsSNN,
    validation_accuracy_by_timestep: list[float],
) -> dict[str, Any]:
    return {
        "model": result.model_name,
        "training_method": "SG-BPTT",
        "encoder_strategy": "jointly-trained-by-SG-BPTT",
        "timesteps": model.timesteps,
        "neurons_evaluated_per_timestep": model.hidden_features,
        "fit_wall_seconds": result.fit_wall_seconds,
        "best_epoch": result.best_epoch,
        "validation_accuracy_by_timestep": validation_accuracy_by_timestep,
        "validation": asdict(result.best_validation),
        "test": asdict(result.test),
        "checkpoint": str(result.best_checkpoint),
    }


def temporal_gtf_summary(
    result: GTFTrainingResult,
    *,
    encoder_strategy: str,
) -> dict[str, Any]:
    summary = gtf_result_summary(result)
    summary["training_method"] = "Temporal-GTF"
    summary["encoder_strategy"] = encoder_strategy
    summary["timesteps"] = result.best_stage
    summary["neurons_evaluated_per_timestep"] = 1
    summary["validation_accuracy_by_timestep"] = [
        row["validation_accuracy"] for row in result.history
    ]
    return summary


def plot_results(summaries: list[dict[str, Any]], output_path: Path) -> None:
    label_names = {
        "binary-bptt": "Bin\nBPTT",
        "ternary-bptt": "Ter\nBPTT",
        "binary-temporal-fixed_random": "Bin T-GTF\nfixed",
        "binary-temporal-bptt_pretrained": "Bin T-GTF\npretrained",
        "ternary-temporal-fixed_random": "Ter T-GTF\nfixed",
        "ternary-temporal-bptt_pretrained": "Ter T-GTF\npretrained",
    }
    labels = [label_names[row["model"]] for row in summaries]
    accuracy = [row["test"]["accuracy"] for row in summaries]
    fit_time = [row["fit_wall_seconds"] for row in summaries]

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].bar(labels, accuracy)
    axes[0, 0].set_ylim(max(0.5, min(accuracy) - 0.1), 1.0)
    axes[0, 0].set(title="Test accuracy", ylabel="Accuracy")
    axes[0, 1].bar(labels, fit_time)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set(title="Fit wall time (log scale)", ylabel="Seconds")

    for spike_kind, axis in (("binary", axes[1, 0]), ("ternary", axes[1, 1])):
        maximum_steps = 0
        for summary in summaries:
            if not summary["model"].startswith(spike_kind):
                continue
            curve = summary["validation_accuracy_by_timestep"]
            maximum_steps = max(maximum_steps, len(curve))
            axis.plot(
                range(1, len(curve) + 1),
                curve,
                marker="o",
                markersize=3,
                label=label_names[summary["model"]].replace("\n", " "),
            )
        axis.set(
            title=f"{spike_kind.capitalize()} validation accuracy by timestep",
            xlabel="Physical timestep",
            ylabel="Accuracy",
            xticks=range(1, maximum_steps + 1),
        )
        axis.legend()

    figure.suptitle("Two moons: recurrent Temporal GTF vs SG-BPTT")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def print_results(summaries: list[dict[str, Any]]) -> None:
    print("\nTemporal training comparison")
    print("model                              test_acc  loss     fit_s   steps  spikes/sample")
    for summary in summaries:
        test = summary["test"]
        print(
            f"{summary['model']:<34} "
            f"{test['accuracy']:.4f}    {test['loss']:.4f}   "
            f"{summary['fit_wall_seconds']:.3f}   "
            f"{summary['timesteps']:<5}  {test['active_spikes_per_sample']:.1f}"
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["runtime"]["device"] = args.device
    if args.bptt_epochs is not None:
        config["bptt_training"]["epochs"] = args.bptt_epochs
    if args.timesteps is not None:
        config["bptt_model"]["timesteps"] = args.timesteps
        config["temporal_gtf"]["timesteps"] = args.timesteps

    seed = int(config["reproducibility"]["seed"])
    deterministic = bool(config["reproducibility"]["deterministic"])
    device = resolve_device(str(config["runtime"]["device"]))
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M_temporal-gtf")
    run_directory = Path(config["project"]["run_root"]) / run_name
    checkpoint_root = Path(config["project"]["checkpoint_root"]) / run_name
    plot_path = Path(config["project"]["plot_root"]) / f"{run_name}.png"
    run_directory.mkdir(parents=True, exist_ok=False)

    data = make_two_moons_data(**config["data"], seed=seed)
    train_features, train_labels = numpy_split(data.train)
    validation_features, validation_labels = numpy_split(data.validation)
    test_features, test_labels = numpy_split(data.test)
    batch_size = int(config["bptt_training"]["batch_size"])

    bptt_models: dict[str, TwoMoonsSNN] = {}
    summaries: list[dict[str, Any]] = []
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
        bptt_models[spike_kind] = model
        summaries.append(
            bptt_summary(
                result,
                model,
                bptt_accuracy_by_timestep(
                    model,
                    data.validation,
                    batch_size=batch_size,
                    device=device,
                ),
            )
        )

    for spike_kind in ("binary", "ternary"):
        for encoder_strategy in config["encoder_strategies"]:
            print(f"Training {spike_kind} Temporal GTF ({encoder_strategy})...")
            model = TemporalGTFClassifier(
                spike_kind=spike_kind,
                seed=seed,
                **config["temporal_gtf"],
            )
            if encoder_strategy == "bptt_pretrained":
                pretrained = bptt_models[spike_kind].encoder
                model.set_encoder(
                    pretrained.weight.detach().cpu().numpy().T,
                    pretrained.bias.detach().cpu().numpy(),
                )
            model_name = f"{spike_kind}-temporal-{encoder_strategy}"
            result = fit_gtf_model(
                model=model,
                model_name=model_name,
                train_features=train_features,
                train_labels=train_labels,
                validation_features=validation_features,
                validation_labels=validation_labels,
                test_features=test_features,
                test_labels=test_labels,
                data_metadata=data.metadata,
                checkpoint_path=checkpoint_root / model_name / "best.pt",
            )
            write_history(run_directory / f"{model_name}_history.csv", result.history)
            summaries.append(
                temporal_gtf_summary(result, encoder_strategy=encoder_strategy)
            )

    output = {
        "run_name": run_name,
        "device": str(device),
        "seed": seed,
        "comparison_scope": (
            "All models use the same physical timestep limit. SG-BPTT evaluates all 64 "
            "hidden neurons per timestep; Temporal GTF greedily selects one neuron-threshold "
            "basis per timestep. Hybrid GTF fit time excludes BPTT encoder pretraining."
        ),
        "models": summaries,
    }
    with (run_directory / "summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=2)
    with (run_directory / "resolved_config.yaml").open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(config, output_file, sort_keys=False)
    plot_results(summaries, plot_path)
    print_results(summaries)
    print(f"Summary: {run_directory / 'summary.json'}")
    print(f"Plot: {plot_path}")
    print(f"Checkpoints: {checkpoint_root}")


if __name__ == "__main__":
    main()
