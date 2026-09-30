"""Compare fixed, BPTT-pretrained, and GTF-selected encoder features."""

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
        default=Path("configs/two_moons_gtf_encoder_strategies.yaml"),
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
        "encoder_strategy": "jointly-trained-by-SG-BPTT",
        "training_method": "SG-BPTT",
        "fit_wall_seconds": result.fit_wall_seconds,
        "best_epoch": result.best_epoch,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "validation": asdict(result.best_validation),
        "test": asdict(result.test),
        "best_checkpoint": str(result.best_checkpoint),
    }


def add_strategy_fields(
    summary: dict[str, Any],
    *,
    strategy: str,
    candidate_features: int,
) -> dict[str, Any]:
    summary["encoder_strategy"] = strategy
    summary["candidate_encoder_features"] = candidate_features
    if strategy == "gtf-projection-pool-selection":
        summary["encoder_learning_scope"] = (
            "GTF selects random projection bases and thresholds; projection weights are not "
            "continuously optimized."
        )
    elif strategy == "bptt-pretrained-frozen":
        summary["encoder_learning_scope"] = (
            "Encoder weights are pretrained by SG-BPTT, then frozen while GTF relearns "
            "threshold bases and decoder coefficients."
        )
    else:
        summary["encoder_learning_scope"] = "Random encoder weights remain fixed."
    return summary


def plot_results(summaries: list[dict[str, Any]], output_path: Path) -> None:
    label_names = {
        "binary-bptt": "Bin\nBPTT",
        "ternary-bptt": "Ter\nBPTT",
        "binary-fixed_random": "Bin\nfixed",
        "binary-bptt_pretrained": "Bin\npretrained",
        "binary-gtf_projection_pool": "Bin\npool",
        "ternary-fixed_random": "Ter\nfixed",
        "ternary-bptt_pretrained": "Ter\npretrained",
        "ternary-gtf_projection_pool": "Ter\npool",
    }
    labels = [label_names[row["model"]] for row in summaries]
    accuracy = [row["test"]["accuracy"] for row in summaries]
    loss = [row["test"]["loss"] for row in summaries]
    fit_time = [row["fit_wall_seconds"] for row in summaries]
    spikes = [row["test"]["active_spikes_per_sample"] for row in summaries]

    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0, 0].bar(labels, accuracy)
    axes[0, 0].set_ylim(max(0.5, min(accuracy) - 0.08), 1.0)
    axes[0, 0].set(title="Test accuracy", ylabel="Accuracy")
    axes[0, 1].bar(labels, loss)
    axes[0, 1].set(title="Test cross entropy", ylabel="Loss")
    axes[1, 0].bar(labels, fit_time)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(title="Fit wall time (log scale)", ylabel="Seconds")
    axes[1, 1].bar(labels, spikes)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set(title="Active spikes per sample (log scale)", ylabel="Events")
    for axis in axes.flat:
        axis.tick_params(axis="x", labelsize=9)
    figure.suptitle("Two moons: GTF encoder strategies")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def print_results(summaries: list[dict[str, Any]]) -> None:
    print("\nEncoder strategy comparison")
    print("model                       test_acc  test_loss  fit_s   basis/spikes")
    for summary in summaries:
        test = summary["test"]
        print(
            f"{summary['model']:<27} "
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
        config["gtf_common"]["max_stages"] = args.gtf_stages

    seed = int(config["reproducibility"]["seed"])
    deterministic = bool(config["reproducibility"]["deterministic"])
    device = resolve_device(str(config["runtime"]["device"]))
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M_gtf-encoder-strategies")
    run_directory = Path(config["project"]["run_root"]) / run_name
    checkpoint_root = Path(config["project"]["checkpoint_root"]) / run_name
    plot_path = Path(config["project"]["plot_root"]) / f"{run_name}.png"
    run_directory.mkdir(parents=True, exist_ok=False)

    data = make_two_moons_data(**config["data"], seed=seed)
    train_features, train_labels = numpy_split(data.train)
    validation_features, validation_labels = numpy_split(data.validation)
    test_features, test_labels = numpy_split(data.test)

    bptt_models: dict[str, TwoMoonsSNN] = {}
    summaries: list[dict[str, Any]] = []
    for spike_kind in ("binary", "ternary"):
        print(f"Training {spike_kind} SG-BPTT encoder...")
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
        summaries.append(bptt_summary(result, model))

    gtf_results: list[GTFTrainingResult] = []
    for spike_kind in ("binary", "ternary"):
        for strategy_name, strategy_config in config["encoder_strategies"].items():
            print(f"Training {spike_kind} GTF with {strategy_name} encoder...")
            model = GTFClassifier(
                spike_kind=spike_kind,
                seed=seed,
                **config["gtf_common"],
                **strategy_config,
            )
            if strategy_name == "bptt_pretrained":
                pretrained = bptt_models[spike_kind].encoder
                model.set_encoder(
                    pretrained.weight.detach().cpu().numpy().T,
                    pretrained.bias.detach().cpu().numpy(),
                )

            model_name = f"{spike_kind}-{strategy_name}"
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
            gtf_results.append(result)
            write_history(run_directory / f"{model_name}_history.csv", result.history)
            strategy_label = {
                "fixed_random": "fixed-random",
                "bptt_pretrained": "bptt-pretrained-frozen",
                "gtf_projection_pool": "gtf-projection-pool-selection",
            }[strategy_name]
            summaries.append(
                add_strategy_fields(
                    gtf_result_summary(result),
                    strategy=strategy_label,
                    candidate_features=int(strategy_config["hidden_features"]),
                )
            )

    output = {
        "run_name": run_name,
        "device": str(device),
        "seed": seed,
        "important_scope_note": (
            "Projection-pool GTF performs discrete encoder basis selection, not continuous "
            "optimization of encoder weights. BPTT-pretrained GTF is a hybrid method."
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
