"""Train and compare binary and ternary SG-BPTT models on two moons."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

from gtf.data import TwoMoonsData, make_two_moons_data
from gtf.snn import TwoMoonsSNN
from gtf.training import TrainingResult, fit_model, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/two_moons_sg_bptt.yaml"),
        help="Experiment YAML file.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Optional quick-run override.")
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


def write_history(path: Path, result: TrainingResult) -> None:
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=result.history[0].keys())
        writer.writeheader()
        writer.writerows(result.history)


def result_summary(result: TrainingResult) -> dict[str, Any]:
    return {
        "model": result.model_name,
        "best_epoch": result.best_epoch,
        "epochs_completed": result.epochs_completed,
        "fit_wall_seconds": result.fit_wall_seconds,
        "train_step_seconds": result.train_step_seconds,
        "seconds_per_training_epoch": result.train_step_seconds / result.epochs_completed,
        "evaluation_seconds": result.evaluation_seconds,
        "best_validation": asdict(result.best_validation),
        "test": asdict(result.test),
        "best_checkpoint": str(result.best_checkpoint),
        "last_checkpoint": str(result.last_checkpoint),
    }


def plot_results(
    *,
    results: list[TrainingResult],
    models: dict[str, TwoMoonsSNN],
    data: TwoMoonsData,
    device: torch.device,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 9))

    for result in results:
        epochs = [row["epoch"] for row in result.history]
        axes[0, 0].plot(
            epochs,
            [row["validation_accuracy"] for row in result.history],
            label=result.model_name,
        )
        axes[0, 1].plot(
            epochs,
            [row["validation_loss"] for row in result.history],
            label=result.model_name,
        )

    axes[0, 0].set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[0, 1].set(title="Validation loss", xlabel="Epoch", ylabel="Cross entropy")
    axes[0, 0].legend()
    axes[0, 1].legend()

    split_features = [data.train.tensors[0], data.validation.tensors[0], data.test.tensors[0]]
    split_labels = [data.train.tensors[1], data.validation.tensors[1], data.test.tensors[1]]
    all_features = torch.cat(split_features)
    all_labels = torch.cat(split_labels)
    x_min, y_min = all_features.min(dim=0).values - 0.5
    x_max, y_max = all_features.max(dim=0).values + 0.5
    grid_x, grid_y = torch.meshgrid(
        torch.linspace(x_min.item(), x_max.item(), 180),
        torch.linspace(y_min.item(), y_max.item(), 180),
        indexing="xy",
    )
    grid = torch.stack([grid_x.ravel(), grid_y.ravel()], dim=1).to(device)

    for axis, result in zip(axes[1], results, strict=True):
        model = models[result.model_name]
        model.eval()
        with torch.no_grad():
            prediction = model(grid).logits.argmax(dim=1).cpu().reshape(grid_x.shape)
        axis.contourf(grid_x, grid_y, prediction, levels=(-0.5, 0.5, 1.5), alpha=0.3)
        axis.scatter(
            all_features[:, 0],
            all_features[:, 1],
            c=all_labels,
            cmap="coolwarm",
            edgecolors="white",
            linewidths=0.3,
            s=12,
        )
        axis.set_title(f"{result.model_name} decision boundary")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def print_comparison(summaries: list[dict[str, Any]]) -> None:
    print("\nSG-BPTT comparison")
    print("model     test_acc  test_loss  train_s  s/epoch  spikes/sample (+/-)")
    for summary in summaries:
        test = summary["test"]
        print(
            f"{summary['model']:<9} "
            f"{test['accuracy']:.4f}    "
            f"{test['loss']:.4f}    "
            f"{summary['train_step_seconds']:.3f}   "
            f"{summary['seconds_per_training_epoch']:.4f}   "
            f"{test['active_spikes_per_sample']:.1f} "
            f"({test['positive_spikes_per_sample']:.1f}/"
            f"{test['negative_spikes_per_sample']:.1f})"
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.device is not None:
        config["runtime"]["device"] = args.device

    seed = int(config["reproducibility"]["seed"])
    deterministic = bool(config["reproducibility"]["deterministic"])
    device = resolve_device(str(config["runtime"]["device"]))
    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M_two-moons_sg-bptt")
    run_directory = Path(config["project"]["run_root"]) / run_name
    checkpoint_root = Path(config["project"]["checkpoint_root"]) / run_name
    plot_path = Path(config["project"]["plot_root"]) / f"{run_name}.png"
    run_directory.mkdir(parents=True, exist_ok=False)

    data = make_two_moons_data(**config["data"], seed=seed)
    results: list[TrainingResult] = []
    models: dict[str, TwoMoonsSNN] = {}

    print(f"Device: {device}")
    print(f"Run directory: {run_directory}")
    for spike_kind in ("binary", "ternary"):
        seed_everything(seed, deterministic=deterministic)
        model = TwoMoonsSNN(spike_kind=spike_kind, **config["model"])
        print(f"Training {spike_kind} SNN...")
        result = fit_model(
            model=model,
            model_name=spike_kind,
            train_data=data.train,
            validation_data=data.validation,
            test_data=data.test,
            data_metadata=data.metadata,
            training_config=config["training"],
            checkpoint_directory=checkpoint_root / spike_kind,
            device=device,
            seed=seed,
        )
        results.append(result)
        models[spike_kind] = model
        write_history(run_directory / f"{spike_kind}_history.csv", result)

    summaries = [result_summary(result) for result in results]
    with (run_directory / "summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(
            {
                "run_name": run_name,
                "device": str(device),
                "torch_version": torch.__version__,
                "models": summaries,
            },
            output_file,
            indent=2,
        )
    with (run_directory / "resolved_config.yaml").open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(config, output_file, sort_keys=False)
    shutil.copy2(args.config, run_directory / "source_config.yaml")

    plot_results(
        results=results,
        models=models,
        data=data,
        device=device,
        output_path=plot_path,
    )
    print_comparison(summaries)
    print(f"Summary: {run_directory / 'summary.json'}")
    print(f"Plot: {plot_path}")
    print(f"Checkpoints: {checkpoint_root}")


if __name__ == "__main__":
    main()
