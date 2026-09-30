"""Dataset utilities used by GTF experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset


@dataclass(frozen=True)
class TwoMoonsData:
    """Train, validation, and test tensors with preprocessing metadata."""

    train: TensorDataset
    validation: TensorDataset
    test: TensorDataset
    metadata: dict[str, Any]


def make_two_moons_data(
    *,
    num_samples: int,
    noise: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> TwoMoonsData:
    """Create deterministic stratified splits and fit scaling on training data only."""
    if num_samples < 10:
        raise ValueError("num_samples must be at least 10")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be less than 1")

    features, labels = make_moons(n_samples=num_samples, noise=noise, random_state=seed)
    indices = np.arange(num_samples)

    train_indices, holdout_indices = train_test_split(
        indices,
        test_size=validation_fraction + test_fraction,
        random_state=seed,
        stratify=labels,
    )
    relative_test_fraction = test_fraction / (validation_fraction + test_fraction)
    validation_indices, test_indices = train_test_split(
        holdout_indices,
        test_size=relative_test_fraction,
        random_state=seed + 1,
        stratify=labels[holdout_indices],
    )

    scaler = StandardScaler().fit(features[train_indices])
    scaled_features = scaler.transform(features).astype(np.float32)
    labels = labels.astype(np.int64)

    def tensor_dataset(selected_indices: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.from_numpy(scaled_features[selected_indices]),
            torch.from_numpy(labels[selected_indices]),
        )

    metadata: dict[str, Any] = {
        "num_samples": num_samples,
        "noise": noise,
        "seed": seed,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "train_indices": train_indices.tolist(),
        "validation_indices": validation_indices.tolist(),
        "test_indices": test_indices.tolist(),
    }

    return TwoMoonsData(
        train=tensor_dataset(train_indices),
        validation=tensor_dataset(validation_indices),
        test=tensor_dataset(test_indices),
        metadata=metadata,
    )
