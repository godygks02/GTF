"""Greedy Temporal Fitting classifiers with binary or ternary spike bases."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

GTFSpikeKind = Literal["binary", "ternary"]


@dataclass(frozen=True)
class GTFMetrics:
    """Classification and selected-basis activity metrics."""

    loss: float
    accuracy: float
    positive_spikes_per_sample: float
    negative_spikes_per_sample: float
    active_spikes_per_sample: float


@dataclass(frozen=True)
class GTFTrainingResult:
    """Result metadata returned by one stagewise GTF fit."""

    model_name: str
    stages_selected: int
    best_stage: int
    fit_wall_seconds: float
    dictionary_seconds: float
    selection_seconds: float
    validation: GTFMetrics
    test: GTFMetrics
    history: list[dict[str, float | int]]
    checkpoint: Path


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def _cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    probabilities = _softmax(logits)
    selected = probabilities[np.arange(labels.size), labels]
    return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())


def _accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((logits.argmax(axis=1) == labels).mean())


def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
    target = np.zeros((labels.size, num_classes), dtype=np.float64)
    target[np.arange(labels.size), labels] = 1.0
    return target


class GTFClassifier:
    """Stagewise classifier using thresholded fixed random membrane features.

    Each selected atom is interpreted as one temporal correction stage. The
    encoder is intentionally fixed in this first implementation so that the
    threshold selection and residual fitting mechanism can be tested directly.
    """

    def __init__(
        self,
        *,
        input_features: int = 2,
        hidden_features: int = 64,
        num_classes: int = 2,
        max_stages: int = 32,
        thresholds_per_feature: int = 32,
        ridge: float = 1e-3,
        sparsity_penalty: float = 0.0,
        shrinkage: float = 1.0,
        line_search_steps: int = 8,
        minimum_improvement: float = 1e-8,
        spike_kind: GTFSpikeKind = "ternary",
        seed: int = 42,
    ) -> None:
        if spike_kind not in {"binary", "ternary"}:
            raise ValueError(f"Unsupported spike kind: {spike_kind}")
        if max_stages < 1 or thresholds_per_feature < 1:
            raise ValueError("max_stages and thresholds_per_feature must be positive")
        if ridge < 0.0 or sparsity_penalty < 0.0:
            raise ValueError("regularization values must be non-negative")
        if not 0.0 < shrinkage <= 1.0:
            raise ValueError("shrinkage must be in (0, 1]")

        self.input_features = input_features
        self.hidden_features = hidden_features
        self.num_classes = num_classes
        self.max_stages = max_stages
        self.thresholds_per_feature = thresholds_per_feature
        self.ridge = ridge
        self.sparsity_penalty = sparsity_penalty
        self.shrinkage = shrinkage
        self.line_search_steps = line_search_steps
        self.minimum_improvement = minimum_improvement
        self.spike_kind = spike_kind
        self.seed = seed

        generator = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (input_features + hidden_features))
        self.encoder_weight = generator.normal(
            loc=0.0,
            scale=scale,
            size=(input_features, hidden_features),
        )
        self.encoder_bias = generator.uniform(-1.0, 1.0, size=hidden_features)

        self.intercept = np.zeros(num_classes, dtype=np.float64)
        self.selected_neurons = np.empty(0, dtype=np.int64)
        self.selected_thresholds = np.empty(0, dtype=np.float64)
        self.coefficients = np.empty((0, num_classes), dtype=np.float64)
        self.history: list[dict[str, float | int]] = []

    def _membrane(self, features: np.ndarray) -> np.ndarray:
        return features.astype(np.float64) @ self.encoder_weight + self.encoder_bias

    def _spike(self, membrane: np.ndarray, threshold: float) -> np.ndarray:
        positive = membrane >= threshold
        if self.spike_kind == "binary":
            return positive.astype(np.float64)
        negative = membrane <= -threshold
        return positive.astype(np.float64) - negative.astype(np.float64)

    def _threshold_candidates(self, membrane: np.ndarray) -> np.ndarray:
        magnitudes = np.abs(membrane)
        nonzero = magnitudes[magnitudes > 1e-12]
        if nonzero.size == 0:
            return np.empty(0, dtype=np.float64)
        quantiles = np.linspace(0.0, 1.0, self.thresholds_per_feature + 2)[1:-1]
        return np.unique(np.quantile(nonzero, quantiles))

    def _build_dictionary(
        self,
        membrane: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        atoms: list[np.ndarray] = []
        neurons: list[int] = []
        thresholds: list[float] = []

        for neuron in range(self.hidden_features):
            feature_membrane = membrane[:, neuron]
            seen_patterns: set[bytes] = set()
            for threshold in self._threshold_candidates(feature_membrane):
                atom = self._spike(feature_membrane, float(threshold))
                if not np.any(atom):
                    continue
                pattern_key = atom.astype(np.int8).tobytes()
                if pattern_key in seen_patterns:
                    continue
                seen_patterns.add(pattern_key)
                atoms.append(atom)
                neurons.append(neuron)
                thresholds.append(float(threshold))

        if not atoms:
            raise RuntimeError("The fixed encoder produced no nonzero spike candidates")
        return (
            np.column_stack(atoms),
            np.asarray(neurons, dtype=np.int64),
            np.asarray(thresholds, dtype=np.float64),
        )

    def _selected_spikes(self, features: np.ndarray) -> np.ndarray:
        if self.selected_neurons.size == 0:
            return np.empty((features.shape[0], 0), dtype=np.float64)
        membrane = self._membrane(features)
        atoms = [
            self._spike(membrane[:, neuron], threshold)
            for neuron, threshold in zip(
                self.selected_neurons,
                self.selected_thresholds,
                strict=True,
            )
        ]
        return np.column_stack(atoms)

    def logits(self, features: np.ndarray) -> np.ndarray:
        """Return class logits after all selected temporal stages."""
        logits = np.broadcast_to(self.intercept, (features.shape[0], self.num_classes)).copy()
        if self.selected_neurons.size:
            logits += self._selected_spikes(features) @ self.coefficients
        return logits

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> GTFMetrics:
        """Evaluate final classification and spike activity."""
        logits = self.logits(features)
        spikes = self._selected_spikes(features)
        positive = float((spikes > 0).sum())
        negative = float((spikes < 0).sum())
        return GTFMetrics(
            loss=_cross_entropy(logits, labels),
            accuracy=_accuracy(logits, labels),
            positive_spikes_per_sample=positive / labels.size,
            negative_spikes_per_sample=negative / labels.size,
            active_spikes_per_sample=(positive + negative) / labels.size,
        )

    def fit(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        validation_features: np.ndarray,
        validation_labels: np.ndarray,
    ) -> dict[str, float | int]:
        """Select spike bases greedily from the functional CE residual."""
        fit_start = time.perf_counter()
        class_counts = np.bincount(train_labels, minlength=self.num_classes).astype(np.float64)
        class_probabilities = (class_counts + 1.0) / (train_labels.size + self.num_classes)
        self.intercept = np.log(class_probabilities)

        dictionary_start = time.perf_counter()
        train_membrane = self._membrane(train_features)
        dictionary, neurons, thresholds = self._build_dictionary(train_membrane)
        dictionary_seconds = time.perf_counter() - dictionary_start

        train_target = _one_hot(train_labels, self.num_classes)
        train_logits = np.broadcast_to(
            self.intercept,
            (train_labels.size, self.num_classes),
        ).copy()
        validation_logits = np.broadcast_to(
            self.intercept,
            (validation_labels.size, self.num_classes),
        ).copy()
        validation_membrane = self._membrane(validation_features)

        selected_dictionary_indices: list[int] = []
        selected_coefficients: list[np.ndarray] = []
        available = np.ones(dictionary.shape[1], dtype=bool)
        denominators = np.square(dictionary).sum(axis=0) + self.ridge
        activity = np.count_nonzero(dictionary, axis=0) / train_labels.size
        current_loss = _cross_entropy(train_logits, train_labels)
        best_validation_loss = _cross_entropy(validation_logits, validation_labels)
        best_stage = 0
        self.history = []
        selection_start = time.perf_counter()

        for _stage in range(1, self.max_stages + 1):
            residual = train_target - _softmax(train_logits)
            correlations = dictionary.T @ residual
            gains = np.square(correlations).sum(axis=1) / denominators
            gains -= self.sparsity_penalty * activity
            gains[~available] = -np.inf
            candidate_index = int(np.argmax(gains))
            if not np.isfinite(gains[candidate_index]) or gains[candidate_index] <= 0.0:
                break

            atom = dictionary[:, candidate_index]
            raw_coefficient = correlations[candidate_index] / denominators[candidate_index]
            validation_atom = self._spike(
                validation_membrane[:, neurons[candidate_index]],
                thresholds[candidate_index],
            )

            accepted_scale = 0.0
            accepted_loss = current_loss
            for line_search_index in range(self.line_search_steps):
                scale = self.shrinkage * (0.5**line_search_index)
                candidate_logits = train_logits + scale * np.outer(atom, raw_coefficient)
                candidate_loss = _cross_entropy(candidate_logits, train_labels)
                if candidate_loss < accepted_loss - self.minimum_improvement:
                    accepted_scale = scale
                    accepted_loss = candidate_loss
                    break

            available[candidate_index] = False
            if accepted_scale == 0.0:
                continue

            coefficient = accepted_scale * raw_coefficient
            train_logits += np.outer(atom, coefficient)
            validation_logits += np.outer(validation_atom, coefficient)
            current_loss = accepted_loss
            selected_dictionary_indices.append(candidate_index)
            selected_coefficients.append(coefficient)

            validation_loss = _cross_entropy(validation_logits, validation_labels)
            self.history.append(
                {
                    "stage": len(selected_dictionary_indices),
                    "train_loss": current_loss,
                    "train_accuracy": _accuracy(train_logits, train_labels),
                    "validation_loss": validation_loss,
                    "validation_accuracy": _accuracy(validation_logits, validation_labels),
                    "gain_score": float(gains[candidate_index]),
                    "line_search_scale": accepted_scale,
                    "active_fraction": float(activity[candidate_index]),
                }
            )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_stage = len(selected_dictionary_indices)

        selection_seconds = time.perf_counter() - selection_start
        if best_stage == 0 and selected_dictionary_indices:
            best_stage = 1

        retained_indices = selected_dictionary_indices[:best_stage]
        self.selected_neurons = neurons[retained_indices]
        self.selected_thresholds = thresholds[retained_indices]
        if best_stage:
            self.coefficients = np.vstack(selected_coefficients[:best_stage])
        else:
            self.coefficients = np.empty((0, self.num_classes), dtype=np.float64)
        self.history = self.history[:best_stage]

        return {
            "candidate_atoms": dictionary.shape[1],
            "stages_attempted": len(selected_dictionary_indices),
            "best_stage": best_stage,
            "dictionary_seconds": dictionary_seconds,
            "selection_seconds": selection_seconds,
            "fit_wall_seconds": time.perf_counter() - fit_start,
        }

    def config(self) -> dict[str, int | float | str]:
        """Return all constructor settings."""
        return {
            "input_features": self.input_features,
            "hidden_features": self.hidden_features,
            "num_classes": self.num_classes,
            "max_stages": self.max_stages,
            "thresholds_per_feature": self.thresholds_per_feature,
            "ridge": self.ridge,
            "sparsity_penalty": self.sparsity_penalty,
            "shrinkage": self.shrinkage,
            "line_search_steps": self.line_search_steps,
            "minimum_improvement": self.minimum_improvement,
            "spike_kind": self.spike_kind,
            "seed": self.seed,
        }

    def save_checkpoint(
        self,
        path: Path,
        *,
        data_metadata: dict[str, Any],
        fit_metadata: dict[str, float | int],
    ) -> None:
        """Save an inference-complete GTF checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "model_type": "GTFClassifier",
                "model_config": self.config(),
                "encoder_weight": torch.from_numpy(self.encoder_weight),
                "encoder_bias": torch.from_numpy(self.encoder_bias),
                "intercept": torch.from_numpy(self.intercept),
                "selected_neurons": torch.from_numpy(self.selected_neurons),
                "selected_thresholds": torch.from_numpy(self.selected_thresholds),
                "coefficients": torch.from_numpy(self.coefficients),
                "history": self.history,
                "data_metadata": data_metadata,
                "fit_metadata": fit_metadata,
            },
            path,
        )


def fit_gtf_model(
    *,
    model: GTFClassifier,
    model_name: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    data_metadata: dict[str, Any],
    checkpoint_path: Path,
) -> GTFTrainingResult:
    """Fit, evaluate, checkpoint, and summarize one GTF model."""
    fit_metadata = model.fit(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
    )
    validation_metrics = model.evaluate(validation_features, validation_labels)
    test_metrics = model.evaluate(test_features, test_labels)
    model.save_checkpoint(
        checkpoint_path,
        data_metadata=data_metadata,
        fit_metadata=fit_metadata,
    )
    return GTFTrainingResult(
        model_name=model_name,
        stages_selected=int(fit_metadata["best_stage"]),
        best_stage=int(fit_metadata["best_stage"]),
        fit_wall_seconds=float(fit_metadata["fit_wall_seconds"]),
        dictionary_seconds=float(fit_metadata["dictionary_seconds"]),
        selection_seconds=float(fit_metadata["selection_seconds"]),
        validation=validation_metrics,
        test=test_metrics,
        history=model.history,
        checkpoint=checkpoint_path,
    )


def gtf_result_summary(result: GTFTrainingResult) -> dict[str, Any]:
    """Convert a result to a JSON-serializable dictionary."""
    return {
        "model": result.model_name,
        "training_method": "GTF-fixed-encoder",
        "stages_selected": result.stages_selected,
        "best_stage": result.best_stage,
        "fit_wall_seconds": result.fit_wall_seconds,
        "dictionary_seconds": result.dictionary_seconds,
        "selection_seconds": result.selection_seconds,
        "validation": asdict(result.validation),
        "test": asdict(result.test),
        "checkpoint": str(result.checkpoint),
    }
