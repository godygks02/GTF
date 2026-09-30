"""Temporal GTF with recurrent membrane state and signed soft reset."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gtf.gtf import (
    GTFClassifier,
    GTFMetrics,
    GTFSpikeKind,
    _accuracy,
    _cross_entropy,
    _one_hot,
    _softmax,
)


class TemporalGTFClassifier(GTFClassifier):
    """Select one residual-correcting spike basis at each real membrane timestep."""

    def __init__(
        self,
        *,
        input_features: int = 2,
        hidden_features: int = 64,
        num_classes: int = 2,
        timesteps: int = 16,
        beta: float = 0.9,
        thresholds_per_feature: int = 32,
        ridge: float = 1e-3,
        sparsity_penalty: float = 0.0,
        shrinkage: float = 1.0,
        line_search_steps: int = 8,
        minimum_improvement: float = 1e-8,
        spike_kind: GTFSpikeKind = "ternary",
        seed: int = 42,
    ) -> None:
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1)")
        super().__init__(
            input_features=input_features,
            hidden_features=hidden_features,
            num_classes=num_classes,
            max_stages=timesteps,
            thresholds_per_feature=thresholds_per_feature,
            ridge=ridge,
            sparsity_penalty=sparsity_penalty,
            shrinkage=shrinkage,
            line_search_steps=line_search_steps,
            minimum_improvement=minimum_improvement,
            spike_kind=spike_kind,
            seed=seed,
        )
        self.timesteps = timesteps
        self.beta = beta

    def _run_temporal_dynamics(
        self,
        features: np.ndarray,
        *,
        return_trace: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        batch_size = features.shape[0]
        current = self._membrane(features)
        membrane = np.zeros_like(current)
        logits = np.broadcast_to(self.intercept, (batch_size, self.num_classes)).copy()
        spikes: list[np.ndarray] = []
        trace: list[np.ndarray] = []

        for neuron, threshold, coefficient in zip(
            self.selected_neurons,
            self.selected_thresholds,
            self.coefficients,
            strict=True,
        ):
            pre_reset = self.beta * membrane + current
            spike = self._spike(pre_reset[:, neuron], float(threshold))
            membrane = pre_reset
            membrane[:, neuron] -= threshold * spike
            logits += np.outer(spike, coefficient)
            spikes.append(spike)
            if return_trace:
                trace.append(logits.copy())

        spike_matrix = (
            np.column_stack(spikes) if spikes else np.empty((batch_size, 0), dtype=np.float64)
        )
        timestep_logits = np.stack(trace) if trace else None
        return logits, spike_matrix, timestep_logits

    def logits(self, features: np.ndarray) -> np.ndarray:
        """Replay learned temporal membrane dynamics and return final logits."""
        logits, _, _ = self._run_temporal_dynamics(features, return_trace=False)
        return logits

    def logits_by_timestep(self, features: np.ndarray) -> np.ndarray:
        """Return cumulative logits after each selected physical timestep."""
        _, _, trace = self._run_temporal_dynamics(features, return_trace=True)
        if trace is None:
            return np.empty((0, features.shape[0], self.num_classes), dtype=np.float64)
        return trace

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> GTFMetrics:
        logits, spikes, _ = self._run_temporal_dynamics(features, return_trace=False)
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
        """Fit one basis per timestep while carrying membrane reset state forward."""
        fit_start = time.perf_counter()
        class_counts = np.bincount(train_labels, minlength=self.num_classes).astype(np.float64)
        class_probabilities = (class_counts + 1.0) / (train_labels.size + self.num_classes)
        self.intercept = np.log(class_probabilities)

        train_current = self._membrane(train_features)
        validation_current = self._membrane(validation_features)
        train_membrane = np.zeros_like(train_current)
        validation_membrane = np.zeros_like(validation_current)
        train_logits = np.broadcast_to(
            self.intercept,
            (train_labels.size, self.num_classes),
        ).copy()
        validation_logits = np.broadcast_to(
            self.intercept,
            (validation_labels.size, self.num_classes),
        ).copy()
        train_target = _one_hot(train_labels, self.num_classes)

        selected_neurons: list[int] = []
        selected_thresholds: list[float] = []
        selected_coefficients: list[np.ndarray] = []
        self.history = []
        current_loss = _cross_entropy(train_logits, train_labels)
        best_validation_loss = _cross_entropy(validation_logits, validation_labels)
        best_timestep = 0
        dictionary_seconds = 0.0
        selection_seconds = 0.0
        candidate_atoms_evaluated = 0

        for timestep in range(1, self.timesteps + 1):
            train_pre_reset = self.beta * train_membrane + train_current
            validation_pre_reset = self.beta * validation_membrane + validation_current

            dictionary_start = time.perf_counter()
            dictionary, neurons, thresholds = self._build_dictionary(train_pre_reset)
            dictionary_seconds += time.perf_counter() - dictionary_start
            candidate_atoms_evaluated += dictionary.shape[1]

            selection_start = time.perf_counter()
            residual = train_target - _softmax(train_logits)
            correlations = dictionary.T @ residual
            denominators = np.square(dictionary).sum(axis=0) + self.ridge
            activity = np.count_nonzero(dictionary, axis=0) / train_labels.size
            gains = np.square(correlations).sum(axis=1) / denominators
            gains -= self.sparsity_penalty * activity
            candidate_order = np.argsort(gains)[::-1]

            accepted_index: int | None = None
            accepted_scale = 0.0
            accepted_loss = current_loss
            accepted_coefficient: np.ndarray | None = None
            for candidate_index in candidate_order:
                if gains[candidate_index] <= 0.0:
                    break
                atom = dictionary[:, candidate_index]
                raw_coefficient = correlations[candidate_index] / denominators[candidate_index]
                for line_search_index in range(self.line_search_steps):
                    scale = self.shrinkage * (0.5**line_search_index)
                    candidate_logits = train_logits + scale * np.outer(atom, raw_coefficient)
                    candidate_loss = _cross_entropy(candidate_logits, train_labels)
                    if candidate_loss < current_loss - self.minimum_improvement:
                        accepted_index = int(candidate_index)
                        accepted_scale = scale
                        accepted_loss = candidate_loss
                        accepted_coefficient = scale * raw_coefficient
                        break
                if accepted_index is not None:
                    break
            selection_seconds += time.perf_counter() - selection_start

            if accepted_index is None or accepted_coefficient is None:
                break

            neuron = int(neurons[accepted_index])
            threshold = float(thresholds[accepted_index])
            train_spike = dictionary[:, accepted_index]
            validation_spike = self._spike(validation_pre_reset[:, neuron], threshold)

            train_membrane = train_pre_reset
            validation_membrane = validation_pre_reset
            train_membrane[:, neuron] -= threshold * train_spike
            validation_membrane[:, neuron] -= threshold * validation_spike
            train_logits += np.outer(train_spike, accepted_coefficient)
            validation_logits += np.outer(validation_spike, accepted_coefficient)
            current_loss = accepted_loss

            selected_neurons.append(neuron)
            selected_thresholds.append(threshold)
            selected_coefficients.append(accepted_coefficient)
            validation_loss = _cross_entropy(validation_logits, validation_labels)
            self.history.append(
                {
                    "timestep": timestep,
                    "train_loss": current_loss,
                    "train_accuracy": _accuracy(train_logits, train_labels),
                    "validation_loss": validation_loss,
                    "validation_accuracy": _accuracy(validation_logits, validation_labels),
                    "gain_score": float(gains[accepted_index]),
                    "line_search_scale": accepted_scale,
                    "active_fraction": float(activity[accepted_index]),
                    "selected_neuron": neuron,
                    "threshold": threshold,
                }
            )
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_timestep = timestep

        if best_timestep == 0 and selected_neurons:
            best_timestep = 1
        self.selected_neurons = np.asarray(selected_neurons[:best_timestep], dtype=np.int64)
        self.selected_thresholds = np.asarray(
            selected_thresholds[:best_timestep],
            dtype=np.float64,
        )
        self.coefficients = (
            np.vstack(selected_coefficients[:best_timestep])
            if best_timestep
            else np.empty((0, self.num_classes), dtype=np.float64)
        )
        self.history = self.history[:best_timestep]

        return {
            "candidate_atoms": candidate_atoms_evaluated,
            "stages_attempted": len(selected_neurons),
            "best_stage": best_timestep,
            "dictionary_seconds": dictionary_seconds,
            "selection_seconds": selection_seconds,
            "fit_wall_seconds": time.perf_counter() - fit_start,
        }

    def config(self) -> dict[str, int | float | str]:
        config = super().config()
        config["timesteps"] = self.timesteps
        config["beta"] = self.beta
        config.pop("max_stages", None)
        return config

    def save_checkpoint(
        self,
        path: Path,
        *,
        data_metadata: dict[str, Any],
        fit_metadata: dict[str, float | int],
    ) -> None:
        """Save temporal dynamics, selected bases, and decoder state."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "model_type": "TemporalGTFClassifier",
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
