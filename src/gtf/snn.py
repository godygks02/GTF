"""Small binary and ternary spiking neural networks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

SpikeKind = Literal["binary", "ternary"]


class _SurrogateStep(torch.autograd.Function):
    """Heaviside forward pass with a fast-sigmoid surrogate derivative."""

    @staticmethod
    def forward(ctx: object, membrane_distance: Tensor, slope: float) -> Tensor:
        ctx.save_for_backward(membrane_distance)  # type: ignore[attr-defined]
        ctx.slope = slope  # type: ignore[attr-defined]
        return (membrane_distance >= 0.0).to(membrane_distance.dtype)

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> tuple[Tensor, None]:
        (membrane_distance,) = ctx.saved_tensors  # type: ignore[attr-defined]
        slope = ctx.slope  # type: ignore[attr-defined]
        surrogate = slope / (1.0 + slope * membrane_distance.abs()).pow(2)
        return grad_output * surrogate, None


def surrogate_step(membrane_distance: Tensor, slope: float) -> Tensor:
    """Apply a hard step in forward and a smooth local derivative in backward."""
    return _SurrogateStep.apply(membrane_distance, slope)


@dataclass(frozen=True)
class SNNOutput:
    """Network output and spike counts accumulated over all timesteps."""

    logits: Tensor
    positive_spikes: Tensor
    negative_spikes: Tensor
    active_spikes: Tensor
    timestep_logits: Tensor | None = None


class TwoMoonsSNN(nn.Module):
    """One-hidden-layer SNN for static two-dimensional classification."""

    def __init__(
        self,
        *,
        input_features: int = 2,
        hidden_features: int = 64,
        num_classes: int = 2,
        timesteps: int = 16,
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_slope: float = 5.0,
        spike_kind: SpikeKind = "binary",
    ) -> None:
        super().__init__()
        if spike_kind not in {"binary", "ternary"}:
            raise ValueError(f"Unsupported spike kind: {spike_kind}")
        if timesteps < 1:
            raise ValueError("timesteps must be positive")
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must be in [0, 1)")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive")

        self.encoder = nn.Linear(input_features, hidden_features)
        self.readout = nn.Linear(hidden_features, num_classes)
        self.hidden_features = hidden_features
        self.timesteps = timesteps
        self.beta = beta
        self.threshold = threshold
        self.surrogate_slope = surrogate_slope
        self.spike_kind = spike_kind

    def _spike(self, membrane: Tensor) -> Tensor:
        positive = surrogate_step(membrane - self.threshold, self.surrogate_slope)
        if self.spike_kind == "binary":
            return positive
        negative = surrogate_step(-membrane - self.threshold, self.surrogate_slope)
        return positive - negative

    def forward(self, features: Tensor, *, return_timestep_logits: bool = False) -> SNNOutput:
        """Unroll membrane dynamics and accumulate readout logits through time."""
        batch_size = features.shape[0]
        current = self.encoder(features)
        membrane = torch.zeros_like(current)
        logits = torch.zeros(
            (batch_size, self.readout.out_features),
            device=features.device,
            dtype=features.dtype,
        )
        positive_spikes = torch.zeros((), device=features.device)
        negative_spikes = torch.zeros((), device=features.device)
        timestep_logits: list[Tensor] = []

        for timestep in range(self.timesteps):
            membrane = self.beta * membrane + current
            spikes = self._spike(membrane)

            # Detaching the reset path is common in SG-BPTT and prevents the hard
            # reset operation from adding a second surrogate-gradient path.
            membrane = membrane - self.threshold * spikes.detach()
            logits = logits + self.readout(spikes)
            positive_spikes = positive_spikes + (spikes > 0).sum()
            negative_spikes = negative_spikes + (spikes < 0).sum()

            if return_timestep_logits:
                timestep_logits.append(logits / float(timestep + 1))

        positive_spikes = positive_spikes.to(features.dtype)
        negative_spikes = negative_spikes.to(features.dtype)
        trace = torch.stack(timestep_logits, dim=0) if timestep_logits else None
        return SNNOutput(
            logits=logits / float(self.timesteps),
            positive_spikes=positive_spikes,
            negative_spikes=negative_spikes,
            active_spikes=positive_spikes + negative_spikes,
            timestep_logits=trace,
        )

    def config(self) -> dict[str, int | float | str]:
        """Return the constructor settings needed to recreate this model."""
        return {
            "input_features": self.encoder.in_features,
            "hidden_features": self.hidden_features,
            "num_classes": self.readout.out_features,
            "timesteps": self.timesteps,
            "beta": self.beta,
            "threshold": self.threshold,
            "surrogate_slope": self.surrogate_slope,
            "spike_kind": self.spike_kind,
        }
