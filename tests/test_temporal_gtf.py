import numpy as np

from gtf.data import make_two_moons_data
from gtf.temporal_gtf import TemporalGTFClassifier


def _numpy_split(dataset: object) -> tuple[np.ndarray, np.ndarray]:
    features, labels = dataset.tensors  # type: ignore[attr-defined]
    return features.numpy(), labels.numpy()


def test_temporal_gtf_reduces_loss_with_real_membrane_steps() -> None:
    data = make_two_moons_data(
        num_samples=300,
        noise=0.15,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=11,
    )
    train_features, train_labels = _numpy_split(data.train)
    validation_features, validation_labels = _numpy_split(data.validation)
    model = TemporalGTFClassifier(
        hidden_features=24,
        timesteps=8,
        thresholds_per_feature=12,
        spike_kind="ternary",
        seed=11,
    )

    initial_loss = model.evaluate(validation_features, validation_labels).loss
    model.fit(train_features, train_labels, validation_features, validation_labels)
    final_metrics = model.evaluate(validation_features, validation_labels)
    trace = model.logits_by_timestep(validation_features)
    train_losses = [float(row["train_loss"]) for row in model.history]

    assert 0 < model.selected_neurons.size <= 8
    assert trace.shape == (model.selected_neurons.size, validation_labels.size, 2)
    assert final_metrics.loss < initial_loss
    assert final_metrics.accuracy >= 0.75
    assert all(
        current < previous
        for previous, current in zip(train_losses, train_losses[1:], strict=False)
    )


def test_signed_soft_reset_changes_next_membrane_state() -> None:
    model = TemporalGTFClassifier(
        input_features=1,
        hidden_features=1,
        num_classes=2,
        timesteps=2,
        beta=1.0 - 1e-6,
        spike_kind="ternary",
    )
    model.set_encoder(np.asarray([[0.0]]), np.asarray([2.0]))
    model.selected_neurons = np.asarray([0, 0])
    model.selected_thresholds = np.asarray([1.0, 3.5])
    model.coefficients = np.asarray([[1.0, -1.0], [1.0, -1.0]])

    trace = model.logits_by_timestep(np.zeros((1, 1)))

    # First positive spike resets 2.0 to 1.0. The next pre-reset membrane is
    # approximately 3.0, below 3.5, so the second timestep stays silent.
    assert np.allclose(trace[0, 0], [1.0, -1.0])
    assert np.allclose(trace[1, 0], [1.0, -1.0])
