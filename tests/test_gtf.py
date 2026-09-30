import numpy as np

from gtf.data import make_two_moons_data
from gtf.gtf import GTFClassifier


def _numpy_split(dataset: object) -> tuple[np.ndarray, np.ndarray]:
    features, labels = dataset.tensors  # type: ignore[attr-defined]
    return features.numpy(), labels.numpy()


def test_gtf_reduces_cross_entropy_on_two_moons() -> None:
    data = make_two_moons_data(
        num_samples=300,
        noise=0.15,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=7,
    )
    train_features, train_labels = _numpy_split(data.train)
    validation_features, validation_labels = _numpy_split(data.validation)
    model = GTFClassifier(
        hidden_features=24,
        max_stages=12,
        thresholds_per_feature=12,
        spike_kind="ternary",
        seed=7,
    )

    initial_loss = model.evaluate(validation_features, validation_labels).loss
    model.fit(train_features, train_labels, validation_features, validation_labels)
    final_metrics = model.evaluate(validation_features, validation_labels)
    train_losses = [float(row["train_loss"]) for row in model.history]

    assert model.selected_neurons.size > 0
    assert final_metrics.loss < initial_loss
    assert final_metrics.accuracy >= 0.75
    assert all(
        current < previous
        for previous, current in zip(train_losses, train_losses[1:], strict=False)
    )


def test_binary_and_ternary_gtf_emit_expected_alphabets() -> None:
    membrane = np.asarray([-2.0, -0.5, 0.5, 2.0])
    binary = GTFClassifier(spike_kind="binary")
    ternary = GTFClassifier(spike_kind="ternary")

    assert set(binary._spike(membrane, 1.0)) <= {0.0, 1.0}
    assert set(ternary._spike(membrane, 1.0)) <= {-1.0, 0.0, 1.0}
