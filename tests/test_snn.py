import pytest

torch = pytest.importorskip("torch")

from gtf.snn import TwoMoonsSNN, surrogate_step  # noqa: E402


def test_surrogate_step_has_nonzero_local_gradient() -> None:
    membrane_distance = torch.tensor([-0.1, 0.0, 0.1], requires_grad=True)
    surrogate_step(membrane_distance, slope=5.0).sum().backward()

    assert membrane_distance.grad is not None
    assert torch.all(membrane_distance.grad > 0)


@pytest.mark.parametrize(
    ("spike_kind", "allowed_values"),
    [("binary", {0.0, 1.0}), ("ternary", {-1.0, 0.0, 1.0})],
)
def test_snn_forward_shape_and_spike_alphabet(
    spike_kind: str,
    allowed_values: set[float],
) -> None:
    model = TwoMoonsSNN(hidden_features=8, timesteps=4, spike_kind=spike_kind)
    features = torch.randn(5, 2)
    output = model(features, return_timestep_logits=True)

    assert output.logits.shape == (5, 2)
    assert output.timestep_logits is not None
    assert output.timestep_logits.shape == (4, 5, 2)
    membrane = torch.tensor([[-2.0, -0.5, 0.5, 2.0]])
    spikes = model._spike(membrane)
    assert set(spikes.flatten().tolist()).issubset(allowed_values)


def test_ternary_snn_emits_negative_spikes() -> None:
    model = TwoMoonsSNN(hidden_features=2, timesteps=2, spike_kind="ternary")
    with torch.no_grad():
        model.encoder.weight.zero_()
        model.encoder.bias.copy_(torch.tensor([-2.0, 2.0]))

    output = model(torch.zeros(3, 2))

    assert output.negative_spikes.item() > 0
    assert output.positive_spikes.item() > 0
