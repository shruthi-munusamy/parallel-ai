"""
Temperature calibration tests.

These cover the pure math (ECE, temperature fitting) with synthetic logits, not
the data-loading path (which needs a network call to Hugging Face and is
exercised manually via `python -m backend.training.calibrate_temperature`).
"""

import torch

from backend.training.calibrate_temperature import (
    _expected_calibration_error,
    _fit_temperature,
)


class TestExpectedCalibrationError:
    def test_perfect_calibration_is_zero(self):
        """Confidence exactly matching accuracy in every bin should score ~0."""
        # Two classes, 100 examples each at confidence 0.9 and 0.5, each bucket's
        # accuracy set to equal its confidence.
        probs = torch.tensor(
            [[0.9, 0.1]] * 90 + [[0.1, 0.9]] * 10  # 90% confident-and-correct at 0.9
            + [[0.5, 0.5]] * 100
        ).numpy()
        targets = torch.tensor([0] * 90 + [0] * 10 + [0] * 50 + [1] * 50).numpy()
        ece = _expected_calibration_error(probs, targets)
        assert ece < 0.05

    def test_overconfident_predictions_score_high(self):
        """Maximally confident but wrong predictions should score near 1."""
        probs = torch.tensor([[0.99, 0.01]] * 50).numpy()
        targets = torch.tensor([1] * 50).numpy()  # always the other class
        ece = _expected_calibration_error(probs, targets)
        assert ece > 0.9


class TestFitTemperature:
    def test_argmax_unchanged_by_scaling(self):
        """Temperature scaling must not change which class wins."""
        torch.manual_seed(0)
        logits = torch.randn(200, 3) * 5  # exaggerated, overconfident logits
        targets = logits.argmax(dim=-1)  # perfectly separable, so argmax = target

        temperature = _fit_temperature(logits, targets)
        scaled = logits / temperature

        assert torch.equal(logits.argmax(dim=-1), scaled.argmax(dim=-1))

    def test_overconfident_logits_fit_temperature_above_one(self):
        """Exaggerated logits with real label noise should soften (T > 1)."""
        torch.manual_seed(0)
        n = 500
        true_logits = torch.randn(n, 3) * 8  # very peaked softmax
        targets = true_logits.argmax(dim=-1).clone()
        # Inject label noise so the model's peaked confidence is miscalibrated.
        noise_idx = torch.randperm(n)[: n // 4]
        targets[noise_idx] = (targets[noise_idx] + 1) % 3

        temperature = _fit_temperature(true_logits, targets)
        assert temperature > 1.0
