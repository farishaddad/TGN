"""
Tests for PRAGMATimeEncoder (Revolut, arXiv:2604.08649, 2026).

Validates:
  - Output shape correctness
  - Gradient flow through the encoder
  - Log-gap monotonicity (larger gaps → larger log-gap values)
  - Calendar features are periodic (same time-of-day → same features)
"""

import math

import torch
import pytest

from tgn_learn.model.time_encoder import PRAGMATimeEncoder


class TestPRAGMAOutputShape:
    """Verify output dimensions for various inputs."""

    @pytest.fixture
    def encoder(self):
        return PRAGMATimeEncoder(time_dim=32)

    def test_1d_input(self, encoder):
        delta_t = torch.tensor([1.0, 10.0, 100.0])
        out = encoder(delta_t)
        assert out.shape == (3, 32)

    def test_2d_input(self, encoder):
        delta_t = torch.tensor([[1.0], [10.0], [100.0]])
        out = encoder(delta_t)
        assert out.shape == (3, 32)

    def test_with_t_abs(self, encoder):
        delta_t = torch.tensor([5.0, 50.0])
        t_abs = torch.tensor([1700000000.0, 1700003600.0])
        out = encoder(delta_t, t_abs=t_abs)
        assert out.shape == (2, 32)

    def test_without_t_abs(self, encoder):
        delta_t = torch.tensor([5.0, 50.0])
        out = encoder(delta_t, t_abs=None)
        assert out.shape == (2, 32)

    def test_single_element(self, encoder):
        delta_t = torch.tensor([42.0])
        out = encoder(delta_t)
        assert out.shape == (1, 32)

    def test_large_batch(self, encoder):
        delta_t = torch.randn(256).abs()
        out = encoder(delta_t)
        assert out.shape == (256, 32)

    @pytest.mark.parametrize("time_dim", [8, 16, 64, 128])
    def test_various_time_dims(self, time_dim):
        enc = PRAGMATimeEncoder(time_dim=time_dim)
        out = enc(torch.tensor([1.0, 2.0, 3.0]))
        assert out.shape == (3, time_dim)


class TestPRAGMAGradientFlow:
    """Verify that gradients flow back through the encoder."""

    def test_gradient_flows_through_proj(self):
        encoder = PRAGMATimeEncoder(time_dim=32)
        delta_t = torch.tensor([1.0, 10.0, 100.0])
        t_abs = torch.tensor([1700000000.0, 1700003600.0, 1700007200.0])

        out = encoder(delta_t, t_abs=t_abs)
        loss = out.sum()
        loss.backward()

        # proj layer should have gradients
        assert encoder.proj.weight.grad is not None
        assert encoder.proj.weight.grad.abs().sum() > 0

    def test_gradient_flows_without_t_abs(self):
        encoder = PRAGMATimeEncoder(time_dim=16)
        delta_t = torch.tensor([5.0, 50.0, 500.0])

        out = encoder(delta_t, t_abs=None)
        loss = out.sum()
        loss.backward()

        assert encoder.proj.weight.grad is not None
        assert encoder.proj.weight.grad.abs().sum() > 0


class TestLogGapMonotonicity:
    """Verify that the log-gap transform is monotonically increasing."""

    def test_monotonic_increasing(self):
        """Larger time gaps should produce larger log-gap values."""
        gaps = torch.tensor([0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0])
        log_gaps = PRAGMATimeEncoder._log_gap(gaps)

        # Should be strictly increasing
        for i in range(len(log_gaps) - 1):
            assert log_gaps[i + 1] > log_gaps[i], (
                f"Log-gap not monotonic: f({gaps[i+1]}) = {log_gaps[i+1]} "
                f"<= f({gaps[i]}) = {log_gaps[i]}"
            )

    def test_zero_gap_is_zero(self):
        """Zero time gap should produce zero log-gap."""
        zero = torch.tensor([0.0])
        assert PRAGMATimeEncoder._log_gap(zero).item() == pytest.approx(0.0, abs=1e-6)

    def test_compression_at_scale(self):
        """Large gaps should be compressed (sublinear growth)."""
        small = PRAGMATimeEncoder._log_gap(torch.tensor([100.0])).item()
        large = PRAGMATimeEncoder._log_gap(torch.tensor([100000.0])).item()
        # 1000x input increase should NOT produce 1000x output increase
        ratio = large / small
        assert ratio < 100, f"Not compressing large gaps: ratio={ratio}"

    def test_negative_input_handled(self):
        """Negative deltas should be handled gracefully (abs applied)."""
        neg = torch.tensor([-10.0, -100.0])
        pos = torch.tensor([10.0, 100.0])
        neg_out = PRAGMATimeEncoder._log_gap(neg)
        pos_out = PRAGMATimeEncoder._log_gap(pos)
        assert torch.allclose(neg_out, pos_out)


class TestCalendarFeaturesPeriodic:
    """Verify calendar features have correct periodicity."""

    def test_24h_periodicity(self):
        """Same time-of-day on different days should give same hour features."""
        day1_noon = torch.tensor([1700006400.0])  # some noon
        day2_noon = torch.tensor([1700006400.0 + 86400.0])  # next day noon

        cal1 = PRAGMATimeEncoder._calendar_features(day1_noon)
        cal2 = PRAGMATimeEncoder._calendar_features(day2_noon)

        # Hour features (first 2) should be identical
        assert torch.allclose(cal1[:, :2], cal2[:, :2], atol=1e-5)

    def test_weekly_periodicity(self):
        """Same time next week should give same day-of-week features."""
        t1 = torch.tensor([1700000000.0])
        t2 = torch.tensor([1700000000.0 + 604800.0])  # +1 week

        cal1 = PRAGMATimeEncoder._calendar_features(t1)
        cal2 = PRAGMATimeEncoder._calendar_features(t2)

        # DoW features (indices 2,3) should be identical
        assert torch.allclose(cal1[:, 2:4], cal2[:, 2:4], atol=1e-5)

    def test_monthly_periodicity(self):
        """Same time next month (30-day cycle) should give same DoM features."""
        t1 = torch.tensor([1700000000.0])
        t2 = torch.tensor([1700000000.0 + 2592000.0])  # +30 days

        cal1 = PRAGMATimeEncoder._calendar_features(t1)
        cal2 = PRAGMATimeEncoder._calendar_features(t2)

        # DoM features (indices 4,5) should be identical
        assert torch.allclose(cal1[:, 4:6], cal2[:, 4:6], atol=1e-5)

    def test_features_bounded(self):
        """All calendar features should be in [-1, 1]."""
        timestamps = torch.linspace(1700000000.0, 1700100000.0, 100)
        cal = PRAGMATimeEncoder._calendar_features(timestamps)
        assert cal.min() >= -1.0
        assert cal.max() <= 1.0

    def test_output_shape(self):
        """Calendar features should be [batch, 6]."""
        t = torch.tensor([1700000000.0, 1700050000.0, 1700100000.0])
        cal = PRAGMATimeEncoder._calendar_features(t)
        assert cal.shape == (3, 6)
