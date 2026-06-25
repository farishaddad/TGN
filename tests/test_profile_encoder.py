"""
Tests for ProfileStateEncoder (PRAGMA, Revolut 2026).

Validates:
  - Output shape correctness
  - Gradient flow through the encoder
  - Different profiles produce different embeddings
"""

import torch
import pytest

from tgn_learn.model.profile_encoder import ProfileStateEncoder


class TestProfileEncoderOutputShape:
    """Verify output dimensions for various configurations."""

    def test_default_dims(self):
        enc = ProfileStateEncoder(profile_dim=6, out_dim=32)
        x = torch.randn(4, 6)
        out = enc(x)
        assert out.shape == (4, 32)

    def test_custom_dims(self):
        enc = ProfileStateEncoder(profile_dim=10, out_dim=64)
        x = torch.randn(8, 10)
        out = enc(x)
        assert out.shape == (8, 64)

    def test_single_sample(self):
        enc = ProfileStateEncoder(profile_dim=6, out_dim=32)
        x = torch.randn(1, 6)
        out = enc(x)
        assert out.shape == (1, 32)

    def test_large_batch(self):
        enc = ProfileStateEncoder(profile_dim=6, out_dim=32)
        x = torch.randn(512, 6)
        out = enc(x)
        assert out.shape == (512, 32)

    @pytest.mark.parametrize("out_dim", [8, 16, 32, 64, 128])
    def test_various_out_dims(self, out_dim):
        enc = ProfileStateEncoder(profile_dim=6, out_dim=out_dim)
        x = torch.randn(4, 6)
        out = enc(x)
        assert out.shape == (4, out_dim)


class TestProfileEncoderGradientFlow:
    """Verify gradients flow through the encoder."""

    def test_gradients_exist(self):
        enc = ProfileStateEncoder(profile_dim=6, out_dim=32)
        x = torch.randn(4, 6, requires_grad=True)
        out = enc(x)
        loss = out.sum()
        loss.backward()

        # Input should have gradient
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_parameter_gradients(self):
        enc = ProfileStateEncoder(profile_dim=6, out_dim=32)
        x = torch.randn(4, 6)
        out = enc(x)
        loss = out.sum()
        loss.backward()

        # All linear layers should have gradients
        for name, param in enc.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


class TestProfileEncoderDifferentiation:
    """Verify that different profiles produce different embeddings."""

    def test_different_profiles_different_outputs(self):
        enc = ProfileStateEncoder(profile_dim=6, out_dim=32)
        enc.eval()

        # High-risk profile: old account, high balance, business
        profile_a = torch.tensor([[1.0, 0.9, 0.9, 1.0, 0.5, 0.866]])
        # Low-risk profile: new account, low balance, consumer
        profile_b = torch.tensor([[0.01, 0.1, 0.1, 0.0, -0.5, -0.866]])

        out_a = enc(profile_a)
        out_b = enc(profile_b)

        # Embeddings should differ
        diff = (out_a - out_b).abs().sum().item()
        assert diff > 0.01, f"Different profiles produced nearly identical embeddings (diff={diff})"

    def test_identical_profiles_identical_outputs(self):
        enc = ProfileStateEncoder(profile_dim=6, out_dim=32)
        enc.eval()

        profile = torch.tensor([[0.5, 0.5, 0.5, 0.0, 0.0, 1.0]])
        out1 = enc(profile)
        out2 = enc(profile)

        assert torch.allclose(out1, out2, atol=1e-6)

    def test_batch_consistency(self):
        """Each sample in a batch should produce the same result as processed individually."""
        enc = ProfileStateEncoder(profile_dim=6, out_dim=32)
        enc.eval()

        profiles = torch.randn(4, 6)
        batch_out = enc(profiles)

        for i in range(4):
            single_out = enc(profiles[i:i+1])
            assert torch.allclose(batch_out[i:i+1], single_out, atol=1e-6)

    def test_sensitive_to_each_dimension(self):
        """Perturbing any single input dimension should change the output."""
        enc = ProfileStateEncoder(profile_dim=6, out_dim=32)
        enc.eval()

        base = torch.zeros(1, 6)
        base_out = enc(base)

        for dim in range(6):
            perturbed = base.clone()
            perturbed[0, dim] = 1.0
            perturbed_out = enc(perturbed)

            diff = (base_out - perturbed_out).abs().sum().item()
            assert diff > 1e-4, f"Encoder insensitive to dimension {dim}"
