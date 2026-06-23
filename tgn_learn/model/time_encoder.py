"""
Learnable Fourier time encoding (Xu et al., 2020).

Maps scalar timestamps into high-dimensional representations using
learnable frequency parameters. This allows the model to discover
which temporal patterns matter for fraud detection.

Key insight: The initial frequencies span many orders of magnitude
(1 to 10^-9), letting the model capture patterns from seconds to years.
"""

import numpy as np
import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    """
    Learnable Fourier time encoding.

    Transforms a scalar time delta into a time_dim-dimensional vector
    using cosine activations with learnable frequencies.

    Args:
        time_dim: Output dimension of the time encoding
    """

    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        self.w = nn.Linear(1, time_dim)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize frequencies spanning multiple time scales."""
        with torch.no_grad():
            # Frequencies from 1 (fast) to 10^-9 (slow)
            freqs = 1.0 / 10 ** np.linspace(0, 9, self.time_dim, dtype=np.float32)
            self.w.weight.copy_(torch.from_numpy(freqs).reshape(self.time_dim, 1))
            self.w.bias.zero_()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Encode timestamps.

        Args:
            t: Tensor of timestamps, shape [batch] or [batch, 1]

        Returns:
            Time encodings, shape [batch, time_dim]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return torch.cos(self.w(t.float()))
