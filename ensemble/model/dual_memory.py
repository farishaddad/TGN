"""
Dual-Track Memory Module (DySA-TGN, DASFAA 2025).

Splits TGN memory into two components:
- StableMemory: Updated slowly via EMA. Encodes long-term behavioural baseline.
- TransientMemory: Standard GRU updated per-event. Encodes real-time deviations.

The fraud signal lives in the transient track (deviation from baseline).
The stable component provides the reference. This eliminates false positives
from legitimate lifestyle changes (which update stable slowly) while genuine
fraud events spike transient.

Output: concat(s_stable, s_transient) — same total dimension as single memory.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DualTrackMemory(nn.Module):
    """Dual-track memory separating stable baseline from transient deviations.

    Args:
        num_nodes: Total node count in the graph
        stable_dim: Dimension of stable baseline memory
        transient_dim: Dimension of transient deviation memory
        msg_dim: Edge feature/message dimension
        time_dim: Time encoding dimension
        stable_update_alpha: EMA decay for stable memory (default: 0.05)
    """

    def __init__(
        self,
        num_nodes: int,
        stable_dim: int = 32,
        transient_dim: int = 32,
        msg_dim: int = 20,
        time_dim: int = 32,
        stable_update_alpha: float = 0.05,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.stable_dim = stable_dim
        self.transient_dim = transient_dim
        self.output_dim = stable_dim + transient_dim
        self.alpha = stable_update_alpha

        # Stable memory — updated via EMA once per epoch
        self.register_buffer(
            "stable_memory",
            torch.zeros(num_nodes, stable_dim),
        )

        # Transient memory — updated per-event via GRU
        self.register_buffer(
            "transient_memory",
            torch.zeros(num_nodes, transient_dim),
        )

        # GRU cell for transient updates
        self.transient_gru = nn.GRUCell(
            input_size=msg_dim + time_dim,
            hidden_size=transient_dim,
        )

        # Projection for message + time encoding into GRU input
        self.msg_proj = nn.Linear(msg_dim + time_dim, msg_dim + time_dim)

        # Last update timestamps
        self.register_buffer(
            "last_update",
            torch.zeros(num_nodes),
        )

    def forward(self, n_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve combined memory for given node IDs.

        Args:
            n_id: Node IDs to look up [batch]

        Returns:
            memory: Combined [stable, transient] memory [batch, output_dim]
            last_update: Last update times for these nodes [batch]
        """
        s_stable = self.stable_memory[n_id]
        s_transient = self.transient_memory[n_id]
        combined = torch.cat([s_stable, s_transient], dim=-1)
        return combined, self.last_update[n_id]

    def update_transient(
        self,
        n_id: torch.Tensor,
        messages: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> None:
        """Update transient memory for nodes that received interactions.

        Args:
            n_id: Node IDs being updated [batch]
            messages: Concatenated [edge_features, time_encoding] [batch, msg_dim + time_dim]
            timestamps: Current timestamps [batch]
        """
        # Project messages
        msg_input = self.msg_proj(messages)

        # Get current transient states
        h = self.transient_memory[n_id]

        # GRU update
        h_new = self.transient_gru(msg_input, h)

        # Write back
        self.transient_memory[n_id] = h_new.detach()
        self.last_update[n_id] = timestamps

    def update_stable_ema(self) -> None:
        """Update stable memory via exponential moving average of transient.

        Call this once per epoch (not per batch). The stable memory slowly
        absorbs the transient state, representing the long-term baseline.
        """
        with torch.no_grad():
            # EMA: stable = (1 - alpha) * stable + alpha * transient_norm
            # We use the L2-normalised transient to avoid scale drift
            transient_norm = torch.nn.functional.normalize(
                self.transient_memory, p=2, dim=-1
            ) * self.stable_dim ** 0.5

            # Only update nodes that have been active (non-zero transient)
            active_mask = self.transient_memory.abs().sum(dim=-1) > 0
            if active_mask.any():
                self.stable_memory[active_mask] = (
                    (1 - self.alpha) * self.stable_memory[active_mask]
                    + self.alpha * transient_norm[active_mask, :self.stable_dim]
                )

    def reset_transient(self) -> None:
        """Reset transient memory to zero (call at epoch start if desired)."""
        self.transient_memory.zero_()

    def reset_all(self) -> None:
        """Reset both memory tracks and timestamps."""
        self.stable_memory.zero_()
        self.transient_memory.zero_()
        self.last_update.zero_()

    def get_deviation_score(self, n_id: torch.Tensor) -> torch.Tensor:
        """Compute deviation of transient from stable baseline.

        Higher values indicate the node's recent behaviour deviates
        significantly from its long-term baseline — a fraud signal.

        Args:
            n_id: Node IDs [batch]

        Returns:
            Deviation scores [batch] (L2 distance, normalised)
        """
        s_stable = self.stable_memory[n_id]
        s_transient = self.transient_memory[n_id]

        # Pad/truncate to match dimensions for comparison
        min_dim = min(self.stable_dim, self.transient_dim)
        deviation = torch.norm(
            s_transient[:, :min_dim] - s_stable[:, :min_dim],
            p=2, dim=-1,
        )
        # Normalise by dimension
        return deviation / (min_dim ** 0.5)
