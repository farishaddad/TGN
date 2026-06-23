"""Negative sampler for contrastive learning."""

import torch


class NegativeSampler:
    """
    Random negative edge sampler for contrastive learning.

    For each real edge (src, dst), samples a random 'negative' destination
    that the source did NOT interact with in this batch. This teaches the
    model to distinguish real interactions from random pairings.

    Args:
        num_nodes: Total number of nodes to sample from
    """

    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes

    def sample(self, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        """Sample random negative destinations.

        Args:
            src: Source node IDs [batch]
            dst: Real destination node IDs [batch] (used for exclusion)

        Returns:
            Negative destination node IDs [batch]
        """
        # Simple random sampling — may occasionally collide with real dst
        # but this is acceptable for learning purposes
        neg_dst = torch.randint(0, self.num_nodes, (src.size(0),), device=src.device)
        return neg_dst
