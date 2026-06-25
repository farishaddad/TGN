"""
Topology-preserving GraphSMOTE (THG-OAFN, PLoS ONE 2025).

Generates synthetic minority samples by interpolating between existing
fraud edge features, then assigns synthetic edges to k-hop neighbours
of their interpolated parents. Preserves graph community structure that
standard SMOTE destroys.

Applied only to training data before TGN training begins.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from tgn_learn.graph import Edge, TemporalGraph


@dataclass
class SMOTEResult:
    """Result of GraphSMOTE augmentation.

    Attributes:
        synthetic_edges: New synthetic fraud edges
        original_count: Number of original fraud edges
        synthetic_count: Number of synthetic edges generated
        target_ratio: Achieved minority ratio
    """

    synthetic_edges: list[Edge]
    original_count: int
    synthetic_count: int
    target_ratio: float


class GraphSMOTE:
    """Topology-preserving oversampling for temporal fraud graphs.

    Unlike standard SMOTE which treats each sample independently,
    GraphSMOTE ensures synthetic edges connect to nodes within the
    k-hop neighbourhood of their parent fraud edges. This preserves
    the community structure that TGN needs to learn from.

    Args:
        k_hop: Neighbourhood radius for synthetic edge placement
        minority_ratio: Target fraud-to-total ratio after augmentation
        random_state: Seed for reproducibility
    """

    def __init__(
        self,
        k_hop: int = 2,
        minority_ratio: float = 0.1,
        random_state: int = 42,
    ):
        self.k_hop = k_hop
        self.minority_ratio = minority_ratio
        self.rng = np.random.default_rng(random_state)

    def augment(self, graph: TemporalGraph) -> SMOTEResult:
        """Generate synthetic fraud edges to balance the dataset.

        Args:
            graph: Input temporal graph with labelled edges

        Returns:
            SMOTEResult with synthetic edges and statistics
        """
        fraud_edges = [e for e in graph.edges if e.label == 1]
        legit_edges = [e for e in graph.edges if e.label == 0]

        if not fraud_edges:
            return SMOTEResult(
                synthetic_edges=[], original_count=0,
                synthetic_count=0, target_ratio=0.0,
            )

        # Calculate how many synthetic samples we need
        total = len(fraud_edges) + len(legit_edges)
        target_fraud_count = int(self.minority_ratio * total / (1 - self.minority_ratio))
        n_synthetic = max(0, target_fraud_count - len(fraud_edges))

        if n_synthetic == 0:
            return SMOTEResult(
                synthetic_edges=[], original_count=len(fraud_edges),
                synthetic_count=0, target_ratio=len(fraud_edges) / total,
            )

        # Build adjacency for k-hop neighbourhood lookup
        adjacency = self._build_adjacency(graph)

        # Generate synthetic edges
        synthetic = []
        for _ in range(n_synthetic):
            # Pick two random fraud edges to interpolate
            idx_a, idx_b = self.rng.choice(len(fraud_edges), size=2, replace=True)
            edge_a = fraud_edges[idx_a]
            edge_b = fraud_edges[idx_b]

            # Interpolate features
            alpha = self.rng.uniform(0.3, 0.7)
            features = alpha * edge_a.features + (1 - alpha) * edge_b.features

            # Place synthetic edge within k-hop neighbourhood of parent
            src = self._pick_neighbour(edge_a.src_id, adjacency)
            dst = self._pick_neighbour(edge_a.dst_id, adjacency)

            # Interpolate timestamp
            t = alpha * edge_a.timestamp + (1 - alpha) * edge_b.timestamp

            synthetic.append(Edge(
                src_id=src, dst_id=dst,
                timestamp=t, features=features, label=1,
            ))

        new_total = total + n_synthetic
        achieved_ratio = (len(fraud_edges) + n_synthetic) / new_total

        return SMOTEResult(
            synthetic_edges=synthetic,
            original_count=len(fraud_edges),
            synthetic_count=n_synthetic,
            target_ratio=achieved_ratio,
        )

    def _build_adjacency(self, graph: TemporalGraph) -> dict[int, set[int]]:
        """Build undirected adjacency dict from graph edges."""
        adj: dict[int, set[int]] = {}
        for edge in graph.edges:
            adj.setdefault(edge.src_id, set()).add(edge.dst_id)
            adj.setdefault(edge.dst_id, set()).add(edge.src_id)
        return adj

    def _pick_neighbour(self, node_id: int, adjacency: dict[int, set[int]]) -> int:
        """Pick a random node within k-hop of node_id."""
        visited = {node_id}
        frontier = {node_id}

        for _ in range(self.k_hop):
            next_frontier: set[int] = set()
            for n in frontier:
                for neighbour in adjacency.get(n, set()):
                    if neighbour not in visited:
                        next_frontier.add(neighbour)
                        visited.add(neighbour)
            frontier = next_frontier
            if not frontier:
                break

        # Pick from all reachable nodes (excluding start)
        candidates = list(visited - {node_id})
        if candidates:
            return int(self.rng.choice(candidates))
        return node_id  # Fallback: self-loop (rare edge case)
