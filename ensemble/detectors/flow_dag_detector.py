"""
Fund-Flow DAG Detector (ETGAT, Wu & Zhang, BDAIE 2025).

Scores transactions based on their position in the fund-flow DAG.
Transactions that are part of long chains (fan-out → fan-in patterns)
get higher scores — these are money laundering indicators.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from tgn_learn.graph import TemporalGraph

from ..graphs.flow_dag import FundFlowDAG
from .base import BaseDetector


class FlowDAGDetector(BaseDetector):
    """Fund-flow DAG path-based anomaly detector.

    Builds an event-centric DAG where transactions are nodes and
    fund-flow chains are edges. Scores based on chain length and
    pass-through topology.

    Args:
        time_window_hours: Max hours between chained events
        min_amount_ratio: Min outflow/inflow ratio for chain edge
    """

    def __init__(
        self,
        time_window_hours: float = 24.0,
        min_amount_ratio: float = 0.5,
    ):
        self._dag = FundFlowDAG(
            time_window_hours=time_window_hours,
            min_amount_ratio=min_amount_ratio,
        )
        self._path_scores: Optional[np.ndarray] = None
        self._edge_to_event: dict[tuple[int, int, float], int] = {}
        self._fitted = False

    @property
    def name(self) -> str:
        return "Fund-Flow Graph"

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, graph: TemporalGraph) -> None:
        """Build the fund-flow DAG and pre-compute path scores."""
        self._dag.build(graph)
        self._path_scores = self._dag.get_path_scores()

        # Build lookup: (src, dst, timestamp) → event_id
        self._edge_to_event.clear()
        for i, edge in enumerate(graph.edges):
            key = (edge.src_id, edge.dst_id, edge.timestamp)
            self._edge_to_event[key] = i

        self._fitted = True

    def score(
        self,
        src: int,
        dst: int,
        timestamp: float,
        features: np.ndarray,
        graph: Optional[TemporalGraph] = None,
    ) -> float:
        """Score based on fund-flow chain position."""
        if not self._fitted or self._path_scores is None:
            return 0.0

        # Look up event ID for this transaction
        key = (src, dst, timestamp)
        event_id = self._edge_to_event.get(key)

        if event_id is not None and event_id < len(self._path_scores):
            return float(self._path_scores[event_id])

        # For unseen transactions, estimate based on structural heuristics
        # Check if src or dst appear as pass-through nodes
        if graph is not None:
            return self._estimate_score(src, dst, timestamp, graph)

        return 0.0

    def _estimate_score(
        self,
        src: int,
        dst: int,
        timestamp: float,
        graph: TemporalGraph,
    ) -> float:
        """Estimate flow score for unseen transactions."""
        # Count how many recent outflows from dst (money passing through)
        recent_outflows = sum(
            1 for e in graph.edges_for_node(dst)
            if e.src_id == dst and e.timestamp > timestamp
            and (e.timestamp - timestamp) < self._dag.time_window
        )

        # Count recent inflows to src (money arriving)
        recent_inflows = sum(
            1 for e in graph.edges_for_node(src)
            if e.dst_id == src and e.timestamp < timestamp
            and (timestamp - e.timestamp) < self._dag.time_window
        )

        # Heuristic: high in+out flow suggests pass-through
        score = min(1.0, (recent_inflows * recent_outflows) / 10.0)
        return score
