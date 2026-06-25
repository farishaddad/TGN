"""
Event-centric Fund-Flow DAG (ETGAT, Wu & Zhang, BDAIE 2025).

Inverts the standard entity graph: transactions become nodes, fund-flow
chains become edges. Makes money-mule paths explicit as graph paths
rather than implicit in entity embeddings.

An edge exists from event i → event j if:
  1. passive_party(i) == active_party(j)  (money flows through)
  2. timestamp(j) - timestamp(i) <= time_window  (within window)

This representation makes layering chains (A→B→C→D) visible as
explicit 3-hop paths in the DAG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from tgn_learn.graph import Edge, TemporalGraph


@dataclass
class FlowNode:
    """A node in the fund-flow DAG (represents a transaction).

    Attributes:
        event_id: Index of the original edge in the entity graph
        src_id: Source account (active party)
        dst_id: Destination account (passive party)
        timestamp: Transaction time
        amount: Transaction amount
        features: Edge features from the entity graph
        label: Fraud label (0/1/-1)
    """

    event_id: int
    src_id: int
    dst_id: int
    timestamp: float
    amount: float
    features: np.ndarray
    label: int = -1


@dataclass
class FlowEdge:
    """An edge in the fund-flow DAG (represents fund continuity).

    Connects event_i to event_j when dst(i) == src(j) within time_window.
    """

    from_event: int
    to_event: int
    time_delta: float  # seconds between events


class FundFlowDAG:
    """Event-centric fund-flow directed acyclic graph.

    Constructs a DAG where each transaction is a node and edges
    represent fund-flow continuity (money passing through accounts).

    Args:
        time_window_hours: Max hours between chained transactions (default: 24)
        min_amount_ratio: Min ratio of outflow/inflow to consider a chain (default: 0.5)
    """

    def __init__(
        self,
        time_window_hours: float = 24.0,
        min_amount_ratio: float = 0.5,
    ):
        self.time_window = time_window_hours * 3600.0
        self.min_amount_ratio = min_amount_ratio
        self.nodes: list[FlowNode] = []
        self.edges: list[FlowEdge] = []
        self._built = False

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def build(self, graph: TemporalGraph) -> "FundFlowDAG":
        """Build the fund-flow DAG from an entity-centric temporal graph.

        Args:
            graph: Source TemporalGraph (entity-centric)

        Returns:
            self (for chaining)
        """
        entity_edges = graph.edges

        # Create flow nodes (one per transaction)
        self.nodes = []
        for i, edge in enumerate(entity_edges):
            amount = float(np.exp(edge.features[0]) - 1) if edge.features is not None else 0.0
            self.nodes.append(FlowNode(
                event_id=i,
                src_id=edge.src_id,
                dst_id=edge.dst_id,
                timestamp=edge.timestamp,
                amount=amount,
                features=edge.features,
                label=edge.label,
            ))

        # Build index: dst_id → list of (event_idx, timestamp, amount)
        # This lets us efficiently find outflows from any account
        outflow_index: dict[int, list[tuple[int, float, float]]] = {}
        for i, node in enumerate(self.nodes):
            outflow_index.setdefault(node.src_id, []).append(
                (i, node.timestamp, node.amount)
            )

        # Sort each account's outflows by timestamp
        for account in outflow_index:
            outflow_index[account].sort(key=lambda x: x[1])

        # Build edges: for each transaction, find subsequent outflows
        # from the same destination account (money flowing through)
        self.edges = []
        for i, node in enumerate(self.nodes):
            # Look for outflows from node.dst_id after node.timestamp
            outflows = outflow_index.get(node.dst_id, [])
            for j, t_j, amount_j in outflows:
                if j == i:
                    continue
                time_delta = t_j - node.timestamp
                if time_delta <= 0:
                    continue  # Must be strictly after
                if time_delta > self.time_window:
                    break  # Sorted by time, so no more valid

                # Optional: check amount ratio (outflow should be substantial)
                if node.amount > 0 and amount_j / node.amount < self.min_amount_ratio:
                    continue

                self.edges.append(FlowEdge(
                    from_event=i,
                    to_event=j,
                    time_delta=time_delta,
                ))

        self._built = True
        return self

    def get_chain_length(self, event_id: int) -> int:
        """Get the longest chain starting from a given event.

        Useful for identifying multi-hop laundering paths.
        """
        if not self._built:
            raise RuntimeError("DAG not built yet — call build() first")

        # BFS/DFS for longest path from event_id
        adj: dict[int, list[int]] = {}
        for edge in self.edges:
            adj.setdefault(edge.from_event, []).append(edge.to_event)

        visited = set()
        max_depth = 0

        def dfs(node: int, depth: int):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            if node in adj:
                for next_node in adj[node]:
                    if next_node not in visited:
                        visited.add(next_node)
                        dfs(next_node, depth + 1)
                        visited.discard(next_node)

        visited.add(event_id)
        dfs(event_id, 0)
        return max_depth

    def get_path_scores(self) -> np.ndarray:
        """Compute path-based anomaly scores for each event node.

        Events that are part of longer chains get higher scores.
        This is a simplified version of the ETGAT path attention.

        Returns:
            scores: [num_nodes] array of path-based anomaly indicators
        """
        if not self._built:
            raise RuntimeError("DAG not built yet")

        # Count in-degree and out-degree
        in_degree = np.zeros(self.num_nodes)
        out_degree = np.zeros(self.num_nodes)
        for edge in self.edges:
            out_degree[edge.from_event] += 1
            in_degree[edge.to_event] += 1

        # Events with both high in-degree and out-degree are "pass-through"
        # nodes in laundering chains — score = in * out normalised
        max_product = max((in_degree * out_degree).max(), 1.0)
        scores = (in_degree * out_degree) / max_product

        return scores

    def get_edge_index(self) -> tuple[np.ndarray, np.ndarray]:
        """Get edge index in COO format for PyG compatibility.

        Returns:
            (src_indices, dst_indices) each of shape [num_edges]
        """
        if not self.edges:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        src = np.array([e.from_event for e in self.edges], dtype=np.int64)
        dst = np.array([e.to_event for e in self.edges], dtype=np.int64)
        return src, dst

    def get_node_features(self) -> np.ndarray:
        """Get feature matrix for all flow nodes.

        Returns:
            [num_nodes, feat_dim] array of node features
        """
        if not self.nodes:
            return np.array([])

        return np.stack([n.features for n in self.nodes])

    def get_node_labels(self) -> np.ndarray:
        """Get label array for all flow nodes.

        Returns:
            [num_nodes] array of labels (0/1/-1)
        """
        return np.array([n.label for n in self.nodes])
