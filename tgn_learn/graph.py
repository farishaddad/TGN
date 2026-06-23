"""
Core graph data structures for temporal fraud detection.

This module provides TemporalGraph, Node, and Edge — the foundational building
blocks for representing financial transaction networks as temporal graphs.

Key concepts:
- Nodes represent entities (accounts, merchants, devices)
- Edges represent interactions (transactions, logins) with timestamps
- The TemporalGraph maintains temporal ordering for TGN training

Ported and simplified from SAIL_FSOS for learning purposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import TemporalData


# Default edge feature dimension (matches SAIL_V2)
EDGE_FEAT_DIM = 20


@dataclass
class Node:
    """
    A node in the temporal graph representing an entity.

    Attributes:
        node_id: Unique integer identifier
        node_type: Category of entity (e.g. 'account', 'merchant', 'device')
        features: Optional static feature vector for this node
        metadata: Arbitrary key-value metadata (name, category, etc.)
    """

    node_id: int
    node_type: str = "account"
    features: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.node_id < 0:
            raise ValueError(f"node_id must be non-negative, got {self.node_id}")
        if not self.node_type:
            raise ValueError("node_type cannot be empty")

    def __hash__(self):
        return hash(self.node_id)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.node_id == other.node_id


@dataclass
class Edge:
    """
    A temporal edge representing an interaction between two nodes.

    Attributes:
        src_id: Source node ID (e.g. the account initiating a transaction)
        dst_id: Destination node ID (e.g. the merchant receiving payment)
        timestamp: Unix timestamp of when this interaction occurred
        features: Edge feature vector (amount, channel, etc.)
        label: Ground truth label (0=legit, 1=fraud, -1=unknown)
        edge_type: Category of interaction (e.g. 'purchase', 'transfer')
        metadata: Additional info (pattern_type, source_dataset, etc.)
    """

    src_id: int
    dst_id: int
    timestamp: float
    features: np.ndarray = field(default_factory=lambda: np.zeros(EDGE_FEAT_DIM, dtype=np.float32))
    label: int = -1  # -1 = unknown/unlabeled
    edge_type: str = "transaction"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.src_id < 0 or self.dst_id < 0:
            raise ValueError(f"Node IDs must be non-negative, got src={self.src_id}, dst={self.dst_id}")
        if self.timestamp < 0:
            raise ValueError(f"Timestamp must be non-negative, got {self.timestamp}")
        if self.label not in (-1, 0, 1):
            raise ValueError(f"Label must be -1, 0, or 1, got {self.label}")
        # Ensure features are the right shape
        if self.features is not None:
            self.features = np.asarray(self.features, dtype=np.float32)
            if self.features.ndim != 1:
                raise ValueError(f"Features must be 1-D, got shape {self.features.shape}")


class TemporalGraph:
    """
    A temporal graph that maintains nodes and time-ordered edges.

    This is the primary data structure for the TGN learning app. It stores
    all nodes and edges, maintains temporal ordering of edges, and can
    convert to PyG's TemporalData format for model training.

    Example:
        >>> g = TemporalGraph()
        >>> g.add_node(Node(0, 'account', metadata={'name': 'Alice'}))
        >>> g.add_node(Node(1, 'merchant', metadata={'name': 'CoffeeShop'}))
        >>> g.add_edge(Edge(src_id=0, dst_id=1, timestamp=1000.0, label=0))
        >>> print(g.summary())
        TemporalGraph: 2 nodes, 1 edges, time range [1000.0, 1000.0]
    """

    def __init__(self):
        self._nodes: dict[int, Node] = {}
        self._edges: list[Edge] = []
        self._sorted = True  # Track if edges need re-sorting

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, node: Node) -> None:
        """Add a node to the graph. Overwrites if node_id already exists."""
        self._nodes[node.node_id] = node

    def get_node(self, node_id: int) -> Optional[Node]:
        """Retrieve a node by ID, or None if not found."""
        return self._nodes.get(node_id)

    def has_node(self, node_id: int) -> bool:
        """Check if a node exists in the graph."""
        return node_id in self._nodes

    @property
    def nodes(self) -> list[Node]:
        """All nodes in the graph."""
        return list(self._nodes.values())

    @property
    def num_nodes(self) -> int:
        """Total number of nodes."""
        return len(self._nodes)

    def nodes_by_type(self, node_type: str) -> list[Node]:
        """Get all nodes of a specific type."""
        return [n for n in self._nodes.values() if n.node_type == node_type]

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(self, edge: Edge) -> None:
        """
        Add a temporal edge to the graph.

        Automatically creates nodes if they don't exist.
        Marks the edge list as potentially unsorted.
        """
        # Auto-create nodes if needed
        if edge.src_id not in self._nodes:
            self._nodes[edge.src_id] = Node(edge.src_id)
        if edge.dst_id not in self._nodes:
            self._nodes[edge.dst_id] = Node(edge.dst_id)

        # Check if temporal ordering is maintained
        if self._edges and edge.timestamp < self._edges[-1].timestamp:
            self._sorted = False

        self._edges.append(edge)

    def add_edges(self, edges: list[Edge]) -> None:
        """Add multiple edges at once."""
        for edge in edges:
            self.add_edge(edge)

    @property
    def edges(self) -> list[Edge]:
        """All edges in temporal order."""
        if not self._sorted:
            self._edges.sort(key=lambda e: e.timestamp)
            self._sorted = True
        return self._edges

    @property
    def num_edges(self) -> int:
        """Total number of edges."""
        return len(self._edges)

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    @property
    def time_range(self) -> tuple[float, float]:
        """Return (min_timestamp, max_timestamp) across all edges."""
        if not self._edges:
            return (0.0, 0.0)
        edges = self.edges  # Ensures sorted
        return (edges[0].timestamp, edges[-1].timestamp)

    @property
    def fraud_rate(self) -> float:
        """Fraction of labeled edges that are fraudulent."""
        labeled = [e for e in self._edges if e.label >= 0]
        if not labeled:
            return 0.0
        return sum(1 for e in labeled if e.label == 1) / len(labeled)

    @property
    def num_fraud(self) -> int:
        """Number of fraudulent edges."""
        return sum(1 for e in self._edges if e.label == 1)

    @property
    def num_legit(self) -> int:
        """Number of legitimate edges."""
        return sum(1 for e in self._edges if e.label == 0)

    def edges_in_range(self, t_start: float, t_end: float) -> list[Edge]:
        """Return edges within a time window [t_start, t_end]."""
        return [e for e in self.edges if t_start <= e.timestamp <= t_end]

    def edges_for_node(self, node_id: int) -> list[Edge]:
        """Return all edges involving a specific node."""
        return [e for e in self._edges if e.src_id == node_id or e.dst_id == node_id]

    def node_types(self) -> dict[str, int]:
        """Count of nodes by type."""
        counts: dict[str, int] = {}
        for node in self._nodes.values():
            counts[node.node_type] = counts.get(node.node_type, 0) + 1
        return counts

    def edge_types(self) -> dict[str, int]:
        """Count of edges by type."""
        counts: dict[str, int] = {}
        for edge in self._edges:
            counts[edge.edge_type] = counts.get(edge.edge_type, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Conversion to PyG TemporalData
    # ------------------------------------------------------------------

    def to_pyg_temporal_data(self) -> TemporalData:
        """
        Convert this graph to PyG's TemporalData format.

        This is the native input format for TGN training with PyG.
        Edges are sorted by timestamp before conversion.

        Returns:
            TemporalData with fields:
                - src: source node IDs [num_edges]
                - dst: destination node IDs [num_edges]
                - t: timestamps [num_edges]
                - msg: edge feature matrix [num_edges, feat_dim]
                - y: labels [num_edges] (-1 for unlabeled)
        """
        edges = self.edges  # Sorted by timestamp

        if not edges:
            return TemporalData(
                src=torch.empty(0, dtype=torch.long),
                dst=torch.empty(0, dtype=torch.long),
                t=torch.empty(0, dtype=torch.float),
                msg=torch.empty(0, EDGE_FEAT_DIM, dtype=torch.float),
                y=torch.empty(0, dtype=torch.long),
            )

        src = torch.tensor([e.src_id for e in edges], dtype=torch.long)
        dst = torch.tensor([e.dst_id for e in edges], dtype=torch.long)
        t = torch.tensor([e.timestamp for e in edges], dtype=torch.float)
        msg = torch.tensor(
            np.stack([e.features for e in edges]),
            dtype=torch.float,
        )
        y = torch.tensor([e.label for e in edges], dtype=torch.long)

        return TemporalData(src=src, dst=dst, t=t, msg=msg, y=y)

    # ------------------------------------------------------------------
    # Temporal splitting
    # ------------------------------------------------------------------

    def temporal_split(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> tuple["TemporalGraph", "TemporalGraph", "TemporalGraph"]:
        """
        Split graph into train/val/test by temporal ordering (chronological).

        This preserves the causal structure: training on past, validating
        on near-future, testing on far-future.

        Args:
            train_ratio: Fraction of edges for training (default 0.70)
            val_ratio: Fraction of edges for validation (default 0.15)

        Returns:
            (train_graph, val_graph, test_graph) — three TemporalGraph instances
        """
        edges = self.edges  # Sorted
        n = len(edges)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_edges = edges[:train_end]
        val_edges = edges[train_end:val_end]
        test_edges = edges[val_end:]

        def _build_subgraph(edge_list: list[Edge]) -> "TemporalGraph":
            g = TemporalGraph()
            # Copy relevant nodes
            node_ids = set()
            for e in edge_list:
                node_ids.add(e.src_id)
                node_ids.add(e.dst_id)
            for nid in node_ids:
                if nid in self._nodes:
                    g.add_node(self._nodes[nid])
            g._edges = edge_list
            g._sorted = True
            return g

        return _build_subgraph(train_edges), _build_subgraph(val_edges), _build_subgraph(test_edges)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary of the graph."""
        t_min, t_max = self.time_range
        lines = [
            f"TemporalGraph: {self.num_nodes} nodes, {self.num_edges} edges",
            f"  Time range: [{t_min:.1f}, {t_max:.1f}]",
            f"  Node types: {self.node_types()}",
            f"  Edge types: {self.edge_types()}",
            f"  Fraud: {self.num_fraud} ({self.fraud_rate:.2%})",
            f"  Legit: {self.num_legit}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"TemporalGraph(nodes={self.num_nodes}, edges={self.num_edges})"

    def __len__(self) -> int:
        return self.num_edges
