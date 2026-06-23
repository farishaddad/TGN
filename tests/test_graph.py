"""Tests for core graph data structures."""

import numpy as np
import pytest

from tgn_learn.graph import Edge, Node, TemporalGraph, EDGE_FEAT_DIM


class TestNode:
    """Tests for Node creation and validation."""

    def test_basic_creation(self):
        node = Node(node_id=0, node_type="account")
        assert node.node_id == 0
        assert node.node_type == "account"
        assert node.features is None
        assert node.metadata == {}

    def test_with_features(self):
        feats = np.ones(10, dtype=np.float32)
        node = Node(node_id=1, node_type="merchant", features=feats)
        assert np.array_equal(node.features, feats)

    def test_with_metadata(self):
        node = Node(node_id=2, node_type="device", metadata={"ip": "1.2.3.4"})
        assert node.metadata["ip"] == "1.2.3.4"

    def test_negative_id_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            Node(node_id=-1, node_type="account")

    def test_empty_type_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            Node(node_id=0, node_type="")

    def test_hash_and_eq(self):
        n1 = Node(node_id=5, node_type="account")
        n2 = Node(node_id=5, node_type="merchant")  # Same ID, different type
        assert n1 == n2  # Equality is by ID
        assert hash(n1) == hash(n2)

        n3 = Node(node_id=6, node_type="account")
        assert n1 != n3


class TestEdge:
    """Tests for Edge creation and validation."""

    def test_basic_creation(self):
        edge = Edge(src_id=0, dst_id=1, timestamp=1000.0)
        assert edge.src_id == 0
        assert edge.dst_id == 1
        assert edge.timestamp == 1000.0
        assert edge.label == -1  # Default: unlabeled
        assert edge.features.shape == (EDGE_FEAT_DIM,)

    def test_with_features(self):
        feats = np.random.randn(EDGE_FEAT_DIM).astype(np.float32)
        edge = Edge(src_id=0, dst_id=1, timestamp=1.0, features=feats)
        assert np.allclose(edge.features, feats)

    def test_label_values(self):
        Edge(src_id=0, dst_id=1, timestamp=1.0, label=0)  # legit
        Edge(src_id=0, dst_id=1, timestamp=1.0, label=1)  # fraud
        Edge(src_id=0, dst_id=1, timestamp=1.0, label=-1)  # unknown

    def test_invalid_label_raises(self):
        with pytest.raises(ValueError, match="Label must be"):
            Edge(src_id=0, dst_id=1, timestamp=1.0, label=2)

    def test_negative_node_id_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            Edge(src_id=-1, dst_id=1, timestamp=1.0)

    def test_negative_timestamp_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            Edge(src_id=0, dst_id=1, timestamp=-1.0)

    def test_features_coerced_to_float32(self):
        feats = np.ones(EDGE_FEAT_DIM, dtype=np.float64)
        edge = Edge(src_id=0, dst_id=1, timestamp=1.0, features=feats)
        assert edge.features.dtype == np.float32

    def test_2d_features_raises(self):
        feats = np.ones((2, EDGE_FEAT_DIM), dtype=np.float32)
        with pytest.raises(ValueError, match="1-D"):
            Edge(src_id=0, dst_id=1, timestamp=1.0, features=feats)


class TestTemporalGraph:
    """Tests for TemporalGraph operations."""

    def _make_graph(self) -> TemporalGraph:
        """Create a small test graph."""
        g = TemporalGraph()
        g.add_node(Node(0, "account", metadata={"name": "Alice"}))
        g.add_node(Node(1, "merchant", metadata={"name": "Shop"}))
        g.add_node(Node(2, "account", metadata={"name": "Bob"}))

        g.add_edge(Edge(src_id=0, dst_id=1, timestamp=100.0, label=0))
        g.add_edge(Edge(src_id=2, dst_id=1, timestamp=200.0, label=0))
        g.add_edge(Edge(src_id=0, dst_id=1, timestamp=300.0, label=1))
        return g

    def test_add_nodes(self):
        g = TemporalGraph()
        g.add_node(Node(0, "account"))
        g.add_node(Node(1, "merchant"))
        assert g.num_nodes == 2
        assert g.has_node(0)
        assert g.has_node(1)
        assert not g.has_node(2)

    def test_add_edges_auto_creates_nodes(self):
        g = TemporalGraph()
        g.add_edge(Edge(src_id=5, dst_id=10, timestamp=1.0))
        assert g.has_node(5)
        assert g.has_node(10)
        assert g.num_nodes == 2

    def test_temporal_ordering(self):
        """Edges should always be returned in temporal order."""
        g = TemporalGraph()
        g.add_edge(Edge(src_id=0, dst_id=1, timestamp=300.0))
        g.add_edge(Edge(src_id=0, dst_id=1, timestamp=100.0))
        g.add_edge(Edge(src_id=0, dst_id=1, timestamp=200.0))

        edges = g.edges
        timestamps = [e.timestamp for e in edges]
        assert timestamps == sorted(timestamps)

    def test_time_range(self):
        g = self._make_graph()
        assert g.time_range == (100.0, 300.0)

    def test_time_range_empty(self):
        g = TemporalGraph()
        assert g.time_range == (0.0, 0.0)

    def test_fraud_rate(self):
        g = self._make_graph()
        # 1 fraud out of 3 labeled edges = 33.3%
        assert abs(g.fraud_rate - 1 / 3) < 1e-6

    def test_node_types(self):
        g = self._make_graph()
        types = g.node_types()
        assert types == {"account": 2, "merchant": 1}

    def test_nodes_by_type(self):
        g = self._make_graph()
        accounts = g.nodes_by_type("account")
        assert len(accounts) == 2

    def test_edges_for_node(self):
        g = self._make_graph()
        edges = g.edges_for_node(0)
        assert len(edges) == 2  # Alice has 2 edges

    def test_edges_in_range(self):
        g = self._make_graph()
        edges = g.edges_in_range(150.0, 250.0)
        assert len(edges) == 1
        assert edges[0].timestamp == 200.0

    def test_summary(self):
        g = self._make_graph()
        s = g.summary()
        assert "3 nodes" in s
        assert "3 edges" in s

    def test_repr(self):
        g = self._make_graph()
        assert "nodes=3" in repr(g)
        assert "edges=3" in repr(g)

    def test_len(self):
        g = self._make_graph()
        assert len(g) == 3


class TestTemporalGraphPyGConversion:
    """Tests for converting TemporalGraph to PyG TemporalData."""

    def test_conversion_basic(self):
        g = TemporalGraph()
        g.add_edge(Edge(src_id=0, dst_id=1, timestamp=1.0, label=0))
        g.add_edge(Edge(src_id=1, dst_id=2, timestamp=2.0, label=1))

        data = g.to_pyg_temporal_data()

        assert data.src.shape == (2,)
        assert data.dst.shape == (2,)
        assert data.t.shape == (2,)
        assert data.msg.shape == (2, EDGE_FEAT_DIM)
        assert data.y.shape == (2,)

        # Check values
        assert data.src.tolist() == [0, 1]
        assert data.dst.tolist() == [1, 2]
        assert data.t.tolist() == [1.0, 2.0]
        assert data.y.tolist() == [0, 1]

    def test_conversion_empty_graph(self):
        g = TemporalGraph()
        data = g.to_pyg_temporal_data()
        assert data.src.shape == (0,)
        assert data.msg.shape == (0, EDGE_FEAT_DIM)

    def test_conversion_preserves_temporal_order(self):
        g = TemporalGraph()
        g.add_edge(Edge(src_id=0, dst_id=1, timestamp=5.0))
        g.add_edge(Edge(src_id=0, dst_id=1, timestamp=1.0))
        g.add_edge(Edge(src_id=0, dst_id=1, timestamp=3.0))

        data = g.to_pyg_temporal_data()
        assert data.t.tolist() == [1.0, 3.0, 5.0]

    def test_conversion_preserves_features(self):
        feats = np.arange(EDGE_FEAT_DIM, dtype=np.float32)
        g = TemporalGraph()
        g.add_edge(Edge(src_id=0, dst_id=1, timestamp=1.0, features=feats))

        data = g.to_pyg_temporal_data()
        assert np.allclose(data.msg[0].numpy(), feats)


class TestTemporalSplit:
    """Tests for temporal train/val/test splitting."""

    def test_split_sizes(self):
        g = TemporalGraph()
        for i in range(100):
            g.add_edge(Edge(src_id=i % 10, dst_id=(i + 1) % 10, timestamp=float(i)))

        train, val, test = g.temporal_split(0.70, 0.15)
        assert train.num_edges == 70
        assert val.num_edges == 15
        assert test.num_edges == 15

    def test_split_preserves_ordering(self):
        g = TemporalGraph()
        for i in range(100):
            g.add_edge(Edge(src_id=0, dst_id=1, timestamp=float(i)))

        train, val, test = g.temporal_split()

        # Train should have earliest edges
        assert train.edges[-1].timestamp < val.edges[0].timestamp
        # Val should come before test
        assert val.edges[-1].timestamp < test.edges[0].timestamp

    def test_split_nodes_included(self):
        g = TemporalGraph()
        g.add_node(Node(0, "account"))
        g.add_node(Node(1, "merchant"))
        for i in range(10):
            g.add_edge(Edge(src_id=0, dst_id=1, timestamp=float(i)))

        train, val, test = g.temporal_split()
        # All subgraphs should have the relevant nodes
        assert train.has_node(0) and train.has_node(1)
