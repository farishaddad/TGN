"""Tests for CSV ingestion module."""

from io import StringIO

import pandas as pd
import pytest

from tgn_learn.ingestion import CSVIngester
from tgn_learn.ingestion.csv_ingester import ColumnMapping


SAMPLE_CSV = """source_id,target_id,timestamp,amount,label
acct_001,merch_42,1700000000,150.00,0
acct_002,merch_42,1700001000,5000.00,1
acct_001,merch_99,1700002000,25.50,0
acct_003,merch_42,1700003000,300.00,0
acct_002,merch_99,1700004000,7500.00,1
"""

SAMPLE_CSV_DATETIME = """source_id,target_id,timestamp,amount,label
alice,shop_a,2024-01-15 10:30:00,100.00,0
bob,shop_b,2024-01-15 11:00:00,200.00,1
"""

SAMPLE_CSV_MISSING = """source_id,target_id,timestamp,amount,label
acct_001,merch_42,1700000000,150.00,0
acct_002,,1700001000,500.00,0
,merch_42,1700002000,25.00,0
acct_003,merch_42,,300.00,0
"""


class TestCSVIngester:
    """Tests for CSVIngester."""

    def test_basic_ingestion(self):
        ingester = CSVIngester()
        result = ingester.ingest(StringIO(SAMPLE_CSV))

        assert result.num_edges == 5
        assert result.num_source_nodes == 3  # acct_001, acct_002, acct_003
        assert result.num_target_nodes == 2  # merch_42, merch_99
        assert result.fraud_rate == pytest.approx(2 / 5)

    def test_graph_structure(self):
        ingester = CSVIngester()
        result = ingester.ingest(StringIO(SAMPLE_CSV))
        graph = result.graph

        assert graph.num_nodes == 5  # 3 accounts + 2 merchants
        assert graph.num_edges == 5
        # Temporal ordering preserved
        timestamps = [e.timestamp for e in graph.edges]
        assert timestamps == sorted(timestamps)

    def test_node_types(self):
        ingester = CSVIngester()
        result = ingester.ingest(StringIO(SAMPLE_CSV))
        graph = result.graph

        types = graph.node_types()
        assert types["account"] == 3
        assert types["merchant"] == 2

    def test_datetime_parsing(self):
        ingester = CSVIngester()
        result = ingester.ingest(StringIO(SAMPLE_CSV_DATETIME))

        assert result.num_edges == 2
        # Timestamps should be parsed to Unix epoch
        graph = result.graph
        assert graph.edges[0].timestamp > 0
        assert graph.edges[0].timestamp < graph.edges[1].timestamp

    def test_missing_values_handled(self):
        ingester = CSVIngester()
        result = ingester.ingest(StringIO(SAMPLE_CSV_MISSING))

        # Only the first row should survive (others have missing required fields)
        assert result.num_edges == 1
        assert len(result.warnings) > 0
        assert "Dropped" in result.warnings[0]

    def test_missing_required_columns_raises(self):
        bad_csv = "col_a,col_b\n1,2\n"
        ingester = CSVIngester()
        with pytest.raises(ValueError, match="Missing required columns"):
            ingester.ingest(StringIO(bad_csv))

    def test_custom_column_mapping(self):
        csv_data = """src,dst,ts,amt,is_fraud
a,b,1700000000,100,0
c,d,1700001000,200,1
"""
        mapping = ColumnMapping(
            source_id="src", target_id="dst",
            timestamp="ts", amount="amt", label="is_fraud",
        )
        ingester = CSVIngester(mapping)
        result = ingester.ingest(StringIO(csv_data))

        assert result.num_edges == 2
        assert result.graph.num_fraud == 1

    def test_no_label_column(self):
        csv_data = """source_id,target_id,timestamp,amount
a,b,1700000000,100
c,d,1700001000,200
"""
        mapping = ColumnMapping(label=None)
        ingester = CSVIngester(mapping)
        result = ingester.ingest(StringIO(csv_data))

        assert result.num_edges == 2
        # All should be unlabeled
        for edge in result.graph.edges:
            assert edge.label == -1

    def test_from_dataframe(self):
        df = pd.DataFrame({
            "source_id": ["a", "b", "a"],
            "target_id": ["x", "x", "y"],
            "timestamp": [1000.0, 2000.0, 3000.0],
            "amount": [50.0, 100.0, 75.0],
        })
        ingester = CSVIngester()
        result = ingester.ingest(df)

        assert result.num_edges == 3
        assert result.num_source_nodes == 2
        assert result.num_target_nodes == 2

    def test_features_encoded(self):
        ingester = CSVIngester()
        result = ingester.ingest(StringIO(SAMPLE_CSV))

        for edge in result.graph.edges:
            assert edge.features.shape[0] == 20
            # log_amount should be > 0 for positive amounts
            assert edge.features[0] > 0

    def test_summary_string(self):
        ingester = CSVIngester()
        result = ingester.ingest(StringIO(SAMPLE_CSV))
        s = str(result)
        assert "5 transactions" in s
        assert "Fraud rate" in s

    def test_channel_column(self):
        csv_data = """source_id,target_id,timestamp,amount,channel
a,b,1700000000,100,online
c,d,1700001000,200,pos
"""
        ingester = CSVIngester()
        result = ingester.ingest(StringIO(csv_data))

        # Channel should be encoded in feature[6]
        edges = result.graph.edges
        assert edges[0].features[6] == pytest.approx(0.5)  # online
        assert edges[1].features[6] == pytest.approx(0.2)  # pos
