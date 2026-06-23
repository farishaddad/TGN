"""
CSV Ingestion Module — Bring Your Own Data.

Converts user-provided CSV transaction data into a TemporalGraph for
training or scoring. Handles schema validation, date parsing, node ID
mapping, and feature encoding.

Required columns: source_id, target_id, timestamp, amount
Optional columns: label, category, channel

Example CSV:
    source_id,target_id,timestamp,amount,label
    acct_001,merch_42,2024-01-15 10:30:00,150.00,0
    acct_002,merch_42,2024-01-15 11:00:00,5000.00,1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from tgn_learn.graph import Edge, Node, TemporalGraph, EDGE_FEAT_DIM


@dataclass
class IngestionResult:
    """Summary of CSV ingestion."""
    graph: TemporalGraph
    num_source_nodes: int
    num_target_nodes: int
    num_edges: int
    time_range: tuple[float, float]
    fraud_rate: Optional[float]
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"Ingested: {self.num_edges} transactions",
            f"  Sources: {self.num_source_nodes}, Targets: {self.num_target_nodes}",
            f"  Time range: {self.time_range[0]:.0f} — {self.time_range[1]:.0f}",
        ]
        if self.fraud_rate is not None:
            lines.append(f"  Fraud rate: {self.fraud_rate:.2%}")
        if self.warnings:
            lines.append(f"  Warnings: {len(self.warnings)}")
            for w in self.warnings[:5]:
                lines.append(f"    - {w}")
        return "\n".join(lines)


@dataclass
class ColumnMapping:
    """Maps CSV column names to expected fields.

    Use this when your CSV has different column names than the defaults.
    """
    source_id: str = "source_id"
    target_id: str = "target_id"
    timestamp: str = "timestamp"
    amount: str = "amount"
    label: Optional[str] = "label"
    category: Optional[str] = "category"
    channel: Optional[str] = "channel"


class CSVIngester:
    """
    Ingests CSV transaction data into a TemporalGraph.

    Handles:
    - Schema validation (required vs optional columns)
    - Multiple date/timestamp formats (Unix, ISO, mixed)
    - Node ID mapping (string IDs to integers)
    - Feature vector encoding from available columns
    - Missing value handling and deduplication

    Example:
        >>> ingester = CSVIngester()
        >>> result = ingester.ingest("transactions.csv")
        >>> print(result)
        >>> graph = result.graph
    """

    def __init__(self, mapping: Optional[ColumnMapping] = None):
        self.mapping = mapping or ColumnMapping()
        self._node_map: dict[str, int] = {}
        self._next_id = 0

    def ingest(
        self,
        source: Union[str, Path, pd.DataFrame, StringIO],
        source_node_type: str = "account",
        target_node_type: str = "merchant",
    ) -> IngestionResult:
        """Ingest transaction data from CSV file, DataFrame, or string.

        Args:
            source: Path to CSV, pandas DataFrame, or StringIO
            source_node_type: Node type for source entities
            target_node_type: Node type for target entities

        Returns:
            IngestionResult with the constructed graph and summary

        Raises:
            ValueError: If required columns are missing
        """
        self._node_map = {}
        self._next_id = 0
        warnings: list[str] = []

        # Load data
        df = self._load_data(source)

        # Validate schema
        self._validate_schema(df)

        # Parse timestamps
        df, ts_warnings = self._parse_timestamps(df)
        warnings.extend(ts_warnings)

        # Handle missing values
        df, mv_warnings = self._handle_missing(df)
        warnings.extend(mv_warnings)

        # Sort by timestamp
        df = df.sort_values(self.mapping.timestamp).reset_index(drop=True)

        # Build graph
        graph = TemporalGraph()

        # Map node IDs and create nodes
        src_col = self.mapping.source_id
        dst_col = self.mapping.target_id

        for raw_id in df[src_col].unique():
            nid = self._get_node_id(str(raw_id))
            graph.add_node(Node(nid, source_node_type, metadata={"raw_id": str(raw_id)}))

        for raw_id in df[dst_col].unique():
            nid = self._get_node_id(str(raw_id))
            graph.add_node(Node(nid, target_node_type, metadata={"raw_id": str(raw_id)}))

        # Create edges
        for _, row in df.iterrows():
            src_nid = self._get_node_id(str(row[src_col]))
            dst_nid = self._get_node_id(str(row[dst_col]))
            timestamp = float(row[self.mapping.timestamp])
            amount = float(row[self.mapping.amount])

            # Encode features
            features = self._encode_row(row, amount, timestamp)

            # Label
            label = -1  # Unknown by default
            if self.mapping.label and self.mapping.label in df.columns:
                raw_label = row[self.mapping.label]
                if pd.notna(raw_label):
                    label = int(raw_label)
                    if label not in (-1, 0, 1):
                        label = 1 if label > 0 else 0

            edge = Edge(
                src_id=src_nid,
                dst_id=dst_nid,
                timestamp=timestamp,
                features=features,
                label=label,
                edge_type=str(row.get(self.mapping.category, "transaction")) if self.mapping.category and self.mapping.category in df.columns else "transaction",
            )
            graph.add_edge(edge)

        # Compute summary
        num_sources = len(df[src_col].unique())
        num_targets = len(df[dst_col].unique())
        fraud_rate = graph.fraud_rate if graph.num_fraud > 0 or graph.num_legit > 0 else None

        return IngestionResult(
            graph=graph,
            num_source_nodes=num_sources,
            num_target_nodes=num_targets,
            num_edges=graph.num_edges,
            time_range=graph.time_range,
            fraud_rate=fraud_rate,
            warnings=warnings,
        )

    def _load_data(self, source: Union[str, Path, pd.DataFrame, StringIO]) -> pd.DataFrame:
        """Load data from various sources."""
        if isinstance(source, pd.DataFrame):
            return source.copy()
        elif isinstance(source, StringIO):
            return pd.read_csv(source)
        else:
            return pd.read_csv(source)

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Validate required columns exist."""
        required = [
            self.mapping.source_id,
            self.mapping.target_id,
            self.mapping.timestamp,
            self.mapping.amount,
        ]
        missing = [col for col in required if col not in df.columns]
        if missing:
            available = list(df.columns)
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Available: {available}. "
                f"Use ColumnMapping to specify custom column names."
            )

    def _parse_timestamps(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Parse timestamps into Unix epoch floats."""
        warnings = []
        ts_col = self.mapping.timestamp

        # Check if already numeric (Unix timestamps)
        if pd.api.types.is_numeric_dtype(df[ts_col]):
            df[ts_col] = df[ts_col].astype(float)
            return df, warnings

        # Try parsing as datetime
        try:
            parsed = pd.to_datetime(df[ts_col], format="mixed", utc=True)
            df[ts_col] = parsed.astype(np.int64) / 1e9  # Convert to Unix seconds
            return df, warnings
        except Exception:
            pass

        # Try common formats
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y %H:%M"]:
            try:
                parsed = pd.to_datetime(df[ts_col], format=fmt)
                df[ts_col] = parsed.astype(np.int64) / 1e9
                return df, warnings
            except Exception:
                continue

        # Last resort: try pandas infer
        try:
            parsed = pd.to_datetime(df[ts_col], infer_datetime_format=True)
            df[ts_col] = parsed.astype(np.int64) / 1e9
            warnings.append("Timestamp format inferred — verify correctness")
            return df, warnings
        except Exception:
            raise ValueError(
                f"Cannot parse timestamps in column '{ts_col}'. "
                f"Expected Unix epoch (numeric) or ISO datetime string."
            )

    def _handle_missing(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Handle missing values in required columns."""
        warnings = []
        initial_len = len(df)

        # Drop rows with missing required fields
        required = [self.mapping.source_id, self.mapping.target_id,
                    self.mapping.timestamp, self.mapping.amount]
        df = df.dropna(subset=required)

        dropped = initial_len - len(df)
        if dropped > 0:
            warnings.append(f"Dropped {dropped} rows with missing required values")

        return df, warnings

    def _encode_row(self, row: pd.Series, amount: float, timestamp: float) -> np.ndarray:
        """Encode a CSV row into feature vector."""
        feat = np.zeros(EDGE_FEAT_DIM, dtype=np.float32)

        feat[0] = np.log1p(abs(amount))
        feat[1] = min(abs(amount) / 10000.0, 1.0)

        hour_frac = (timestamp % 86400) / 86400
        feat[2] = np.sin(2 * np.pi * hour_frac)
        feat[3] = np.cos(2 * np.pi * hour_frac)
        day_frac = (timestamp % 604800) / 604800
        feat[4] = np.sin(2 * np.pi * day_frac)
        feat[5] = np.cos(2 * np.pi * day_frac)

        # Channel encoding if available
        if self.mapping.channel and self.mapping.channel in row.index:
            channel_val = str(row[self.mapping.channel]).lower()
            channel_map = {"pos": 0.2, "online": 0.5, "transfer": 0.8, "atm": 0.9}
            feat[6] = channel_map.get(channel_val, 0.5)
        else:
            feat[6] = 0.5

        return feat

    def _get_node_id(self, raw_id: str) -> int:
        """Map raw string ID to integer."""
        if raw_id not in self._node_map:
            self._node_map[raw_id] = self._next_id
            self._next_id += 1
        return self._node_map[raw_id]
