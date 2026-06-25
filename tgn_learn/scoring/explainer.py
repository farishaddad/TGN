"""
Fraud explainer — generates human-readable explanations for fraud scores.

Produces 2–4 bullet-point reasons for each scoring decision, mapping model
signals to domain-meaningful language. Designed for demo audiences who are
fraud domain experts but not ML experts.

Example:
    >>> explainer = FraudExplainer()
    >>> signals = explainer.explain(result, edge, graph)
    >>> for s in signals:
    ...     print(f"{s.icon} {s.title}: {s.detail}")
    🔴 Velocity Burst: 8 transactions in last 5 minutes (typical: 0-1)
    🔴 Amount Spike: £2,400 is 12.4x above account average
    🟡 New Merchant: First ever transaction with this merchant
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from tgn_learn.graph import Edge, TemporalGraph
from tgn_learn.scoring.scorer import ScoringResult


@dataclass
class FraudSignal:
    """A single explainable signal contributing to a fraud score.

    Attributes:
        icon: Emoji indicator (🔴 high risk, 🟡 moderate, 🟢 low)
        title: Short label (e.g. "Velocity Burst")
        detail: One-sentence human-readable explanation
        contribution: Approximate contribution to the score [0.0, 1.0]
    """

    icon: str
    title: str
    detail: str
    contribution: float = 0.0


class FraudExplainer:
    """Generate human-readable explanations for fraud scores.

    Analyses the transaction context (graph history, amount patterns,
    temporal patterns) and produces ranked FraudSignal objects explaining
    why the model scored a transaction the way it did.

    This is a rule-based explainer that maps observable graph/transaction
    features to domain-meaningful language. It does NOT require access to
    model internals — it works purely from the graph context and edge features.

    Args:
        velocity_window_seconds: Time window for velocity checks (default: 300s = 5min)
        amount_spike_threshold: Multiplier above average to flag (default: 5x)
        unusual_hour_start: Start of unusual hour range (default: 23)
        unusual_hour_end: End of unusual hour range (default: 6)
    """

    def __init__(
        self,
        velocity_window_seconds: float = 300.0,
        amount_spike_threshold: float = 5.0,
        unusual_hour_start: int = 23,
        unusual_hour_end: int = 6,
    ):
        self.velocity_window = velocity_window_seconds
        self.amount_spike_threshold = amount_spike_threshold
        self.unusual_hour_start = unusual_hour_start
        self.unusual_hour_end = unusual_hour_end

    def explain(
        self,
        result: ScoringResult,
        edge: Edge,
        graph: TemporalGraph,
    ) -> list[FraudSignal]:
        """Generate explanation signals for a scored transaction.

        Returns a ranked list of FraudSignal objects, sorted by
        contribution (highest first). Typically returns 2–4 signals.

        Args:
            result: The scoring result for this transaction
            edge: The Edge being scored
            graph: The full TemporalGraph (for context lookups)

        Returns:
            List of FraudSignal objects, sorted by contribution descending
        """
        signals: list[FraudSignal] = []

        # --- Check velocity anomaly ---
        velocity_signal = self._check_velocity(edge, graph)
        if velocity_signal:
            signals.append(velocity_signal)

        # --- Check amount anomaly ---
        amount_signal = self._check_amount_spike(edge, graph)
        if amount_signal:
            signals.append(amount_signal)

        # --- Check new merchant/recipient ---
        new_merchant_signal = self._check_new_merchant(edge, graph)
        if new_merchant_signal:
            signals.append(new_merchant_signal)

        # --- Check time-of-day anomaly ---
        time_signal = self._check_unusual_time(edge)
        if time_signal:
            signals.append(time_signal)

        # --- Check amount rounding (money laundering indicator) ---
        rounding_signal = self._check_round_amount(edge)
        if rounding_signal:
            signals.append(rounding_signal)

        # --- If score is high but no signals found, add a generic one ---
        if not signals and result.risk_score >= 0.60:
            signals.append(FraudSignal(
                icon="🟡",
                title="TGN Memory Signal",
                detail="Model's temporal memory flagged unusual pattern in account history",
                contribution=0.5,
            ))

        return sorted(signals, key=lambda s: -s.contribution)

    def _check_velocity(self, edge: Edge, graph: TemporalGraph) -> Optional[FraudSignal]:
        """Check for transaction velocity burst from the source account."""
        t_start = edge.timestamp - self.velocity_window
        t_end = edge.timestamp

        # Get recent edges from same source within the time window
        recent = [
            e for e in graph.edges_for_node(edge.src_id)
            if e.src_id == edge.src_id and t_start <= e.timestamp <= t_end
        ]

        if len(recent) >= 3:
            window_min = int(self.velocity_window / 60)
            return FraudSignal(
                icon="🔴",
                title="Velocity Burst",
                detail=f"{len(recent)} transactions in last {window_min} minutes (typical: 0-1)",
                contribution=0.4,
            )
        return None

    def _check_amount_spike(self, edge: Edge, graph: TemporalGraph) -> Optional[FraudSignal]:
        """Check if this transaction amount is abnormally high for the account."""
        # Get amount from edge features (feature[0] is log_amount)
        this_amount = self._get_amount(edge)
        if this_amount <= 0:
            return None

        # Calculate account's average amount from history
        account_edges = [
            e for e in graph.edges_for_node(edge.src_id)
            if e.src_id == edge.src_id and e.timestamp < edge.timestamp
        ]

        if len(account_edges) < 3:
            # Not enough history to determine average
            return None

        amounts = [self._get_amount(e) for e in account_edges]
        amounts = [a for a in amounts if a > 0]
        if not amounts:
            return None

        avg_amount = np.mean(amounts)
        if avg_amount > 0 and this_amount > self.amount_spike_threshold * avg_amount:
            ratio = this_amount / avg_amount
            return FraudSignal(
                icon="🔴",
                title="Amount Spike",
                detail=f"£{this_amount:,.0f} is {ratio:.1f}x above account average (£{avg_amount:,.0f})",
                contribution=0.3,
            )
        return None

    def _check_new_merchant(self, edge: Edge, graph: TemporalGraph) -> Optional[FraudSignal]:
        """Check if this is the first interaction with the destination."""
        prior = [
            e for e in graph.edges_for_node(edge.src_id)
            if e.src_id == edge.src_id
            and e.dst_id == edge.dst_id
            and e.timestamp < edge.timestamp
        ]

        if len(prior) == 0:
            dst_node = graph.get_node(edge.dst_id)
            dst_label = "merchant" if dst_node and dst_node.node_type == "merchant" else "recipient"
            return FraudSignal(
                icon="🟡",
                title="New Merchant",
                detail=f"First ever transaction with this {dst_label} (ID: {edge.dst_id})",
                contribution=0.15,
            )
        return None

    def _check_unusual_time(self, edge: Edge) -> Optional[FraudSignal]:
        """Check if the transaction occurs at an unusual hour."""
        # Convert timestamp to hour of day
        hour = (edge.timestamp % 86400) / 3600

        if hour >= self.unusual_hour_start or hour < self.unusual_hour_end:
            hour_display = int(hour)
            minute_display = int((hour - hour_display) * 60)
            return FraudSignal(
                icon="🟡",
                title="Unusual Time",
                detail=f"Transaction at {hour_display:02d}:{minute_display:02d} (outside normal hours)",
                contribution=0.15,
            )
        return None

    def _check_round_amount(self, edge: Edge) -> Optional[FraudSignal]:
        """Check for suspiciously round amounts (money laundering indicator)."""
        amount = self._get_amount(edge)
        if amount >= 1000 and amount % 1000 == 0:
            return FraudSignal(
                icon="🟡",
                title="Round Amount",
                detail=f"Exactly £{amount:,.0f} — round amounts are a layering indicator",
                contribution=0.10,
            )
        return None

    @staticmethod
    def _get_amount(edge: Edge) -> float:
        """Extract the transaction amount from edge features.

        Feature[0] is log_amount: amount = exp(feature[0]) - 1
        """
        if edge.features is not None and len(edge.features) > 0:
            return float(np.exp(edge.features[0]) - 1)
        return 0.0
