"""
Device/Account Event Graph (Saldaña-Ulloa et al., Algorithms 2024).

Temporal graph of registration and account-change events (not transactions).
Provides early-warning signals that precede card fraud — device registration,
card binding, address changes often happen minutes before an attack.

Fusing card registration, device registration, and bank account registration
events consistently outperforms single-event-type graphs for fraud detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class EventType(str, Enum):
    """Account/device event types tracked."""

    CARD_BIND = "CARD_BIND"
    DEVICE_REG = "DEVICE_REG"
    ADDR_CHANGE = "ADDR_CHANGE"
    PHONE_CHANGE = "PHONE_CHANGE"
    BENEFICIARY_ADD = "BENEFICIARY_ADD"


# Feature dimension for device events
DEVICE_EVENT_FEAT_DIM = 8


@dataclass
class DeviceEvent:
    """A single device/account event.

    Attributes:
        account_id: Account that triggered the event
        event_type: Type of event
        timestamp: When the event occurred
        features: Feature vector [DEVICE_EVENT_FEAT_DIM]
        related_entity: Optional related entity (device ID, card ID, etc.)
        is_suspicious: Whether this event is labelled suspicious
    """

    account_id: int
    event_type: EventType
    timestamp: float
    features: np.ndarray
    related_entity: Optional[int] = None
    is_suspicious: bool = False


@dataclass
class DeviceEventEdge:
    """An edge in the device event graph.

    Connects account → related_entity (or account → account for
    beneficiary additions).
    """

    src_id: int
    dst_id: int
    event_type: EventType
    timestamp: float
    features: np.ndarray


class DeviceEventGraph:
    """Temporal graph of registration and account-change events.

    Unlike the transaction graph (which models money flow), this graph
    models administrative/setup events. Patterns like rapid device
    registration followed by card binding are strong fraud precursors.

    Args:
        num_accounts: Expected number of accounts (for pre-allocation)
    """

    def __init__(self, num_accounts: int = 1000):
        self.num_accounts = num_accounts
        self.events: list[DeviceEvent] = []
        self.edges: list[DeviceEventEdge] = []
        self._next_entity_id = num_accounts  # Entity IDs start after accounts

    @property
    def num_events(self) -> int:
        return len(self.events)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def add_event(
        self,
        account_id: int,
        event_type: EventType,
        timestamp: float,
        features: Optional[np.ndarray] = None,
        related_entity: Optional[int] = None,
        is_suspicious: bool = False,
    ) -> DeviceEvent:
        """Add a device/account event to the graph.

        Args:
            account_id: Account performing the action
            event_type: Type of event
            timestamp: When it happened
            features: Feature vector (auto-generated if None)
            related_entity: Related device/card/account ID
            is_suspicious: Label for this event

        Returns:
            The created DeviceEvent
        """
        if features is None:
            features = self._build_default_features(event_type, timestamp)

        if related_entity is None:
            related_entity = self._next_entity_id
            self._next_entity_id += 1

        event = DeviceEvent(
            account_id=account_id,
            event_type=event_type,
            timestamp=timestamp,
            features=features,
            related_entity=related_entity,
            is_suspicious=is_suspicious,
        )
        self.events.append(event)

        # Create edge: account → related entity
        edge = DeviceEventEdge(
            src_id=account_id,
            dst_id=related_entity,
            event_type=event_type,
            timestamp=timestamp,
            features=features,
        )
        self.edges.append(edge)

        return event

    def get_events_for_account(
        self,
        account_id: int,
        event_type: Optional[EventType] = None,
        time_window: Optional[tuple[float, float]] = None,
    ) -> list[DeviceEvent]:
        """Get events for a specific account, optionally filtered.

        Args:
            account_id: Account to look up
            event_type: Filter by type (None = all types)
            time_window: (start, end) timestamp filter

        Returns:
            List of matching DeviceEvent objects
        """
        results = []
        for event in self.events:
            if event.account_id != account_id:
                continue
            if event_type is not None and event.event_type != event_type:
                continue
            if time_window is not None:
                if event.timestamp < time_window[0] or event.timestamp > time_window[1]:
                    continue
            results.append(event)
        return results

    def get_velocity(
        self,
        account_id: int,
        window_seconds: float = 300.0,
        reference_time: Optional[float] = None,
    ) -> int:
        """Count events within a time window for an account.

        High velocity (many events in short window) is a fraud precursor.

        Args:
            account_id: Account to check
            window_seconds: Time window in seconds
            reference_time: End of window (default: latest event time)

        Returns:
            Number of events in window
        """
        if reference_time is None:
            if not self.events:
                return 0
            reference_time = max(e.timestamp for e in self.events)

        t_start = reference_time - window_seconds
        count = sum(
            1 for e in self.events
            if e.account_id == account_id
            and t_start <= e.timestamp <= reference_time
        )
        return count

    def compute_risk_features(self, account_id: int, timestamp: float) -> dict[str, float]:
        """Compute risk features from device event history.

        Returns features useful for the ensemble's RF head or meta-learner.

        Args:
            account_id: Account to analyse
            timestamp: Current time reference

        Returns:
            Dict of named features
        """
        account_events = self.get_events_for_account(account_id)

        # Velocity features
        velocity_5m = self.get_velocity(account_id, 300.0, timestamp)
        velocity_1h = self.get_velocity(account_id, 3600.0, timestamp)
        velocity_24h = self.get_velocity(account_id, 86400.0, timestamp)

        # Type diversity
        types_seen = set(e.event_type for e in account_events if e.timestamp <= timestamp)
        type_diversity = len(types_seen) / len(EventType)

        # Recency of last event
        recent_events = [e for e in account_events if e.timestamp <= timestamp]
        if recent_events:
            last_event_age = timestamp - max(e.timestamp for e in recent_events)
        else:
            last_event_age = float("inf")

        # New device registration in last hour
        new_devices = sum(
            1 for e in account_events
            if e.event_type == EventType.DEVICE_REG
            and timestamp - 3600 <= e.timestamp <= timestamp
        )

        return {
            "device_velocity_5m": float(velocity_5m),
            "device_velocity_1h": float(velocity_1h),
            "device_velocity_24h": float(velocity_24h),
            "event_type_diversity": type_diversity,
            "last_event_age_seconds": min(last_event_age, 1e6),
            "new_devices_1h": float(new_devices),
        }

    def _build_default_features(self, event_type: EventType, timestamp: float) -> np.ndarray:
        """Generate a default feature vector for an event."""
        feat = np.zeros(DEVICE_EVENT_FEAT_DIM, dtype=np.float32)
        # One-hot encode event type
        type_idx = list(EventType).index(event_type)
        feat[type_idx] = 1.0
        # Time-of-day features
        hour_frac = (timestamp % 86400) / 86400
        feat[5] = np.sin(2 * np.pi * hour_frac)
        feat[6] = np.cos(2 * np.pi * hour_frac)
        # Day-of-week
        day_frac = (timestamp % 604800) / 604800
        feat[7] = np.sin(2 * np.pi * day_frac)
        return feat
