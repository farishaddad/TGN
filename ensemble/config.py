"""
Unified configuration for the ensemble fraud detection system.

Extends TGNConfig with ensemble-specific settings. All new config
fields default to values that reproduce the existing single-model
behaviour — enabling ensemble features is opt-in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnsembleConfig:
    """Top-level ensemble configuration.

    Controls which ensemble components are active. All flags default
    to False so the system degrades gracefully to single-TGN mode.

    Attributes:
        # Phase 1
        use_multiscale_time: Use MultiScaleTimeEncoder instead of single-scale
        use_rf_head: Fit RF scoring head post-training
        rf_n_estimators: Trees in RF ensemble
        rf_max_depth: Max tree depth (None=unlimited)
        use_graph_smote: Apply topology-preserving oversampling
        smote_k_hop: k-hop neighbourhood for synthetic edges
        smote_minority_ratio: Target minority class ratio

        # Phase 2
        use_dual_memory: Split memory into stable + transient tracks
        stable_memory_dim: Dimension of stable baseline memory
        transient_memory_dim: Dimension of transient deviation memory
        stable_update_alpha: EMA decay for stable memory updates
        use_flow_dag: Build event-centric fund-flow DAG
        flow_dag_time_window_hours: Max hours between chained events
        use_device_events: Include device/account event graph

        # Phase 3
        use_lambda_inference: Enable batch+realtime Lambda architecture
        cache_max_age_seconds: Max staleness for cached embeddings
        batch_refresh_interval: Seconds between batch embedding refreshes

        # Phase 4
        enabled_detectors: List of active detector names
        meta_learner_n_estimators: Trees in LightGBM meta-learner
        use_two_hurdle_filter: Enable FP suppression filter
        recon_threshold: Reconstruction score threshold (95th pct)
        deviation_threshold: Sigma threshold for deviation filter

        # Phase 5
        use_drift_detection: Enable latent-space drift monitoring
        drift_cusum_threshold: CUSUM statistic trigger threshold
        use_threshold_adaptation: Adapt risk thresholds per-segment
    """

    # --- Phase 1: Quick Wins ---
    use_multiscale_time: bool = False
    use_rf_head: bool = False
    rf_n_estimators: int = 200
    rf_max_depth: Optional[int] = 10
    use_graph_smote: bool = False
    smote_k_hop: int = 2
    smote_minority_ratio: float = 0.1

    # --- Phase 2: Dual-Track Memory + Event Graphs ---
    use_dual_memory: bool = False
    stable_memory_dim: int = 32
    transient_memory_dim: int = 32
    stable_update_alpha: float = 0.05
    use_flow_dag: bool = False
    flow_dag_time_window_hours: float = 24.0
    use_device_events: bool = False

    # --- Phase 3: Lambda Inference ---
    use_lambda_inference: bool = False
    cache_max_age_seconds: float = 3600.0
    batch_refresh_interval: float = 3600.0

    # --- Phase 4: Ensemble Detectors + Meta-Learner ---
    enabled_detectors: list[str] = field(default_factory=lambda: [
        "tgn_memory",
        "rf_structural",
        "flow_dag",
        "semantic",
        "drift_monitor",
    ])
    meta_learner_n_estimators: int = 500
    use_two_hurdle_filter: bool = False
    recon_threshold: float = 0.95
    deviation_threshold: float = 3.0

    # --- Phase 5: Adaptive Maintenance ---
    use_drift_detection: bool = False
    drift_cusum_threshold: float = 5.0
    use_threshold_adaptation: bool = False

    def phase1_enabled(self) -> bool:
        """Any Phase 1 feature active."""
        return self.use_multiscale_time or self.use_rf_head or self.use_graph_smote

    def phase2_enabled(self) -> bool:
        """Any Phase 2 feature active."""
        return self.use_dual_memory or self.use_flow_dag or self.use_device_events

    def full_ensemble_enabled(self) -> bool:
        """Full multi-detector ensemble is active."""
        return len(self.enabled_detectors) > 1


@dataclass
class RiskThresholds:
    """Configurable thresholds for risk tier classification.

    These may be adapted per-segment by the ThresholdAdapter (Phase 5).
    """

    medium: float = 0.30
    high: float = 0.60
    critical: float = 0.85
