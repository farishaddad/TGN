# Ensemble TGN Fraud Detection — Design Document

---

## 0. Isolation Architecture — Demo Separation

**The ensemble must be developed as a fully independent codebase alongside the original,
not as modifications to it. The two systems must be demoable independently.**

### Directory Structure

```
TGN/
├── tgn_learn/              ← ORIGINAL — do not modify any existing file
│   ├── model/
│   ├── training/
│   ├── scoring/
│   └── ...
│
├── ensemble/               ← NEW PACKAGE — mirror structure, fully independent
│   ├── __init__.py
│   ├── model/              (DualTrackMemory, MultiScaleTimeEncoder, RFHead, etc.)
│   ├── graphs/             (FundFlowDAG, DeviceEventGraph)
│   ├── embedding/          (EmbeddingCache, BatchEmbedder, RTEmbedder)
│   ├── detectors/          (BaseDetector + 5 implementations)
│   ├── training/           (EnsembleTrainer, GraphSMOTE)
│   ├── scoring/            (EnsembleScorer, EnsembleScoringResult)
│   └── maintenance/        (DriftDetector, ThresholdAdapter)
│
├── app/
│   ├── main.py             ← add mode selector (see below)
│   ├── pages/              ← ORIGINAL pages — do not modify
│   │   ├── 1_Generate_Data.py
│   │   ├── 2_Explore_Graph.py
│   │   ├── 3_Train_Model.py
│   │   ├── 4_Score_Transactions.py
│   │   └── 5_Upload_CSV.py
│   └── ensemble_pages/     ← NEW — parallel pages for ensemble demo
│       ├── 1_Generate_Data.py      (adds device/account event graphs)
│       ├── 2_Explore_Graph.py      (adds multi-graph view)
│       ├── 3_Train_Ensemble.py     (trains all detectors)
│       ├── 4_Score_Transactions.py (shows detector breakdown + explainer)
│       ├── 5_Upload_CSV.py         (same as original)
│       ├── 6_Pattern_Visualiser.py (NEW)
│       └── 7_Why_Ensemble.py       (NEW)
└── docs/
```

### Mode Selector in app/main.py

```python
# app/main.py — add at the top before page routing
mode = st.sidebar.radio(
    "Demo Mode",
    ["Standard TGN", "Ensemble TGN"],
    help="Standard: original single-model TGN. Ensemble: multi-detector approach.",
)
st.session_state["demo_mode"] = mode

# Route to correct page directory based on mode
pages_dir = "app/ensemble_pages" if mode == "Ensemble TGN" else "app/pages"
```

### Key Rule for Kiro
> `tgn_learn/` is read-only for ensemble work. All ensemble code lives in `ensemble/`.
> The ensemble package imports FROM `tgn_learn` where reuse makes sense
> (e.g. `from tgn_learn.graph import TemporalGraph, Edge`) but never modifies it.
> No file under `tgn_learn/` or `app/pages/` is changed during ensemble development.

---

**Target codebase:** `/Users/fahaddad/Documents/TGN`
**Agent:** Kiro
**Research basis:** 18 research papers reviewed June 2026

---

## 1. Executive Summary

This document specifies a redesign of the existing `TGNFraudDetector` from a single-model scorer into a **multi-layer ensemble** that combines five specialised detectors, a Lambda inference architecture, and an adaptive meta-learner. Each enhancement is directly grounded in a specific research paper. The redesign is structured as additive phases so Kiro can implement and validate incrementally without breaking the existing codebase.

**Goal**: Maximise AUPRC (primary metric) at ≤100ms P99 latency on the existing Streamlit app workflow.

---

## 2. Current Architecture Snapshot

```
tgn_learn/
├── model/
│   ├── tgn.py            # TGNFraudDetector — single model
│   ├── embedder.py       # GraphAttentionEmbedding
│   ├── time_encoder.py   # Single-scale Fourier time encoding
│   ├── heads.py          # LinkPredictor (MLP) + NodeClassifier (MLP)
│   ├── neighbor_loader.py
│   └── config.py         # TGNConfig (memory_dim=64, embedding_dim=64)
├── training/
│   ├── trainer.py        # TGNTrainer — single training loop
│   ├── sampler.py        # NegativeSampler
│   ├── metrics.py        # FraudMetrics
│   └── mint.py           # Multi-Network Training
├── scoring/
│   └── scorer.py         # Scorer — single model, isotonic calibration
│                         # Fixed thresholds: medium=0.30, high=0.60, critical=0.85
└── graph.py              # TemporalGraph, Node, Edge (EDGE_FEAT_DIM=20)
```

### Current limitations mapped to research findings

| Limitation | Evidence | Paper Source |
|---|---|---|
| Single-scale Fourier time encoding | Misses fraud patterns at minute, hour, day, week, month scales simultaneously | TempReasoner (Scientific Reports 2026) |
| MLP LinkPredictor ignores subgraph structure | Cannot detect bridge edges, triangle closure with fraud clusters | TGN-SEAL (EPJ Data Science 2026) |
| Single GRU memory conflates stable behaviour with transient deviations | Concept drift causes false positives; lifestyle changes look like fraud | DySA-TGN (DASFAA 2025) |
| MLP NodeClassifier on imbalanced data | Majority-class bias at <2% fraud rate; no feature importances for explainability | NID-TGN (SPACE 2024) |
| Single entity-centric graph only | Misses device/account registration signals that precede card fraud | Salda&ntilde;a-Ulloa et al. (Algorithms 2024) |
| No event-centric fund-flow graph | Money-mule chains invisible in entity graph | Wu & Zhang (BDAIE 2025 / ETGAT) |
| Live GNN inference per transaction | Neighbourhood explosion → high latency at scale | BRIGHT (CIKM 2022, eBay) |
| No concept drift detection | CloudWatch metric monitoring misses structural/relational drift | TGNN-CDD (TGNN paper 2) |
| SMOTE not topology-preserving | Standard oversampling destroys graph community structure | THG-OAFN (PLoS ONE 2025) |
| No semantic per-relation-type encoding | Transaction type (CNP/contactless/online/recurring) treated uniformly | HTGNN (ICAART 2025) |

---

## 3. Target Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  LAYER 0 — MULTI-MODAL GRAPH CONSTRUCTION                          │
│  tgn_learn/graphs/                                                 │
│  ├── entity_graph.py        (existing TemporalGraph, refactored)   │
│  ├── device_event_graph.py  (card/device/account registration)     │
│  └── flow_dag.py            (event-centric fund-flow DAG / ETGAT)  │
└────────────────────────┬───────────────────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────────────────┐
│  LAYER 1 — EMBEDDING (Lambda architecture)                         │
│  tgn_learn/embedding/                                              │
│  ├── batch_embedder.py      (offline pre-compute entity embeddings)│
│  ├── rt_embedder.py         (real-time delta lookup)               │
│  ├── buyer_subgraph.py      (card-centric sub-model)               │
│  ├── seller_subgraph.py     (merchant-centric sub-model)           │
│  └── embedding_cache.py     (dict-based cache, Redis in prod)      │
└────────────────────────┬───────────────────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────────────────┐
│  LAYER 2 — SPECIALISED DETECTORS (parallel)                        │
│  tgn_learn/detectors/                                              │
│  ├── tgn_detector.py        (existing TGN memory deviation)        │
│  ├── rf_detector.py         (weighted RF on TGN embeddings)        │
│  ├── flow_dag_detector.py   (ETGAT path anomaly scoring)           │
│  ├── semantic_detector.py   (HTGNN per-relation-type encoding)     │
│  └── drift_monitor.py       (TGNN-CDD latent space autoencoder)    │
└────────────────────────┬───────────────────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────────────────┐
│  LAYER 3 — ENSEMBLE FUSION                                         │
│  tgn_learn/ensemble/                                               │
│  ├── meta_learner.py        (LightGBM on detector scores + feats)  │
│  └── calibrator.py          (isotonic calibration, moved here)     │
└────────────────────────┬───────────────────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────────────────┐
│  LAYER 4 — DECISION + FP FILTER                                    │
│  tgn_learn/scoring/scorer.py  (extended)                           │
│  ├── two_hurdle_filter()    (deviation score FP suppression)       │
│  └── risk_tier_classifier() (updated thresholds + segment logic)   │
└────────────────────────┬───────────────────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────────────────┐
│  LAYER 5 — ADAPTIVE MAINTENANCE                                    │
│  tgn_learn/maintenance/                                            │
│  ├── drift_detector.py      (latent-space CUSUM / ADWIN trigger)   │
│  ├── graph_smote.py         (topology-preserving oversampling)     │
│  └── threshold_adapter.py   (per-segment threshold calibration)    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 4. Phased Implementation Plan

### Phase 1 — Quick Wins (no architecture change)
*Expected: 1–2 days. No breaking changes.*

**1A: Replace time_encoder.py with multi-scale variant**
- **File**: `tgn_learn/model/time_encoder.py`
- **Change**: Keep existing `TimeEncoder` class, add `MultiScaleTimeEncoder` alongside it
- **What it does**: Creates separate sinusoidal encoders for five temporal scales (minute, hour, day, week, month), fuses with learned scale weights via `nn.Linear`
- **Source**: TempReasoner (Scientific Reports 2026)

```python
# New class to add in time_encoder.py
class MultiScaleTimeEncoder(nn.Module):
    """Multi-scale temporal encoding (TempReasoner, 2026).
    
    Runs separate Fourier encoders at five temporal granularities and
    fuses with learned scale weights. Detects fraud patterns that
    manifest at different timescales simultaneously (minute-level
    card-testing bursts AND week-level ring coordination).
    
    Scales: [60s, 3600s, 86400s, 604800s, 2592000s]
    """
    SCALES = [60.0, 3600.0, 86400.0, 604800.0, 2592000.0]  # sec per unit
    
    def __init__(self, time_dim: int):
        super().__init__()
        self.encoders = nn.ModuleList([
            TimeEncoder(time_dim) for _ in self.SCALES
        ])
        self.scale_weights = nn.Linear(len(self.SCALES) * time_dim, time_dim)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        scale_encs = [enc(t / s) for enc, s in zip(self.encoders, self.SCALES)]
        fused = torch.cat(scale_encs, dim=-1)
        return self.scale_weights(fused)
```

- **Config change** in `TGNConfig`: add `use_multiscale_time: bool = True`
- **Integration**: In `tgn_learn/model/embedder.py`, swap `TimeEncoder` → `MultiScaleTimeEncoder` when `config.use_multiscale_time` is True

---

**1B: Replace MLP LinkPredictor with Weighted Random Forest**
- **Files**: `tgn_learn/model/heads.py`, `tgn_learn/scoring/scorer.py`
- **What it does**: After TGN generates embeddings, feed `z_src ⊕ z_dst ⊕ edge_features` into a `RandomForestClassifier(class_weight='balanced', n_estimators=200)` instead of the MLP. The RF handles class imbalance natively and produces feature importances for explainability.
- **Source**: NID-TGN (SPACE 2024)
- **Note**: The TGN still trains with MLP heads for the contrastive loss (link + node). The RF is the **inference-time scoring head** fitted on TGN-generated embeddings post-training.

```python
# New file: tgn_learn/model/rf_head.py
class RFScoringHead:
    """Random Forest scoring head on TGN embeddings (NID-TGN, 2024).
    
    Fitted after TGN training on the validation set. At inference,
    receives concatenated [z_src, z_dst, edge_features] and outputs
    fraud probability with feature importances.
    
    Handles class imbalance via class_weight='balanced' — no resampling
    needed, which preserves temporal ordering.
    """
    def __init__(self, n_estimators: int = 200, max_depth: int = 10):
        from sklearn.ensemble import RandomForestClassifier
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight='balanced',
            n_jobs=-1,
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray): ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
    def feature_importances(self) -> dict: ...  # for explainability
```

- **Scorer changes**: `scorer.py` — add `rf_head: Optional[RFScoringHead]` attribute; when present, use it instead of the MLP for final score
- **Training change**: After `TGNTrainer.train()` completes, auto-fit the RF head on val embeddings

---

**1C: Add GraphSMOTE to training oversampling**
- **File**: new `tgn_learn/training/graph_smote.py`
- **What it does**: Before training, identify fraud-labelled edges, generate synthetic minority samples by interpolating between existing fraud edge features (SMOTE), then assign synthetic edges to k-hop neighbours of their interpolated parents (topology preservation). Applies only to training data.
- **Source**: THG-OAFN (PLoS ONE 2025)
- **Config**: `TrainingConfig` — add `use_graph_smote: bool = False`, `smote_k_hop: int = 2`

---

### Phase 2 — Dual-Track Memory + Event Graph (core architecture)
*Expected: 3–4 days. New modules, TGN changes are backwards-compatible.*

**2A: Dual-Track Memory Module**
- **File**: new `tgn_learn/model/dual_memory.py`
- **What it does**: Splits TGN memory into two components:
  - `StableMemory`: Updated slowly (once per epoch, via exponential moving average of recent states). Encodes the cardholder's long-term behavioural baseline.
  - `TransientMemory`: Standard GRU, updated per-event. Encodes real-time deviations from the stable baseline.
  - Concatenated output: `s_i(t) = concat(s_stable_i, s_transient_i(t))`
- **Source**: DySA-TGN DGC-MVFF (DASFAA 2025)

```python
# tgn_learn/model/dual_memory.py
class DualTrackMemory(nn.Module):
    """Dual-track memory separating stable baseline from transient deviations.
    
    Based on DySA-TGN's time-invariant / time-varying decomposition (2025).
    
    The fraud signal lives in s_transient (deviation from baseline).
    The stable component provides the reference. This eliminates false
    positives from legitimate lifestyle changes, which update s_stable
    slowly while genuine fraud events spike s_transient.
    
    Args:
        num_nodes: Total node count
        stable_dim: Dimension of stable baseline memory
        transient_dim: Dimension of transient deviation memory  
        raw_msg_dim: Edge feature dimension
        time_dim: Time encoding dimension
        stable_update_alpha: EMA decay for stable memory (default: 0.05)
    """
    def __init__(self, num_nodes, stable_dim, transient_dim,
                 raw_msg_dim, time_dim, stable_update_alpha=0.05):
        super().__init__()
        self.stable = nn.Embedding(num_nodes, stable_dim)  # slow-updating
        self.transient_gru = nn.GRUCell(raw_msg_dim + time_dim, transient_dim)
        self.transient_memory = nn.Embedding(num_nodes, transient_dim)
        self.alpha = stable_update_alpha
        self.output_dim = stable_dim + transient_dim
    
    def forward(self, n_id) -> tuple[torch.Tensor, torch.Tensor]:
        s_stable = self.stable(n_id)
        s_transient = self.transient_memory(n_id)
        return torch.cat([s_stable, s_transient], dim=-1), None
    
    def update_transient(self, n_id, messages, timestamps): ...
    def update_stable_ema(self): ...  # call once per epoch
```

- **Integration**: `TGNFraudDetector.__init__` — add `use_dual_memory: bool` to `TGNConfig`; when True, replace `TGNMemory` with `DualTrackMemory`
- **Memory dimensions**: `TGNConfig` — add `stable_memory_dim: int = 32`, `transient_memory_dim: int = 32` (total = 64, same as before)

---

**2B: Event-Centric Fund-Flow Graph (ETGAT)**
- **File**: new `tgn_learn/graphs/flow_dag.py`
- **What it does**: Constructs a second graph where **each transaction is a node** and **edges represent fund-flow chains** (edge from event i to event j if the passive party of i is the active party of j, within 24 hours). Trains a small GAT on this graph to score path-level anomalies.
- **Source**: Wu & Zhang ETGAT (BDAIE 2025)

```python
# tgn_learn/graphs/flow_dag.py
class FundFlowDAG:
    """Event-centric fund-flow DAG (ETGAT, Wu & Zhang 2025).
    
    Inverts the standard entity graph: transactions become nodes,
    fund-flow chains become edges. Makes money-mule paths explicit
    as graph paths rather than implicit in entity embeddings.
    
    An edge exists from event i -> event j if:
      1. passive_party(i) == active_party(j)  [money flows through]
      2. timestamp(j) - timestamp(i) <= time_window  [within window]
    
    Args:
        time_window_hours: Max hours between chained transactions (default: 24)
    """
    def __init__(self, time_window_hours: float = 24.0):
        self.time_window = time_window_hours * 3600
    
    def build(self, edges: list[Edge]) -> 'FundFlowDAG': ...
    def to_pyg_data(self) -> Data: ...  # PyG Data object for GAT
    def get_node_features(self, edge: Edge) -> np.ndarray: ...
```

- **New model**: `tgn_learn/detectors/flow_dag_detector.py` — a 2-layer GAT trained on the flow DAG with node-level fraud labels (node = transaction, label = fraud indicator)

---

**2C: Multi-Graph Device/Account Event Graph**
- **File**: new `tgn_learn/graphs/device_event_graph.py`
- **What it does**: Builds a second entity graph where edges represent device registration, card binding, address change, and account modification events (not transactions). Provides early-warning signals that precede fraud.
- **Source**: Salda&ntilde;a-Ulloa et al. Algorithms 2024

```python
# tgn_learn/graphs/device_event_graph.py  
class DeviceEventGraph(TemporalGraph):
    """Temporal graph of registration and account-change events.
    
    Based on empirical finding (Algorithms 2024) that graphs fusing
    card registration, device registration, and bank account registration
    events consistently outperform single-event graphs.
    
    Event types tracked:
      - CARD_BIND: New card linked to account
      - DEVICE_REG: New device registered
      - ADDR_CHANGE: Billing address modification
      - PHONE_CHANGE: Phone number modification
      - BENEFICIARY_ADD: New transfer beneficiary added
    """
    EVENT_TYPES = ['CARD_BIND', 'DEVICE_REG', 'ADDR_CHANGE', 
                   'PHONE_CHANGE', 'BENEFICIARY_ADD']
    
    def add_event(self, account_id: int, event_type: str, 
                  timestamp: float, features: np.ndarray): ...
```

- **Integration**: `TemporalGraph` in `graph.py` already has `edge_type` on `Edge` — extend `generators/banksim.py` and `generators/paysim.py` to also emit device/account events with appropriate edge types

---

### Phase 3 — Lambda Inference Architecture
*Expected: 2–3 days. Biggest latency impact.*

**3A: Batch Entity Embedder**
- **File**: new `tgn_learn/embedding/batch_embedder.py`
- **What it does**: Pre-computes node embeddings for all entities (cards, merchants) offline via the full multi-hop TGN. Stores results in a dict-based cache (mimics Redis in the local context).
- **Source**: BRIGHT Lambda architecture (CIKM 2022, eBay)

```python
# tgn_learn/embedding/batch_embedder.py
class BatchEntityEmbedder:
    """Offline batch pre-computation of entity embeddings (BRIGHT, 2022).
    
    Decouples the expensive multi-hop TGN computation from real-time
    scoring. Run hourly (or triggered by drift detection). Results
    stored in EmbeddingCache, retrieved at inference time.
    
    This is the 'batch layer' of the Lambda architecture:
      - Full TGN neighbourhood aggregation
      - Multi-scale time encoding
      - Buyer subgraph + seller subgraph embeddings
    
    At inference, only a lightweight delta is computed on top of
    the cached embeddings (real-time layer).
    """
    def __init__(self, model: TGNFraudDetector, cache: 'EmbeddingCache'): ...
    
    def run(self, graph: TemporalGraph, 
            batch_size: int = 512) -> int: ...  # returns nodes updated
    
    def schedule_refresh(self, interval_seconds: int = 3600): ...
```

**3B: Real-Time Embedder**
- **File**: new `tgn_learn/embedding/rt_embedder.py`
- **What it does**: For each new transaction at inference time, retrieves pre-computed entity embeddings from cache and runs only the lightweight scoring head. No multi-hop GNN traversal at runtime.

```python
# tgn_learn/embedding/rt_embedder.py
class RTEmbedder:
    """Real-time embedding lookup + lightweight delta (BRIGHT, 2022).
    
    At inference:
      1. Retrieve z_src, z_dst from EmbeddingCache
      2. Apply lightweight temporal delta (last 5 transactions only)
      3. Pass to scoring heads
    
    No full neighbourhood traversal at inference time.
    P99 latency target: <20ms for this step.
    """
    def __init__(self, model: TGNFraudDetector, cache: 'EmbeddingCache'): ...
    
    def embed(self, src: int, dst: int, t: float,
              edge_features: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]: ...
```

**3C: Embedding Cache**
- **File**: new `tgn_learn/embedding/embedding_cache.py`

```python
# tgn_learn/embedding/embedding_cache.py
class EmbeddingCache:
    """In-memory embedding cache (Redis-compatible interface).
    
    Stores pre-computed entity embeddings with timestamps.
    Local implementation uses a dict; production replaces with Redis.
    
    Interface matches aioredis / redis-py so swapping is a one-line change.
    """
    def get(self, node_id: int) -> Optional[np.ndarray]: ...
    def set(self, node_id: int, embedding: np.ndarray, 
            timestamp: float): ...
    def is_stale(self, node_id: int, max_age_seconds: float) -> bool: ...
    def save(self, path: str): ...  # persist to disk
    def load(self, path: str): ...
```

---

### Phase 4 — Ensemble Meta-Learner
*Expected: 2 days.*

**4A: Detector Registry**
- **File**: new `tgn_learn/detectors/__init__.py`
- **What it does**: Defines a standard `BaseDetector` interface that all detectors implement. The ensemble can call any detector uniformly.

```python
# tgn_learn/detectors/base.py
class BaseDetector(ABC):
    """Standard interface for all fraud detectors in the ensemble."""
    
    @abstractmethod
    def score(self, src: int, dst: int, t: float,
              features: np.ndarray) -> float:
        """Return fraud probability in [0.0, 1.0]."""
        ...
    
    @abstractmethod
    def fit(self, graph: TemporalGraph): ...
    
    @property
    @abstractmethod 
    def name(self) -> str: ...
```

Implement `BaseDetector` for each detector:
- `TGNDetector` (wraps existing `TGNFraudDetector` + `RTEmbedder`)
- `RFDetector` (wraps `RFScoringHead`)
- `FlowDAGDetector` (wraps ETGAT GAT on fund-flow DAG)
- `SemanticDetector` (per-relation-type encoder, simplified HTGNN)
- `DriftMonitorDetector` (anomaly score from autoencoder reconstruction error)

**4B: LightGBM Meta-Learner**
- **File**: new `tgn_learn/ensemble/meta_learner.py`
- **What it does**: Takes all detector scores + raw transaction features as input, outputs calibrated fraud probability. Trained on chronological holdout after all detectors are fitted.

```python
# tgn_learn/ensemble/meta_learner.py
class EnsembleMetaLearner:
    """LightGBM meta-learner over detector scores (stacking ensemble).
    
    Input features per transaction:
      - detector_scores: [tgn_score, rf_score, flow_dag_score,
                          semantic_score, drift_score]  (5 floats)
      - raw_features: [amount_log, mcc_code, channel, time_sin, time_cos,
                       velocity_5m, velocity_1h, velocity_24h]  (8 floats)
      - structural_features: [common_neighbours, path_distance,
                               is_first_interaction, bridge_score]  (4 floats)
      Total: 17 features
    
    Why LightGBM over simple averaging:
      - Each detector has complementary failure modes
      - Learns that flow_dag=0.8 + tgn=0.3 ≠ tgn=0.8 + flow_dag=0.3
      - Handles heterogeneous score scales
      - Feature importances show which detector matters for which fraud type
    
    Training: Chronological split of labelled transactions after all
    detectors are individually trained. Use AUPRC as objective.
    """
    def __init__(self, n_estimators: int = 500):
        self.model = None  # LightGBM classifier
    
    def fit(self, detector_scores: np.ndarray, 
            raw_features: np.ndarray,
            labels: np.ndarray): ...
    
    def predict_proba(self, detector_scores: np.ndarray,
                      raw_features: np.ndarray) -> np.ndarray: ...
    
    def feature_importances(self) -> dict[str, float]: ...
```

**4C: Ensemble Scorer (replaces existing Scorer)**
- **File**: extend `tgn_learn/scoring/scorer.py`
- Add `EnsembleScorer` class that orchestrates all layers
- Backwards-compatible: existing `Scorer` class remains unchanged; `EnsembleScorer` is new

```python
# tgn_learn/scoring/scorer.py — additions
class EnsembleScorer:
    """Full ensemble scoring pipeline.
    
    Orchestrates: embedding cache → parallel detectors → meta-learner
    → two-hurdle filter → risk tier classification.
    
    Drop-in replacement for Scorer in app/pages/4_Score_Transactions.py.
    """
    def __init__(
        self,
        detectors: list[BaseDetector],
        meta_learner: EnsembleMetaLearner,
        embedding_cache: EmbeddingCache,
        thresholds: Optional[RiskThresholds] = None,
    ): ...
    
    def score_transaction(
        self, src: int, dst: int, amount: float,
        timestamp: float, edge_features: Optional[np.ndarray] = None,
    ) -> EnsembleScoringResult: ...
```

---

### Phase 5 — Drift Detection + Adaptive Maintenance
*Expected: 1–2 days.*

**5A: Latent-Space Drift Monitor**
- **File**: new `tgn_learn/maintenance/drift_detector.py`
- **What it does**: Trains an autoencoder on TGN node embeddings from normal-labelled transactions. Monitors reconstruction error in the embedding space. Uses CUSUM to detect when the error distribution shifts significantly (structural or relational drift).
- **Source**: TGNN-CDD (hi-28)

```python
# tgn_learn/maintenance/drift_detector.py
class LatentSpaceDriftDetector:
    """Autoencoder-based concept drift detection in embedding space.
    
    Based on TGNN-CDD (2025): monitors distribution shifts in learned
    TGN node representations rather than raw features. More sensitive
    than CloudWatch metric monitoring — detects structural drift
    (new fraud topology) and relational drift (new entity relationships)
    that raw feature monitors miss.
    
    Trigger conditions:
      - CUSUM statistic exceeds threshold → flag drift
      - On drift: expand temporal receptive field (more historical context)
      - On significant drift: trigger fine-tuning of TGN
    
    Three drift types tracked:
      - FEATURE_DRIFT: Node embedding distribution shifts
      - STRUCTURAL_DRIFT: Graph connectivity patterns change
      - RELATIONAL_DRIFT: Inter-entity relationship dynamics change
    """
    class DriftType(Enum):
        FEATURE = "feature"
        STRUCTURAL = "structural"
        RELATIONAL = "relational"
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 32): ...
    def fit_normal(self, normal_embeddings: np.ndarray): ...
    def check(self, current_embeddings: np.ndarray
              ) -> Optional['DriftEvent']: ...
```

**5B: Topology-Preserving GraphSMOTE**
*(implemented in Phase 1C as a stub, full implementation here)*
- **File**: `tgn_learn/training/graph_smote.py`
- The complete implementation with k-hop neighbourhood inheritance

**5C: Threshold Adapter**
- **File**: new `tgn_learn/maintenance/threshold_adapter.py`
- Simple implementation: monitors daily FP rate and recall per card segment, applies exponential smoothing to adjust the `medium`/`high`/`critical` thresholds in `RiskThresholds`
- Source: hi-26 RL optimisation module (simplified — no RL needed for initial version)

---

## 5. New Module Map

```
tgn_learn/
├── model/
│   ├── tgn.py              (modified — add use_dual_memory flag)
│   ├── dual_memory.py      (NEW — DualTrackMemory)
│   ├── time_encoder.py     (modified — add MultiScaleTimeEncoder)
│   ├── rf_head.py          (NEW — RFScoringHead)
│   ├── heads.py            (unchanged)
│   ├── embedder.py         (modified — swap to MultiScaleTimeEncoder)
│   ├── neighbor_loader.py  (unchanged)
│   └── config.py           (extended — new fields)
│
├── graphs/                 (NEW package)
│   ├── __init__.py
│   ├── flow_dag.py         (ETGAT event-centric DAG)
│   └── device_event_graph.py (registration event graph)
│
├── embedding/              (NEW package)
│   ├── __init__.py
│   ├── batch_embedder.py   (offline pre-compute)
│   ├── rt_embedder.py      (real-time lookup)
│   └── embedding_cache.py  (dict → Redis interface)
│
├── detectors/              (NEW package)
│   ├── __init__.py
│   ├── base.py             (BaseDetector ABC)
│   ├── tgn_detector.py     (wraps existing TGN)
│   ├── rf_detector.py      (weighted RF on embeddings)
│   ├── flow_dag_detector.py (ETGAT GAT)
│   ├── semantic_detector.py (HTGNN per-relation encoding)
│   └── drift_monitor.py    (autoencoder reconstruction error)
│
├── ensemble/               (NEW package)
│   ├── __init__.py
│   ├── meta_learner.py     (LightGBM stacking)
│   └── calibrator.py       (isotonic, moved from scorer.py)
│
├── maintenance/            (NEW package)
│   ├── __init__.py
│   ├── drift_detector.py   (latent-space CUSUM)
│   ├── graph_smote.py      (topology-preserving oversampling)
│   └── threshold_adapter.py (per-segment threshold calibration)
│
├── training/
│   ├── trainer.py          (modified — GraphSMOTE hook, RF head fit)
│   ├── metrics.py          (unchanged)
│   ├── sampler.py          (unchanged)
│   └── mint.py             (unchanged)
│
├── scoring/
│   └── scorer.py           (extended — add EnsembleScorer)
│
├── ingestion/
│   └── csv_ingester.py     (unchanged)
│
├── generators/
│   ├── banksim.py          (extended — emit device events)
│   └── paysim.py           (extended — emit device events)
│
└── graph.py                (extended — add edge_type='device_event')
```

---

## 6. Config Changes

### TGNConfig (model/config.py)
```python
@dataclass
class TGNConfig:
    # Existing
    memory_dim: int = 64
    embedding_dim: int = 64
    time_dim: int = 32
    edge_feat_dim: int = 20
    num_neighbors: int = 10
    num_heads: int = 2
    dropout: float = 0.1
    
    # Phase 1A — multi-scale time encoding
    use_multiscale_time: bool = True
    
    # Phase 2A — dual-track memory
    use_dual_memory: bool = False      # enable when ready
    stable_memory_dim: int = 32        # must sum to memory_dim
    transient_memory_dim: int = 32
    stable_update_alpha: float = 0.05  # EMA decay
    
    # Phase 3 — Lambda architecture
    use_lambda_inference: bool = False  # enable when cache ready
    cache_max_age_seconds: float = 3600.0
```

### TrainingConfig (training/config.py)
```python
@dataclass  
class TrainingConfig:
    # Existing fields unchanged
    ...
    
    # Phase 1B — RF head
    fit_rf_head: bool = True
    rf_n_estimators: int = 200
    
    # Phase 1C — GraphSMOTE
    use_graph_smote: bool = False
    smote_k_hop: int = 2
    smote_minority_ratio: float = 0.1  # target minority ratio
```

---

## 7. Key Interface Contracts

### Two-Hurdle FP Filter
Implement in `scorer.py`. High score alone is NOT sufficient to flag HIGH risk:

```python
def two_hurdle_filter(
    reconstruction_score: float,   # how poorly TGN predicted this edge type
    deviation_score: float,         # (score - rolling_mean) / rolling_std
    recon_threshold: float = 0.95,  # 95th percentile
    deviation_threshold: float = 3.0,  # 3-sigma
) -> str:  # 'HIGH' | 'MEDIUM' | 'PASS'
    """TFLAG-inspired false-positive suppression.
    
    Only flag HIGH RISK when BOTH:
      1. reconstruction_score > recon_threshold (unusual event)
      2. deviation_score > deviation_threshold (statistically anomalous)
    
    Flag MEDIUM when reconstruction is high but deviation is within normal
    range (unusual but within cardholder's evolving baseline — e.g. holiday
    spending surge). This eliminates the largest source of false positives.
    """
```

### Structural Features for RF/Meta-Learner
Compute at scoring time and append to feature vectors:

```python
def compute_structural_features(
    src: int, dst: int, graph: TemporalGraph
) -> dict[str, float]:
    """DRNL-inspired structural features (TGN-SEAL, 2026).
    
    Returns:
        common_neighbours: count of nodes connected to both src and dst
        path_distance: shortest historical path length between src and dst
        is_first_ever_interaction: 1.0 if src and dst never transacted before
        bridge_score: estimated increase in graph betweenness if edge added
    """
```

---

## 8. Backwards Compatibility Rules

1. **All existing `app/pages/*.py` continue to work unchanged** — `EnsembleScorer` is additive; the original `Scorer` class stays in `scorer.py`
2. **Existing `TGNFraudDetector` is not modified** — new features are added via composition and config flags (all new flags default to the old behaviour)
3. **Existing checkpoints load without changes** — `DualTrackMemory` is opt-in via `use_dual_memory=True` in config
4. **`learn/*.py` scripts still run** — no changes to the `learn/` directory
5. **pyproject.toml additions only** — add `lightgbm`, no version bumps to existing deps unless required

### New dependencies (add to pyproject.toml)
```toml
lightgbm = ">=4.0"          # meta-learner (Apache 2.0)
# All other new functionality uses already-installed deps:
# torch, torch_geometric, scikit-learn, numpy — all present
```

---

## 9. Testing Checklist for Kiro

For each phase, implement these tests in `tests/`:

### Phase 1 tests
- [ ] `test_multiscale_time_encoder.py` — output shape, gradient flow, different scales produce different outputs
- [ ] `test_rf_head.py` — fits on synthetic embeddings, AUC-PR > random baseline, feature importances non-zero
- [ ] `test_graph_smote.py` — synthetic edges are k-hop-connected to parent edges, fraud ratio increases

### Phase 2 tests
- [ ] `test_dual_memory.py` — stable memory updates slower than transient, gradients flow through both
- [ ] `test_flow_dag.py` — DAG has correct edges (passive→active chains within 24h), no future leakage
- [ ] `test_device_event_graph.py` — events ingested and retrievable, edge_type populated

### Phase 3 tests
- [ ] `test_embedding_cache.py` — get/set/is_stale/save/load round-trip
- [ ] `test_batch_embedder.py` — all nodes in graph get embeddings, no future leakage
- [ ] `test_rt_embedder.py` — latency <20ms per transaction on CPU

### Phase 4 tests
- [ ] `test_meta_learner.py` — trains on synthetic detector scores, AUPRC > any single detector
- [ ] `test_ensemble_scorer.py` — score_transaction returns EnsembleScoringResult, each detector score accessible
- [ ] `test_two_hurdle_filter.py` — high reconstruction + low deviation → MEDIUM (not HIGH)

### Phase 5 tests
- [ ] `test_drift_detector.py` — fit on normal embeddings, shifted distribution triggers DriftEvent
- [ ] `test_threshold_adapter.py` — thresholds update in correct direction when FP rate increases

---

## 10. Streamlit App Updates

After Phase 4, update `app/pages/3_Train_Model.py`:
- Add toggle: "Use Ensemble (experimental)" → uses `EnsembleScorer`
- Show per-detector scores in `4_Score_Transactions.py` result view
- Add "Drift Status" panel showing reconstruction error over time

After Phase 5, update `app/pages/3_Train_Model.py`:
- Show GraphSMOTE toggle in training config
- Display drift events in a timeline chart

No changes needed to pages 1, 2, or 5.

---

## 11. Implementation Order for Kiro

```
1. tgn_learn/model/time_encoder.py        (MultiScaleTimeEncoder)
2. tgn_learn/model/rf_head.py             (RFScoringHead)
3. tgn_learn/training/graph_smote.py      (GraphSMOTE stub)
4. tgn_learn/model/config.py              (add new fields)
5. tgn_learn/model/embedder.py            (hook in MultiScaleTimeEncoder)
6. tgn_learn/training/trainer.py          (add RF head fit step)
7. tests/ for Phase 1 ─────────────────── VALIDATE HERE

8. tgn_learn/model/dual_memory.py         (DualTrackMemory)
9. tgn_learn/model/tgn.py                 (add use_dual_memory flag)
10. tgn_learn/graphs/__init__.py
11. tgn_learn/graphs/flow_dag.py          (FundFlowDAG)
12. tgn_learn/graphs/device_event_graph.py
13. tgn_learn/generators/banksim.py       (emit device events)
14. tests/ for Phase 2 ─────────────────── VALIDATE HERE

15. tgn_learn/embedding/embedding_cache.py
16. tgn_learn/embedding/batch_embedder.py
17. tgn_learn/embedding/rt_embedder.py
18. tests/ for Phase 3 ─────────────────── VALIDATE HERE

19. tgn_learn/detectors/base.py
20. tgn_learn/detectors/tgn_detector.py
21. tgn_learn/detectors/rf_detector.py
22. tgn_learn/detectors/flow_dag_detector.py
23. tgn_learn/detectors/semantic_detector.py
24. tgn_learn/detectors/drift_monitor.py
25. tgn_learn/ensemble/meta_learner.py
26. tgn_learn/ensemble/calibrator.py
27. tgn_learn/scoring/scorer.py            (add EnsembleScorer)
28. tests/ for Phase 4 ─────────────────── VALIDATE HERE

29. tgn_learn/maintenance/drift_detector.py
30. tgn_learn/maintenance/graph_smote.py   (full implementation)
31. tgn_learn/maintenance/threshold_adapter.py
32. tests/ for Phase 5 ─────────────────── VALIDATE HERE

33. app/pages/3_Train_Model.py             (ensemble toggle)
34. app/pages/4_Score_Transactions.py      (show per-detector scores)
```

---

## 12. Research Provenance

| Feature | Paper | Venue | Confidence |
|---|---|---|---|
| MultiScaleTimeEncoder | Aldawsari — TempReasoner | Scientific Reports 2026 | Architecture ideas sound; performance claims unverified |
| RFScoringHead | Sai et al. — NID-TGN | SPACE 2024 (Springer) | ✓ Credible |
| GraphSMOTE | Wei & Lee — THG-OAFN | PLoS ONE 2025 | ✓ Credible, code on GitHub |
| DualTrackMemory | Wan & Long — DySA-TGN | DASFAA 2025 (Springer) | ✓ Credible |
| FundFlowDAG (ETGAT) | Wu & Zhang | BDAIE 2025 (ACM) | ✓ Credible |
| DeviceEventGraph | Salda&ntilde;a-Ulloa et al. | Algorithms 2024 (MDPI) | ✓ Credible, open data |
| BatchEmbedder / Lambda | Lu et al. — BRIGHT | CIKM 2022 (ACM/eBay) | ✓ Production-deployed |
| EmbeddingCache | Lu et al. — BRIGHT | CIKM 2022 (ACM/eBay) | ✓ Production-deployed |
| BuyerSubgraph + SellerSubgraph | Chen & Yang — C2GAT | Frontiers in AI 2026 (Ant Group) | ✓ Production-deployed |
| SemanticDetector | Nguyen & Le — HTGNN | ICAART 2025 | ✓ Credible |
| LatentSpaceDriftDetector | (hi-28 — TGNN-CDD) | Architecture ideas only | ⚠️ Performance claims unverified |
| ThresholdAdapter | (hi-26 — scalable GNN) | Architecture ideas only | ⚠️ Performance claims unverified |
| EnsembleMetaLearner | Standard stacking | — | Ensemble design by synthesis |
| TwoHurdleFilter | Jiang et al. — TFLAG | arXiv 2501.06997 | ✓ Credible (P=1.0 DARPA E3) |

---

*Design document version 1.0 — June 2026.*
*Reviewed against codebase at `/Users/fahaddad/Documents/TGN` (commit state at document creation).*
