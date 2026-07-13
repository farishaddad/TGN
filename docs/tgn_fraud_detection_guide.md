
# Temporal Graph Networks for Financial Fraud Detection
## Technical Implementation Guide — Greenfield Architecture

---

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Graph Schema Design](#2-graph-schema-design)
3. [TGN Model Architecture (PyTorch Geometric)](#3-tgn-model-architecture)
4. [Multi-Source Data Fusion Pipeline](#4-multi-source-data-fusion-pipeline)
5. [Risk Quantification Framework](#5-risk-quantification-framework)
6. [Rules Engine Integration](#6-rules-engine-integration)
7. [Transfer Learning Extensibility](#7-transfer-learning-extensibility)
8. [Implementation Roadmap](#8-implementation-roadmap)

---

## 1. Architecture Overview

### Design Philosophy

The system follows a **lambda architecture pattern** — batch training with real-time inference — built around three core principles:

- **Temporal-first**: Every node and edge carries timestamps. No snapshot discretisation; events stream continuously.
- **Heterogeneous**: Multiple node types (accounts, devices, merchants, external threat indicators) and edge types (transfers, logins, alerts).
- **Transfer-ready**: Node/edge feature schemas use institution-agnostic representations from day one, so domain adaptation across institutions requires fine-tuning, not re-architecture.

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                  │
├──────────────┬──────────────┬──────────────┬────────────────────────┤
│  Bank Txn    │  Login/      │  External    │  Watchlists /          │
│  Feeds       │  Device      │  Threat      │  SARs /                │
│  (SWIFT/ISO) │  Events      │  Intel       │  Sanctions             │
└──────┬───────┴──────┬───────┴──────┬───────┴────────┬───────────────┘
       │              │              │                │
       ▼              ▼              ▼                ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STREAMING INGESTION LAYER (Kafka)                       │
│  • Schema validation  • Deduplication  • Timestamp normalisation     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
┌──────────────────────┐          ┌──────────────────────────────────┐
│   BATCH PIPELINE     │          │   REAL-TIME PIPELINE              │
│                      │          │                                    │
│  Graph Construction  │          │  Streaming Graph Updates           │
│  ──────────────────  │          │  ──────────────────────            │
│  • Full rebuild      │          │  • Incremental edge insertion      │
│  • Feature eng.      │          │  • TGN memory state update         │
│  • Label curation    │          │  • Node embedding refresh          │
│                      │          │                                    │
│  TGN Training        │          │  TGN Inference                     │
│  ──────────────────  │          │  ──────────────────────            │
│  • Temporal split    │          │  • Sub-100ms per txn               │
│  • Leak-free eval    │          │  • Risk score computation          │
│  • Model checkpoint  │          │  • Confidence intervals            │
└──────────┬───────────┘          └──────────────┬───────────────────┘
           │                                     │
           │    ┌────────────────────────┐        │
           └───►│   MODEL REGISTRY       │◄───────┘
                │   (MLflow / W&B)       │
                └────────────┬───────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 RISK QUANTIFICATION LAYER                            │
│                                                                      │
│  • Raw TGN anomaly score → calibrated probability                    │
│  • Temporal pattern features → pattern risk profile                  │
│  • Ensemble with XGBoost on TGN embeddings                           │
│  • Conformal prediction for uncertainty bounds                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 RULES ENGINE INTEGRATION                             │
│                                                                      │
│  • TGN risk scores exposed as real-time feature store                │
│  • Dynamic threshold rules (auto-tuned from model performance)       │
│  • Pattern-based rules generated from TGN cluster analysis           │
│  • Human-in-the-loop escalation with explainability                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Framework Recommendation: PyTorch Geometric (PyG)

**Why PyG over DGL for this use case:**

| Criteria | PyG | DGL |
|----------|-----|-----|
| TGN native support | `TGNMemory` built-in with examples | Requires custom implementation |
| Heterogeneous graphs | `HeteroData` + `to_hetero()` auto-conversion | Good support, but more verbose |
| Community & research | Larger research community, faster paper implementations | Strong industry adoption (AWS) |
| Real-time serving | Pairs well with TorchServe / Triton | Better Kafka integration via DGL-KE |
| Transfer learning | Modular design makes weight sharing easier | Less modular architecture |

**Verdict**: PyG is the better starting point. Its `TGNMemory` module + `TemporalData` loader maps directly to the Rossi et al. (2020) paper, and the heterogeneous graph support is cleaner for multi-source fusion.

---

## 2. Graph Schema Design

### Heterogeneous Temporal Graph Schema

```
NODE TYPES                          EDGE TYPES
──────────                          ──────────
Account (bank account)              transfer: Account → Account
  • account_id                      login: Account → Device
  • account_type (enum)             merchant_txn: Account → Merchant
  • creation_date                   alert_link: Account → ThreatIndicator
  • jurisdiction (ISO 3166)         beneficiary_link: Account → Account
  • risk_tier (initial static)      shared_device: Device → Device (inferred)
                                    
Device (login device)               
  • device_fingerprint              
  • device_type                     
  • geo_location                    
  • first_seen / last_seen          
                                    
Merchant (payee)                    
  • merchant_category_code (MCC)    
  • merchant_risk_rating            
  • country_code                    
                                    
ThreatIndicator (external intel)    
  • indicator_type (IP/IBAN/name)   
  • source (FinCEN/FATF/internal)   
  • severity_score                  
  • valid_from / valid_to           
```

### Edge Feature Schema (Common Across Edge Types)

```python
TEMPORAL_EDGE_FEATURES = {
    # Universal temporal features
    'timestamp': float,              # Unix epoch (seconds)
    'amount': float,                 # Normalised transaction amount (0-1)
    'amount_log': float,             # log(amount + 1)
    
    # Contextual features
    'channel': int,                  # Encoding: online=0, branch=1, ATM=2, mobile=3
    'cross_border': bool,            # Domestic vs international
    'currency_pair': int,            # Encoded currency pair
    
    # Derived temporal features (computed at ingestion)
    'time_since_last_txn': float,    # Seconds since this account's last transaction
    'txn_velocity_1h': float,        # Number of txns in last hour for source
    'amount_deviation': float,       # Std devs from account's rolling mean
    
    # Transfer-learning-ready (institution-agnostic)
    'normalised_amount_percentile': float,  # Within-institution percentile
    'time_of_day_sin': float,        # Cyclical encoding: sin(2π * hour/24)
    'time_of_day_cos': float,        # Cyclical encoding: cos(2π * hour/24)
    'day_of_week_sin': float,        # Cyclical encoding: sin(2π * day/7)
    'day_of_week_cos': float,        # Cyclical encoding: cos(2π * day/7)
}
```

---

## 3. TGN Model Architecture

### Core TGN Components

The TGN architecture from Rossi et al. (2020) has five key modules. Here's how each maps to fraud detection:

```
┌──────────────────────────────────────────────────────┐
│                   TGN ARCHITECTURE                    │
│                                                       │
│  1. RAW MESSAGE FUNCTION                              │
│     • Combines source memory, dest memory,            │
│       edge features, and time encoding                │
│     • For fraud: captures "what happened" per txn     │
│                                                       │
│  2. MESSAGE AGGREGATOR                                │
│     • Combines multiple messages to same node         │
│     • "last" aggregator for fraud (recency matters)   │
│                                                       │
│  3. MEMORY UPDATER (GRU)                              │
│     • Updates node state after each interaction       │
│     • For fraud: tracks evolving account behaviour    │
│                                                       │
│  4. TEMPORAL EMBEDDING MODULE                         │
│     • Graph attention over temporal neighbourhood     │
│     • For fraud: captures relational context          │
│                                                       │
│  5. LINK PREDICTOR / NODE CLASSIFIER                  │
│     • Downstream task head                            │
│     • For fraud: anomaly score output                 │
│                                                       │
└──────────────────────────────────────────────────────┘
```

### PyG Implementation

```python
"""
TGN-based Fraud Detection Model
Framework: PyTorch Geometric
Pattern: Batch training, streaming inference
"""

import torch
import torch.nn as nn
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
)
from torch_geometric.data import TemporalData


# ──────────────────────────────────────────────
# 1. TIME ENCODER
# ──────────────────────────────────────────────
class TimeEncoder(nn.Module):
    """Learnable Fourier time encoding (Xu et al., 2020)"""
    
    def __init__(self, time_dim: int):
        super().__init__()
        self.w = nn.Linear(1, time_dim)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.w.weight = nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.w.out_features)))
            .float().reshape(self.w.out_features, -1)
        )
        self.w.bias = nn.Parameter(torch.zeros(self.w.out_features))
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(dim=-1) if t.dim() == 1 else t
        return torch.cos(self.w(t))


# ──────────────────────────────────────────────
# 2. GRAPH ATTENTION EMBEDDING
# ──────────────────────────────────────────────
class GraphAttentionEmbedding(nn.Module):
    """
    Temporal graph attention for computing node embeddings
    from their temporal neighbourhood.
    """
    
    def __init__(self, in_channels: int, out_channels: int, msg_dim: int, time_enc: TimeEncoder):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.w.out_features
        self.conv = TransformerConv(
            in_channels, out_channels // 2,  # multi-head
            heads=2,
            dropout=0.1,
            edge_dim=edge_dim,
        )
    
    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


# ──────────────────────────────────────────────
# 3. ANOMALY SCORING HEAD
# ──────────────────────────────────────────────
class FraudScoringHead(nn.Module):
    """
    Dual-purpose head:
      - Link prediction (is this transaction anomalous?)
      - Node classification (is this account fraudulent?)
    
    Outputs calibrated risk scores via temperature scaling.
    """
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Link-level scoring (transaction anomaly)
        self.link_predictor = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )
        # Node-level scoring (account risk)
        self.node_classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )
        # Learnable temperature for calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward_link(self, z_src, z_dst):
        """Score a transaction edge"""
        h = torch.cat([z_src, z_dst], dim=-1)
        logit = self.link_predictor(h)
        return torch.sigmoid(logit / self.temperature)
    
    def forward_node(self, z):
        """Score an account node"""
        logit = self.node_classifier(z)
        return torch.sigmoid(logit / self.temperature)


# ──────────────────────────────────────────────
# 4. FULL TGN FRAUD DETECTION MODEL
# ──────────────────────────────────────────────
class TGNFraudDetector(nn.Module):
    """
    Complete TGN-based fraud detection model.
    
    Architecture decisions:
    - Memory dim: 100 (sufficient for account state)
    - Time dim: 100 (captures cyclical patterns)
    - Embedding dim: 128 (balance of expressiveness and speed)
    - Message aggregator: 'last' (recency matters in fraud)
    - Memory updater: GRU (better gradient flow than RNN)
    - Embedding module: Graph Attention (captures relational context)
    """
    
    def __init__(
        self,
        num_nodes: int,
        edge_feat_dim: int,
        memory_dim: int = 100,
        time_dim: int = 100,
        embedding_dim: int = 128,
        num_neighbors: int = 10,
    ):
        super().__init__()
        self.num_neighbors = num_neighbors
        
        # TGN Memory (PyG built-in)
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=edge_feat_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=IdentityMessage(
                raw_msg_dim=edge_feat_dim,
                memory_dim=memory_dim,
                time_dim=time_dim,
            ),
            aggregator_module=LastAggregator(),
        )
        
        # Embedding computation
        self.time_enc = TimeEncoder(time_dim)
        self.gnn = GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=edge_feat_dim,
            time_enc=self.time_enc,
        )
        
        # Scoring heads
        self.scorer = FraudScoringHead(embedding_dim)
        
        # Neighbour sampler will be set externally
        self.neighbor_loader = None  # Set during training setup
    
    def compute_embedding(self, n_id, t=None):
        """
        Compute temporal embedding for a set of node IDs.
        Fetches memory states and aggregates over temporal neighbours.
        """
        z, last_update = self.memory(n_id)
        
        if self.neighbor_loader is not None and t is not None:
            # Sample temporal neighbours
            n_id_expanded, edge_index, e_id, t_neigh = (
                self.neighbor_loader(n_id, t)
            )
            z_expanded, last_update_expanded = self.memory(n_id_expanded)
            
            # Compute attention over temporal neighbourhood
            z_expanded = self.gnn(
                z_expanded, last_update_expanded,
                edge_index, t_neigh,
                self.neighbor_loader.e_feat[e_id],
            )
            z = z_expanded[:n_id.size(0)]
        
        return z
    
    def forward(self, src, dst, t, msg, neg_dst=None):
        """
        Forward pass for a batch of interactions.
        
        Args:
            src: Source node IDs [batch_size]
            dst: Destination node IDs [batch_size]
            t: Timestamps [batch_size]
            msg: Edge features [batch_size, edge_feat_dim]
            neg_dst: Negative destination samples (for contrastive training)
        
        Returns:
            pos_score: Anomaly score for real edges
            neg_score: Anomaly score for negative edges
            node_scores: Account-level risk scores
        """
        n_id = torch.cat([src, dst]).unique()
        
        # Compute embeddings
        z_src = self.compute_embedding(src, t)
        z_dst = self.compute_embedding(dst, t)
        
        # Link-level anomaly scores
        pos_score = self.scorer.forward_link(z_src, z_dst)
        
        neg_score = None
        if neg_dst is not None:
            z_neg = self.compute_embedding(neg_dst, t)
            neg_score = self.scorer.forward_link(z_src, z_neg)
        
        # Node-level risk scores
        node_scores = self.scorer.forward_node(z_src)
        
        # Update memory with this batch
        self.memory.update_state(src, dst, t, msg)
        
        return pos_score, neg_score, node_scores


# ──────────────────────────────────────────────
# 5. TRAINING LOOP (TEMPORAL-AWARE)
# ──────────────────────────────────────────────
def train_epoch(model, data: TemporalData, optimizer, criterion, 
                batch_size=200, device='cuda'):
    """
    Temporal-aware training with strict chronological ordering.
    
    CRITICAL: No temporal leakage — each batch only sees past interactions.
    Uses the 70/15/15 temporal split recommended by TGB benchmark.
    """
    model.train()
    model.memory.reset_state()
    
    total_loss = 0
    num_batches = 0
    
    # Data is pre-sorted by timestamp
    for batch_start in range(0, data.num_events, batch_size):
        batch_end = min(batch_start + batch_size, data.num_events)
        
        src = data.src[batch_start:batch_end].to(device)
        dst = data.dst[batch_start:batch_end].to(device)
        t = data.t[batch_start:batch_end].to(device)
        msg = data.msg[batch_start:batch_end].to(device)
        label = data.y[batch_start:batch_end].to(device)
        
        # Sample negative destinations (non-existent edges)
        neg_dst = torch.randint(
            0, data.num_nodes, (src.size(0),),
            dtype=torch.long, device=device
        )
        
        optimizer.zero_grad()
        
        pos_score, neg_score, node_scores = model(src, dst, t, msg, neg_dst)
        
        # Combined loss: contrastive (link) + supervised (node classification)
        link_loss = criterion(pos_score.squeeze(), torch.ones_like(pos_score.squeeze()))
        link_loss += criterion(neg_score.squeeze(), torch.zeros_like(neg_score.squeeze()))
        
        # Weighted BCE for class imbalance (fraud is ~0.1-1% of transactions)
        fraud_weight = (label == 0).sum().float() / max((label == 1).sum().float(), 1)
        weight = torch.where(label == 1, fraud_weight, torch.ones_like(label.float()))
        node_loss = nn.functional.binary_cross_entropy(
            node_scores.squeeze(), label.float(), weight=weight
        )
        
        loss = 0.5 * link_loss + 0.5 * node_loss
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # CRITICAL: Detach memory after backprop to prevent 
        # computational graph from growing across batches
        model.memory.detach()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches
```

### Temporal Data Splitting (Leak-Free)

```python
def temporal_train_val_test_split(data: TemporalData, 
                                   train_ratio=0.70, 
                                   val_ratio=0.15):
    """
    Strict temporal split — no future information leaks into training.
    
    This is CRITICAL for fraud detection. As shown by Kapoor & Narayanan (2023),
    data leakage from improper splitting has affected 294 research papers.
    """
    num_events = data.num_events
    
    # Sort by timestamp (should already be sorted, but verify)
    sorted_idx = data.t.argsort()
    
    train_end = int(num_events * train_ratio)
    val_end = int(num_events * (train_ratio + val_ratio))
    
    train_data = data[sorted_idx[:train_end]]
    val_data = data[sorted_idx[train_end:val_end]]
    test_data = data[sorted_idx[val_end:]]
    
    # Verify no temporal leakage
    assert train_data.t.max() <= val_data.t.min(), "Temporal leakage: train→val"
    assert val_data.t.max() <= test_data.t.min(), "Temporal leakage: val→test"
    
    return train_data, val_data, test_data
```

---

## 4. Multi-Source Data Fusion Pipeline

### Source Integration Architecture

```
                    ┌──────────────────────────────────┐
                    │        KAFKA TOPIC DESIGN          │
                    │                                    │
                    │  txn.raw.bank_transfers            │
                    │  txn.raw.card_payments             │
                    │  event.raw.device_logins           │
                    │  intel.raw.threat_indicators        │
                    │  intel.raw.sanctions_updates        │
                    │  ─────────────────────────          │
                    │  graph.events.unified  ◄── merged   │
                    └──────────────────────────────────┘
```

### Unified Event Schema

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np


class EventType(Enum):
    TRANSFER = "transfer"
    CARD_PAYMENT = "card_payment"
    LOGIN = "login"
    THREAT_INDICATOR = "threat_indicator"
    SANCTIONS_UPDATE = "sanctions_update"
    ACCOUNT_UPDATE = "account_update"


@dataclass
class UnifiedGraphEvent:
    """
    Canonical event format that all sources are normalised into.
    This is the single schema consumed by the graph construction layer.
    """
    event_id: str
    event_type: EventType
    timestamp: float                     # Unix epoch (seconds, float for sub-second)
    
    # Graph topology
    src_node_id: str                     # Canonical node ID (type-prefixed)
    dst_node_id: str                     # e.g., "account:GB12345", "device:fp_abc"
    src_node_type: str
    dst_node_type: str
    edge_type: str
    
    # Feature vector (fixed-length, padded)
    edge_features: np.ndarray            # Shape: [edge_feat_dim]
    
    # Labels (if available, otherwise -1)
    label: int = -1                      # 0=legitimate, 1=fraud, -1=unlabelled
    
    # Provenance
    source_system: str = ""
    confidence: float = 1.0              # For external intel with varying reliability


class MultiSourceFusionPipeline:
    """
    Transforms raw events from multiple sources into UnifiedGraphEvent stream.
    
    Key design decisions:
    - Node IDs are type-prefixed to avoid collisions across namespaces
    - Edge features are zero-padded to fixed length for TGN compatibility
    - External threat intel creates edges to existing accounts (not new txns)
    - Confidence scores from external sources feed into risk quantification
    """
    
    def __init__(self, edge_feat_dim: int = 20, node_registry=None):
        self.edge_feat_dim = edge_feat_dim
        self.node_registry = node_registry or NodeRegistry()
        self.feature_encoder = FeatureEncoder(edge_feat_dim)
    
    def process_bank_transfer(self, raw_event: dict) -> UnifiedGraphEvent:
        """Transform SWIFT/ISO 20022 bank transfer into graph event."""
        
        src_id = f"account:{raw_event['debtor_iban']}"
        dst_id = f"account:{raw_event['creditor_iban']}"
        
        self.node_registry.register(src_id, 'account', raw_event.get('debtor_meta', {}))
        self.node_registry.register(dst_id, 'account', raw_event.get('creditor_meta', {}))
        
        features = self.feature_encoder.encode_transfer(raw_event)
        
        return UnifiedGraphEvent(
            event_id=raw_event['transaction_id'],
            event_type=EventType.TRANSFER,
            timestamp=raw_event['value_date_epoch'],
            src_node_id=src_id,
            dst_node_id=dst_id,
            src_node_type='account',
            dst_node_type='account',
            edge_type='transfer',
            edge_features=features,
            label=raw_event.get('fraud_label', -1),
            source_system='core_banking',
        )
    
    def process_device_login(self, raw_event: dict) -> UnifiedGraphEvent:
        """Transform device login event into graph event."""
        
        src_id = f"account:{raw_event['account_id']}"
        dst_id = f"device:{raw_event['device_fingerprint']}"
        
        self.node_registry.register(dst_id, 'device', {
            'device_type': raw_event.get('device_type'),
            'geo': raw_event.get('geo_location'),
        })
        
        features = self.feature_encoder.encode_login(raw_event)
        
        return UnifiedGraphEvent(
            event_id=raw_event['session_id'],
            event_type=EventType.LOGIN,
            timestamp=raw_event['login_time_epoch'],
            src_node_id=src_id,
            dst_node_id=dst_id,
            src_node_type='account',
            dst_node_type='device',
            edge_type='login',
            edge_features=features,
            source_system='authentication',
        )
    
    def process_threat_indicator(self, raw_event: dict) -> list[UnifiedGraphEvent]:
        """
        Transform external threat intelligence into graph edges.
        
        This is the key multi-source fusion step: external indicators
        create edges between ThreatIndicator nodes and matching accounts.
        
        A single threat indicator may generate MULTIPLE graph events
        (one per matched account).
        """
        
        indicator_id = f"threat:{raw_event['indicator_type']}:{raw_event['indicator_value']}"
        
        self.node_registry.register(indicator_id, 'threat_indicator', {
            'severity': raw_event['severity_score'],
            'source': raw_event['source'],
            'valid_from': raw_event['valid_from'],
        })
        
        # Match against existing accounts
        matched_accounts = self.node_registry.match_accounts(
            indicator_type=raw_event['indicator_type'],
            indicator_value=raw_event['indicator_value'],
        )
        
        events = []
        for account_id in matched_accounts:
            features = self.feature_encoder.encode_threat_link(
                raw_event, 
                match_confidence=raw_event.get('match_confidence', 0.8)
            )
            
            events.append(UnifiedGraphEvent(
                event_id=f"{raw_event['indicator_id']}_{account_id}",
                event_type=EventType.THREAT_INDICATOR,
                timestamp=raw_event['published_epoch'],
                src_node_id=account_id,
                dst_node_id=indicator_id,
                src_node_type='account',
                dst_node_type='threat_indicator',
                edge_type='alert_link',
                edge_features=features,
                source_system=raw_event['source'],
                confidence=raw_event.get('match_confidence', 0.8),
            ))
        
        return events


class FeatureEncoder:
    """
    Encodes raw event data into fixed-length feature vectors.
    
    Design for transfer learning:
    - Uses relative/normalised features (percentiles, z-scores)
    - Cyclical time encoding (sin/cos, institution-agnostic)
    - Avoids absolute values that vary across institutions
    """
    
    def __init__(self, feat_dim: int = 20):
        self.feat_dim = feat_dim
        self.amount_scaler = None  # Fitted during batch training
    
    def encode_transfer(self, event: dict) -> np.ndarray:
        features = np.zeros(self.feat_dim, dtype=np.float32)
        
        amount = event.get('amount', 0)
        timestamp = event.get('value_date_epoch', 0)
        
        features[0] = np.log1p(amount)                              # Log amount
        features[1] = self._amount_percentile(amount)                # Percentile
        features[2] = float(event.get('cross_border', False))       # Cross-border flag
        features[3] = self._encode_channel(event.get('channel', ''))  # Channel
        features[4:6] = self._cyclical_time(timestamp)               # Time of day (sin/cos)
        features[6:8] = self._cyclical_day(timestamp)                # Day of week (sin/cos)
        features[8] = event.get('time_since_last_txn', 0) / 86400   # Days since last txn
        features[9] = event.get('txn_velocity_1h', 0) / 100         # Normalised velocity
        features[10] = event.get('amount_deviation', 0)              # Std devs from mean
        features[11] = self._encode_currency(event.get('currency', 'GBP'))
        
        return features
    
    def _cyclical_time(self, epoch: float) -> np.ndarray:
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
        hour_frac = dt.hour + dt.minute / 60
        return np.array([
            np.sin(2 * np.pi * hour_frac / 24),
            np.cos(2 * np.pi * hour_frac / 24),
        ])
    
    def _cyclical_day(self, epoch: float) -> np.ndarray:
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
        return np.array([
            np.sin(2 * np.pi * dt.weekday() / 7),
            np.cos(2 * np.pi * dt.weekday() / 7),
        ])
    
    def _amount_percentile(self, amount: float) -> float:
        """Returns institution-relative percentile (transfer-learning friendly)."""
        if self.amount_scaler is not None:
            return self.amount_scaler.transform([[amount]])[0][0]
        return min(np.log1p(amount) / 15.0, 1.0)  # Rough normalisation fallback
    
    def _encode_channel(self, channel: str) -> float:
        mapping = {'online': 0.0, 'branch': 0.25, 'atm': 0.5, 'mobile': 0.75, 'api': 1.0}
        return mapping.get(channel.lower(), 0.5)
    
    def _encode_currency(self, currency: str) -> float:
        # Simple hash-based encoding; replace with learned embedding for production
        return (hash(currency) % 100) / 100.0
```

### Streaming Graph Construction (Kafka → PyG TemporalData)

```python
import json
from collections import defaultdict
from kafka import KafkaConsumer
from torch_geometric.data import TemporalData


class StreamingGraphBuilder:
    """
    Builds and incrementally updates a PyG TemporalData object
    from a Kafka stream of UnifiedGraphEvents.
    
    For real-time inference:
    - Maintains an in-memory node ID mapping (str → int)
    - Appends new events to the temporal data structure
    - Triggers TGN memory updates on each new event
    """
    
    def __init__(self, kafka_config: dict, model: TGNFraudDetector):
        self.consumer = KafkaConsumer(
            'graph.events.unified',
            **kafka_config,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        )
        self.model = model
        self.node_map = {}  # str → int mapping
        self.next_node_id = 0
        
        # Accumulate events for batch updates
        self.event_buffer = []
        self.buffer_size = 50  # Mini-batch for efficiency
    
    def _get_or_create_node_id(self, node_str: str) -> int:
        if node_str not in self.node_map:
            self.node_map[node_str] = self.next_node_id
            self.next_node_id += 1
        return self.node_map[node_str]
    
    def process_event(self, event: UnifiedGraphEvent) -> dict:
        """
        Process a single event:
        1. Map string IDs to integer IDs
        2. Update TGN memory
        3. Compute risk score
        
        Returns risk assessment dict.
        """
        src_int = self._get_or_create_node_id(event.src_node_id)
        dst_int = self._get_or_create_node_id(event.dst_node_id)
        
        src = torch.tensor([src_int], dtype=torch.long)
        dst = torch.tensor([dst_int], dtype=torch.long)
        t = torch.tensor([event.timestamp], dtype=torch.float)
        msg = torch.tensor(event.edge_features, dtype=torch.float).unsqueeze(0)
        
        with torch.no_grad():
            self.model.eval()
            pos_score, _, node_score = self.model(src, dst, t, msg)
        
        return {
            'event_id': event.event_id,
            'transaction_risk_score': pos_score.item(),
            'account_risk_score': node_score.item(),
            'source_account': event.src_node_id,
            'destination_account': event.dst_node_id,
            'timestamp': event.timestamp,
            'source_confidence': event.confidence,
        }
```

---

## 5. Risk Quantification Framework

### Overview

Raw TGN anomaly scores are **not directly usable** by a rules engine. They need to be:

1. **Calibrated** — converted from model logits to true probabilities
2. **Contextualised** — combined with pattern-level features
3. **Bounded** — uncertainty quantified via conformal prediction
4. **Profiled** — aggregated into entity-level risk profiles

```
┌─────────────────────────────────────────────────────────────┐
│              RISK QUANTIFICATION PIPELINE                     │
│                                                               │
│  RAW TGN                                                      │
│  SCORES ──► CALIBRATION ──► PATTERN ──► RISK ──► RULES       │
│              (Platt/       PROFILING   INDEX   ENGINE         │
│               Isotonic)    (temporal   (multi-  API           │
│                            clusters)   factor)                │
│                                                               │
│  CONFORMAL PREDICTION ──────────────────────► UNCERTAINTY     │
│  (coverage guarantee)                         BOUNDS          │
└─────────────────────────────────────────────────────────────┘
```

### Step 1: Score Calibration

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import numpy as np


class TGNScoreCalibrator:
    """
    Calibrates raw TGN scores to true fraud probabilities.
    
    Why this matters:
    - A TGN score of 0.7 doesn't mean 70% probability of fraud
    - Rules engines need calibrated probabilities for threshold decisions
    - Calibration enables meaningful comparison across model versions
    
    Uses isotonic regression (non-parametric) because:
    - Fraud score distributions are highly non-linear
    - No assumption about score distribution shape
    - Better than Platt scaling for imbalanced classes
    """
    
    def __init__(self):
        self.calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds='clip'
        )
        self.is_fitted = False
    
    def fit(self, raw_scores: np.ndarray, true_labels: np.ndarray):
        """
        Fit calibrator on validation set (NEVER on training set).
        
        Args:
            raw_scores: TGN anomaly scores from validation set
            true_labels: Ground truth fraud labels (0/1)
        """
        self.calibrator.fit(raw_scores, true_labels)
        self.is_fitted = True
    
    def calibrate(self, raw_score: float) -> float:
        """Convert raw TGN score to calibrated probability."""
        if not self.is_fitted:
            return raw_score  # Fallback: uncalibrated
        return float(self.calibrator.predict([raw_score])[0])
    
    def calibrate_batch(self, raw_scores: np.ndarray) -> np.ndarray:
        """Calibrate a batch of scores."""
        if not self.is_fitted:
            return raw_scores
        return self.calibrator.predict(raw_scores)
```

### Step 2: Temporal Pattern Profiling

```python
from collections import deque
from typing import Dict, List, Tuple
import numpy as np


class TemporalPatternProfiler:
    """
    Builds entity-level risk profiles from streams of calibrated scores.
    
    Captures patterns that individual transaction scores miss:
    - Velocity anomalies (sudden burst of activity)
    - Escalation patterns (scores increasing over time)
    - Network contagion (neighbours becoming risky)
    - Periodic anomalies (activity at unusual times)
    """
    
    def __init__(self, window_sizes: list[int] = [3600, 86400, 604800]):
        """
        Args:
            window_sizes: Time windows in seconds [1h, 24h, 7d]
        """
        self.windows = window_sizes
        self.entity_histories: Dict[str, deque] = {}
        self.max_history = max(window_sizes) * 2  # Keep 2x longest window
    
    def update(self, entity_id: str, timestamp: float, 
               calibrated_score: float, amount: float = 0.0):
        """Record a new event for an entity."""
        if entity_id not in self.entity_histories:
            self.entity_histories[entity_id] = deque(maxlen=10000)
        
        self.entity_histories[entity_id].append({
            'timestamp': timestamp,
            'score': calibrated_score,
            'amount': amount,
        })
    
    def compute_profile(self, entity_id: str, current_time: float) -> dict:
        """
        Compute comprehensive risk profile for an entity.
        
        Returns features that the rules engine can use directly.
        """
        history = self.entity_histories.get(entity_id, deque())
        
        if len(history) < 2:
            return self._empty_profile()
        
        profile = {}
        
        for window in self.windows:
            window_name = self._window_name(window)
            window_events = [
                e for e in history 
                if current_time - e['timestamp'] <= window
            ]
            
            if not window_events:
                profile.update(self._empty_window_profile(window_name))
                continue
            
            scores = [e['score'] for e in window_events]
            amounts = [e['amount'] for e in window_events]
            timestamps = [e['timestamp'] for e in window_events]
            
            # Velocity features
            profile[f'{window_name}_txn_count'] = len(window_events)
            profile[f'{window_name}_total_amount'] = sum(amounts)
            
            # Score aggregation features
            profile[f'{window_name}_mean_score'] = np.mean(scores)
            profile[f'{window_name}_max_score'] = np.max(scores)
            profile[f'{window_name}_score_std'] = np.std(scores) if len(scores) > 1 else 0
            profile[f'{window_name}_high_score_ratio'] = (
                sum(1 for s in scores if s > 0.5) / len(scores)
            )
            
            # Escalation detection (is risk trending upward?)
            if len(scores) >= 3:
                recent_half = scores[len(scores)//2:]
                earlier_half = scores[:len(scores)//2]
                profile[f'{window_name}_escalation'] = (
                    np.mean(recent_half) - np.mean(earlier_half)
                )
            else:
                profile[f'{window_name}_escalation'] = 0.0
            
            # Inter-event timing
            if len(timestamps) >= 2:
                intervals = np.diff(sorted(timestamps))
                profile[f'{window_name}_mean_interval'] = np.mean(intervals)
                profile[f'{window_name}_min_interval'] = np.min(intervals)
                # Burstiness: coefficient of variation of inter-event times
                profile[f'{window_name}_burstiness'] = (
                    np.std(intervals) / np.mean(intervals) 
                    if np.mean(intervals) > 0 else 0
                )
            else:
                profile[f'{window_name}_mean_interval'] = 0
                profile[f'{window_name}_min_interval'] = 0
                profile[f'{window_name}_burstiness'] = 0
        
        return profile
    
    def _window_name(self, seconds: int) -> str:
        if seconds == 3600: return 'w1h'
        if seconds == 86400: return 'w24h'
        if seconds == 604800: return 'w7d'
        return f'w{seconds}s'
    
    def _empty_profile(self) -> dict:
        profile = {}
        for w in self.windows:
            profile.update(self._empty_window_profile(self._window_name(w)))
        return profile
    
    def _empty_window_profile(self, window_name: str) -> dict:
        return {
            f'{window_name}_txn_count': 0,
            f'{window_name}_total_amount': 0.0,
            f'{window_name}_mean_score': 0.0,
            f'{window_name}_max_score': 0.0,
            f'{window_name}_score_std': 0.0,
            f'{window_name}_high_score_ratio': 0.0,
            f'{window_name}_escalation': 0.0,
            f'{window_name}_mean_interval': 0.0,
            f'{window_name}_min_interval': 0.0,
            f'{window_name}_burstiness': 0.0,
        }
```

### Step 3: Composite Risk Index

```python
class CompositeRiskIndex:
    """
    Combines TGN scores, pattern profiles, and external intelligence
    into a single actionable risk index.
    
    Architecture: XGBoost on TGN embeddings + pattern features
    (proven approach: NVIDIA's fraud detection blueprint uses this exact pattern)
    
    The TGN handles relational/temporal patterns.
    XGBoost handles the tabular risk quantification on top.
    Best of both worlds.
    """
    
    def __init__(self):
        self.xgb_model = None
        self.feature_names = None
    
    def build_feature_vector(
        self,
        calibrated_txn_score: float,
        calibrated_account_score: float,
        pattern_profile: dict,
        external_intel: dict,
        tgn_embedding: np.ndarray,
    ) -> np.ndarray:
        """
        Assemble complete feature vector for risk scoring.
        
        Feature groups:
        1. TGN scores (calibrated) — 2 features
        2. Pattern profile — ~30 features (10 per time window)
        3. External intelligence — ~5 features
        4. TGN embedding — 128 features (from graph attention layer)
        """
        features = []
        
        # Group 1: Calibrated TGN scores
        features.extend([calibrated_txn_score, calibrated_account_score])
        
        # Group 2: Pattern profile (sorted keys for consistency)
        for key in sorted(pattern_profile.keys()):
            features.append(pattern_profile[key])
        
        # Group 3: External intelligence
        features.extend([
            external_intel.get('sanctions_proximity', 0.0),
            external_intel.get('threat_indicator_count', 0),
            external_intel.get('highest_threat_severity', 0.0),
            external_intel.get('jurisdiction_risk_score', 0.0),
            external_intel.get('pep_proximity', 0.0),
        ])
        
        # Group 4: TGN embedding (rich learned representation)
        features.extend(tgn_embedding.tolist())
        
        return np.array(features, dtype=np.float32)
    
    def compute_risk_index(self, feature_vector: np.ndarray) -> dict:
        """
        Compute final risk index with uncertainty bounds.
        
        Returns:
            risk_score: float [0, 1] — calibrated fraud probability
            risk_tier: str — LOW / MEDIUM / HIGH / CRITICAL
            confidence_lower: float — conformal prediction lower bound
            confidence_upper: float — conformal prediction upper bound
            contributing_factors: list — top features driving the score
        """
        if self.xgb_model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        import xgboost as xgb
        
        dmatrix = xgb.DMatrix(feature_vector.reshape(1, -1))
        raw_score = self.xgb_model.predict(dmatrix)[0]
        
        # Apply conformal prediction bounds
        lower, upper = self._conformal_bounds(raw_score)
        
        # Determine risk tier
        tier = self._compute_tier(raw_score)
        
        # Feature importance for this prediction (SHAP-based)
        contributing_factors = self._explain_prediction(feature_vector)
        
        return {
            'risk_score': float(raw_score),
            'risk_tier': tier,
            'confidence_lower': float(lower),
            'confidence_upper': float(upper),
            'contributing_factors': contributing_factors,
        }
    
    def _compute_tier(self, score: float) -> str:
        """
        Risk tier mapping — configurable thresholds.
        
        These should be tuned based on:
        - Operational capacity (how many alerts can the team handle?)
        - Regulatory requirements (what's the max acceptable FNR?)
        - Historical fraud rates
        """
        if score >= 0.85:
            return 'CRITICAL'  # Auto-block or immediate review
        elif score >= 0.60:
            return 'HIGH'      # Priority queue for investigation
        elif score >= 0.30:
            return 'MEDIUM'    # Enhanced monitoring
        else:
            return 'LOW'       # Standard processing
    
    def _conformal_bounds(self, score: float, alpha: float = 0.1) -> tuple:
        """
        Conformal prediction interval.
        Guarantees coverage with probability 1-alpha.
        
        This is critical for the rules engine — it tells you
        not just "is this risky?" but "how sure are we?"
        """
        # Simplified version; production uses full conformal prediction
        # with nonconformity scores from calibration set
        margin = 0.05 + 0.1 * score * (1 - score)  # Wider at uncertainty peak
        return (max(0, score - margin), min(1, score + margin))
    
    def _explain_prediction(self, features: np.ndarray) -> list:
        """Top contributing features using SHAP values."""
        # Placeholder — use shap.TreeExplainer in production
        return [
            {'feature': 'w1h_escalation', 'contribution': 0.15},
            {'feature': 'tgn_txn_score', 'contribution': 0.12},
        ]
```

---

## 6. Rules Engine Integration

### Integration Pattern

The TGN system doesn't **replace** the existing rules engine — it **enhances** it by providing a new class of features that capture patterns invisible to static rules.

```
┌──────────────────────────────────────────────────────────────────┐
│                    RULES ENGINE ARCHITECTURE                      │
│                                                                   │
│  EXISTING RULES (keep these):                                     │
│  ├── Velocity rules (>N txns in T time)                           │
│  ├── Amount threshold rules (>£X)                                 │
│  ├── Blocklist rules (sanctioned entities)                        │
│  └── Country/MCC rules                                            │
│                                                                   │
│  NEW TGN-ENHANCED RULES:                                          │
│  ├── Risk score threshold: risk_index > 0.85 → BLOCK              │
│  ├── Escalation rule: w1h_escalation > 0.3 → ESCALATE             │
│  ├── Network contagion: >2 HIGH-risk neighbours → REVIEW          │
│  ├── Confidence rule: confidence_upper < 0.3 → ALLOW              │
│  ├── Pattern cluster: matches known fraud pattern → HOLD           │
│  └── Dynamic threshold: auto-adjusted by FPR target               │
│                                                                   │
│  DECISION FUSION:                                                  │
│  • Any rule triggers → action                                     │
│  • TGN scores break ties between conflicting rules                │
│  • Confidence bounds determine hold vs. block                      │
└──────────────────────────────────────────────────────────────────┘
```

### Feature Store API

```python
from typing import Optional
import redis
import json


class FraudFeatureStore:
    """
    Real-time feature store exposing TGN-derived features to the rules engine.
    
    The rules engine queries this store for every transaction decision.
    Features are updated asynchronously as the TGN processes events.
    
    Storage: Redis (sub-millisecond reads, TTL for staleness management)
    """
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port, db=0)
        self.ttl_seconds = 86400 * 7  # Features expire after 7 days of inactivity
    
    def update_transaction_score(
        self,
        event_id: str,
        risk_assessment: dict,
    ):
        """Store risk assessment for a specific transaction."""
        key = f"txn_risk:{event_id}"
        self.redis.setex(key, self.ttl_seconds, json.dumps(risk_assessment))
    
    def update_entity_profile(
        self, 
        entity_id: str, 
        risk_profile: dict,
    ):
        """Store/update entity-level risk profile."""
        key = f"entity_risk:{entity_id}"
        self.redis.setex(key, self.ttl_seconds, json.dumps(risk_profile))
    
    def get_entity_risk(self, entity_id: str) -> Optional[dict]:
        """
        Called by the rules engine on every transaction.
        
        Returns:
            Risk profile dict or None if entity is unknown.
        """
        key = f"entity_risk:{entity_id}"
        data = self.redis.get(key)
        return json.loads(data) if data else None
    
    def get_transaction_risk(self, event_id: str) -> Optional[dict]:
        """Called by the rules engine for transaction-level decisions."""
        key = f"txn_risk:{event_id}"
        data = self.redis.get(key)
        return json.loads(data) if data else None
    
    def get_network_risk(self, entity_id: str, depth: int = 1) -> dict:
        """
        Compute network-level risk for an entity.
        
        This is where TGN truly shines over traditional approaches:
        - Aggregates risk across 1-hop and 2-hop neighbours
        - Identifies "guilty by association" patterns
        - Detects coordinated fraud rings
        """
        # Simplified; production version queries the graph store
        neighbours = self._get_neighbours(entity_id)
        
        neighbour_risks = []
        for n_id in neighbours:
            risk = self.get_entity_risk(n_id)
            if risk:
                neighbour_risks.append(risk.get('risk_score', 0))
        
        return {
            'entity_id': entity_id,
            'neighbour_count': len(neighbours),
            'mean_neighbour_risk': (
                sum(neighbour_risks) / len(neighbour_risks) 
                if neighbour_risks else 0
            ),
            'max_neighbour_risk': max(neighbour_risks) if neighbour_risks else 0,
            'high_risk_neighbour_count': sum(1 for r in neighbour_risks if r > 0.6),
        }


# ──────────────────────────────────────────────
# RULES ENGINE ADAPTER
# ──────────────────────────────────────────────

class TGNRulesEngineAdapter:
    """
    Adapter that translates TGN risk assessments into rules engine actions.
    
    This is the integration boundary — the rules engine doesn't need to
    know anything about TGNs, graph theory, or neural networks. It just
    sees features and thresholds.
    """
    
    def __init__(self, feature_store: FraudFeatureStore, config: dict = None):
        self.feature_store = feature_store
        self.config = config or self._default_config()
    
    def evaluate_transaction(self, transaction: dict) -> dict:
        """
        Evaluate a transaction against TGN-enhanced rules.
        
        Args:
            transaction: Raw transaction dict from the payments pipeline
            
        Returns:
            Decision dict with action, reasons, and scores
        """
        entity_id = f"account:{transaction['source_account']}"
        event_id = transaction['transaction_id']
        
        # Fetch TGN-derived features
        txn_risk = self.feature_store.get_transaction_risk(event_id)
        entity_risk = self.feature_store.get_entity_risk(entity_id)
        network_risk = self.feature_store.get_network_risk(entity_id)
        
        # Evaluate rules
        decisions = []
        
        # Rule 1: Hard score threshold
        if txn_risk and txn_risk.get('risk_score', 0) >= self.config['block_threshold']:
            decisions.append({
                'rule': 'TGN_SCORE_CRITICAL',
                'action': 'BLOCK',
                'reason': f"Transaction risk score {txn_risk['risk_score']:.3f} "
                          f"exceeds threshold {self.config['block_threshold']}",
                'score': txn_risk['risk_score'],
            })
        
        # Rule 2: Escalation pattern
        if entity_risk:
            escalation = entity_risk.get('w1h_escalation', 0)
            if escalation > self.config['escalation_threshold']:
                decisions.append({
                    'rule': 'TGN_ESCALATION_PATTERN',
                    'action': 'HOLD',
                    'reason': f"Risk escalation of {escalation:.3f} detected "
                              f"in last hour",
                    'score': escalation,
                })
        
        # Rule 3: Network contagion
        if network_risk:
            if network_risk['high_risk_neighbour_count'] >= self.config['contagion_min_neighbours']:
                decisions.append({
                    'rule': 'TGN_NETWORK_CONTAGION',
                    'action': 'REVIEW',
                    'reason': f"{network_risk['high_risk_neighbour_count']} high-risk "
                              f"connected entities detected",
                    'score': network_risk['mean_neighbour_risk'],
                })
        
        # Rule 4: Confidence-based allow
        if txn_risk and txn_risk.get('confidence_upper', 1.0) < self.config['safe_upper_bound']:
            decisions.append({
                'rule': 'TGN_HIGH_CONFIDENCE_SAFE',
                'action': 'ALLOW',
                'reason': f"High confidence low-risk: upper bound "
                          f"{txn_risk['confidence_upper']:.3f}",
                'score': txn_risk['risk_score'],
            })
        
        # Resolve conflicting decisions (highest severity wins)
        final_action = self._resolve_decisions(decisions)
        
        return {
            'transaction_id': event_id,
            'final_action': final_action,
            'triggered_rules': decisions,
            'txn_risk': txn_risk,
            'entity_risk': entity_risk,
            'network_risk': network_risk,
        }
    
    def _resolve_decisions(self, decisions: list) -> str:
        """Resolve conflicting rule decisions. Strictest action wins."""
        severity_order = {'BLOCK': 4, 'HOLD': 3, 'REVIEW': 2, 'ALLOW': 1}
        if not decisions:
            return 'ALLOW'
        return max(decisions, key=lambda d: severity_order.get(d['action'], 0))['action']
    
    def _default_config(self) -> dict:
        return {
            'block_threshold': 0.85,
            'hold_threshold': 0.60,
            'review_threshold': 0.30,
            'escalation_threshold': 0.3,
            'contagion_min_neighbours': 2,
            'safe_upper_bound': 0.15,
        }
```

### Dynamic Threshold Tuning

```python
class DynamicThresholdTuner:
    """
    Auto-tunes rules engine thresholds based on operational targets.
    
    Runs as a batch job (e.g., daily) and adjusts thresholds to meet:
    - Target false positive rate (FPR) — operational capacity constraint
    - Target false negative rate (FNR) — regulatory requirement
    - Alert volume budget — maximum alerts per day
    
    This is what makes the system self-improving:
    As the TGN model gets better, thresholds can tighten automatically.
    """
    
    def __init__(self, target_fpr: float = 0.01, target_fnr: float = 0.05,
                 max_daily_alerts: int = 500):
        self.target_fpr = target_fpr
        self.target_fnr = target_fnr
        self.max_daily_alerts = max_daily_alerts
    
    def tune(self, val_scores: np.ndarray, val_labels: np.ndarray,
             daily_volume: int) -> dict:
        """
        Find optimal thresholds given operational constraints.
        
        Uses binary search on threshold space to find the operating point
        that satisfies both FPR and FNR targets.
        """
        from sklearn.metrics import precision_recall_curve, roc_curve
        
        fpr, tpr, thresholds_roc = roc_curve(val_labels, val_scores)
        fnr = 1 - tpr
        
        # Find threshold meeting FNR target
        fnr_candidates = thresholds_roc[fnr <= self.target_fnr]
        block_threshold = fnr_candidates[0] if len(fnr_candidates) > 0 else 0.9
        
        # Estimate alert volume at this threshold
        alert_rate = (val_scores >= block_threshold).mean()
        projected_alerts = alert_rate * daily_volume
        
        # If too many alerts, relax to meet capacity
        if projected_alerts > self.max_daily_alerts:
            target_rate = self.max_daily_alerts / daily_volume
            sorted_scores = np.sort(val_scores)[::-1]
            cutoff_idx = int(len(sorted_scores) * target_rate)
            block_threshold = sorted_scores[min(cutoff_idx, len(sorted_scores) - 1)]
        
        return {
            'block_threshold': float(block_threshold),
            'hold_threshold': float(block_threshold * 0.7),
            'review_threshold': float(block_threshold * 0.35),
            'projected_daily_alerts': int(projected_alerts),
            'estimated_fpr': float(fpr[np.searchsorted(thresholds_roc, block_threshold)]),
            'estimated_fnr': float(fnr[np.searchsorted(thresholds_roc, block_threshold)]),
        }
```

---

## 7. Transfer Learning Extensibility

### Design Decisions for Cross-Institution Transfer

The architecture is designed from day one to support transfer learning across institutions. Key decisions:

```
WHAT TRANSFERS                         WHAT DOESN'T TRANSFER
──────────────                         ──────────────────────
• GNN attention weights                • Node ID mappings
  (relational pattern recognition)     • Absolute amount thresholds
• Time encoder parameters              • Institution-specific node features
  (cyclical temporal patterns)         • Account-level memory states
• Message function weights             • Calibration parameters
  (interaction representation)         
• Fraud scoring head (partial)         
  (universal fraud indicators)         
```

### Feature Engineering for Transfer

```python
class TransferReadyFeatureConfig:
    """
    Feature design principles that enable cross-institution transfer:
    
    1. RELATIVE over ABSOLUTE: Use percentiles, z-scores, ratios
    2. CYCLICAL over CATEGORICAL: sin/cos encoding for time
    3. BEHAVIOURAL over NOMINAL: "deviation from norm" over "specific value"
    4. UNIVERSAL over LOCAL: MCC codes over merchant names
    """
    
    # These features transfer well across institutions
    TRANSFERABLE_FEATURES = [
        'normalised_amount_percentile',   # Within-institution relative
        'time_of_day_sin', 'time_of_day_cos',  # Universal cyclical
        'day_of_week_sin', 'day_of_week_cos',  # Universal cyclical
        'amount_deviation_zscore',         # Relative to account behaviour
        'txn_velocity_percentile',         # Relative to account norm
        'cross_border_flag',               # Universal binary
        'channel_encoding',                # Universal categorical
        'time_since_last_txn_normalised',  # Relative to account rhythm
    ]
    
    # These features are institution-specific (don't transfer)
    LOCAL_FEATURES = [
        'raw_amount',                      # Currency/scale dependent
        'account_balance',                 # Institution specific
        'branch_code',                     # Local geography
        'product_type_encoding',           # Product catalogue specific
    ]


class DomainAdapter(nn.Module):
    """
    Lightweight domain adaptation layer for transfer learning.
    
    When deploying a pre-trained TGN to a new institution:
    1. Freeze GNN and time encoder weights
    2. Train only this adapter + new calibration layer
    3. Fine-tune end-to-end with small learning rate
    
    This typically requires 10-20% of the data needed for
    training from scratch.
    """
    
    def __init__(self, source_feat_dim: int, target_feat_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(target_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, source_feat_dim),
        )
    
    def forward(self, target_features: torch.Tensor) -> torch.Tensor:
        """Map target institution features to source feature space."""
        return self.adapter(target_features)
```

---

## 8. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)
- [ ] Set up PyG environment with TGN dependencies
- [ ] Implement graph schema and `UnifiedGraphEvent` data model
- [ ] Build `FeatureEncoder` with transfer-ready features
- [ ] Implement `TGNFraudDetector` model class
- [ ] Train on synthetic or public dataset (DGraph or IEEE-CIS)
- [ ] Implement temporal train/val/test split with leak verification

### Phase 2: Multi-Source Fusion (Weeks 5-8)
- [ ] Build `MultiSourceFusionPipeline` for bank transaction feeds
- [ ] Add device login event processing
- [ ] Integrate external threat intelligence feed
- [ ] Implement `StreamingGraphBuilder` with Kafka
- [ ] Build node registry with cross-source entity resolution
- [ ] End-to-end test with multi-source synthetic data

### Phase 3: Risk Quantification (Weeks 9-12)
- [ ] Implement `TGNScoreCalibrator` (isotonic regression)
- [ ] Build `TemporalPatternProfiler` with multi-window features
- [ ] Train `CompositeRiskIndex` (XGBoost on TGN embeddings)
- [ ] Implement conformal prediction for uncertainty bounds
- [ ] Add SHAP-based explainability for contributing factors
- [ ] Validate risk tiers against historical fraud data

### Phase 4: Rules Engine Integration (Weeks 13-16)
- [ ] Deploy `FraudFeatureStore` (Redis)
- [ ] Build `TGNRulesEngineAdapter` with configurable rules
- [ ] Implement `DynamicThresholdTuner`
- [ ] Integration testing with rules engine
- [ ] A/B test TGN-enhanced rules vs. existing rules
- [ ] Documentation and operational runbooks

### Phase 5: Transfer Learning Prep (Weeks 17-20)
- [ ] Refactor features to ensure `TransferReadyFeatureConfig` compliance
- [ ] Implement `DomainAdapter` module
- [ ] Test transfer to synthetic second institution
- [ ] Benchmark: transfer learning vs. training from scratch
- [ ] Document transfer learning playbook

---

## Key References

1. **Rossi et al. (2020)** — *Temporal Graph Networks for Deep Learning on Dynamic Graphs* (the foundational TGN paper)
2. **Kim et al. (2024)** — *Temporal Graph Networks for Graph Anomaly Detection in Financial Networks* (TGN applied to DGraph financial dataset)
3. **Kapoor & Narayanan (2023)** — *Leakage and the Reproducibility Crisis in ML-Based Science* (critical reading on temporal data leakage)
4. **NVIDIA Financial Fraud Blueprint** — GNN embeddings + XGBoost ensemble pattern
5. **PyG TGN Documentation** — `torch_geometric.nn.models.tgn.TGNMemory`
6. **TGB Benchmark** — Temporal Graph Benchmark for standardised evaluation
7. **T-GNNExplainer (Xia et al.)** — Model-agnostic explainability for temporal GNNs

---

*Created February 2026 | Framework: PyTorch Geometric | Python 3.10+*
