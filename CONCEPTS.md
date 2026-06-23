# Core Concepts

A brief introduction to TGN, temporal graphs, and MiNT for newcomers.

## Temporal Graphs

A **temporal graph** represents interactions between entities over time.
Unlike static graphs where edges are permanent, temporal edges have timestamps
and represent events that happened at specific moments.

In fraud detection:
- **Nodes** = entities (accounts, merchants, devices)
- **Edges** = interactions (transactions, logins, transfers)
- **Timestamps** = when each interaction occurred
- **Features** = properties of each interaction (amount, channel, etc.)

This temporal structure is critical because fraud has temporal signatures:
- Card testing: rapid bursts of tiny transactions
- Account takeover: sudden change in spending pattern
- Money laundering: layered flows through intermediaries

## Temporal Graph Networks (TGN)

TGN (Rossi et al., 2020) is a framework for learning on temporal graphs.
Its key innovation is **memory** — each node maintains a state vector that
evolves with every interaction.

### Architecture Components

1. **Memory Module**: Per-node state vectors updated via GRU
2. **Message Function**: Computes what information to pass from an interaction
3. **Aggregator**: Combines multiple messages (we use LastAggregator)
4. **Embedding Module**: Graph attention over temporal neighbors
5. **Scoring Heads**: Map embeddings to predictions

### Why TGN for Fraud?

- **Sequential awareness**: Captures behavioral patterns over time
- **Memory**: Remembers account history without explicit features
- **Inductive**: Can score new accounts never seen during training
- **Real-time**: Updates with each new transaction during inference

## Combined Loss Function

We use two loss components:

### Contrastive Link Loss (unsupervised)
For each real edge, sample a random "negative" edge. Train the model to
score real edges higher than fake ones. This learns **structural** patterns
(who normally transacts with whom).

### Supervised Node Loss (supervised)
For labeled nodes, predict fraud/legit using weighted BCE. Class weighting
handles the extreme imbalance (typically 1-5% fraud). This learns
**behavioral** patterns (what fraud looks like).

Combined: `loss = 0.5 * link_loss + 0.5 * node_loss`

## MiNT: Multi-Network Training

MiNT trains a single TGN across multiple transaction networks simultaneously.

### Why Multi-Network?

Fraud patterns are universal:
- Card testing looks similar across any payment network
- Account takeover follows the same temporal signature everywhere
- Money laundering uses similar layering strategies

By training on 5+ networks, the model learns **generalizable** fraud features
rather than network-specific ones.

### Pipeline

1. Generate N synthetic networks with different parameters
2. Accumulate gradients across all networks before optimizer step
3. Evaluate zero-shot on completely unseen networks
4. Fine-tune: freeze backbone (memory + GNN), retrain scoring heads
5. Compare against single-network trained models

### Fine-Tuning Strategy

When deploying to a new network:
1. Start with MiNT-pretrained weights
2. Freeze all parameters except scoring heads
3. Train heads on small labeled dataset from target network
4. This adapts the decision boundary without losing learned features

## Risk Tiers

The scoring system maps continuous probabilities to discrete actions:

| Score Range | Tier | Action |
|---|---|---|
| 0.00 - 0.30 | LOW | Allow automatically |
| 0.30 - 0.60 | MEDIUM | Flag for review |
| 0.60 - 0.85 | HIGH | Hold for investigation |
| 0.85 - 1.00 | CRITICAL | Block immediately |

Score calibration (isotonic regression) ensures scores represent true
probabilities: a score of 0.7 means ~70% probability of fraud.

## Key Metrics

- **AUC-PR** (primary): Area under precision-recall curve. Best for imbalanced data.
- **AUC-ROC**: Area under ROC curve. Less informative with extreme imbalance.
- **Precision**: Of flagged transactions, what % are actually fraud?
- **Recall**: Of all fraud, what % did we catch?
- **F1**: Harmonic mean of precision and recall.

## References

- Rossi et al. (2020) "Temporal Graph Networks for Deep Learning on Dynamic Graphs"
- Xu et al. (2020) "Inductive Representation Learning on Temporal Graphs" (TGAT)
- ScalingTGNs/MiNT: Multi-network pre-training across Ethereum token networks
