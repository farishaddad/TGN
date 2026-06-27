# Production AWS Architecture — TGN Ensemble Fraud Detection
*June 2026*

---

## Overview

End-to-end production architecture for deploying the TGN Ensemble fraud detection system on AWS. The architecture is designed around three constraints: **<100ms P99 inference latency**, **50M+ active entities**, and **1,000–10,000 TPS throughput**.

The design implements the **Lambda architecture pattern** (BRIGHT, eBay, CIKM 2022) — expensive multi-hop TGN computations run in batch offline, real-time inference retrieves pre-computed embeddings from cache and runs only a lightweight scoring head.

---

## Architecture Diagram

```
Core Banking System
      │ transaction event
      ▼
API Gateway (HTTP API) → Lambda (scoring orchestrator)
                              │
              ┌───────────────┼──────────────────────┐
              ▼               ▼                      ▼
        ElastiCache      Feature Store          Kinesis Data Streams
        (Redis/Valkey)   (profile + velocity)        │
        embeddings       SageMaker                   ▼
              │                              Lambda (graph event processor)
              └──────────────┐                   │
                             ▼               Neptune DB
                  SageMaker TGN            (property graph)
                  Fast Path                    │
                  ml.g4dn.2xlarge × 3      Neptune Analytics
                  Pass 1 (~20ms)           (community detection,
                             │              vector similarity)
                  score ≥ 0.3?
                             │ yes (~3% of txns)
                             ▼
                  SageMaker SEAL
                  ml.g4dn.xlarge × 2
                  Pass 2 (+50ms)
                             │
                  Risk Score + Tier + Explanation
                             │
              ┌──────────────┘
              ▼
        API Gateway response → Core Banking
              │
              ▼
        DocumentDB (fraud case record)
        QuickSight / Analyst Dashboard

─── Training Path ─────────────────────────────────────
S3 (raw data) → Glue ETL → SageMaker Pipelines
→ Training Jobs → Model Registry → Endpoint Update

─── Monitoring ─────────────────────────────────────────
CloudWatch + Model Monitor → EventBridge
→ Retrain trigger on drift detection
```

---

## Service-by-Service Breakdown

### Layer 1: Transaction Ingestion

#### Amazon Kinesis Data Streams
Real-time event ingestion at 1,000–10,000 TPS. Each transaction arrives as a timestamped event tuple. Kinesis provides:
- Ordering guarantees within a shard
- Ordered delivery to the graph construction layer
- Replay capability for retraining and backfill
- On-Demand mode scales automatically without pre-provisioning shards

#### AWS Lambda (graph event processor)
Consumes from Kinesis. Transforms each raw transaction into a graph event (node lookups, edge feature encoding, multi-scale time encoding via `PRAGMATimeEncoder`), and writes to Neptune. Serverless — cost-proportional to volume.

---

### Layer 2: Graph Store

#### Amazon Neptune (database)
Persistent property graph store for the full transaction graph: cardholders, merchants, devices, account-change events. Used for:
- **Subgraph extraction** at inference time — TGN-SEAL k=2 hop traversal for flagged transactions (Pass 2 only)
- **Fund-flow DAG queries** — ETGAT money chain tracing via Gremlin traversals
- **Fraud analyst workbench** — ad-hoc investigation queries

Neptune natively supports property graphs with temporal edge properties, handles billions of edges, and provides Gremlin for the multi-hop traversals TGN requires.

#### Amazon Neptune Analytics
In-memory analytics engine for the full graph. Used for:
- Community detection — surfaces fraud rings automatically
- Vector similarity search on TGN embeddings — "find transactions similar to this flagged one"
- GraphStorm integration for offline batch scoring and embedding enrichment
- Analyst notebook for interactive fraud investigation

---

### Layer 3: Node Memory Store

#### Amazon ElastiCache for Redis (Valkey)
Stores pre-computed entity embeddings from the BRIGHT Lambda architecture. Two data types:

| Data | Description | TTL |
|---|---|---|
| TGN memory vectors `s_i(t)` | 128-dim float vectors per cardholder/merchant, updated after each transaction | 90 days inactivity |
| Batch embeddings | Richer multi-hop embeddings pre-computed hourly, used as base for real-time scoring | 2 hours |

Redis gives sub-2ms P99 lookup for 50M+ active entities. The Lambda architecture means real-time inference retrieves pre-computed embeddings from Redis rather than traversing Neptune live — this keeps TGN embedding latency under 20ms.

---

### Layer 4: ML Inference

#### Amazon SageMaker Real-Time Inference — Two Endpoints

| Endpoint | Model | Instance | Latency target |
|---|---|---|---|
| **TGN Fast Path** | TGN embedder + RF scorer (Pass 1) | ml.g4dn.2xlarge × 3 | ~20ms |
| **SEAL Structural** | TGN-SEAL DGCNN (Pass 2) | ml.g4dn.xlarge × 2 | ~50ms additional |

The SEAL endpoint is only invoked for ~3% of transactions scoring ≥0.3 in Pass 1. Separate endpoints allow independent scaling and prevent structural enrichment from consuming fast-path capacity.

**Two-pass scoring logic:**
```
Pass 1 (100% of transactions):
  TGN memory + RF decoder → score
  score < 0.3  → APPROVE
  score ≥ 0.3  → Pass 2

Pass 2 (~3% of transactions):
  Neptune subgraph extraction → SEAL DGCNN → structural score
  combined_score = α·tgn_score + (1-α)·seal_score
  + two-hurdle deviation filter (TFLAG pattern)
  0.3–0.7 → Soft decline / 3DS challenge
  > 0.7   → Hard decline / case management alert
```

#### Amazon SageMaker Batch Transform
Runs the BRIGHT batch embedder hourly:
1. Takes last hour's transactions
2. Runs full multi-hop TGN neighbourhood aggregation
3. Writes updated entity embeddings back to ElastiCache

This is the "batch layer" of the Lambda architecture — the expensive multi-hop computation runs offline, keeping real-time inference fast and predictable.

#### Amazon SageMaker Feature Store
Online + offline feature store for:
- **Node profile features** — account age, balance quantile, spending limit (inputs to `ProfileStateEncoder`)
- **Rolling velocity counts** — 5min/1hr/24hr transaction counts per card
- **Deviation reference statistics** — rolling mean and std of anomaly scores per card (TFLAG two-hurdle filter)

Online store: sub-10ms lookup at inference time. Offline store: feeds training pipeline with training/serving feature parity.

---

### Layer 5: Training & Model Management

#### Amazon SageMaker Training Jobs
Two training cadences:

| Job | Frequency | Duration | Instance | Purpose |
|---|---|---|---|---|
| Fine-tune | Weekly (4×/month) | ~8h | ml.g4dn.12xlarge | Update TGN weights on recent labelled transactions |
| Full retrain | Monthly | ~48h | ml.g4dn.12xlarge | Full ensemble retraining: TGN + RF head + meta-learner + drift autoencoder |

#### Amazon SageMaker Pipelines
Orchestrates the full MLOps workflow:
```
Data prep → Graph construction → TGN training → RF head fitting
→ Meta-learner training → Validation → Endpoint update
```
Triggered weekly by EventBridge. Gate: AUPRC must be ≥ previous version before endpoint update.

#### Amazon SageMaker Model Registry
Versioned model store. Every trained model registered with:
- Training metrics (AUPRC, F1, FP rate per segment, FP rate per fraud pattern)
- Training data hash (reproducibility)
- Approval gate: human review required for production promotion

#### Amazon S3
Raw transaction history, training datasets, graph snapshots, model artifacts.

| Tier | Age | Storage Class |
|---|---|---|
| Hot | 0–90 days | S3 Standard |
| Warm | 90–365 days | S3-IA |
| Cold | 365+ days | S3 Glacier |

#### AWS Glue
ETL pipeline that transforms raw transaction records (tabular) into heterogeneous graph format (Neptune-compatible CSV). Runs nightly. Also builds training datasets with strict chronological splits (no random shuffle — prevents temporal leakage).

---

### Layer 6: Drift Detection & Adaptive Maintenance

#### Amazon CloudWatch + SageMaker Model Monitor
Monitors:
- Rolling AUPRC on labelled feedback (chargebacks received 30–90 days post-transaction)
- Embedding distribution drift (TGNN-CDD autoencoder reconstruction error)
- Endpoint latency P99 and error rate
- ElastiCache memory utilisation and cache hit rate
- Feature distribution drift (SageMaker Model Monitor built-in)

#### Amazon EventBridge
Event-driven triggers:
- `CloudWatch alarm (AUPRC drop > 5%)` → EventBridge → SageMaker Pipeline (retrain)
- `Cron (hourly)` → EventBridge → Batch Transform (embedding refresh)
- `Cron (daily)` → EventBridge → Glue ETL (feature store update)

---

### Layer 7: API & Integration

#### Amazon API Gateway (HTTP API)
Exposes the fraud scoring endpoint to core banking. Routes:
- `POST /score` → Lambda → ElastiCache + SageMaker → response <100ms P99
- `POST /score/batch` → Lambda → SageMaker Batch Transform (async, for post-transaction AML review)

Provides: IAM/Cognito authentication, throttling, request logging, canary deployment for model updates.

#### AWS Lambda (scoring orchestrator)
Thin orchestration layer:
1. Receive transaction event from API Gateway
2. Retrieve entity embeddings from ElastiCache (~2ms)
3. Retrieve profile features from SageMaker Feature Store (~8ms)
4. Invoke TGN Fast Path endpoint (~20ms)
5. Apply two-hurdle deviation filter
6. If score ≥ 0.3: invoke SEAL endpoint (+50ms)
7. Apply meta-learner fusion
8. Return risk tier + score + explanation signals

Decouples routing logic from ML models — routing can update without redeploying models.

---

### Layer 8: Fraud Analyst Tooling

#### Amazon QuickSight (or CloudFront + AppSync + React)
Business dashboard:
- Real-time feed of flagged transactions with risk tier and explanation signals
- Historical fraud pattern trends (by type, segment, time-of-day)
- Graph visualisation of detected fraud rings (Neptune Analytics community detection)
- "Find similar" — vector similarity search on TGN embeddings

#### Amazon DocumentDB
Fraud case records — enriched flagged transactions with analyst notes, resolution status, chargeback outcomes. Closes the feedback loop: confirmed fraud → training label → weekly fine-tune.

---

## Cost Estimate (eu-west-1, On-Demand)

| Component | Service | Est. Monthly |
|---|---|---|
| Transaction ingestion | Kinesis On-Demand (1K TPS) | ~$300 |
| Node memory store | ElastiCache Redis 200GB | ~$900 |
| TGN inference (Fast Path) | SageMaker g4dn.2xlarge × 3, 24/7 | ~$2,500 |
| Structural inference (SEAL) | SageMaker g4dn.xlarge × 2, 24/7 | ~$1,200 |
| Training (weekly + monthly) | SageMaker g4dn.12xlarge | ~$700 |
| Graph store | Neptune db.r6g.2xlarge × 3, 2TB | ~$1,700 |
| Feature store | SageMaker Feature Store 5TB | ~$1,500 |
| Observability | CloudWatch + Model Monitor | ~$200 |
| **Total (On-Demand)** | | **~$9,000/month** |

**With 1-year Reserved Instances on compute:** ~$5,500/month (~$66K/year)

AWS Pricing Calculator: https://calculator.aws/#/estimate?id=f74396b7e9337634ecd19c93cb2f5ecdfbc9462f

---

## Design Decisions: Why This Combination

### Neptune + ElastiCache (not just Neptune)
Neptune is authoritative but P99 Gremlin query latency at inference scale is 50–200ms — too slow for the fast path. ElastiCache gives sub-2ms for the 97% of transactions that don't need live graph traversal. The Lambda pattern (BRIGHT, eBay CIKM 2022) separates batch pre-computation from real-time lookup.

### Two SageMaker endpoints (not one)
Collocating SEAL with the fast-path TGN wastes GPU capacity on the 97% of transactions that never need structural enrichment. Separate endpoints allow independent auto-scaling: fast-path handles full TPS, SEAL runs on smaller instances since it sees only ~3% of volume.

### SageMaker Feature Store (not just Redis)
Redis stores dynamic embeddings updated per transaction. Feature Store stores slower-changing profile features (account age, balance quantile) and the rolling statistics needed for the two-hurdle deviation filter. Keeping them separate maintains Redis leanness and provides offline feature consistency for training/serving parity — a common source of AUC degradation in production.

---

## Research Foundations

| Architectural pattern | Source paper |
|---|---|
| Lambda inference (batch + RT) | BRIGHT — Lu et al., eBay, CIKM 2022 |
| Two-pass scoring + two-hurdle FP filter | TFLAG — Jiang et al., arXiv 2501.06997, 2025 |
| TGN-SEAL structural enrichment | Sajadi et al., EPJ Data Science, 2026 |
| PRAGMA time encoding | Revolut PRAGMA team, arXiv 2604.08649, 2026 |
| Profile state encoder | Revolut PRAGMA team, arXiv 2604.08649, 2026 |
| GraphSMOTE training oversampling | THG-OAFN — Wei & Lee, PLoS ONE, 2025 |
| Buyer/seller subgraph modules | C2GAT — Chen & Yang, Ant Group, Front. AI, 2026 |

---

*Architecture version 1.0 — June 2026*
*Companion documents: ENSEMBLE_DESIGN.md, DEMO_READINESS.md, KIRO.md*
