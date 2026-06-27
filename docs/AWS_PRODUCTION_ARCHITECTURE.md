# Production AWS Architecture — TGN Fraud Detection + Rule Engine

> **Target codebase:** `/Users/fahaddad/Documents/TGN`
> **Last updated:** June 2026

---

## Architecture Overview

```
                    ┌─────────────────────────────────┐
                    │         API GATEWAY              │
                    │   (REST + WebSocket endpoints)   │
                    └────────────────┬────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                       │
    ┌─────────▼──────┐    ┌──────────▼──────┐    ┌─────────▼──────┐
    │  Lambda         │    │  Lambda          │    │  Lambda         │
    │  score_tx       │    │  rule_eval       │    │  rule_crud      │
    │  (TGN inference)│    │  (rule engine)   │    │  (manage rules) │
    └─────────┬──────┘    └──────────┬──────┘    └─────────┬──────┘
              │                      │                       │
    ┌─────────▼──────────────────────▼───────────────────────▼──────┐
    │                    CORE DATA LAYER                             │
    │  ElastiCache Redis  │  DynamoDB Rules  │  SageMaker Endpoint  │
    │  (embedding cache)  │  (rule store)    │  (TGN model serving) │
    └────────────────────────────────────────────────────────────────┘
```

---

## Services, Why They're Needed, and How They Work Together

### 1. Amazon API Gateway
**Why:** Single entry point for all transaction scoring requests — handles auth, rate limiting, and routing.
- REST endpoint for synchronous scoring (`POST /score`)
- WebSocket endpoint for the Streamlit app's real-time updates
- Throttles to prevent model overload (e.g. 10k RPS max)
- Passes requests to Lambda functions downstream

---

### 2. AWS Lambda — 3 separate functions

| Function | Purpose |
|----------|---------|
| `score-transaction` | Calls SageMaker for TGN inference, pulls embeddings from ElastiCache |
| `rule-evaluator` | Loads rule set from DynamoDB, runs `RuleEvaluator.evaluate()` against transaction dict |
| `rule-crud` | CRUD operations on rules (create/update/enable/disable) from the Rule Builder UI |

**Why separate:** Rule evaluation is fast (~1ms, pure Python logic) and should never be blocked waiting for the TGN model (~50ms). Each function scales independently and the rule function can `BLOCK` a transaction **before** the model is ever invoked.

**How they work together:** API Gateway routes `/score` → `score-transaction` Lambda, which first calls `rule-evaluator` synchronously. If the rule returns `BLOCK`, it short-circuits and never touches SageMaker. Otherwise, both results are fused in `HybridScorer` and returned.

---

### 3. Amazon SageMaker Real-Time Endpoint
**Why:** Hosts the trained TGN PyTorch model with GPU-backed inference. Lambda alone can't run a GNN — 512MB memory limit and no GPU.
- Model artifact (`.pt` checkpoint) lives in S3
- Endpoint auto-scales with Application Auto Scaling based on invocation count
- Returns `risk_score` + confidence interval back to the Lambda

**How it connects:** `score-transaction` Lambda calls `sagemaker-runtime:InvokeEndpoint` with the transaction feature vector. Response is the TGN embedding + fraud probability.

---

### 4. Amazon ElastiCache (Redis)
**Why:** The TGN's Lambda architecture pre-computes entity embeddings offline. At inference time, you need <1ms lookup of the source/destination node's current embedding — Redis delivers this.
- Keys: `emb:{node_id}` → float32 vector
- TTL: 24 hours (refreshed by the batch job)
- Reduces SageMaker payload: only the delta needs recomputing, not the full neighbourhood

**How it connects:** `score-transaction` Lambda fetches `src_emb` and `dst_emb` from Redis before calling SageMaker, passing them as pre-computed features. This cuts P99 latency from ~200ms to ~50ms.

---

### 5. Amazon DynamoDB — 2 tables

| Table | Contents |
|-------|---------|
| `fraud-rules` | Rule definitions (id, conditions JSON, verdict, priority, enabled flag) |
| `score-audit` | Every scoring decision (tx_id, tgn_score, rule_verdict, final_tier, timestamp) |

**Why DynamoDB:** Rules need single-digit ms reads (rule-evaluator is on the hot path). DynamoDB with DAX gives microsecond reads for the active rule set.
- Rule Builder UI writes to `fraud-rules` via `rule-crud` Lambda
- All scoring events written to `score-audit` for compliance + model monitoring

---

### 6. Amazon DAX (DynamoDB Accelerator)
**Why:** The `rule-evaluator` Lambda reads the full active rule set on every invocation. DAX caches this in-memory so Lambda doesn't hammer DynamoDB on each of 10k+ transactions/sec.
- Rule set cached with 5-minute TTL (rules don't change frequently)
- Cache invalidated when `rule-crud` updates a rule

---

### 7. Amazon S3 — 3 buckets

| Bucket | Contents |
|--------|---------|
| `tgn-model-artifacts` | Trained `.pt` checkpoints, versioned with SageMaker Model Registry |
| `tgn-transaction-data` | Raw CSV uploads from the "Upload CSV" page, partitioned by date |
| `tgn-embeddings-batch` | Pre-computed embedding outputs from the batch job |

**How it connects:** SageMaker loads the model from `tgn-model-artifacts`. EventBridge triggers the batch embedding job when new data lands in `tgn-transaction-data`.

---

### 8. AWS Batch + ECR
**Why:** The `BatchEmbedder` (from `ensemble/embedding/batch_embedder.py`) runs the full graph and recomputes all node embeddings overnight. This is too long for Lambda (15-minute limit) and needs GPU.
- Dockerised PyTorch job stored in ECR
- Runs nightly via EventBridge Scheduler
- Outputs embedding vectors to S3 → synced to ElastiCache by a post-job Lambda

---

### 9. Amazon EventBridge
**Why:** Orchestrates the async workflows without tight coupling.
- `cron(0 2 * * ? *)` → triggers nightly AWS Batch embedding job
- `S3 PutObject` event → triggers retraining pipeline when new labelled data arrives
- `DynamoDB Streams` → triggers cache invalidation when rules change

---

### 10. Amazon SageMaker Pipelines (retraining)
**Why:** When new fraud labels arrive, the TGN model needs retraining. Pipelines automates the `TGNTrainer` workflow end-to-end with versioning.
- Steps: Data validation → Training → Evaluation (AUPRC check) → Conditional model registration → Endpoint update
- Model Registry holds all versions; only `Approved` models get deployed
- Triggered by EventBridge when labelled data volume exceeds a threshold

---

### 11. Amazon CloudWatch + CloudWatch Alarms
**Why:** The `drift_monitor.py` and `threshold_adapter.py` in the ensemble need operational signals to detect concept drift in production.

| Metric | What it detects |
|--------|----------------|
| `FraudRate_7d` rolling average | Sudden drop/spike = population shift |
| `TGNScore_p95` | Model confidence degradation |
| `RuleHitRate` per rule ID | Rules becoming stale/over-triggering |
| `SageMakerEndpoint/Latency` | Inference slowdown |

Alarms → SNS → email/Slack when drift exceeds thresholds.

---

### 12. Amazon Cognito
**Why:** The Rule Builder UI must be access-controlled — analysts should be able to create rules, but only risk managers should be able to enable `BLOCK` verdict rules.
- User pools: `analyst` role (read/write rules, test), `risk_manager` role (enable BLOCK)
- JWT tokens passed to API Gateway → Lambda validates before rule mutations

---

### 13. AWS Secrets Manager
**Why:** Lambda functions need credentials for Redis, SageMaker endpoint ARNs, and DynamoDB table names without hardcoding.
- Secrets rotated automatically
- Fetched at Lambda cold start and cached in-memory for the function lifetime

---

## Request Flow — End to End

```
1. Transaction arrives at API Gateway (POST /score)
       │
2. rule-evaluator Lambda reads active rules from DAX/DynamoDB (~1ms)
       │
3.  ├── BLOCK verdict? → return CRITICAL immediately (skip steps 4–5)
    └── PASS/ESCALATE? → continue
       │
4. score-transaction Lambda fetches src/dst embeddings from ElastiCache (~1ms)
       │
5. SageMaker endpoint runs TGN inference (~40ms)
       │
6. HybridScorer fuses TGN score + rule verdict → final RiskTier
       │
7. Result written to DynamoDB audit table (async, fire-and-forget)
       │
8. Response returned to caller with risk_tier + explanation signals
```

**Total P99 latency target: ~50ms** (BLOCK path: ~2ms)

---

## Infrastructure as Code (CDK)

All infrastructure deployed via **AWS CDK** (Python) — one stack per layer:

```
cdk/
├── stacks/
│   ├── api_stack.py          # API Gateway + Lambda functions
│   ├── data_stack.py         # DynamoDB + ElastiCache + DAX
│   ├── ml_stack.py           # SageMaker endpoint + Model Registry
│   ├── batch_stack.py        # AWS Batch + ECR
│   └── monitoring_stack.py   # CloudWatch + Alarms + SNS
```

---

## Service Summary

| Service | Layer | Role |
|---------|-------|------|
| API Gateway | Ingress | Auth, rate limiting, routing |
| Lambda (×3) | Compute | Rule eval, TGN scoring, rule CRUD |
| SageMaker Endpoint | ML | GPU TGN model inference |
| ElastiCache Redis | Cache | Sub-ms embedding lookup |
| DynamoDB | Storage | Rule store + audit log |
| DAX | Cache | Rule set microsecond reads |
| S3 (×3) | Storage | Models, data, embeddings |
| AWS Batch + ECR | Compute | Nightly embedding pre-computation |
| EventBridge | Orchestration | Async workflow triggers |
| SageMaker Pipelines | MLOps | Automated retraining + deployment |
| CloudWatch | Observability | Drift detection + latency monitoring |
| Cognito | Security | Role-based access to Rule Builder |
| Secrets Manager | Security | Credential management |
