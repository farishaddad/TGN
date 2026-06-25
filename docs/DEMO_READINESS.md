# Demo Readiness — TGN Ensemble Fraud Detection
**Purpose:** Make the project AWS-demo-ready with synthetic data and a UI that showcases the fraud detection approach to a technical audience (Accenture SAIL/FS-OS teams)
*June 2026*

---

## 1. What "Demo Ready" Means

A live AWS demo for Accenture SAIL/FS-OS needs to achieve three things:

1. **Explain the approach** — the audience needs to understand *why* this is better than a standard ML model, without needing to read research papers
2. **Show it working** — generate data → train → score transactions → reveal which fraud patterns were detected and *why*
3. **Survive a live demo** — reliable, fast, no broken states, reproducible results with a fixed seed

The existing app has the right bones (5-page Streamlit flow) but is currently a **learning app**, not a **showcase app**. This document specifies what to change, add, and fix for the demo.

---

## 2. Current State Assessment

### What works well ✅
- BankSim generator with 5 configurable fraud patterns
- Live training with loss/AUC-PR chart
- Risk tier scoring (LOW/MEDIUM/HIGH/CRITICAL) with confidence intervals
- Isotonic calibration
- Temporal chronological split

### What's missing for a demo ❌

| Gap | Impact on demo |
|---|---|
| Scoring page shows only raw score + tier — no *explanation* of why | Audience can't understand the model's reasoning |
| No graph visualisation of detected fraud (only "Explore Graph" for input graph) | The graph-based nature of the approach is invisible |
| Training is too slow at default settings for a live demo (~2 min) | Demo awkwardness; audience disengages |
| No "fraud pattern walkthrough" mode | Can't narrate specific fraud stories |
| Page 4 scores 10 hardcoded transactions — not compelling | Need curated demo scenarios |
| No comparison of TGN vs. baseline (logistic regression / XGBoost) | Can't justify the approach vs. what they already use |
| No AWS-specific narrative (where would each component run?) | Misses the AWS positioning opportunity |
| No ensemble view — single model only | Doesn't showcase the research depth |
| App title/branding is generic | Should be branded for the SAIL / Accenture audience |

---

## 3. Changes Required

### 3.1 Immediate Fixes (Day 1 — before any ensemble work)

**FIX 1: Add a Demo Mode to the generator (page 1)**

Add a "Demo Mode" toggle that generates a pre-seeded scenario with named accounts and a pre-scripted fraud story. This avoids "generate random stuff" and lets the presenter narrate specific events.

```python
# Add to app/pages/1_Generate_Data.py
DEMO_SCENARIOS = {
    "Card Testing Ring": {
        "description": "A compromised card is tested with 8 micro-transactions before a £2,400 purchase",
        "config": GeneratorConfig(num_accounts=50, num_merchants=15, 
                                  num_transactions=2000, fraud_rate=0.04, seed=42),
        "patterns": ["card_testing", "account_takeover"],
        "highlight_account": 7,  # the compromised card
    },
    "Money Laundering Network": {
        "description": "£180k layered through 6 intermediary accounts over 48 hours",
        "config": GeneratorConfig(num_accounts=80, num_merchants=20,
                                  num_transactions=3000, fraud_rate=0.03, seed=99),
        "patterns": ["money_laundering"],
    },
    "Bust-Out Fraud": {
        "description": "Account builds credit history over 3 months, then maxes out instantly",
        "config": GeneratorConfig(num_accounts=60, num_merchants=18,
                                  num_transactions=2500, fraud_rate=0.025, seed=77),
        "patterns": ["bust_out", "synthetic_identity"],
    },
}
```

**FIX 2: Speed up training for demo (page 3)**

Add a "Demo Mode" preset that uses a smaller graph but still demonstrates the concepts:
- 1,000 transactions (not 5,000)
- 10 epochs with batch_size=100
- Reduce to ~15–20 seconds total
- Pre-train a model and cache it in `checkpoints/demo_model.pt` so the presenter can click "Load Pre-trained" and skip training entirely

```python
# Add to app/pages/3_Train_Model.py
if st.button("⚡ Load Pre-trained Demo Model"):
    # Load checkpoint from checkpoints/demo_model.pt
    scorer = Scorer.from_checkpoint("checkpoints/demo_model.pt")
    st.session_state["trained_model"] = scorer.model
    st.success("Pre-trained model loaded! Ready to score.")
```

**FIX 3: Include a pre-generated demo checkpoint in the repo**

Add `checkpoints/demo_model.pt` (generated with seed=42, 5,000 transactions, 20 epochs). This means the demo can show training for those who want to see it live, OR skip straight to scoring.

---

### 3.2 Scoring Page Redesign (Page 4) — highest priority

The current scoring page is functionally correct but visually inert. Replace it with a three-panel layout:

```
┌────────────────────────────────────────────────────────────────┐
│  Transaction Scoring & Fraud Pattern Detection                  │
├────────────────┬───────────────────────────────────────────────┤
│  INPUT         │  RISK ASSESSMENT        │  WHY: EXPLANATION   │
│                │                         │                      │
│  Source: 007   │  ████████████ 0.84      │  🔴 Card Testing     │
│  Merchant: 42  │  CRITICAL               │  8 micro-txns in 4m │
│  Amount: £2400 │                         │  → anomalous burst   │
│  Time: 14:23   │  Confidence: 0.79-0.89  │                      │
│                │                         │  🔴 New Merchant     │
│  [SCORE]       │  TGN Memory: HIGH       │  First ever visit    │
│                │  Graph Struct: MED      │  to merchant 42      │
│                │  Temporal: HIGH         │                      │
│                │                         │  🟡 Amount Spike     │
│                │                         │  12.4x normal avg    │
└────────────────┴───────────────────────┴──────────────────────┘
│  FRAUD PATTERN TIMELINE (for source account 007)               │
│  [mini chart: 30-day history + this transaction marked red]    │
└────────────────────────────────────────────────────────────────┘
```

**Explanation logic to implement:**

```python
# New file: tgn_learn/scoring/explainer.py
class FraudExplainer:
    """Generate human-readable explanations for fraud scores.
    
    Produces 2-4 bullet-point reasons for each scoring decision,
    mapping model signals to domain-meaningful language.
    
    Designed for demo audiences who are fraud domain experts
    but not ML experts.
    """
    def explain(
        self, 
        result: ScoringResult,
        edge: Edge,
        graph: TemporalGraph,
    ) -> list[FraudSignal]:
        """
        Returns a ranked list of FraudSignal objects, each with:
          - icon: emoji (🔴/🟡/🟢)
          - title: short label ("Card Testing Burst")
          - detail: one-sentence explanation
          - contribution: float 0-1 (how much this signal contributed)
        """
        signals = []
        
        # Check velocity anomaly
        recent = graph.get_edges_from(edge.src_id, window_seconds=300)
        if len(recent) >= 3:
            signals.append(FraudSignal(
                icon="🔴",
                title="Velocity Burst",
                detail=f"{len(recent)} transactions in last 5 minutes (typical: 0-1)",
                contribution=0.4,
            ))
        
        # Check amount anomaly
        avg_amount = graph.get_average_amount(edge.src_id)
        this_amount = np.exp(edge.features[0]) - 1
        if avg_amount > 0 and this_amount > 5 * avg_amount:
            signals.append(FraudSignal(
                icon="🔴",
                title="Amount Spike",
                detail=f"£{this_amount:.0f} is {this_amount/avg_amount:.1f}x above account average",
                contribution=0.3,
            ))
        
        # Check new merchant
        prior_txns = graph.get_edges_from_to(edge.src_id, edge.dst_id)
        if len(prior_txns) == 0:
            signals.append(FraudSignal(
                icon="🟡",
                title="New Merchant",
                detail="First ever transaction with this merchant",
                contribution=0.15,
            ))
        
        # Check time-of-day anomaly
        hour = (edge.timestamp % 86400) / 3600
        if hour < 6 or hour > 23:
            signals.append(FraudSignal(
                icon="🟡",
                title="Unusual Time",
                detail=f"Transaction at {hour:.0f}:00 (outside normal hours)",
                contribution=0.15,
            ))
        
        return sorted(signals, key=lambda s: -s.contribution)
```

---

### 3.3 New Page: Fraud Pattern Visualiser (Page 6)

Add a new page dedicated to showing the fraud pattern that was detected, as a graph visualisation with animated highlighting.

**Page concept:**
```
Page 6 — Fraud Pattern Visualiser

[ Select Pattern: Card Testing ▼ ]  [ Show Step by Step ]

Step 1/4: Normal transactions (grey)
[Network graph of 20 nodes, grey edges]

Step 2/4: Card testing begins
[3 small edges highlighted red → micro-transactions fan out from account 7]

Step 3/4: TGN memory updates — anomaly score rises
[Node 7 pulses red, memory vector shown as bar chart updating]

Step 4/4: Large transaction — BLOCKED
[Final red edge from account 7 to merchant 42, CRITICAL banner]

"Why TGN caught this and a simple model wouldn't:"
→ "A logistic regression model would score each transaction independently.
   Transaction #1 (£0.50 to merchant A) looks completely normal.
   TGN's memory module tracked that 8 micro-transactions happened in 4 minutes,
   making transaction #9 (£2,400) instantly recognisable as card testing."
```

**Implementation**: Use `plotly.graph_objects` with `networkx` layout for the graph visualisation. Animate by adding edge traces incrementally using Streamlit's re-run mechanism.

**File**: `app/pages/6_Pattern_Visualiser.py`

---

### 3.4 New Page: TGN vs. Baseline Comparison (Page 7)

A three-part comparison page. Three columns: Logistic Regression (no graph),
PRAGMA-style Sequence Model (Revolut, arXiv:2604.08649), TGN Ensemble (this system).

```
Page 7 — Why TGN?

┌──────────────────┬─────────────────────┬───────────────────────┐
│ Logistic         │ Sequence Model      │ TGN Ensemble          │
│ Regression       │ (PRAGMA-style,      │ (this system)         │
│                  │  Revolut 2026)      │                       │
├──────────────────┼─────────────────────┼───────────────────────┤
│ AUC-PR: ~0.41   │ AUC-PR: ~0.71       │ AUC-PR: ~0.84         │
│ F1:     ~0.48   │ F1:     ~0.67       │ F1:     ~0.79         │
├──────────────────┼─────────────────────┼───────────────────────┤
│ Card Testing: ❌ │ Card Testing: ✅     │ Card Testing: ✅       │
│ Money Mule:   ❌ │ Money Mule:   ⚠️*  │ Money Mule:   ✅       │
│ Bust-Out:     ⚠️│ Bust-Out:     ✅     │ Bust-Out:     ✅       │
│ Account TKO:  ❌ │ Account TKO:  ✅    │ Account TKO:  ✅       │
└──────────────────┴─────────────────────┴───────────────────────┘

* Revolut's own PRAGMA paper (Section 3.4.5) states their sequence model
  performs BELOW task-specific baseline on AML due to relational complexity.
```

**File**: `app/pages/7_Why_TGN.py`

**Column implementations:**

*Column 1 — Logistic Regression:*
`sklearn.LogisticRegression(class_weight='balanced')` on raw edge features only.
No graph. Represents typical bank baseline.

*Column 2 — PRAGMA-style Sequence Model:*
2-layer PyTorch Transformer on last 50 transactions per account, using
`PRAGMATimeEncoder` from `tgn_learn/model/time_encoder.py` (already implemented).
No graph structure — purely sequential. Label as "Sequence Model (PRAGMA-style, Revolut 2026)".

*Column 3 — TGN Ensemble* from `session_state["trained_model"]`.

**PRAGMA limitation panel to add below the comparison table:**

```python
PRAGMA_LIMITATION_QUOTE = """
**Why Revolut's PRAGMA sequence model struggles with money laundering:**

From the PRAGMA paper (arXiv:2604.08649, Revolut Research, April 2026),
Section 3.4.5 — "Limitations in Highly Relational Tasks: Anti-Money Laundering":

> "AML remains a challenging task... The highly relational nature of money
> laundering — where signals span multiple accounts and multi-hop paths —
> limits the effectiveness of per-user sequence models."

PRAGMA performs **below their own task-specific baseline** on AML.

**Why TGN succeeds here:** TGN models the full transaction graph. The
laundering chain A → B → C → D is an explicit path that TGN's memory
propagation traverses. A sequence model sees A, B, C, D as separate
users with no connection between them.

💡 **PRAGMA + TGN are complementary:**
   Use PRAGMA for individual behavioural fraud (card testing, bust-out).
   Use TGN for network fraud (AML, synthetic identity rings).
   The Ensemble combines both via the LightGBM meta-learner.
"""
```

**Per-fraud-type breakdown:**
Use `Edge.metadata["pattern_type"]` (already stored in BankSim) to compute
AUC-PR separately for each fraud pattern and show as a grouped bar chart.

---

### 3.5 Ensemble View in Scoring Page

When the ensemble is implemented (Phase 4 of `ENSEMBLE_DESIGN.md`), the scoring page should show a **detector breakdown panel**:

```
┌─────────────────────────────────────────────────────┐
│  ENSEMBLE DETECTOR BREAKDOWN                        │
│                                                     │
│  TGN Memory Detector     ████████░░  0.84  HIGH     │
│  RF Structural           ██████░░░░  0.61  MED      │
│  Fund-Flow Graph         █████████░  0.91  CRITICAL │
│  Semantic Patterns       ████░░░░░░  0.41  LOW      │
│  Drift Monitor           ███░░░░░░░  0.30  LOW      │
│                                                     │
│  → Fund-flow topology most suspicious               │
│  → Consistent with: money mule chain (3-hop)        │
└─────────────────────────────────────────────────────┘
```

---

### 3.6 AWS Architecture Page (Page 8)

One page showing where each component runs in AWS, so the audience can visualise the production path.

```
Page 8 — AWS Deployment Architecture
[Static HTML/Plotly diagram showing:]

Data Stream → Kinesis → Lambda (graph construction)
                              ↓
                         ElastiCache (node memory)
                              ↓
                         SageMaker RT Inference (TGN GPU)
                              ↓
                         Decision API → Core Banking
                              
Offline:
  S3 (graph storage) → SageMaker Training → SageMaker Model Registry
  CloudWatch (drift monitoring) → Lambda (retrain trigger)
  
"This Streamlit app is the local development equivalent of this architecture.
  Each page corresponds to a component in the production system."
```

**File**: `app/pages/8_AWS_Architecture.py` — static HTML rendered inside Streamlit via `st.components.v1.html()`

---

### 3.7 Main App Landing Page Update

Update `app/main.py` to:
1. Add a title card: **"TGN Fraud Detection — Temporal Graph Networks for Financial Crime"**
2. Add a 3-step "how it works" diagram visible on the landing page
3. Add sidebar navigation with section labels:
   - **Setup**: Generate Data, Explore Graph
   - **Model**: Train Model, Why TGN?
   - **Inference**: Score Transactions, Pattern Visualiser
   - **Deploy**: Upload CSV, AWS Architecture

---

## 4. BankSim Generator Enhancements

The existing 5 fraud patterns need richer narrative metadata for the demo — specifically, *pattern names and descriptions* that get surfaced in the explanation UI.

**Add to `tgn_learn/generators/banksim.py`:**

```python
# Pattern metadata for explanation UI
PATTERN_METADATA = {
    "card_testing": {
        "display_name": "Card Testing Ring",
        "description": "Stolen card validated with micro-transactions before large purchase",
        "typical_signals": ["velocity_burst", "amount_escalation", "new_merchant"],
        "detection_window": "5 minutes",
        "reference_paper": "Wu & Zhang (BDAIE 2025) — Event-centric graph detects fund-flow chain",
    },
    "money_laundering": {
        "display_name": "Money Laundering",
        "description": "Funds layered through intermediary accounts to obscure origin",
        "typical_signals": ["chain_topology", "round_amount", "rapid_forwarding"],
        "detection_window": "24–48 hours",
        "reference_paper": "Salda&ntilde;a-Ulloa et al. (Algorithms 2024) — Multi-graph fusion detects layering",
    },
    "bust_out": {
        "display_name": "Bust-Out Fraud",
        "description": "Account builds legitimate history before sudden maxed-out fraud",
        "typical_signals": ["baseline_deviation", "account_age", "limit_exhaustion"],
        "detection_window": "3–6 months",
        "reference_paper": "DySA-TGN (DASFAA 2025) — Dual-track memory separates baseline from deviation",
    },
    "account_takeover": {
        "display_name": "Account Takeover",
        "description": "Compromised credentials lead to rapid high-value transactions",
        "typical_signals": ["location_shift", "device_change", "velocity_burst"],
        "detection_window": "Minutes",
        "reference_paper": "TFLAG (arXiv 2025) — Self-supervised deviation network flags novel behaviour",
    },
    "synthetic_identity": {
        "display_name": "Synthetic Identity Fraud",
        "description": "Fabricated identity makes immediate large purchases",
        "typical_signals": ["new_account", "large_first_transaction", "no_history"],
        "detection_window": "Account creation",
        "reference_paper": "AnomalyGFM (KDD 2025) — Zero-shot detection via neighbourhood residuals",
    },
}
```

---

## 5. Demo Script (Presenter Guide)

Add this as `docs/DEMO_SCRIPT.md` (Kiro should also create this file):

### Opening (2 min)
> "This is a Temporal Graph Network for payment fraud detection. Unlike traditional ML models that score each transaction in isolation, TGN understands the *network* — who each account has transacted with, how recently, and in what patterns. Let me show you a card testing attack."

### Step 1: Generate Data (1 min)
- Click Demo Mode → "Card Testing Ring"
- Show the stats: 2,000 transactions, 4% fraud
- "We've injected a card testing attack — account 7 is about to test a stolen card"

### Step 2: Load Pre-trained Model (30 sec)
- Click "Load Pre-trained Demo Model"
- "Training takes about 20 seconds for this size — or I can show you live if you prefer"

### Step 3: Score a Transaction (2 min)
- Go to Score Transactions
- Score account 7's large transaction
- Point to the explanation panel: "CRITICAL — here's why: 8 micro-transactions in 4 minutes, first visit to this merchant, amount is 12× the account average"
- "A logistic regression would have scored each of those 8 transactions as LOW RISK — they're all £0.50 or less. TGN's memory module flagged the *pattern*."

### Step 4: Pattern Visualiser (2 min)
- Go to Pattern Visualiser
- Step through card testing animation
- "Watch the memory vector on account 7 update with each transaction — by step 8 the model knows something is wrong before the large transaction even arrives"

### Step 5: Why TGN (1 min)
- Go to Why TGN page
- Show the three-column table (Logistic Regression → PRAGMA-style → TGN Ensemble)
- "AUC-PR 0.84 vs 0.41 for logistic regression — that's expected. But look at the middle column."
- "This is a sequence foundation model — the same architecture Revolut published as PRAGMA
   in April 2026. It's pre-trained on user transaction history. It catches card testing and
   bust-out really well. But see the money mule row? ⚠️"
- Click to expand the PRAGMA limitation panel:
  "This is Revolut's own words from their paper: *'AML remains challenging — the highly
   relational nature of money laundering limits per-user sequence models.'* They score
   below their own baseline on AML."
- "TGN catches it because it sees the network, not just the individual. The laundering
   chain A → B → C → D is a graph path — you need a graph model to see it."
- "PRAGMA and TGN aren't competing. They're complementary: PRAGMA for individual
   behavioural fraud, TGN for network fraud. Our ensemble uses both."

### AWS (1 min)
- Go to AWS Architecture page
- "In production, this is what the stack looks like on AWS..."

---

## 6. Ensemble Demo Integration

Once `ENSEMBLE_DESIGN.md` Phase 1 and 4 are complete, add the following to the demo flow:

**New Step after scoring:**
- Show the detector breakdown panel
- "The ensemble runs 5 specialised detectors in parallel. Each one was inspired by a different research paper. The fund-flow graph detector — which models transactions as nodes in a chain — flags this as CRITICAL because it can see the complete card-testing chain as an explicit path in the graph."

---

## 7. Implementation Checklist for Kiro

### Immediate (demo-unblocking, 2–3 days)
- [ ] Add `DEMO_SCENARIOS` dict and demo mode toggle to `1_Generate_Data.py`
- [ ] Add "Load Pre-trained Model" button to `3_Train_Model.py`
- [ ] Generate and save `checkpoints/demo_model.pt` (seed=42, 5K txns, 20 epochs)
- [ ] Create `tgn_learn/scoring/explainer.py` with `FraudExplainer` and `FraudSignal`
- [ ] Redesign `4_Score_Transactions.py` with 3-panel layout + explanation panel
- [ ] Create `app/pages/6_Pattern_Visualiser.py` with step-through animation
- [ ] Create `app/pages/7_Why_TGN.py` with baseline comparison
- [ ] Update `app/main.py` with branding + navigation sections
- [ ] Add `PATTERN_METADATA` to `banksim.py`

### Before first external demo
- [ ] Add `app/pages/8_AWS_Architecture.py` with architecture diagram
- [ ] Create `docs/DEMO_SCRIPT.md`
- [ ] Test full flow end-to-end with seed=42 (results must be reproducible)
- [ ] Verify app runs on Python 3.11 + PyTorch CPU (no GPU required for demo)

### After ensemble (Phase 4 of ENSEMBLE_DESIGN.md)
- [ ] Add detector breakdown panel to `4_Score_Transactions.py`
- [ ] Show per-detector scores in the explanation panel
- [ ] Add "Research basis" toggle showing which paper each detector is from

---

## 8. AWS Deployment for Demo

The simplest AWS demo setup:

```
1. Amazon EC2 (t3.xlarge, 4 vCPU, 16GB RAM)
   - Runs: streamlit run app/main.py
   - Pre-loaded with demo_model.pt
   - No GPU needed (inference is fast enough on CPU for demo scale)

2. Security Group: port 8501 open to demo IP range
3. Streamlit config: server.headless=true, server.port=8501

Cost: ~$0.17/hour (t3.xlarge on-demand, eu-west-1)
For a 2-hour demo window: ~$0.35
```

For production-like AWS demo (optional):
```
- SageMaker Studio: run the Streamlit app as a SageMaker app
- Or SageMaker JumpStart: deploy via a custom endpoint + API Gateway
```

---

*Document version 1.0 — June 2026*
*Companion to: ENSEMBLE_DESIGN.md*
