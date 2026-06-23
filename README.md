# TGN Learn — Temporal Graph Network Fraud Detection

A hands-on learning app for understanding and applying **Temporal Graph Networks (TGN)** to fraud detection. Built for teams wanting to incorporate TGN into their fraud detection products.

## What You'll Learn

- How temporal graphs represent financial transaction networks
- How TGN memory enables learning from sequential interactions
- How contrastive + supervised loss detects fraud patterns
- How MiNT (Multi-Network Training) enables transfer learning
- How to score transactions and classify risk tiers

## Quick Start (< 5 minutes)

```bash
# Clone and setup
cd ~/Documents/TGN
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Launch the interactive app
streamlit run app/main.py

# Or run progressive learning scripts
python learn/01_data_generation.py
```

## Project Structure

```
TGN/
├── tgn_learn/                 # Core library
│   ├── graph.py               # TemporalGraph, Node, Edge data structures
│   ├── generators/            # Synthetic fraud data generators
│   │   ├── banksim.py         # BankSim: 5 fraud patterns
│   │   ├── paysim.py          # PaySim: mobile money patterns
│   │   └── registry.py        # Generator discovery
│   ├── model/                 # TGN model components
│   │   ├── tgn.py             # TGNFraudDetector (main model)
│   │   ├── embedder.py        # GraphAttentionEmbedding
│   │   ├── time_encoder.py    # Learnable Fourier encoding
│   │   ├── heads.py           # LinkPredictor, NodeClassifier
│   │   └── neighbor_loader.py # Temporal neighbor sampling
│   ├── training/              # Training pipeline
│   │   ├── trainer.py         # TGNTrainer with early stopping
│   │   ├── mint.py            # Multi-Network Training
│   │   └── metrics.py         # FraudMetrics (AUC-PR, F1, etc.)
│   ├── scoring/               # Inference and risk assessment
│   │   └── scorer.py          # Scorer with calibration & risk tiers
│   └── ingestion/             # CSV data ingestion
│       └── csv_ingester.py    # Bring-your-own-data pathway
├── app/                       # Streamlit interactive dashboard
│   ├── main.py                # App entry point
│   └── pages/                 # Multi-page app
│       ├── 1_Generate_Data.py
│       ├── 2_Explore_Graph.py
│       ├── 3_Train_Model.py
│       ├── 4_Score_Transactions.py
│       └── 5_Upload_CSV.py
├── learn/                     # Progressive learning scripts
│   ├── 01_data_generation.py
│   ├── 02_graph_construction.py
│   ├── 03_tgn_architecture.py
│   ├── 04_training_loop.py
│   ├── 05_inference_scoring.py
│   └── 06_mint_transfer.py
├── tests/                     # Unit tests (pytest)
├── pyproject.toml             # Project dependencies
└── Makefile                   # Common commands
```

## Streamlit App Pages

| Page | Description |
|------|-------------|
| Generate Data | Configure and generate synthetic fraud networks |
| Explore Graph | Interactive graph visualization with filtering |
| Train Model | Train TGN with live loss/metric charts |
| Score Transactions | Score individual or batch transactions |
| Upload CSV | Bring your own transaction data |

## Progressive Scripts

Each script is self-contained and heavily documented:

| Script | Topic |
|--------|-------|
| `01_data_generation.py` | Synthetic fraud networks, fraud patterns |
| `02_graph_construction.py` | Node/Edge/TemporalGraph, PyG conversion |
| `03_tgn_architecture.py` | TGN components: memory, attention, heads |
| `04_training_loop.py` | Combined loss, temporal splitting, metrics |
| `05_inference_scoring.py` | Scoring, calibration, risk tiers |
| `06_mint_transfer.py` | Multi-network training, transfer learning |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   TGN Fraud Detector                 │
├─────────────────────────────────────────────────────┤
│  TGNMemory          │  Embedding        │  Heads    │
│  ┌─────────┐        │  ┌────────────┐   │  ┌─────┐ │
│  │ Per-node │ ──────►│  │ Transformer│──►│  │Link │ │
│  │ state    │        │  │ Conv       │   │  │Pred │ │
│  │ vectors  │        │  │ (attention)│   │  ├─────┤ │
│  └────┬────┘        │  └────────────┘   │  │Node │ │
│       │             │        ▲           │  │Cls  │ │
│  ┌────┴────┐        │  ┌────┴────┐      │  └─────┘ │
│  │ Identity │        │  │ Time    │      │          │
│  │ Message  │        │  │ Encoder │      │          │
│  └─────────┘        │  └─────────┘      │          │
├─────────────────────────────────────────────────────┤
│  Input: (src, dst, timestamp, edge_features)        │
│  Output: (link_score, node_score)                   │
└─────────────────────────────────────────────────────┘
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.1
- PyG (torch-geometric) >= 2.4
- Streamlit >= 1.28

## Development

```bash
# Run all tests
make test

# Run the app
make app

# Run learning scripts
make learn

# Lint
make lint
```

## License

MIT
