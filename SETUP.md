# Setup Guide

## Prerequisites

- Python 3.10+ (tested with 3.11, 3.12, 3.14)
- pip (comes with Python)
- ~2GB disk space for dependencies (PyTorch)

## Installation

### macOS / Linux

```bash
cd ~/Documents/TGN

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package with all dependencies
pip install -e ".[dev]"
```

### Windows

```powershell
cd ~\Documents\TGN

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install
pip install -e ".[dev]"
```

## Verify Installation

```bash
# Quick check
python -c "from tgn_learn.graph import TemporalGraph; print('OK')"

# Run tests
pytest tests/ -v

# Generate sample data
python -m tgn_learn.generators --type banksim --transactions 1000
```

## Running the App

```bash
streamlit run app/main.py
```

Opens at http://localhost:8501

## Running Learning Scripts

```bash
python learn/01_data_generation.py
python learn/02_graph_construction.py
python learn/03_tgn_architecture.py
python learn/04_training_loop.py
python learn/05_inference_scoring.py
python learn/06_mint_transfer.py
```

## Common Issues

### PyTorch Installation

If `pip install` fails on torch, install PyTorch first:

```bash
# CPU only (fastest install)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Then install the rest
pip install -e ".[dev]"
```

### torch-geometric

If PyG fails to install, try:

```bash
pip install torch-geometric --find-links https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

### Memory Issues

The default config uses small dimensions (64-dim memory/embedding) suitable
for laptops. If you get OOM errors, reduce further:

```python
from tgn_learn.model import TGNConfig
config = TGNConfig(memory_dim=32, embedding_dim=32, time_dim=16)
```

### Streamlit Port Conflict

If port 8501 is busy:

```bash
streamlit run app/main.py --server.port 8502
```
