"""
TGN Fraud Detection Learning App — Main entry point.

Run with: streamlit run app/main.py
"""

import streamlit as st

st.set_page_config(
    page_title="TGN Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("TGN Fraud Detection Learning App")
st.markdown("""
Welcome to the **Temporal Graph Network** fraud detection learning environment.

Use the sidebar to navigate between pages:

- **Generate Data** — Create synthetic fraud networks with configurable parameters
- **Explore Graph** — Visualize the transaction graph interactively
- **Train Model** — Train a TGN model and watch metrics improve
- **Score Transactions** — Score individual transactions for fraud risk
- **Upload CSV** — Bring your own transaction data

---

### Quick Start

1. Go to **Generate Data** and click "Generate" (or use Quick Start)
2. Explore the graph in **Explore Graph**
3. Train a model in **Train Model**
4. Score transactions in **Score Transactions**

### What you'll learn

- How temporal graphs represent financial transaction networks
- How TGN memory enables learning from sequential interactions
- How contrastive + supervised loss detects fraud patterns
- How MiNT enables transfer learning across networks
""")
