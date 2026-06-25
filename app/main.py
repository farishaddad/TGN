"""
TGN Fraud Detection — Temporal Graph Networks for Financial Crime.

Main entry point for the Streamlit demo application.
Run with: streamlit run app/main.py
"""

import streamlit as st

st.set_page_config(
    page_title="TGN Fraud Detection — Temporal Graph Networks",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Title Card
# ---------------------------------------------------------------------------

st.markdown(
    "<h1 style='text-align: center; margin-bottom: 0;'>"
    "🛡️ TGN Fraud Detection"
    "</h1>"
    "<p style='text-align: center; font-size: 1.2em; color: #555; margin-top: 5px;'>"
    "Temporal Graph Networks for Financial Crime Detection"
    "</p>",
    unsafe_allow_html=True,
)

st.divider()

# ---------------------------------------------------------------------------
# How It Works — 3-step diagram
# ---------------------------------------------------------------------------

st.markdown("### How It Works")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        "<div style='text-align:center; padding:20px; "
        "background:#f0f7ff; border-radius:10px; min-height:180px;'>"
        "<h2 style='margin:0;'>1️⃣</h2>"
        "<h4>Build the Graph</h4>"
        "<p style='color:#555; font-size:0.9em;'>"
        "Transactions become edges. Accounts and merchants become nodes. "
        "Temporal ordering is preserved."
        "</p></div>",
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        "<div style='text-align:center; padding:20px; "
        "background:#f0fff0; border-radius:10px; min-height:180px;'>"
        "<h2 style='margin:0;'>2️⃣</h2>"
        "<h4>Learn Patterns</h4>"
        "<p style='color:#555; font-size:0.9em;'>"
        "TGN's memory module learns each account's behavioural baseline. "
        "Deviations signal potential fraud."
        "</p></div>",
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        "<div style='text-align:center; padding:20px; "
        "background:#fff5f5; border-radius:10px; min-height:180px;'>"
        "<h2 style='margin:0;'>3️⃣</h2>"
        "<h4>Score & Explain</h4>"
        "<p style='color:#555; font-size:0.9em;'>"
        "Each transaction gets a risk score with human-readable explanations "
        "of why it was flagged."
        "</p></div>",
        unsafe_allow_html=True,
    )

st.markdown("")

# ---------------------------------------------------------------------------
# Navigation Guide (sidebar sections)
# ---------------------------------------------------------------------------

st.markdown("### Pages")

col_setup, col_model, col_inference, col_deploy = st.columns(4)

with col_setup:
    st.markdown("**📊 Setup**")
    st.markdown("- **Generate Data** — Create synthetic fraud networks")
    st.markdown("- **Explore Graph** — Visualise the transaction network")

with col_model:
    st.markdown("**🧠 Model**")
    st.markdown("- **Train Model** — Train TGN with live metrics")
    st.markdown("- **Why TGN?** — Compare against baselines")

with col_inference:
    st.markdown("**🔍 Inference**")
    st.markdown("- **Score Transactions** — Score with explanations")
    st.markdown("- **Pattern Visualiser** — Step-through fraud animations")

with col_deploy:
    st.markdown("**📤 Deploy**")
    st.markdown("- **Upload CSV** — Ingest your own data")

st.divider()

# ---------------------------------------------------------------------------
# Quick Start
# ---------------------------------------------------------------------------

st.markdown("### Quick Start")

st.markdown("""
1. Go to **Generate Data** → toggle **Demo Mode** → select "Card Testing Ring"
2. Go to **Train Model** → click **Load Pre-trained Demo Model**
3. Go to **Score Transactions** → click a preset → see the explanation panel
4. Go to **Pattern Visualiser** → step through the attack animation
""")

# ---------------------------------------------------------------------------
# Sidebar: Mode Selector + Branding
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Mode")
    mode = st.radio(
        "Select demo mode:",
        options=["Standard TGN", "Ensemble TGN"],
        index=0,
        help=(
            "Standard TGN: single model scoring. "
            "Ensemble TGN: multi-detector system with per-model breakdown."
        ),
        key="app_mode",
    )

    if mode == "Ensemble TGN":
        st.info(
            "**Ensemble mode** active. Use the ensemble pages below "
            "for multi-detector scoring and pattern visualisation."
        )
        st.markdown("**Ensemble Pages:**")
        st.markdown("- [Generate Data (Ensemble)](ensemble_pages/1_Generate_Data)")
        st.markdown("- [Score Transactions (Ensemble)](ensemble_pages/4_Score_Transactions)")
        st.markdown("- [Pattern Visualiser (Ensemble)](ensemble_pages/6_Pattern_Visualiser)")

    st.markdown("---")
    st.markdown(
        "<p style='color:#888; font-size:0.8em; text-align:center;'>"
        "TGN Fraud Detection v0.2<br>"
        "Temporal Graph Networks<br>"
        "for Financial Crime"
        "</p>",
        unsafe_allow_html=True,
    )
