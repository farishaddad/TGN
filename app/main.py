"""
TGN Fraud Detection — main entry point.

Uses st.navigation() (Streamlit ≥ 1.36) to register both standard and
ensemble pages so Streamlit Cloud can route to them correctly.

Run with:
    streamlit run app/main.py
"""

import streamlit as st

st.set_page_config(
    page_title="TGN Fraud Detection — Temporal Graph Networks",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Page registry — both sections always defined, shown as sidebar sections
# ---------------------------------------------------------------------------

standard_pages = [
    st.Page("pages/1_Generate_Data.py",    title="Generate Data",       icon="📊"),
    st.Page("pages/2_Explore_Graph.py",    title="Explore Graph",        icon="🔍"),
    st.Page("pages/3_Train_Model.py",      title="Train Model",          icon="🧠"),
    st.Page("pages/4_Score_Transactions.py", title="Score Transactions", icon="🎯"),
    st.Page("pages/5_Upload_CSV.py",       title="Upload CSV",           icon="📤"),
]

ensemble_pages = [
    st.Page("ensemble_pages/1_Generate_Data.py",      title="Generate Data",        icon="📊"),
    st.Page("ensemble_pages/4_Score_Transactions.py", title="Score Transactions",   icon="🎯"),
    st.Page("ensemble_pages/6_Pattern_Visualiser.py", title="Pattern Visualiser",   icon="🎭"),
]

pg = st.navigation(
    {
        "🔷 Standard TGN": standard_pages,
        "🔶 Ensemble TGN": ensemble_pages,
    }
)

# ---------------------------------------------------------------------------
# Sidebar branding (shown on every page)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        "<p style='color:#888; font-size:0.8em; text-align:center; margin-top:20px;'>"
        "TGN Fraud Detection v0.2<br>"
        "Temporal Graph Networks<br>"
        "for Financial Crime"
        "</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Run whichever page the user navigated to
# ---------------------------------------------------------------------------

pg.run()
