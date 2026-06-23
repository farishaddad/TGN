"""
Streamlit page: Upload CSV for ingestion and scoring.
"""

import streamlit as st
import pandas as pd

from tgn_learn.ingestion import CSVIngester

st.header("Upload CSV Data")

st.markdown("""
Upload your own transaction data as CSV. Required columns:
- `source_id` — Account/entity initiating the transaction
- `target_id` — Recipient account/merchant
- `timestamp` — Unix timestamp or ISO datetime
- `amount` — Transaction amount

Optional: `label` (0=legit, 1=fraud), `channel`, `category`
""")

uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded is not None:
    # Preview
    df = pd.read_csv(uploaded)
    st.subheader("Preview")
    st.dataframe(df.head(10))

    st.write(f"**Rows:** {len(df)} | **Columns:** {list(df.columns)}")

    # Ingest
    if st.button("Ingest Data", type="primary"):
        uploaded.seek(0)
        ingester = CSVIngester()
        try:
            result = ingester.ingest(uploaded)
            st.session_state["graph"] = result.graph
            st.session_state["ingestion_result"] = result

            st.success("Data ingested successfully!")
            st.write(str(result))

            if result.warnings:
                for w in result.warnings:
                    st.warning(w)

            st.info("Graph is now available for training and scoring in other pages.")

        except ValueError as e:
            st.error(f"Ingestion failed: {e}")

# Show sample CSV format
with st.expander("Sample CSV Format"):
    sample = pd.DataFrame({
        "source_id": ["acct_001", "acct_002", "acct_001", "acct_003"],
        "target_id": ["merch_42", "merch_42", "merch_99", "merch_42"],
        "timestamp": [1700000000, 1700001000, 1700002000, 1700003000],
        "amount": [150.00, 5000.00, 25.50, 300.00],
        "label": [0, 1, 0, 0],
    })
    st.dataframe(sample)
    st.download_button(
        "Download Sample CSV",
        sample.to_csv(index=False),
        "sample_transactions.csv",
        "text/csv",
    )
