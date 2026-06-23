"""
Streamlit page: Train TGN model with live metrics.
"""

import streamlit as st
import plotly.graph_objects as go

from tgn_learn.model import TGNConfig
from tgn_learn.training import TGNTrainer, TrainingConfig

st.header("Train TGN Model")

if "graph" not in st.session_state:
    st.warning("No graph generated yet. Go to **Generate Data** first.")
    st.stop()

graph = st.session_state["graph"]
st.info(f"Training on graph: {graph.num_nodes} nodes, {graph.num_edges} edges, "
        f"{graph.fraud_rate:.1%} fraud rate")

# --- Config ---
st.subheader("Training Configuration")
col1, col2, col3 = st.columns(3)

with col1:
    epochs = st.slider("Epochs", 5, 100, 20)
    batch_size = st.slider("Batch Size", 50, 500, 200, step=50)

with col2:
    lr = st.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3)
    patience = st.slider("Early Stopping Patience", 3, 20, 10)

with col3:
    memory_dim = st.select_slider("Memory Dim", [32, 64, 128], value=64)
    embedding_dim = st.select_slider("Embedding Dim", [32, 64, 128], value=64)

# --- Train ---
if st.button("Start Training", type="primary"):
    train_config = TrainingConfig(
        epochs=epochs, batch_size=batch_size,
        learning_rate=lr, patience=patience,
    )
    model_config = TGNConfig(
        memory_dim=memory_dim, embedding_dim=embedding_dim,
    )

    trainer = TGNTrainer(train_config, model_config)

    # Progress containers
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_container = st.empty()

    # Live charts
    losses = []
    val_auc_prs = []

    def on_epoch(record):
        epoch_num = record["epoch"]
        losses.append(record["train_loss"])
        val_auc_prs.append(record.get("val_auc_pr", 0))

        progress_bar.progress(epoch_num / epochs)
        status_text.text(
            f"Epoch {epoch_num}/{epochs} | "
            f"Loss={record['train_loss']:.4f} | "
            f"Val AUC-PR={record.get('val_auc_pr', 0):.4f}"
        )

        # Update charts
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=losses, mode="lines+markers", name="Train Loss",
            line=dict(color="steelblue"),
        ))
        fig.add_trace(go.Scatter(
            y=val_auc_prs, mode="lines+markers", name="Val AUC-PR",
            yaxis="y2", line=dict(color="green"),
        ))
        fig.update_layout(
            height=300,
            margin=dict(t=30, b=30),
            yaxis=dict(title="Loss"),
            yaxis2=dict(title="AUC-PR", overlaying="y", side="right", range=[0, 1]),
            legend=dict(x=0.01, y=0.99),
        )
        metrics_container.plotly_chart(fig, use_container_width=True)

    results = trainer.train(graph, verbose=False, callback=on_epoch)

    progress_bar.progress(1.0)
    status_text.text("Training complete!")

    # Store results
    st.session_state["trained_model"] = results["model"]
    st.session_state["train_results"] = results

    # Final results table
    st.subheader("Results")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Best Validation Metrics:**")
        m = results["best_metrics"]
        st.write({
            "AUC-PR": f"{m.auc_pr:.4f}",
            "AUC-ROC": f"{m.auc_roc:.4f}",
            "Precision": f"{m.precision:.4f}",
            "Recall": f"{m.recall:.4f}",
            "F1": f"{m.f1:.4f}",
        })

    with col2:
        st.markdown("**Test Metrics:**")
        m = results["test_metrics"]
        st.write({
            "AUC-PR": f"{m.auc_pr:.4f}",
            "AUC-ROC": f"{m.auc_roc:.4f}",
            "Precision": f"{m.precision:.4f}",
            "Recall": f"{m.recall:.4f}",
            "F1": f"{m.f1:.4f}",
        })

    st.success("Model saved to session. Go to **Score Transactions** to use it.")

# Show existing results if available
elif "train_results" in st.session_state:
    results = st.session_state["train_results"]
    st.subheader("Previous Training Results")

    losses = [r["train_loss"] for r in results["history"]]
    val_aucs = [r.get("val_auc_pr", 0) for r in results["history"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=losses, mode="lines", name="Train Loss"))
    fig.add_trace(go.Scatter(y=val_aucs, mode="lines", name="Val AUC-PR", yaxis="y2"))
    fig.update_layout(
        height=300,
        yaxis=dict(title="Loss"),
        yaxis2=dict(title="AUC-PR", overlaying="y", side="right", range=[0, 1]),
    )
    st.plotly_chart(fig, use_container_width=True)

    m = results["best_metrics"]
    st.write(f"Best Val AUC-PR: {m.auc_pr:.4f} | Test AUC-PR: {results['test_metrics'].auc_pr:.4f}")
