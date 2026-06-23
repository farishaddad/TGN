"""
TGN Training Engine.

Implements the complete training loop with:
- Combined contrastive (link prediction) + supervised (node classification) loss
- Temporal data splitting (chronological train/val/test)
- Early stopping on validation AUC-PR
- Checkpoint save/load
- Per-epoch metrics tracking

Simplified from Sail-v3's TGNTrainer for educational use.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader

from tgn_learn.graph import TemporalGraph
from tgn_learn.model import TGNConfig, TGNFraudDetector

from .config import TrainingConfig
from .metrics import FraudMetrics
from .sampler import NegativeSampler

logger = logging.getLogger(__name__)


class TGNTrainer:
    """
    Complete training engine for TGN fraud detection.

    Handles model construction, training loop with mixed loss,
    validation with early stopping, and checkpointing.

    Example:
        >>> from tgn_learn.training import TGNTrainer, TrainingConfig
        >>> trainer = TGNTrainer(TrainingConfig(epochs=20))
        >>> results = trainer.train(graph)
        >>> print(results['best_metrics'])
    """

    def __init__(self, config: TrainingConfig, model_config: Optional[TGNConfig] = None):
        self.config = config
        self.model_config = model_config or TGNConfig()

        # Resolve device
        if config.device == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_str = config.device
        self.device = torch.device(device_str)

        # State
        self.model: Optional[TGNFraudDetector] = None
        self.optimizer: Optional[Adam] = None
        self.neg_sampler: Optional[NegativeSampler] = None
        self.history: list[dict] = []
        self.best_val_metric = 0.0
        self.epochs_without_improvement = 0

    def train(
        self,
        graph: TemporalGraph,
        verbose: bool = True,
        callback: Optional[callable] = None,
    ) -> dict:
        """
        Train the TGN model on a temporal graph.

        Args:
            graph: TemporalGraph with labeled edges
            verbose: Print progress to stdout
            callback: Optional function called each epoch with metrics dict

        Returns:
            Dictionary with:
                - 'model': trained TGNFraudDetector
                - 'history': list of per-epoch metrics dicts
                - 'best_metrics': FraudMetrics at best validation epoch
                - 'test_metrics': FraudMetrics on test set (if available)
        """
        # Split graph temporally
        train_graph, val_graph, test_graph = graph.temporal_split(
            self.config.train_ratio, self.config.val_ratio
        )

        # Convert to PyG format
        train_data = train_graph.to_pyg_temporal_data()
        val_data = val_graph.to_pyg_temporal_data()
        test_data = test_graph.to_pyg_temporal_data()

        # Determine number of nodes
        all_nodes = set()
        for data in [train_data, val_data, test_data]:
            if data.src.numel() > 0:
                all_nodes.update(data.src.tolist())
                all_nodes.update(data.dst.tolist())
        num_nodes = max(all_nodes) + 1 if all_nodes else graph.num_nodes

        # Initialize model
        self.model = TGNFraudDetector(num_nodes, self.model_config).to(self.device)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.neg_sampler = NegativeSampler(num_nodes)

        if verbose:
            print(f"Training TGN on {graph.num_edges} edges ({num_nodes} nodes)")
            print(f"  Split: train={train_data.src.size(0)}, "
                  f"val={val_data.src.size(0)}, test={test_data.src.size(0)}")
            print(f"  Device: {self.device}")
            print(f"  Parameters: {self.model.total_parameters:,}")
            print()

        # Create data loaders
        train_loader = TemporalDataLoader(train_data, batch_size=self.config.batch_size)
        val_loader = TemporalDataLoader(val_data, batch_size=self.config.batch_size)

        # Training loop
        self.history = []
        self.best_val_metric = 0.0
        self.epochs_without_improvement = 0
        best_metrics = FraudMetrics()

        for epoch in range(1, self.config.epochs + 1):
            # Train one epoch
            train_metrics = self._train_epoch(train_loader)

            # Evaluate on validation set
            val_metrics = self._evaluate(val_loader, prefix="val")

            # Combine into epoch record
            epoch_record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_link_loss": train_metrics["link_loss"],
                "train_node_loss": train_metrics["node_loss"],
                **{f"val_{k}": v for k, v in val_metrics.to_dict().items()},
            }
            self.history.append(epoch_record)

            # Early stopping check
            if val_metrics.auc_pr > self.best_val_metric:
                self.best_val_metric = val_metrics.auc_pr
                self.epochs_without_improvement = 0
                best_metrics = val_metrics
                self._save_checkpoint("best_model.pt")
            else:
                self.epochs_without_improvement += 1

            if verbose:
                print(
                    f"Epoch {epoch:3d}/{self.config.epochs} | "
                    f"Loss={train_metrics['loss']:.4f} | "
                    f"Val: {val_metrics}"
                )

            if callback:
                callback(epoch_record)

            if self.epochs_without_improvement >= self.config.patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch} "
                          f"(no improvement for {self.config.patience} epochs)")
                break

        # Evaluate on test set
        test_metrics = FraudMetrics()
        if test_data.src.size(0) > 0:
            test_loader = TemporalDataLoader(test_data, batch_size=self.config.batch_size)
            test_metrics = self._evaluate(test_loader, prefix="test")
            if verbose:
                print(f"\nTest: {test_metrics}")

        return {
            "model": self.model,
            "history": self.history,
            "best_metrics": best_metrics,
            "test_metrics": test_metrics,
        }

    def _train_epoch(self, loader: TemporalDataLoader) -> dict:
        """Train one epoch with combined loss."""
        self.model.train()
        self.model.reset_memory()
        self.model.neighbor_loader.reset()

        total_loss = 0.0
        total_link_loss = 0.0
        total_node_loss = 0.0
        num_batches = 0

        alpha = self.config.link_loss_weight
        beta = self.config.node_loss_weight

        for batch in loader:
            src = batch.src.to(self.device)
            dst = batch.dst.to(self.device)
            t = batch.t.to(self.device)
            msg = batch.msg.to(self.device)
            labels = batch.y.to(self.device) if hasattr(batch, "y") else None

            neg_dst = self.neg_sampler.sample(src, dst)

            self.optimizer.zero_grad()

            # Forward pass
            pos_score, neg_score, node_scores = self.model(src, dst, t, msg, neg_dst)

            # --- Link prediction loss (contrastive) ---
            link_loss = F.binary_cross_entropy_with_logits(
                pos_score.squeeze(), torch.ones(src.size(0), device=self.device)
            ) + F.binary_cross_entropy_with_logits(
                neg_score.squeeze(), torch.zeros(src.size(0), device=self.device)
            )

            # --- Node classification loss (supervised, class-weighted) ---
            node_loss = torch.tensor(0.0, device=self.device)
            if labels is not None:
                labelled_mask = labels >= 0
                if labelled_mask.any():
                    node_out = node_scores[labelled_mask]
                    node_labels = labels[labelled_mask].float()

                    # Class weighting for imbalance
                    n_pos = node_labels.sum().clamp(min=1)
                    n_neg = (1 - node_labels).sum().clamp(min=1)
                    pos_weight = (n_neg / n_pos).clamp(max=50)

                    node_loss = F.binary_cross_entropy_with_logits(
                        node_out.squeeze(),
                        node_labels,
                        pos_weight=pos_weight.expand_as(node_labels),
                    )

            # Combined loss
            loss = alpha * link_loss + beta * node_loss
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Update neighbor loader
            self.model.neighbor_loader.insert(src, dst)
            self.model.detach_memory()

            total_loss += loss.item()
            total_link_loss += link_loss.item()
            total_node_loss += node_loss.item()
            num_batches += 1

        return {
            "loss": total_loss / max(num_batches, 1),
            "link_loss": total_link_loss / max(num_batches, 1),
            "node_loss": total_node_loss / max(num_batches, 1),
        }

    @torch.no_grad()
    def _evaluate(self, loader: TemporalDataLoader, prefix: str = "val") -> FraudMetrics:
        """Evaluate on validation or test set."""
        self.model.eval()

        all_scores = []
        all_labels = []

        for batch in loader:
            src = batch.src.to(self.device)
            dst = batch.dst.to(self.device)
            t = batch.t.to(self.device)
            msg = batch.msg.to(self.device)
            labels = batch.y.to(self.device) if hasattr(batch, "y") else None

            _, _, node_scores = self.model(src, dst, t, msg)

            if labels is not None:
                scores_np = node_scores.sigmoid().cpu().numpy().flatten()
                labels_np = labels.cpu().numpy().flatten()
                all_scores.append(scores_np)
                all_labels.append(labels_np)

            # Keep memory updated for temporal consistency
            self.model.neighbor_loader.insert(src, dst)
            self.model.detach_memory()

        if not all_scores:
            return FraudMetrics()

        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)

        return FraudMetrics.compute(scores, labels)

    def _save_checkpoint(self, filename: str) -> str:
        """Save model checkpoint."""
        path = Path(self.config.checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)
        filepath = path / filename

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_config": self.model_config,
            "training_config": self.config,
            "best_val_metric": self.best_val_metric,
            "epoch": len(self.history),
        }, filepath)

        return str(filepath)

    @classmethod
    def load_checkpoint(cls, filepath: str, device: str = "auto") -> TGNFraudDetector:
        """Load a model from checkpoint.

        Args:
            filepath: Path to checkpoint file
            device: Device to load model onto

        Returns:
            Loaded TGNFraudDetector model
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        model_config = checkpoint["model_config"]

        # We need num_nodes from the state dict
        memory_key = "memory.memory"
        num_nodes = checkpoint["model_state_dict"][memory_key].size(0)

        model = TGNFraudDetector(num_nodes, model_config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model
