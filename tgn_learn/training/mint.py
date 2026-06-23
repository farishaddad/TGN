"""
MiNT (Multi-Network Training) — Transfer learning across multiple networks.

Trains a shared TGN model across multiple synthetic fraud networks,
demonstrating that temporal fraud patterns generalize. The model is
then evaluated zero-shot on unseen networks and optionally fine-tuned.

Key concepts:
- Gradient accumulation across networks (shared optimizer step)
- Zero-shot transfer to unseen networks
- Fine-tuning: freeze backbone, train classifier on target network

Based on ScalingTGNs/MiNT: Multi-network pre-training across Ethereum
token networks showing 70/15/15 temporal splits and transfer to unseen graphs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader

from tgn_learn.generators import BankSimGenerator, PaySimGenerator
from tgn_learn.generators.base import GeneratorConfig
from tgn_learn.graph import TemporalGraph
from tgn_learn.model import TGNConfig, TGNFraudDetector
from tgn_learn.training.metrics import FraudMetrics
from tgn_learn.training.sampler import NegativeSampler

logger = logging.getLogger(__name__)


@dataclass
class MiNTConfig:
    """Configuration for MiNT multi-network training.

    Attributes:
        num_train_networks: Number of synthetic training networks
        num_test_networks: Number of unseen test networks
        epochs: Training epochs per round
        batch_size: Edges per batch
        learning_rate: Optimizer learning rate
        fine_tune_epochs: Epochs for fine-tuning on target network
        fine_tune_lr: Learning rate for fine-tuning (typically lower)
        transactions_per_network: Edges in each synthetic network
        accounts_per_network: Accounts per network
        fraud_rate: Fraud rate in generated networks
        seed: Base random seed
    """
    num_train_networks: int = 5
    num_test_networks: int = 2
    epochs: int = 10
    batch_size: int = 200
    learning_rate: float = 1e-3
    fine_tune_epochs: int = 5
    fine_tune_lr: float = 1e-4
    transactions_per_network: int = 1000
    accounts_per_network: int = 100
    fraud_rate: float = 0.05
    seed: int = 42


@dataclass
class MiNTResult:
    """Results from MiNT training and evaluation."""
    train_history: list[dict] = field(default_factory=list)
    zero_shot_metrics: dict[str, FraudMetrics] = field(default_factory=dict)
    fine_tuned_metrics: dict[str, FraudMetrics] = field(default_factory=dict)
    single_network_metrics: dict[str, FraudMetrics] = field(default_factory=dict)

    def summary(self) -> str:
        lines = ["=" * 60, "MiNT Multi-Network Training Results", "=" * 60, ""]

        if self.train_history:
            final = self.train_history[-1]
            lines.append(f"Training: {len(self.train_history)} epochs, "
                        f"final loss={final.get('avg_loss', 0):.4f}")
            lines.append("")

        lines.append("Zero-shot transfer to unseen networks:")
        for name, m in self.zero_shot_metrics.items():
            lines.append(f"  {name}: {m}")
        lines.append("")

        if self.fine_tuned_metrics:
            lines.append("After fine-tuning:")
            for name, m in self.fine_tuned_metrics.items():
                lines.append(f"  {name}: {m}")
            lines.append("")

        if self.single_network_metrics:
            lines.append("Single-network baselines (no transfer):")
            for name, m in self.single_network_metrics.items():
                lines.append(f"  {name}: {m}")

        lines.append("=" * 60)
        return "\n".join(lines)


class MiNTTrainer:
    """
    Multi-Network Training (MiNT) for TGN fraud detection.

    Trains a shared TGN across multiple synthetic networks using gradient
    accumulation, then evaluates zero-shot transfer and fine-tuning on
    unseen networks.

    Example:
        >>> trainer = MiNTTrainer(MiNTConfig())
        >>> result = trainer.run()
        >>> print(result.summary())
    """

    def __init__(self, config: MiNTConfig, model_config: Optional[TGNConfig] = None):
        self.config = config
        self.model_config = model_config or TGNConfig()
        self.device = torch.device("cpu")

    def run(self, verbose: bool = True) -> MiNTResult:
        """Execute full MiNT pipeline: generate, train, evaluate, fine-tune.

        Args:
            verbose: Print progress to stdout

        Returns:
            MiNTResult with all metrics
        """
        result = MiNTResult()

        # Step 1: Generate networks
        if verbose:
            print("Step 1: Generating synthetic networks...")

        train_networks = self._generate_networks(
            self.config.num_train_networks, seed_offset=0, prefix="train"
        )
        test_networks = self._generate_networks(
            self.config.num_test_networks, seed_offset=100, prefix="test"
        )

        if verbose:
            print(f"  Training: {len(train_networks)} networks, "
                  f"~{self.config.transactions_per_network} edges each")
            print(f"  Test: {len(test_networks)} networks (unseen)")
            print()

        # Step 2: Multi-network training
        if verbose:
            print("Step 2: Multi-network training...")

        # Determine max nodes across all networks
        all_networks = list(train_networks.values()) + list(test_networks.values())
        max_nodes = max(g.num_nodes for g in all_networks)
        # Pad to account for potential node IDs
        num_nodes = max_nodes + 50

        model = TGNFraudDetector(num_nodes, self.model_config).to(self.device)
        optimizer = Adam(model.parameters(), lr=self.config.learning_rate)
        neg_sampler = NegativeSampler(num_nodes)

        for epoch in range(1, self.config.epochs + 1):
            epoch_losses = []

            for name, graph in train_networks.items():
                loss = self._train_on_network(model, optimizer, neg_sampler, graph)
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses)
            result.train_history.append({"epoch": epoch, "avg_loss": float(avg_loss)})

            if verbose:
                print(f"  Epoch {epoch}/{self.config.epochs} | Avg Loss={avg_loss:.4f}")

        if verbose:
            print()

        # Step 3: Zero-shot evaluation on unseen networks
        if verbose:
            print("Step 3: Zero-shot evaluation on unseen networks...")

        for name, graph in test_networks.items():
            metrics = self._evaluate_on_network(model, graph)
            result.zero_shot_metrics[name] = metrics
            if verbose:
                print(f"  {name}: {metrics}")

        if verbose:
            print()

        # Step 4: Fine-tune on target network
        if verbose:
            print("Step 4: Fine-tuning on target networks...")

        for name, graph in test_networks.items():
            fine_tuned_model = self._fine_tune(model, graph)
            metrics = self._evaluate_on_network(fine_tuned_model, graph)
            result.fine_tuned_metrics[name] = metrics
            if verbose:
                print(f"  {name} (fine-tuned): {metrics}")

        if verbose:
            print()

        # Step 5: Single-network baselines
        if verbose:
            print("Step 5: Single-network baselines (for comparison)...")

        for name, graph in test_networks.items():
            metrics = self._train_single_network(graph)
            result.single_network_metrics[name] = metrics
            if verbose:
                print(f"  {name} (trained from scratch): {metrics}")

        if verbose:
            print()
            print(result.summary())

        return result

    def _generate_networks(
        self, count: int, seed_offset: int, prefix: str
    ) -> dict[str, TemporalGraph]:
        """Generate multiple synthetic networks with different seeds."""
        networks = {}
        generators = [BankSimGenerator, PaySimGenerator]

        for i in range(count):
            seed = self.config.seed + seed_offset + i
            gen_cls = generators[i % len(generators)]
            config = GeneratorConfig(
                num_accounts=self.config.accounts_per_network,
                num_merchants=20,
                num_transactions=self.config.transactions_per_network,
                fraud_rate=self.config.fraud_rate,
                seed=seed,
            )
            gen = gen_cls(config)
            graph = gen.generate()
            networks[f"{prefix}_{i}_{gen.name}"] = graph

        return networks

    def _train_on_network(
        self,
        model: TGNFraudDetector,
        optimizer: Adam,
        neg_sampler: NegativeSampler,
        graph: TemporalGraph,
    ) -> float:
        """Train one pass on a single network (gradient accumulation step)."""
        model.train()
        model.reset_memory()
        model.neighbor_loader.reset()

        # Use 70% of edges for training pass
        train_graph, _, _ = graph.temporal_split(0.70, 0.15)
        data = train_graph.to_pyg_temporal_data()

        if data.src.size(0) == 0:
            return 0.0

        loader = TemporalDataLoader(data, batch_size=self.config.batch_size)
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            src = batch.src.to(self.device)
            dst = batch.dst.to(self.device)
            t = batch.t.to(self.device)
            msg = batch.msg.to(self.device)
            labels = batch.y.to(self.device)
            neg_dst = neg_sampler.sample(src, dst)

            optimizer.zero_grad()

            pos_score, neg_score, node_scores = model(src, dst, t, msg, neg_dst)

            # Combined loss
            link_loss = F.binary_cross_entropy_with_logits(
                pos_score.squeeze(), torch.ones(src.size(0), device=self.device)
            ) + F.binary_cross_entropy_with_logits(
                neg_score.squeeze(), torch.zeros(src.size(0), device=self.device)
            )

            node_loss = torch.tensor(0.0, device=self.device)
            labelled_mask = labels >= 0
            if labelled_mask.any():
                n_pos = labels[labelled_mask].float().sum().clamp(min=1)
                n_neg = (1 - labels[labelled_mask].float()).sum().clamp(min=1)
                pos_weight = (n_neg / n_pos).clamp(max=50)
                node_loss = F.binary_cross_entropy_with_logits(
                    node_scores[labelled_mask].squeeze(),
                    labels[labelled_mask].float(),
                    pos_weight=pos_weight.expand_as(labels[labelled_mask].float()),
                )

            loss = 0.5 * link_loss + 0.5 * node_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            model.neighbor_loader.insert(src, dst)
            model.detach_memory()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _evaluate_on_network(
        self, model: TGNFraudDetector, graph: TemporalGraph
    ) -> FraudMetrics:
        """Evaluate model on the test split of a network."""
        model.eval()
        model.reset_memory()
        model.neighbor_loader.reset()

        # Use last 15% as test
        _, _, test_graph = graph.temporal_split(0.70, 0.15)
        # But first warm up memory with train+val data
        train_graph, val_graph, _ = graph.temporal_split(0.70, 0.15)

        # Warm up
        for g in [train_graph, val_graph]:
            data = g.to_pyg_temporal_data()
            if data.src.size(0) > 0:
                loader = TemporalDataLoader(data, batch_size=self.config.batch_size)
                for batch in loader:
                    src = batch.src.to(self.device)
                    dst = batch.dst.to(self.device)
                    t = batch.t.to(self.device)
                    msg = batch.msg.to(self.device)
                    model(src, dst, t, msg)
                    model.neighbor_loader.insert(src, dst)
                    model.detach_memory()

        # Evaluate on test
        test_data = test_graph.to_pyg_temporal_data()
        if test_data.src.size(0) == 0:
            return FraudMetrics()

        loader = TemporalDataLoader(test_data, batch_size=self.config.batch_size)
        all_scores, all_labels = [], []

        for batch in loader:
            src = batch.src.to(self.device)
            dst = batch.dst.to(self.device)
            t = batch.t.to(self.device)
            msg = batch.msg.to(self.device)
            labels = batch.y.to(self.device)

            _, _, node_scores = model(src, dst, t, msg)
            all_scores.append(node_scores.sigmoid().cpu().numpy().flatten())
            all_labels.append(labels.cpu().numpy().flatten())

            model.neighbor_loader.insert(src, dst)
            model.detach_memory()

        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        return FraudMetrics.compute(scores, labels)

    def _fine_tune(
        self, base_model: TGNFraudDetector, target_graph: TemporalGraph
    ) -> TGNFraudDetector:
        """Fine-tune: freeze backbone (memory+GNN), train only scoring heads."""
        import copy
        model = copy.deepcopy(base_model).to(self.device)

        # Freeze memory and GNN parameters
        for param in model.memory.parameters():
            param.requires_grad = False
        for param in model.gnn.parameters():
            param.requires_grad = False

        # Only optimize scoring heads
        head_params = list(model.link_pred.parameters()) + list(model.node_pred.parameters())
        optimizer = Adam(head_params, lr=self.config.fine_tune_lr)
        neg_sampler = NegativeSampler(model.num_nodes)

        # Train on target network's training split
        train_graph, _, _ = target_graph.temporal_split(0.70, 0.15)
        data = train_graph.to_pyg_temporal_data()

        if data.src.size(0) == 0:
            return model

        loader = TemporalDataLoader(data, batch_size=self.config.batch_size)

        for epoch in range(self.config.fine_tune_epochs):
            model.train()
            model.reset_memory()
            model.neighbor_loader.reset()

            for batch in loader:
                src = batch.src.to(self.device)
                dst = batch.dst.to(self.device)
                t = batch.t.to(self.device)
                msg = batch.msg.to(self.device)
                labels = batch.y.to(self.device)
                neg_dst = neg_sampler.sample(src, dst)

                optimizer.zero_grad()
                pos_score, neg_score, node_scores = model(src, dst, t, msg, neg_dst)

                link_loss = F.binary_cross_entropy_with_logits(
                    pos_score.squeeze(), torch.ones(src.size(0), device=self.device)
                ) + F.binary_cross_entropy_with_logits(
                    neg_score.squeeze(), torch.zeros(src.size(0), device=self.device)
                )

                node_loss = torch.tensor(0.0, device=self.device)
                labelled_mask = labels >= 0
                if labelled_mask.any():
                    n_pos = labels[labelled_mask].float().sum().clamp(min=1)
                    n_neg = (1 - labels[labelled_mask].float()).sum().clamp(min=1)
                    pos_weight = (n_neg / n_pos).clamp(max=50)
                    node_loss = F.binary_cross_entropy_with_logits(
                        node_scores[labelled_mask].squeeze(),
                        labels[labelled_mask].float(),
                        pos_weight=pos_weight.expand_as(labels[labelled_mask].float()),
                    )

                loss = 0.5 * link_loss + 0.5 * node_loss
                loss.backward()
                optimizer.step()

                model.neighbor_loader.insert(src, dst)
                model.detach_memory()

        return model

    def _train_single_network(self, graph: TemporalGraph) -> FraudMetrics:
        """Train a fresh model on a single network as baseline."""
        num_nodes = graph.num_nodes + 50
        model = TGNFraudDetector(num_nodes, self.model_config).to(self.device)
        optimizer = Adam(model.parameters(), lr=self.config.learning_rate)
        neg_sampler = NegativeSampler(num_nodes)

        # Train for same number of epochs
        for epoch in range(self.config.epochs):
            self._train_on_network(model, optimizer, neg_sampler, graph)

        return self._evaluate_on_network(model, graph)
