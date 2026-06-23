"""
TGNFraudDetector — Complete TGN model for fraud detection.

Assembles TGNMemory (PyG), GraphAttentionEmbedding, and dual scoring
heads into a single model. This is the core architecture that learns
temporal patterns from transaction graphs.

Architecture:
    1. TGNMemory: Maintains per-node state vectors updated with each interaction
    2. GraphAttentionEmbedding: Aggregates neighbor information via attention
    3. LinkPredictor: Scores individual transactions for anomaly
    4. NodeClassifier: Scores accounts for overall risk

Ported from Sail-v3's TGNFraudDetector with simplified configuration.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator

from .config import TGNConfig
from .embedder import GraphAttentionEmbedding
from .heads import LinkPredictor, NodeClassifier
from .neighbor_loader import LastNeighborLoader


class TGNFraudDetector(nn.Module):
    """
    Complete TGN-based fraud detection model.

    Combines temporal memory, graph attention, and scoring heads
    for both transaction-level and account-level fraud detection.

    Args:
        num_nodes: Total number of nodes in the graph
        config: Model configuration (dimensions, neighbors, etc.)

    Example:
        >>> config = TGNConfig(memory_dim=64, embedding_dim=64)
        >>> model = TGNFraudDetector(num_nodes=1000, config=config)
        >>> # Forward pass with a batch of interactions
        >>> pos_score, neg_score, node_scores = model(src, dst, t, msg, neg_dst)
    """

    def __init__(self, num_nodes: int, config: TGNConfig):
        super().__init__()
        self.config = config
        self.num_nodes = num_nodes

        # TGN Memory: maintains per-node state vectors
        # IdentityMessage just passes raw edge features as messages
        # LastAggregator keeps only the most recent message per node
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=config.edge_feat_dim,
            memory_dim=config.memory_dim,
            time_dim=config.time_dim,
            message_module=IdentityMessage(
                raw_msg_dim=config.edge_feat_dim,
                memory_dim=config.memory_dim,
                time_dim=config.time_dim,
            ),
            aggregator_module=LastAggregator(),
        )

        # Graph attention embedding for contextual representations
        self.gnn = GraphAttentionEmbedding(
            in_channels=config.memory_dim,
            out_channels=config.embedding_dim,
            msg_dim=config.edge_feat_dim,
            time_dim=config.time_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
        )

        # Dual scoring heads
        self.link_pred = LinkPredictor(config.embedding_dim)
        self.node_pred = NodeClassifier(config.embedding_dim)

        # Neighbor loader (set up externally during training)
        self.neighbor_loader = LastNeighborLoader(
            num_nodes=num_nodes,
            size=config.num_neighbors,
        )

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        t: torch.Tensor,
        msg: torch.Tensor,
        neg_dst: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """
        Forward pass for a batch of interactions.

        Args:
            src: Source node IDs [batch]
            dst: Destination node IDs [batch]
            t: Timestamps [batch]
            msg: Edge features [batch, edge_feat_dim]
            neg_dst: Negative destination node IDs [batch] (for contrastive loss)

        Returns:
            pos_score: Anomaly logits for real edges [batch, 1]
            neg_score: Anomaly logits for negative edges [batch, 1] or None
            node_scores: Account risk logits for source nodes [batch, 1]
        """
        # Get memory states for all involved nodes
        n_id = torch.cat([src, dst]).unique()
        z, last_update = self.memory(n_id)

        # Build association map: node_id -> index in z
        assoc = torch.full(
            (n_id.max() + 1,), -1, dtype=torch.long, device=src.device
        )
        assoc[n_id] = torch.arange(n_id.size(0), device=src.device)

        # Look up embeddings for source and destination
        z_src = z[assoc[src]]
        z_dst = z[assoc[dst]]

        # Link-level anomaly scores
        pos_score = self.link_pred(z_src, z_dst)

        # Negative edge scores (for contrastive learning)
        neg_score = None
        if neg_dst is not None:
            n_id_neg = neg_dst.unique()
            z_neg_all, _ = self.memory(n_id_neg)
            assoc_neg = torch.full(
                (n_id_neg.max() + 1,), 0, dtype=torch.long, device=src.device
            )
            assoc_neg[n_id_neg] = torch.arange(n_id_neg.size(0), device=src.device)
            z_neg = z_neg_all[assoc_neg[neg_dst]]
            neg_score = self.link_pred(z_src, z_neg)

        # Node-level risk scores
        node_scores = self.node_pred(z_src)

        # Update memory with this batch of interactions
        # TGNMemory stores last_update as int64, so cast timestamps
        self.memory.update_state(src, dst, t.long(), msg)

        return pos_score, neg_score, node_scores

    def detach_memory(self):
        """Detach memory from computation graph (call between batches)."""
        self.memory.detach()

    def reset_memory(self):
        """Reset all memory states to zero (call at epoch start)."""
        self.memory.reset_state()

    @property
    def total_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
