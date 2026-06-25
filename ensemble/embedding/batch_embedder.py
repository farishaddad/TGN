"""
Batch Entity Embedder (BRIGHT Lambda architecture, CIKM 2022).

Offline pre-computation of entity embeddings via the full multi-hop TGN.
Results stored in EmbeddingCache, retrieved at inference time.

This is the 'batch layer' of the Lambda architecture:
- Full TGN neighbourhood aggregation
- Multi-scale time encoding
- Buyer subgraph + seller subgraph embeddings

Run hourly (or triggered by drift detection).
At inference, only a lightweight delta is computed on top of cached embeddings.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from tgn_learn.graph import TemporalGraph
from tgn_learn.model import TGNFraudDetector

from .embedding_cache import EmbeddingCache


class BatchEmbedder:
    """Offline batch pre-computation of entity embeddings.

    Decouples expensive multi-hop TGN computation from real-time scoring.

    Args:
        model: Trained TGN model
        cache: EmbeddingCache to store results
        device: Torch device for computation
        batch_size: Nodes per forward pass
    """

    def __init__(
        self,
        model: TGNFraudDetector,
        cache: EmbeddingCache,
        device: str = "cpu",
        batch_size: int = 512,
    ):
        self.model = model
        self.cache = cache
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def run(
        self,
        graph: TemporalGraph,
        node_ids: Optional[list[int]] = None,
    ) -> int:
        """Pre-compute embeddings for all (or specified) nodes.

        Processes the graph chronologically to build up TGN memory,
        then extracts embeddings for each node.

        Args:
            graph: Full temporal graph
            node_ids: Specific nodes to embed (None = all nodes)

        Returns:
            Number of nodes updated in cache
        """
        # Reset model memory for clean computation
        self.model.reset_memory()

        # Process all edges chronologically to build up memory
        edges = graph.edges
        n_edges = len(edges)

        for start in range(0, n_edges, self.batch_size):
            end = min(start + self.batch_size, n_edges)
            batch_edges = edges[start:end]

            src = torch.tensor([e.src_id for e in batch_edges], dtype=torch.long, device=self.device)
            dst = torch.tensor([e.dst_id for e in batch_edges], dtype=torch.long, device=self.device)
            t = torch.tensor([e.timestamp for e in batch_edges], dtype=torch.float, device=self.device)
            msg = torch.tensor(
                np.stack([e.features for e in batch_edges]),
                dtype=torch.float, device=self.device,
            )

            # Forward pass updates memory
            self.model(src, dst, t, msg)
            self.model.detach_memory()

        # Extract embeddings from memory for target nodes
        if node_ids is None:
            node_ids = list(range(graph.num_nodes))

        import time as time_mod
        current_time = time_mod.time()
        nodes_updated = 0

        for start in range(0, len(node_ids), self.batch_size):
            end = min(start + self.batch_size, len(node_ids))
            batch_ids = node_ids[start:end]

            n_id = torch.tensor(batch_ids, dtype=torch.long, device=self.device)
            z, _ = self.model.memory(n_id)
            embeddings = z.cpu().numpy()

            for i, nid in enumerate(batch_ids):
                self.cache.set(nid, embeddings[i], timestamp=current_time)
                nodes_updated += 1

        return nodes_updated

    def run_incremental(
        self,
        graph: TemporalGraph,
        since_timestamp: float,
    ) -> int:
        """Update embeddings only for nodes active since a given timestamp.

        More efficient than full recomputation — only refreshes nodes
        that have had recent interactions.

        Args:
            graph: Full temporal graph
            since_timestamp: Only update nodes active after this time

        Returns:
            Number of nodes updated
        """
        # Find active nodes
        active_nodes = set()
        for edge in graph.edges:
            if edge.timestamp >= since_timestamp:
                active_nodes.add(edge.src_id)
                active_nodes.add(edge.dst_id)

        if not active_nodes:
            return 0

        return self.run(graph, node_ids=sorted(active_nodes))

    def refresh_stale(self, graph: TemporalGraph) -> int:
        """Refresh all stale embeddings in the cache.

        Returns:
            Number of nodes refreshed
        """
        stale_nodes = self.cache.get_stale_nodes()
        if not stale_nodes:
            return 0
        return self.run(graph, node_ids=stale_nodes)
