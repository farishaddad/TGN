"""
Last-neighbor loader for temporal subgraph construction.

During TGN inference, we need to look up the most recent neighbors
for each node to construct a local subgraph for message passing.
This module maintains a circular buffer of recent neighbors per node.
"""

import torch


class LastNeighborLoader:
    """
    Stores the K most recent neighbors for each node in a circular buffer.

    Used during inference to efficiently construct temporal subgraphs
    without scanning the full edge list.

    Args:
        num_nodes: Total number of nodes in the graph
        size: Number of recent neighbors to store per node
        device: Torch device
    """

    def __init__(self, num_nodes: int, size: int, device=None):
        self.size = size
        self.device = device or torch.device("cpu")
        self.neighbors = torch.empty(
            (num_nodes, size), dtype=torch.long, device=self.device
        ).fill_(-1)
        self.edge_ids = torch.empty(
            (num_nodes, size), dtype=torch.long, device=self.device
        ).fill_(-1)
        self._assoc = torch.zeros(num_nodes, dtype=torch.long, device=self.device)
        self.cur_e_id = 0

    def insert(self, src: torch.Tensor, dst: torch.Tensor):
        """Insert new edges into the neighbor buffer.

        For each edge (src, dst), we store dst as a neighbor of src
        AND src as a neighbor of dst (undirected view).

        Args:
            src: Source node IDs [batch]
            dst: Destination node IDs [batch]
        """
        neighbors = torch.cat([src, dst], dim=0)
        nodes = torch.cat([dst, src], dim=0)
        edge_ids = torch.arange(
            self.cur_e_id,
            self.cur_e_id + src.size(0),
            device=self.device,
        ).repeat(2)
        self.cur_e_id += src.numel()

        for node, neighbor, e_id in zip(
            nodes.tolist(), neighbors.tolist(), edge_ids.tolist()
        ):
            if node < self.neighbors.size(0):
                idx = int(self._assoc[node].item()) % self.size
                self.neighbors[node, idx] = neighbor
                self.edge_ids[node, idx] = e_id
                self._assoc[node] += 1

    def __call__(self, n_id: torch.Tensor):
        """Get neighbors and edge IDs for given nodes.

        Args:
            n_id: Node IDs to query [num_query]

        Returns:
            (neighbor_ids, edge_ids) — flattened valid neighbors
        """
        neighbors = self.neighbors[n_id]
        edge_ids = self.edge_ids[n_id]
        mask = neighbors >= 0
        return neighbors[mask], edge_ids[mask]

    def reset(self):
        """Clear all stored neighbors."""
        self.neighbors.fill_(-1)
        self.edge_ids.fill_(-1)
        self._assoc.zero_()
        self.cur_e_id = 0
