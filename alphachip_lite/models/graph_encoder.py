"""
AlphaChip-Lite: Graph Encoder
===============================
Encodes the netlist connectivity graph into node embeddings used by the policy.

Architecture:
  - Node features: normalised [width, height]
  - Adjacency: weighted NxN matrix (normalised)
  - Graph convolution: 2-layer graph attention / spectral GCN
  - Output: (N, embed_dim) node embeddings

The embeddings are then aggregated (mean pool) into a single context vector
that conditions the policy on the netlist topology.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Simple symmetric Graph Convolutional layer
# ---------------------------------------------------------------------------

class GCNLayer(nn.Module):
    """
    Symmetric GCN: H' = σ(Â H W)
    where Â = D^{-1/2} (A + I) D^{-1/2}  (normalised adjacency).
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        x:         (N, in_dim)
        adj_norm:  (N, N) normalised adjacency
        """
        # H' = A_norm H W
        out = adj_norm @ x                   # (N, in_dim)
        out = self.linear(out)               # (N, out_dim)
        out = self.bn(out)
        out = F.elu(out)
        return self.dropout(out)


def normalise_adjacency(adj: torch.Tensor, self_loops: bool = True) -> torch.Tensor:
    """
    Compute the symmetrically normalised adjacency matrix:
      Â = D^{-1/2} (A + I) D^{-1/2}
    adj: (N, N) weighted adjacency (unnormalised).
    """
    if self_loops:
        adj = adj + torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
    # Degree
    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    # Normalise: D^{-1/2} A D^{-1/2}
    return deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)


# ---------------------------------------------------------------------------
# Graph Encoder
# ---------------------------------------------------------------------------

class GraphEncoder(nn.Module):
    """
    Encodes the netlist graph into a fixed-size context vector.

    Input:
      - node_features:  (N, node_feat_dim)  — [width, height] (normalised)
      - adj:            (N, N)              — weighted adjacency (unnormalised)

    Output:
      - node_embeds:    (N, embed_dim)      — per-node embeddings
      - graph_embed:    (embed_dim,)        — global mean-pooled embedding
    """

    def __init__(
        self,
        node_feat_dim: int = 2,
        hidden_dim: int = 64,
        embed_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(node_feat_dim, hidden_dim)

        # GCN layers
        gcn_dims = [hidden_dim] + [hidden_dim] * (n_layers - 1) + [embed_dim]
        self.gcn_layers = nn.ModuleList([
            GCNLayer(gcn_dims[i], gcn_dims[i + 1], dropout=dropout)
            for i in range(n_layers)
        ])

        self.embed_dim = embed_dim

    def forward(
        self,
        node_features: torch.Tensor,    # (N, feat_dim)
        adj: torch.Tensor,              # (N, N)
    ) -> tuple:
        adj_norm = normalise_adjacency(adj)

        x = F.elu(self.input_proj(node_features))

        for layer in self.gcn_layers:
            x = layer(x, adj_norm)

        # graph-level: mean pool
        graph_embed = x.mean(dim=0)    # (embed_dim,)
        return x, graph_embed          # (N, embed_dim), (embed_dim,)


# ---------------------------------------------------------------------------
# Positional embedding for partially placed macros
# ---------------------------------------------------------------------------

class PlacementEncoder(nn.Module):
    """
    Encodes the per-macro partial placement (x, y, placed?) into embeddings
    that are concatenated to the GCN node features.
    """

    def __init__(self, in_dim: int = 3, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, partial_placement: torch.Tensor) -> torch.Tensor:
        """
        partial_placement: (N, 3)  — [norm_x, norm_y, placed?]
        Returns:           (N, out_dim)
        """
        return self.net(partial_placement)
