"""
AlphaChip-Lite: Policy and Value Network
==========================================
Actor-Critic network for the PPO algorithm.

Architecture:
  ┌──────────────────────────────────────────────────┐
  │  Grid encoder (CNN)                              │
  │    input: (1, rows, cols) occupancy grid         │
  │    output: grid_embed (grid_embed_dim,)          │
  ├──────────────────────────────────────────────────┤
  │  Graph encoder (GCN)                             │
  │    input: node_features (N, 2), adj (N, N)       │
  │    output: graph_embed (embed_dim,)              │
  ├──────────────────────────────────────────────────┤
  │  Macro feature embedding (MLP)                   │
  │    input: macro_feat (3,)                        │
  │    output: macro_embed (macro_embed_dim,)        │
  ├──────────────────────────────────────────────────┤
  │  Combined context                                │
  │    concat → MLP → context (hidden_dim,)          │
  ├──────────────────────────────────────────────────┤
  │  Policy head                                     │
  │    context → logits (cols * rows)                │
  │    masked softmax → action probs                 │
  ├──────────────────────────────────────────────────┤
  │  Value head                                      │
  │    context → scalar value                        │
  └──────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from alphachip_lite.models.graph_encoder import GraphEncoder, PlacementEncoder


# ---------------------------------------------------------------------------
# CNN Grid Encoder
# ---------------------------------------------------------------------------

class GridEncoder(nn.Module):
    """
    Encodes the (1, rows, cols) occupancy grid into a compact feature vector.
    Uses a small CNN with residual connections.
    """

    def __init__(self, rows: int, cols: int, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Compute flatten size
        dummy = torch.zeros(1, 1, rows, cols)
        flat = self.conv(dummy).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """grid: (B, 1, rows, cols) → (B, out_dim)"""
        x = self.conv(grid)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Actor-Critic Policy Network
# ---------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    """
    Combined Actor-Critic network for PPO-based macro placement.

    Supports action masking to prevent placing macros in invalid cells.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        num_macros: int,
        grid_embed_dim:  int = 128,
        graph_embed_dim: int = 128,
        macro_embed_dim: int = 64,
        hidden_dim:      int = 256,
        gcn_hidden_dim:  int = 64,
        n_gcn_layers:    int = 2,
        dropout:         float = 0.1,
    ):
        super().__init__()

        self.rows = rows
        self.cols = cols
        self.n_actions = rows * cols
        self.num_macros = num_macros

        # --- Encoders ---
        self.grid_encoder = GridEncoder(rows, cols, out_dim=grid_embed_dim)

        self.graph_encoder = GraphEncoder(
            node_feat_dim=2,
            hidden_dim=gcn_hidden_dim,
            embed_dim=graph_embed_dim,
            n_layers=n_gcn_layers,
            dropout=dropout,
        )

        self.placement_encoder = PlacementEncoder(in_dim=3, out_dim=32)
        self.placement_pool = nn.Linear(32, graph_embed_dim)  # pool N → 1

        self.macro_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, macro_embed_dim),
            nn.ReLU(),
        )

        # --- Context fusion ---
        fused_dim = grid_embed_dim + graph_embed_dim + macro_embed_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # --- Actor head ---
        self.actor_head = nn.Linear(hidden_dim, self.n_actions)

        # --- Critic head ---
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(
        self,
        grid: torch.Tensor,                       # (B, 1, rows, cols)
        macro_feat: torch.Tensor,                 # (B, 3)
        partial_placement: torch.Tensor,          # (B, N, 3)
        node_features: torch.Tensor,              # (N, 2)   — shared across batch
        adj: torch.Tensor,                        # (N, N)   — shared
        action_mask: Optional[torch.Tensor] = None,  # (B, n_actions) bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          action_probs:  (B, n_actions) normalised over valid actions
          log_probs:     (B, n_actions) log of action_probs
          values:        (B,)           critic estimates
        """
        B = grid.shape[0]
        device = grid.device

        # 1. Grid embedding
        grid_emb = self.grid_encoder(grid)                         # (B, grid_dim)

        # 2. Graph embedding (shared for all batch elements — same netlist)
        _, graph_emb = self.graph_encoder(node_features, adj)     # (graph_dim,)
        graph_emb = graph_emb.unsqueeze(0).expand(B, -1)          # (B, graph_dim)

        # Incorporate partial placement into graph context
        pp_emb = self.placement_encoder(partial_placement)         # (B, N, 32)
        pp_ctx = self.placement_pool(pp_emb.mean(dim=1))          # (B, graph_dim)
        graph_emb = graph_emb + pp_ctx

        # 3. Current macro embedding
        macro_emb = self.macro_mlp(macro_feat)                    # (B, macro_dim)

        # 4. Fuse
        context = torch.cat([grid_emb, graph_emb, macro_emb], dim=-1)  # (B, fused)
        context = self.fusion(context)                             # (B, hidden)

        # 5. Actor
        logits = self.actor_head(context)                         # (B, n_actions)
        if action_mask is not None:
            # Mask invalid actions with a large negative value
            logits = logits.masked_fill(~action_mask, -1e9)

        action_probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # 6. Critic
        values = self.critic_head(context).squeeze(-1)            # (B,)

        return action_probs, log_probs, values

    def get_action(
        self,
        grid: torch.Tensor,
        macro_feat: torch.Tensor,
        partial_placement: torch.Tensor,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (or greedily select) an action.
        Returns: actions (B,), log_prob_of_action (B,), values (B,)
        """
        probs, log_probs, values = self.forward(
            grid, macro_feat, partial_placement, node_features, adj, action_mask
        )

        if deterministic:
            actions = probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs=probs)
            actions = dist.sample()

        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        return actions, action_log_probs, values


# ---------------------------------------------------------------------------
# Convenience: collect observation tensors from a FloorplanState
# ---------------------------------------------------------------------------

def obs_from_state(state, node_feat_tensor, adj_tensor, device):
    """
    Convert a FloorplanState to batch-1 tensors suitable for PolicyNetwork.
    Used during rollout collection.
    """
    obs = state.to_tensor_obs()
    grid = obs["grid"].unsqueeze(0).to(device)                     # (1, 1, R, C)
    macro_feat = obs["macro_feat"].unsqueeze(0).to(device)         # (1, 3)
    pp = obs["partial_placement"].unsqueeze(0).to(device)          # (1, N, 3)
    return grid, macro_feat, pp, node_feat_tensor, adj_tensor
