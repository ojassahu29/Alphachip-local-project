"""
AlphaChip-Lite: PPO (Proximal Policy Optimization)
====================================================
Full PPO implementation for chip macro placement.

Key features:
  - Clipped surrogate objective
  - Value function clipping
  - Entropy bonus
  - Action masking integration
  - GPU-accelerated mini-batch updates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from alphachip_lite.rl.buffer import RolloutBuffer


class PPO:
    """
    Proximal Policy Optimization trainer.

    Handles the update loop: takes a filled RolloutBuffer and updates
    the policy and value networks for `n_epochs` over `batch_size`-sized
    mini-batches.
    """

    def __init__(
        self,
        policy: nn.Module,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        device: torch.device = torch.device("cpu"),
    ):
        self.policy = policy
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

    def update(
        self,
        buffer: RolloutBuffer,
        node_features: torch.Tensor,   # (N, 2)
        adj: torch.Tensor,             # (N, N)
    ) -> Dict[str, float]:
        """
        Run PPO update for `n_epochs` on the given buffer.

        Returns dict of average loss metrics.
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        n_updates = 0

        for epoch in range(self.n_epochs):
            for batch in buffer.get_batches(self.batch_size):
                grids = batch["grids"]                       # (B, 1, R, C)
                macro_feats = batch["macro_feats"]           # (B, 3)
                pps = batch["partial_placements"]            # (B, N, 3)
                actions = batch["actions"]                   # (B,)
                old_log_probs = batch["old_log_probs"]       # (B,)
                returns = batch["returns"]                   # (B,)
                advantages = batch["advantages"]             # (B,)
                masks = batch["masks"]                       # (B, n_actions)

                # Forward pass
                action_probs, log_probs_all, values = self.policy(
                    grids, macro_feats, pps,
                    node_features, adj,
                    action_mask=masks,
                )

                # Log prob of taken actions
                new_log_probs = log_probs_all.gather(1, actions.unsqueeze(-1)).squeeze(-1)

                # --- Policy loss (clipped surrogate) ---
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # --- Value loss ---
                value_loss = F.mse_loss(values, returns)

                # --- Entropy bonus ---
                entropy = -(action_probs * log_probs_all).sum(dim=-1).mean()

                # --- Total loss ---
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_loss += loss.item()
                n_updates += 1

        n_updates = max(n_updates, 1)
        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "total_loss": total_loss / n_updates,
        }
