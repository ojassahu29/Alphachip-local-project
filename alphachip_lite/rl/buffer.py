"""
AlphaChip-Lite: Rollout Buffer
================================
Stores transitions collected during PPO rollouts and computes:
  - Generalised Advantage Estimation (GAE)
  - Returns (value targets)

Designed for GPU-batched training.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch


@dataclass
class Transition:
    grid: torch.Tensor             # (1, rows, cols)
    macro_feat: torch.Tensor       # (3,)
    partial_placement: torch.Tensor  # (N, 3)
    action: int
    action_log_prob: float
    reward: float
    value: float
    done: bool
    mask: torch.Tensor             # (n_actions,) bool


class RolloutBuffer:
    """
    Fixed-size rollout buffer for PPO.
    Stores raw transitions and computes advantages after rollout ends.
    """

    def __init__(
        self,
        capacity: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device = torch.device("cpu"),
    ):
        self.capacity = capacity
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.transitions: List[Transition] = []

    def clear(self):
        self.transitions = []

    def add(self, t: Transition):
        self.transitions.append(t)

    def __len__(self):
        return len(self.transitions)

    def is_full(self) -> bool:
        return len(self.transitions) >= self.capacity

    def compute_advantages(self, last_value: float = 0.0):
        """
        Compute GAE advantages and value targets in-place.
        last_value: bootstrap value for the last (incomplete) episode.
        """
        n = len(self.transitions)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        last_gae = 0.0
        last_return = last_value

        for i in reversed(range(n)):
            t = self.transitions[i]
            # Next value
            if i < n - 1:
                next_value = self.transitions[i + 1].value
                next_done = self.transitions[i + 1].done
            else:
                next_value = last_value
                next_done = True

            delta = t.reward + self.gamma * next_value * (1 - float(next_done)) - t.value
            last_gae = delta + self.gamma * self.gae_lambda * (1 - float(next_done)) * last_gae
            advantages[i] = last_gae
            returns[i] = advantages[i] + t.value

        self._advantages = advantages
        self._returns = returns

    def get_batches(self, batch_size: int):
        """
        Yields mini-batches as dicts of tensors.
        Call compute_advantages() first.
        """
        n = len(self.transitions)
        indices = np.random.permutation(n)

        # Normalise advantages
        adv = self._advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for start in range(0, n, batch_size):
            idx = indices[start: start + batch_size]

            grids, macro_feats, pps, actions, log_probs, rets, advs, masks = \
                [], [], [], [], [], [], [], []

            for i in idx:
                t = self.transitions[i]
                grids.append(t.grid.unsqueeze(0))       # (1, 1, R, C)
                macro_feats.append(t.macro_feat.unsqueeze(0))  # (1, 3)
                pps.append(t.partial_placement.unsqueeze(0))   # (1, N, 3)
                actions.append(t.action)
                log_probs.append(t.action_log_prob)
                rets.append(self._returns[i])
                advs.append(adv[i])
                masks.append(t.mask.unsqueeze(0))       # (1, n_actions)

            yield {
                "grids": torch.cat(grids, dim=0).to(self.device),
                "macro_feats": torch.cat(macro_feats, dim=0).to(self.device),
                "partial_placements": torch.cat(pps, dim=0).to(self.device),
                "actions": torch.tensor(actions, dtype=torch.long, device=self.device),
                "old_log_probs": torch.tensor(log_probs, dtype=torch.float32, device=self.device),
                "returns": torch.tensor(rets, dtype=torch.float32, device=self.device),
                "advantages": torch.tensor(advs, dtype=torch.float32, device=self.device),
                "masks": torch.cat(masks, dim=0).to(self.device),
            }
