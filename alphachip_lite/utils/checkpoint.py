"""
AlphaChip-Lite: Checkpoint Manager
====================================
Save and load model checkpoints, training state, and config.
"""

import os
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: str,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict] = None,
):
    """Save full training checkpoint."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "epoch": epoch,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics or {},
        "config": config or {},
    }, path)
    print(f"[Checkpoint] Saved to {path}")


def load_checkpoint(
    path: str,
    policy: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Load checkpoint and return metadata dict."""
    ckpt = torch.load(path, map_location=device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"[Checkpoint] Loaded from {path} (epoch {ckpt.get('epoch', '?')})")
    return ckpt
