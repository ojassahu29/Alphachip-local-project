"""
AlphaChip-Lite: Training Script
=================================
Main PPO training loop for the chip floorplanning RL agent.

Follows the AlphaChip paradigm:
  1. Sequential macro placement (one macro per step)
  2. Collect rollouts via environment interaction
  3. Compute GAE advantages
  4. PPO mini-batch updates on GPU
  5. Curriculum learning on macro ordering
  6. Periodic evaluation + checkpoint saving

Usage:
  python -m alphachip_lite.training.train
  python -m alphachip_lite.training.train --config alphachip_lite/configs/default.yaml
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# Ensure project root on path
ROOT_DIR = str(Path(__file__).resolve().parent.parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from alphachip_lite.netlist import parser as netlist_parser
from alphachip_lite.env.floorplan_env import FloorplanEnv
from alphachip_lite.env.placement_utils import half_perimeter_wirelength
from alphachip_lite.models.policy_network import PolicyNetwork, obs_from_state
from alphachip_lite.rl.buffer import RolloutBuffer, Transition
from alphachip_lite.rl.ppo import PPO
from alphachip_lite.utils.logger import Logger
from alphachip_lite.utils.checkpoint import save_checkpoint
from alphachip_lite.visualization.plot_layout import (
    plot_floorplan,
    plot_training_curves,
)


def collect_rollout(
    env: FloorplanEnv,
    policy: PolicyNetwork,
    buffer: RolloutBuffer,
    node_feat: torch.Tensor,
    adj: torch.Tensor,
    device: torch.device,
    n_steps: int,
):
    """
    Collect `n_steps` transitions by running the environment
    and sampling actions from the policy.
    """
    state = env.reset()
    episode_rewards = []
    episode_wls = []
    ep_reward = 0.0

    policy.eval()
    with torch.no_grad():
        for step in range(n_steps):
            if state.done:
                # Log completed episode
                wl = half_perimeter_wirelength(env.netlist.nets, state.placements)
                episode_rewards.append(ep_reward)
                episode_wls.append(wl)
                ep_reward = 0.0
                state = env.reset()

            # Get observation tensors
            mask_np = state.current_mask()
            mask_t = torch.tensor(mask_np, dtype=torch.bool, device=device).unsqueeze(0)
            obs = state.to_tensor_obs()
            grid = obs["grid"].unsqueeze(0)
            mf = obs["macro_feat"].unsqueeze(0)
            pp = obs["partial_placement"].unsqueeze(0)

            # Sample action
            actions, log_probs, values = policy.get_action(
                grid, mf, pp, node_feat, adj,
                action_mask=mask_t,
                deterministic=False,
            )
            action = actions.item()
            log_prob = log_probs.item()
            value = values.item()

            # Step environment
            next_state, reward, done, info = env.step(action)
            ep_reward += reward

            # Store transition
            buffer.add(Transition(
                grid=obs["grid"],
                macro_feat=obs["macro_feat"],
                partial_placement=obs["partial_placement"],
                action=action,
                action_log_prob=log_prob,
                reward=reward,
                value=value,
                done=done,
                mask=torch.tensor(mask_np, dtype=torch.bool),
            ))

            state = next_state

    # Bootstrap value for last incomplete episode
    last_value = 0.0
    if not state.done:
        with torch.no_grad():
            obs = state.to_tensor_obs()
            _, _, values = policy(
                obs["grid"].unsqueeze(0),
                obs["macro_feat"].unsqueeze(0),
                obs["partial_placement"].unsqueeze(0),
                node_feat, adj,
            )
            last_value = values.item()

    buffer.compute_advantages(last_value)
    return episode_rewards, episode_wls


def main():
    parser = argparse.ArgumentParser(description="AlphaChip-Lite PPO Training")
    parser.add_argument("--config", type=str, default="alphachip_lite/configs/default.yaml")
    args = parser.parse_args()

    # ---------- Config ----------
    config_path = Path(ROOT_DIR) / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Device
    if cfg.get("device", "auto") == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg["device"])

    print("=" * 60)
    print("  AlphaChip-Lite: GPU-Accelerated RL for Chip Floorplanning")
    print("=" * 60)
    print(f"  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  Memory : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # ---------- Netlist ----------
    netlist_path = Path(ROOT_DIR) / cfg["netlist_path"]
    netlist = netlist_parser.load(str(netlist_path))
    print(f"  Netlist: {netlist.name} ({netlist.num_macros} macros, {len(netlist.nets)} nets)")
    print(f"  Canvas : {netlist.canvas.width:.0f} x {netlist.canvas.height:.0f} ({netlist.canvas.columns}x{netlist.canvas.rows} grid)")

    # Precompute fixed tensors
    node_feat = netlist.macro_feature_tensor(device)
    adj = netlist.adjacency_tensor(device)

    # ---------- Environment ----------
    env_cfg = cfg.get("env", {})
    reward_cfg = env_cfg.get("reward", {})
    train_cfg = cfg.get("training", {})
    ppo_cfg = cfg.get("ppo", {})
    model_cfg = cfg.get("model", {})

    num_episodes = train_cfg.get("num_episodes", 500)
    rollout_steps = train_cfg.get("rollout_steps", 256)
    log_interval = train_cfg.get("log_interval", 10)
    save_interval = train_cfg.get("save_interval", 100)
    output_dir = train_cfg.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # ---------- Model ----------
    policy = PolicyNetwork(
        rows=netlist.canvas.rows,
        cols=netlist.canvas.columns,
        num_macros=netlist.num_macros,
        grid_embed_dim=model_cfg.get("grid_embed_dim", 128),
        graph_embed_dim=model_cfg.get("graph_embed_dim", 128),
        macro_embed_dim=model_cfg.get("macro_embed_dim", 64),
        hidden_dim=model_cfg.get("hidden_dim", 256),
        gcn_hidden_dim=model_cfg.get("gcn_hidden_dim", 64),
        n_gcn_layers=model_cfg.get("n_gcn_layers", 2),
        dropout=model_cfg.get("dropout", 0.1),
    ).to(device)

    param_count = sum(p.numel() for p in policy.parameters())
    print(f"  Model  : {param_count:,} parameters")

    # ---------- PPO ----------
    ppo = PPO(
        policy=policy,
        lr=ppo_cfg.get("lr", 3e-4),
        clip_eps=ppo_cfg.get("clip_eps", 0.2),
        value_coef=ppo_cfg.get("value_coef", 0.5),
        entropy_coef=ppo_cfg.get("entropy_coef", 0.01),
        max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
        n_epochs=ppo_cfg.get("n_epochs", 4),
        batch_size=ppo_cfg.get("batch_size", 64),
        device=device,
    )

    # ---------- Buffer ----------
    buffer = RolloutBuffer(
        capacity=rollout_steps,
        gamma=ppo_cfg.get("gamma", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        device=device,
    )

    # ---------- Logger + Metrics ----------
    logger = Logger(output_dir, name="train")
    all_rewards = []
    all_losses = []
    all_wls = []
    best_wl = float("inf")

    print(f"\n{'=' * 60}")
    print(f"  Training for {num_episodes} episodes  |  rollout={rollout_steps}  |  batch={ppo_cfg.get('batch_size', 64)}")
    print(f"{'=' * 60}\n")

    t0 = time.time()

    for episode in range(1, num_episodes + 1):
        # Curriculum: update ordering based on epoch
        env = FloorplanEnv(
            netlist, device,
            ordering=env_cfg.get("ordering", "curriculum"),
            epoch=episode,
            max_epochs=num_episodes,
            reward_kwargs=reward_cfg,
        )

        # Collect rollout
        buffer.clear()
        ep_rewards, ep_wls = collect_rollout(
            env, policy, buffer, node_feat, adj, device, rollout_steps,
        )

        # PPO update
        policy.train()
        metrics = ppo.update(buffer, node_feat, adj)

        # Track metrics
        if ep_rewards:
            avg_reward = np.mean(ep_rewards)
            avg_wl = np.mean(ep_wls)
        else:
            avg_reward = 0.0
            avg_wl = 0.0

        all_rewards.append(avg_reward)
        all_losses.append(metrics["total_loss"])
        all_wls.append(avg_wl)

        # Log
        if episode % log_interval == 0:
            logger.log(episode, {
                "reward": round(avg_reward, 4),
                "wirelength": round(avg_wl, 1),
                "policy_loss": round(metrics["policy_loss"], 4),
                "value_loss": round(metrics["value_loss"], 4),
                "entropy": round(metrics["entropy"], 4),
            })

        # Save best
        if avg_wl > 0 and avg_wl < best_wl:
            best_wl = avg_wl
            save_checkpoint(
                os.path.join(output_dir, "ckpt_best.pt"),
                policy, ppo.optimizer, episode,
                metrics={"best_wl": best_wl},
                config=cfg,
            )

        # Periodic checkpoint
        if episode % save_interval == 0:
            save_checkpoint(
                os.path.join(output_dir, f"ckpt_ep{episode}.pt"),
                policy, ppo.optimizer, episode,
                metrics={"avg_wl": avg_wl},
                config=cfg,
            )

    # ---------- Final outputs ----------
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Training complete in {elapsed:.0f}s  |  Best HPWL: {best_wl:.1f}")
    print(f"{'=' * 60}")

    # Save final checkpoint
    save_checkpoint(
        os.path.join(output_dir, "ckpt_final.pt"),
        policy, ppo.optimizer, num_episodes,
        metrics={"best_wl": best_wl},
        config=cfg,
    )

    # Training curves
    plot_training_curves(
        all_rewards, all_losses, all_wls,
        save_path=os.path.join(output_dir, "training_curves.png"),
    )

    # Final floorplan from one episode
    env = FloorplanEnv(netlist, device, ordering="size")
    state = env.reset()
    policy.eval()
    with torch.no_grad():
        while not state.done:
            mask = state.current_mask()
            mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            grid, mf, pp, nf, ad = obs_from_state(state, node_feat, adj, device)
            actions, _, _ = policy.get_action(grid, mf, pp, nf, ad, mask_t, deterministic=True)
            state, _, _, _ = env.step(actions.item())

    final_wl = half_perimeter_wirelength(netlist.nets, state.placements)
    plot_floorplan(
        netlist, state.placements,
        title=f"Final Placement  (HPWL={final_wl:.0f})",
        save_path=os.path.join(output_dir, "floorplan_final.png"),
    )

    print(f"\n  All outputs saved to {output_dir}/")
    print(f"  Run evaluation:  python -m alphachip_lite.training.evaluate --checkpoint {output_dir}/ckpt_best.pt --compare")


if __name__ == "__main__":
    main()
