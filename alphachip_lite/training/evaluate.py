"""
AlphaChip-Lite: Evaluation Script
====================================
Evaluate a trained (or random) agent and produce floorplan visualizations.

Usage:
  python -m alphachip_lite.training.evaluate                              # random baseline
  python -m alphachip_lite.training.evaluate --checkpoint outputs/ckpt_best.pt  # trained
  python -m alphachip_lite.training.evaluate --compare                    # side-by-side
"""

import argparse
import os
import sys
import random
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
from alphachip_lite.env.placement_utils import half_perimeter_wirelength, order_by_size
from alphachip_lite.models.policy_network import PolicyNetwork, obs_from_state
from alphachip_lite.utils.checkpoint import load_checkpoint
from alphachip_lite.visualization.plot_layout import plot_floorplan, plot_comparison


def evaluate_random(env, num_episodes=10):
    """Run episodes with random valid actions, return average metrics."""
    all_wl = []
    last_placements = None
    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        while not state.done:
            mask = state.current_mask()
            valid = np.where(mask)[0]
            if len(valid) == 0:
                state.current_step += 1
                continue
            action = np.random.choice(valid)
            state, reward, done, info = env.step(action)
            total_reward += reward
        wl = half_perimeter_wirelength(env.netlist.nets, state.placements)
        all_wl.append(wl)
        last_placements = dict(state.placements)
    return {
        "mean_wl": np.mean(all_wl),
        "std_wl": np.std(all_wl),
        "min_wl": np.min(all_wl),
        "placements": last_placements,
    }


def evaluate_trained(env, policy, node_feat, adj, device, num_episodes=10, deterministic=True):
    """Run episodes with the trained policy, return average metrics."""
    policy.eval()
    all_wl = []
    last_placements = None
    with torch.no_grad():
        for ep in range(num_episodes):
            state = env.reset()
            total_reward = 0.0
            while not state.done:
                mask = state.current_mask()
                mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
                grid, mf, pp, nf, ad = obs_from_state(state, node_feat, adj, device)
                actions, _, _ = policy.get_action(
                    grid, mf, pp, nf, ad,
                    action_mask=mask_t,
                    deterministic=deterministic,
                )
                state, reward, done, info = env.step(actions.item())
                total_reward += reward
            wl = half_perimeter_wirelength(env.netlist.nets, state.placements)
            all_wl.append(wl)
            last_placements = dict(state.placements)
    return {
        "mean_wl": np.mean(all_wl),
        "std_wl": np.std(all_wl),
        "min_wl": np.min(all_wl),
        "placements": last_placements,
    }


def main():
    parser = argparse.ArgumentParser(description="AlphaChip-Lite Evaluation")
    parser.add_argument("--config", type=str, default="alphachip_lite/configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint")
    parser.add_argument("--compare", action="store_true", help="Compare random vs trained")
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="outputs/eval")
    args = parser.parse_args()

    # Load config
    config_path = Path(ROOT_DIR) / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Device
    if cfg.get("device", "auto") == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg["device"])
    print(f"[Eval] Device: {device}")

    # Netlist
    netlist_path = Path(ROOT_DIR) / cfg["netlist_path"]
    netlist = netlist_parser.load(str(netlist_path))
    print(f"[Eval] Netlist: {netlist.name} ({netlist.num_macros} macros)")

    # Env
    env = FloorplanEnv(netlist, device, ordering="size")
    os.makedirs(args.output_dir, exist_ok=True)

    # Random baseline
    print("\n--- Random Baseline ---")
    random_res = evaluate_random(env, args.num_episodes)
    print(f"  HPWL: {random_res['mean_wl']:.1f} ± {random_res['std_wl']:.1f}  (best: {random_res['min_wl']:.1f})")
    plot_floorplan(netlist, random_res["placements"],
                   title=f"Random Baseline  (HPWL={random_res['mean_wl']:.0f})",
                   save_path=os.path.join(args.output_dir, "floorplan_random.png"))

    # Trained agent
    if args.checkpoint or args.compare:
        node_feat = netlist.macro_feature_tensor(device)
        adj = netlist.adjacency_tensor(device)
        mc = cfg.get("model", {})
        policy = PolicyNetwork(
            rows=netlist.canvas.rows,
            cols=netlist.canvas.columns,
            num_macros=netlist.num_macros,
            **{k: mc[k] for k in mc if k in PolicyNetwork.__init__.__code__.co_varnames},
        ).to(device)

        if args.checkpoint:
            load_checkpoint(args.checkpoint, policy, device=device)
        else:
            print("[Eval] No checkpoint specified — using untrained policy")

        print("\n--- Trained Agent ---")
        trained_res = evaluate_trained(env, policy, node_feat, adj, device, args.num_episodes)
        print(f"  HPWL: {trained_res['mean_wl']:.1f} ± {trained_res['std_wl']:.1f}  (best: {trained_res['min_wl']:.1f})")
        plot_floorplan(netlist, trained_res["placements"],
                       title=f"Trained Agent  (HPWL={trained_res['mean_wl']:.0f})",
                       save_path=os.path.join(args.output_dir, "floorplan_trained.png"))

        # Comparison
        if args.compare:
            improvement = (random_res["mean_wl"] - trained_res["mean_wl"]) / random_res["mean_wl"] * 100
            print(f"\n  Improvement: {improvement:.1f}%")
            plot_comparison(
                netlist,
                random_res["placements"], trained_res["placements"],
                random_res["mean_wl"], trained_res["mean_wl"],
                save_path=os.path.join(args.output_dir, "comparison.png"),
            )

    print(f"\n[Eval] Results saved in {args.output_dir}/")


if __name__ == "__main__":
    main()
