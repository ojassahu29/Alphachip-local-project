"""
AlphaChip-Lite: Visualization
================================
Plot chip floorplan layouts with macro rectangles, labels, and grid overlay.
Supports comparison between random and trained agent layouts.
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from alphachip_lite.netlist.parser import Netlist


# ---------------------------------------------------------------------------
# Colour palette for macros
# ---------------------------------------------------------------------------

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#E377C2", "#BCBD22", "#17BECF", "#AEC7E8", "#FFBB78",
    "#98DF8A", "#FF9896", "#C5B0D5", "#C49C94", "#F7B6D2",
]


def plot_floorplan(
    netlist: Netlist,
    placements: Dict[int, Tuple[float, float]],
    title: str = "Chip Floorplan",
    save_path: Optional[str] = None,
    show_grid: bool = True,
    show_connections: bool = True,
    figsize: Tuple[int, int] = (10, 10),
    dpi: int = 120,
) -> plt.Figure:
    """
    Plot a chip floorplan showing macro rectangles.

    Parameters
    ----------
    netlist : Netlist
    placements : {macro_id: (cx, cy)}
    title : figure title
    save_path : if set, save figure to this path
    show_grid : overlay the grid cells
    show_connections : draw nets as lines between macro centroids
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    canvas = netlist.canvas

    # Canvas boundary
    ax.set_xlim(0, canvas.width)
    ax.set_ylim(0, canvas.height)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("X (μm)")
    ax.set_ylabel("Y (μm)")

    # Grid overlay
    if show_grid:
        for c in range(canvas.columns + 1):
            x = c * canvas.cell_width
            ax.axvline(x, color="#e0e0e0", linewidth=0.3)
        for r in range(canvas.rows + 1):
            y = r * canvas.cell_height
            ax.axhline(y, color="#e0e0e0", linewidth=0.3)

    # Draw connections first (below macros)
    if show_connections:
        for net in netlist.nets:
            pin_positions = []
            for pid in net.pins:
                if pid in placements:
                    pin_positions.append(placements[pid])
            if len(pin_positions) >= 2:
                alpha = min(0.6, 0.1 + net.weight * 0.1)
                for i in range(len(pin_positions)):
                    for j in range(i + 1, len(pin_positions)):
                        x1, y1 = pin_positions[i]
                        x2, y2 = pin_positions[j]
                        ax.plot([x1, x2], [y1, y2], color="#888888",
                                linewidth=0.5 + net.weight * 0.3,
                                alpha=alpha, zorder=1)

    # Draw macro rectangles
    for macro in netlist.macros:
        if macro.id not in placements:
            continue
        cx, cy = placements[macro.id]
        w, h = macro.width, macro.height
        color = PALETTE[macro.id % len(PALETTE)]

        rect = mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.5",
            facecolor=color,
            edgecolor="black",
            linewidth=1.2,
            alpha=0.8,
            zorder=5,
        )
        ax.add_patch(rect)

        # Label
        fontsize = max(5, min(9, int(min(w, h) / 3)))
        ax.text(cx, cy, macro.name, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="white",
                zorder=6)

    # Canvas border
    border = mpatches.Rectangle((0, 0), canvas.width, canvas.height,
                                fill=False, edgecolor="black", linewidth=2)
    ax.add_patch(border)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[Viz] Saved floorplan to {save_path}")

    return fig


def plot_comparison(
    netlist: Netlist,
    random_placements: Dict[int, Tuple[float, float]],
    trained_placements: Dict[int, Tuple[float, float]],
    random_wl: float,
    trained_wl: float,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (20, 10),
    dpi: int = 120,
) -> plt.Figure:
    """
    Side-by-side comparison of random vs trained agent placements.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    for ax, placements, label, wl in [
        (axes[0], random_placements, "Random Baseline", random_wl),
        (axes[1], trained_placements, "Trained Agent", trained_wl),
    ]:
        canvas = netlist.canvas
        ax.set_xlim(0, canvas.width)
        ax.set_ylim(0, canvas.height)
        ax.set_aspect("equal")
        ax.set_title(f"{label}\nHPWL = {wl:.1f}", fontsize=13, fontweight="bold")
        ax.set_xlabel("X (μm)")
        ax.set_ylabel("Y (μm)")

        # Grid
        for c in range(canvas.columns + 1):
            ax.axvline(c * canvas.cell_width, color="#e0e0e0", linewidth=0.3)
        for r in range(canvas.rows + 1):
            ax.axhline(r * canvas.cell_height, color="#e0e0e0", linewidth=0.3)

        # Connections
        for net in netlist.nets:
            pin_pos = [placements[p] for p in net.pins if p in placements]
            if len(pin_pos) >= 2:
                for i in range(len(pin_pos)):
                    for j in range(i + 1, len(pin_pos)):
                        ax.plot([pin_pos[i][0], pin_pos[j][0]],
                                [pin_pos[i][1], pin_pos[j][1]],
                                color="#888888", linewidth=0.5, alpha=0.4, zorder=1)

        # Macros
        for macro in netlist.macros:
            if macro.id not in placements:
                continue
            cx, cy = placements[macro.id]
            w, h = macro.width, macro.height
            color = PALETTE[macro.id % len(PALETTE)]
            rect = mpatches.FancyBboxPatch(
                (cx - w / 2, cy - h / 2), w, h,
                boxstyle="round,pad=0.5",
                facecolor=color, edgecolor="black",
                linewidth=1.2, alpha=0.8, zorder=5,
            )
            ax.add_patch(rect)
            fontsize = max(5, min(9, int(min(w, h) / 3)))
            ax.text(cx, cy, macro.name, ha="center", va="center",
                    fontsize=fontsize, fontweight="bold", color="white", zorder=6)

        border = mpatches.Rectangle((0, 0), canvas.width, canvas.height,
                                    fill=False, edgecolor="black", linewidth=2)
        ax.add_patch(border)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[Viz] Saved comparison to {save_path}")
    return fig


def plot_training_curves(
    rewards: List[float],
    losses: List[float],
    wirelengths: List[float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 5),
) -> plt.Figure:
    """Plot reward, loss, and wirelength curves during training."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    colors = ["#2196F3", "#F44336", "#4CAF50"]

    for ax, data, label, color in zip(
        axes,
        [rewards, losses, wirelengths],
        ["Episode Reward", "PPO Loss", "HPWL (Wirelength)"],
        colors,
    ):
        ax.plot(data, color=color, alpha=0.3, linewidth=0.5)
        # Smoothed curve
        if len(data) > 10:
            window = min(50, len(data) // 5)
            smoothed = np.convolve(data, np.ones(window) / window, mode="valid")
            ax.plot(range(window - 1, len(data)), smoothed, color=color, linewidth=2)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[Viz] Saved training curves to {save_path}")
    return fig
