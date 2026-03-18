"""
AlphaChip-Lite: Placement Utilities
=====================================
Low-level geometry helpers for the floorplan environment:
  - overlap detection
  - boundary enforcement
  - wirelength computation
  - congestion estimation
  - placement masks
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def macro_cells(x: float, y: float, w: float, h: float,
                cell_w: float, cell_h: float,
                cols: int, rows: int) -> List[Tuple[int, int]]:
    """
    Returns the list of (col, row) grid cells covered by a macro placed at (x, y)
    with size (w, h).  x, y are the centroid in canvas coordinates.
    """
    half_w = w / 2.0
    half_h = h / 2.0
    x0 = int(max(0, (x - half_w) / cell_w))
    y0 = int(max(0, (y - half_h) / cell_h))
    x1 = int(min(cols - 1, math.ceil((x + half_w) / cell_w) - 1))
    y1 = int(min(rows - 1, math.ceil((y + half_h) / cell_h) - 1))
    cells = []
    for c in range(x0, x1 + 1):
        for r in range(y0, y1 + 1):
            cells.append((c, r))
    return cells


def overlaps(x1: float, y1: float, w1: float, h1: float,
             x2: float, y2: float, w2: float, h2: float,
             tolerance: float = 0.001) -> bool:
    """Check if two rectangles (given by center + size) overlap."""
    return (abs(x1 - x2) < (w1 + w2) / 2.0 - tolerance and
            abs(y1 - y2) < (h1 + h2) / 2.0 - tolerance)


def out_of_bounds(x: float, y: float, w: float, h: float,
                  canvas_w: float, canvas_h: float) -> bool:
    """Check if a macro placed at (x,y) with size (w,h) exceeds canvas."""
    half_w, half_h = w / 2.0, h / 2.0
    return (x - half_w < 0 or x + half_w > canvas_w or
            y - half_h < 0 or y + half_h > canvas_h)


# ---------------------------------------------------------------------------
# Placement mask
# ---------------------------------------------------------------------------

def compute_placement_mask(
    macro_idx: int,
    macro_width: float,
    macro_height: float,
    placed_macros: Dict[int, Tuple[float, float, float, float]],
    canvas_w: float,
    canvas_h: float,
    cols: int,
    rows: int,
) -> np.ndarray:
    """
    Returns a flat boolean mask of shape (cols * rows,).
    mask[i] = True iff placing the macro centred at cell i is valid
    (no overlap with placed macros, within bounds).

    placed_macros: {id: (x, y, w, h)} of already placed macros.
    """
    cell_w = canvas_w / cols
    cell_h = canvas_h / rows

    mask = np.ones(cols * rows, dtype=bool)

    for col in range(cols):
        for row in range(rows):
            # Centre of this cell
            cx = (col + 0.5) * cell_w
            cy = (row + 0.5) * cell_h

            # Out-of-bounds check
            if out_of_bounds(cx, cy, macro_width, macro_height, canvas_w, canvas_h):
                mask[row * cols + col] = False
                continue

            # Overlap check with placed macros
            for _, (px, py, pw, ph) in placed_macros.items():
                if overlaps(cx, cy, macro_width, macro_height, px, py, pw, ph):
                    mask[row * cols + col] = False
                    break

    return mask


# ---------------------------------------------------------------------------
# Wirelength
# ---------------------------------------------------------------------------

def half_perimeter_wirelength(
    nets,
    placements: Dict[int, Tuple[float, float]],
) -> float:
    """
    Compute total HPWL (Half-Perimeter Wire Length) for all nets.
    Only includes nets where ALL pins are placed.
    nets: list of Net objects (with .pins and .weight)
    placements: {macro_id: (x, y)}
    """
    total = 0.0
    for net in nets:
        xs, ys = [], []
        skip = False
        for pid in net.pins:
            if pid not in placements:
                skip = True
                break
            x, y = placements[pid]
            xs.append(x)
            ys.append(y)
        if skip or not xs:
            continue
        hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
        total += net.weight * hpwl
    return total


# ---------------------------------------------------------------------------
# Overlap penalty
# ---------------------------------------------------------------------------

def total_overlap_area(
    placements: Dict[int, Tuple[float, float, float, float]],
) -> float:
    """
    Compute total pairwise overlap area among placed macros.
    placements: {id: (x, y, w, h)}
    """
    ids = list(placements.keys())
    total = 0.0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            x1, y1, w1, h1 = placements[ids[i]]
            x2, y2, w2, h2 = placements[ids[j]]
            dx = max(0, (w1 + w2) / 2 - abs(x1 - x2))
            dy = max(0, (h1 + h2) / 2 - abs(y1 - y2))
            total += dx * dy
    return total


# ---------------------------------------------------------------------------
# Simple congestion estimate
# ---------------------------------------------------------------------------

def grid_density(
    placed_macros: Dict[int, Tuple[float, float, float, float]],
    canvas_w: float, canvas_h: float,
    cols: int, rows: int,
) -> np.ndarray:
    """
    Returns a (rows, cols) array with fraction of each cell covered.
    """
    cell_w = canvas_w / cols
    cell_h = canvas_h / rows
    density = np.zeros((rows, cols), dtype=np.float32)

    for _, (x, y, w, h) in placed_macros.items():
        for c in range(cols):
            for r in range(rows):
                # cell bounding box
                cx0, cx1 = c * cell_w, (c + 1) * cell_w
                cy0, cy1 = r * cell_h, (r + 1) * cell_h
                # macro bounding box
                mx0, mx1 = x - w / 2, x + w / 2
                my0, my1 = y - h / 2, y + h / 2
                # intersection
                ix = max(0, min(cx1, mx1) - max(cx0, mx0))
                iy = max(0, min(cy1, my1) - max(cy0, my0))
                density[r, c] += (ix * iy) / (cell_w * cell_h)

    return np.clip(density, 0.0, 1.0)


def congestion_cost(density: np.ndarray, threshold: float = 0.8) -> float:
    """Penalise grid cells above the density threshold."""
    excess = np.maximum(0.0, density - threshold)
    return float(excess.mean())


# ---------------------------------------------------------------------------
# Macro ordering strategies (AlphaChip-inspired)
# ---------------------------------------------------------------------------

def order_by_size(macros, descending: bool = True) -> List[int]:
    """Order macros by area (largest first by default)."""
    areas = [(m.id, m.width * m.height) for m in macros]
    areas.sort(key=lambda t: t[1], reverse=descending)
    return [t[0] for t in areas]


def order_by_connectivity(macros, nets, descending: bool = True) -> List[int]:
    """Order macros by total connection weight (most connected first)."""
    scores = {m.id: 0.0 for m in macros}
    for net in nets:
        for pid in net.pins:
            if pid in scores:
                scores[pid] += net.weight
    ordered = sorted(scores.keys(), key=lambda k: scores[k], reverse=descending)
    return ordered


def curriculum_order(macros, nets, epoch: int, max_epochs: int) -> List[int]:
    """
    Curriculum-learning ordering:
      early epochs → order by size (simpler)
      later epochs  → order by connectivity (harder, more realistic)
    """
    progress = epoch / max(1, max_epochs)
    if progress < 0.4:
        return order_by_size(macros)
    return order_by_connectivity(macros, nets)
