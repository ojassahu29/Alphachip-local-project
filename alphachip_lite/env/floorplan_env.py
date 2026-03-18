"""
AlphaChip-Lite: Floorplan Environment
=======================================
A Gym-compatible environment for sequential macro placement on a chip canvas.
Inspired by the circuit_training environment (Google Research).

Key design decisions:
- One macro is placed per step (sequential, as in AlphaChip).
- State = (grid_occupancy, current_macro_features, partial_placement).
- Action = integer index into the flat (cols * rows) grid.
- Reward = weighted combination of HPWL, overlap, and boundary penalties.
- Invalid actions are masked out via placement mask.
"""

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import gym
    from gym import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

from alphachip_lite.env.placement_utils import (
    compute_placement_mask,
    half_perimeter_wirelength,
    total_overlap_area,
    grid_density,
    congestion_cost,
    curriculum_order,
    order_by_size,
)
from alphachip_lite.netlist.parser import Netlist, Macro


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------

class FloorplanState:
    """Encapsulates the full observation of the environment at one step."""

    def __init__(self, netlist: Netlist, macro_order: List[int], device: torch.device):
        self.netlist = netlist
        self.macro_order = macro_order          # placement sequence (list of macro IDs)
        self.current_step = 0                   # index into macro_order
        self.device = device

        canvas = netlist.canvas
        self.canvas_w = canvas.width
        self.canvas_h = canvas.height
        self.cols = canvas.columns
        self.rows = canvas.rows
        self.cell_w = canvas.cell_width
        self.cell_h = canvas.cell_height

        # Placement state: macro_id → (x, y)
        self.placements: Dict[int, Tuple[float, float]] = {}
        # Also store (x, y, w, h) for overlap computations
        self.rect_placements: Dict[int, Tuple[float, float, float, float]] = {}

        # Grid occupancy: (1, rows, cols) tensor — fraction of each cell covered
        self.grid = np.zeros((self.rows, self.cols), dtype=np.float32)

    @property
    def done(self) -> bool:
        return self.current_step >= len(self.macro_order)

    @property
    def current_macro(self) -> Optional[Macro]:
        if self.done:
            return None
        mid = self.macro_order[self.current_step]
        return self.netlist.macros[mid]

    def place(self, macro: Macro, col: int, row: int):
        """Place macro at (col, row) grid cell."""
        cx = (col + 0.5) * self.cell_w
        cy = (row + 0.5) * self.cell_h
        mid = macro.id
        self.placements[mid] = (cx, cy)
        self.rect_placements[mid] = (cx, cy, macro.width, macro.height)
        self._update_grid(cx, cy, macro.width, macro.height, value=1.0)
        self.current_step += 1

    def _update_grid(self, x, y, w, h, value=1.0):
        xmin = max(0, (x - w / 2) / self.cell_w)
        xmax = min(self.cols, (x + w / 2) / self.cell_w)
        ymin = max(0, (y - h / 2) / self.cell_h)
        ymax = min(self.rows, (y + h / 2) / self.cell_h)
        for c in range(int(xmin), int(math.ceil(xmax))):
            for r in range(int(ymin), int(math.ceil(ymax))):
                if 0 <= c < self.cols and 0 <= r < self.rows:
                    self.grid[r, c] = min(1.0, self.grid[r, c] + value)

    def current_mask(self) -> np.ndarray:
        """Return flat boolean mask for the current macro."""
        m = self.current_macro
        if m is None:
            return np.zeros(self.cols * self.rows, dtype=bool)
        return compute_placement_mask(
            macro_idx=m.id,
            macro_width=m.width,
            macro_height=m.height,
            placed_macros=self.rect_placements,
            canvas_w=self.canvas_w,
            canvas_h=self.canvas_h,
            cols=self.cols,
            rows=self.rows,
        )

    def to_tensor_obs(self) -> Dict[str, torch.Tensor]:
        """Convert state to tensors for the neural network."""
        dev = self.device
        m = self.current_macro

        # Grid occupancy: (1, rows, cols)
        grid_t = torch.tensor(self.grid, dtype=torch.float32, device=dev).unsqueeze(0)

        # Macro features: normalised [width, height, step_ratio]
        if m is not None:
            mf = torch.tensor([
                m.width / self.canvas_w,
                m.height / self.canvas_h,
                self.current_step / max(1, len(self.macro_order)),
            ], dtype=torch.float32, device=dev)
        else:
            mf = torch.zeros(3, dtype=torch.float32, device=dev)

        # Partial placements encoded as (N, 3) → [norm_x, norm_y, placed?]
        N = self.netlist.num_macros
        pp = np.zeros((N, 3), dtype=np.float32)
        for mid, (px, py) in self.placements.items():
            if mid < N:
                pp[mid, 0] = px / self.canvas_w
                pp[mid, 1] = py / self.canvas_h
                pp[mid, 2] = 1.0
        pp_t = torch.tensor(pp, dtype=torch.float32, device=dev)

        return {"grid": grid_t, "macro_feat": mf, "partial_placement": pp_t}


# ---------------------------------------------------------------------------
# Reward calculator
# ---------------------------------------------------------------------------

class RewardCalculator:
    """
    Computes the placement reward after each macro is placed.

    Reward = - (wl_weight * normalised_wirelength
               + overlap_weight * normalised_overlap
               + boundary_weight * boundary_violations
               + congestion_weight * congestion)
    """

    def __init__(
        self,
        wl_weight: float = 1.0,
        overlap_weight: float = 1.0,
        boundary_weight: float = 0.5,
        congestion_weight: float = 0.1,
        normalise_wl: float = 1000.0,
    ):
        self.wl_weight = wl_weight
        self.overlap_weight = overlap_weight
        self.boundary_weight = boundary_weight
        self.congestion_weight = congestion_weight
        self.normalise_wl = normalise_wl

    def __call__(self, state: FloorplanState, is_final: bool = False) -> float:
        nl = state.netlist

        # Wirelength (only for fully-placed nets)
        wl = half_perimeter_wirelength(nl.nets, state.placements)
        wl_norm = wl / max(1.0, self.normalise_wl)

        # Overlap penalty
        overlap = total_overlap_area(state.rect_placements)
        overlap_norm = overlap / max(1.0, state.canvas_w * state.canvas_h * 0.01)

        # Congestion
        density = grid_density(state.rect_placements,
                               state.canvas_w, state.canvas_h,
                               state.cols, state.rows)
        cong = congestion_cost(density)

        # Boundary: penalise macros placed close to canvas edge (0 for now, mask handles it)
        boundary = 0.0

        reward = -(
            self.wl_weight * wl_norm
            + self.overlap_weight * overlap_norm
            + self.boundary_weight * boundary
            + self.congestion_weight * cong
        )
        return float(reward), {
            "wirelength": wl,
            "overlap": overlap,
            "congestion": cong,
        }


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class FloorplanEnv:
    """
    AlphaChip-Lite Floorplan Environment.

    Episode:
      - reset() → state
      - step(action: int) → (next_state, reward, done, info)

    Compatible with Gym interface if gym is installed.
    """

    def __init__(
        self,
        netlist: Netlist,
        device: torch.device,
        ordering: str = "size",           # 'size' | 'connectivity' | 'curriculum'
        epoch: int = 0,
        max_epochs: int = 100,
        reward_kwargs: Optional[Dict] = None,
    ):
        self.netlist = netlist
        self.device = device
        self.ordering = ordering
        self.epoch = epoch
        self.max_epochs = max_epochs
        self.reward_fn = RewardCalculator(**(reward_kwargs or {}))

        self.state: Optional[FloorplanState] = None

        # Gym-compatible spaces (if gym is installed)
        if GYM_AVAILABLE:
            n_actions = netlist.canvas.columns * netlist.canvas.rows
            self.action_space = spaces.Discrete(n_actions)

    def _get_macro_order(self) -> List[int]:
        macros = self.netlist.macros
        nets = self.netlist.nets
        if self.ordering == "size":
            return order_by_size(macros)
        elif self.ordering == "connectivity":
            from alphachip_lite.env.placement_utils import order_by_connectivity
            return order_by_connectivity(macros, nets)
        elif self.ordering == "curriculum":
            return curriculum_order(macros, nets, self.epoch, self.max_epochs)
        else:
            return list(range(len(macros)))

    def reset(self) -> FloorplanState:
        order = self._get_macro_order()
        self.state = FloorplanState(self.netlist, order, self.device)
        return self.state

    def step(self, action: int) -> Tuple[FloorplanState, float, bool, Dict]:
        assert self.state is not None, "Call reset() first."
        state = self.state

        m = state.current_macro
        if m is None:
            return state, 0.0, True, {}

        col = action % state.cols
        row = action // state.cols

        # Validate action (silently clamp to nearest valid)
        mask = state.current_mask()
        if not mask[action]:
            # Fallback: pick centre of canvas
            valid = np.where(mask)[0]
            if len(valid) == 0:
                # No valid placement — penalise and skip
                state.current_step += 1
                return state, -2.0, state.done, {"invalid": True}
            action = valid[len(valid) // 2]
            col = action % state.cols
            row = action // state.cols

        state.place(m, col, row)

        is_final = state.done
        reward, info = self.reward_fn(state, is_final)
        return state, reward, state.done, info

    def get_action_mask(self) -> np.ndarray:
        """Return valid action mask for the current step."""
        if self.state is None:
            raise RuntimeError("Call reset() first.")
        return self.state.current_mask()

    def render_text(self):
        """Simple ASCII grid render."""
        if self.state is None:
            return
        grid = np.full((self.state.rows, self.state.cols), '.', dtype='<U1')
        for mid, (cx, cy) in self.state.placements.items():
            c = int(cx / self.state.cell_w)
            r = int(cy / self.state.cell_h)
            c = min(c, self.state.cols - 1)
            r = min(r, self.state.rows - 1)
            m = self.netlist.macros[mid]
            grid[r, c] = m.name[0].upper()

        for r in range(self.state.rows - 1, -1, -1):
            print(' '.join(grid[r]))
        print()


# ---------------------------------------------------------------------------
# Vectorised environment (for batched PPO training on GPU)
# ---------------------------------------------------------------------------

class VecFloorplanEnv:
    """
    Runs `n_envs` FloorplanEnv instances in parallel (Python-level parallelism).
    Returns batched tensors ready for the GPU.
    """

    def __init__(self, netlist: Netlist, n_envs: int, device: torch.device, **env_kwargs):
        self.envs = [FloorplanEnv(netlist, device, **env_kwargs) for _ in range(n_envs)]
        self.n_envs = n_envs
        self.device = device
        self.states = [None] * n_envs

    def reset(self):
        self.states = [env.reset() for env in self.envs]
        return self.states

    def step(self, actions: List[int]):
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        next_states, rewards, dones, infos = zip(*results)
        self.states = list(next_states)
        # Auto reset finished envs
        for i, done in enumerate(dones):
            if done:
                self.states[i] = self.envs[i].reset()
        return self.states, list(rewards), list(dones), list(infos)

    def get_action_masks(self) -> torch.Tensor:
        """Returns (n_envs, cols*rows) boolean tensor."""
        masks = [env.get_action_mask() for env in self.envs]
        return torch.tensor(np.stack(masks), dtype=torch.bool, device=self.device)
