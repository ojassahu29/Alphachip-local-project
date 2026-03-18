"""
AlphaChip-Lite: Netlist Parser
================================
Loads netlists from JSON or the circuit_training protobuf text format.
Outputs a structured representation used by the environment.
"""

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Macro:
    id: int
    name: str
    width: float
    height: float
    # list of macro IDs this macro is connected to
    connections: List[int] = field(default_factory=list)
    # initial placement (may be None = unplaced)
    x: Optional[float] = None
    y: Optional[float] = None
    fixed: bool = False


@dataclass
class Net:
    weight: float
    pins: List[int]  # macro IDs involved in this net


@dataclass
class Canvas:
    width: float
    height: float
    columns: int
    rows: int

    @property
    def cell_width(self) -> float:
        return self.width / self.columns

    @property
    def cell_height(self) -> float:
        return self.height / self.rows


@dataclass
class Netlist:
    name: str
    canvas: Canvas
    macros: List[Macro]
    nets: List[Net]

    @property
    def num_macros(self) -> int:
        return len(self.macros)

    def adjacency_matrix(self) -> np.ndarray:
        """Weighted NxN adjacency matrix (N = num_macros)."""
        n = self.num_macros
        adj = np.zeros((n, n), dtype=np.float32)
        for net in self.nets:
            w = net.weight
            for i in range(len(net.pins)):
                for j in range(i + 1, len(net.pins)):
                    a, b = net.pins[i], net.pins[j]
                    if 0 <= a < n and 0 <= b < n:
                        adj[a][b] += w
                        adj[b][a] += w
        return adj

    def adjacency_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(self.adjacency_matrix(), dtype=torch.float32, device=device)

    def macro_feature_matrix(self) -> np.ndarray:
        """
        Returns an (N, 2) matrix of normalised [width, height] per macro.
        Use for node features in the GNN.
        """
        feats = np.array([[m.width, m.height] for m in self.macros], dtype=np.float32)
        # normalise by canvas size
        feats[:, 0] /= self.canvas.width
        feats[:, 1] /= self.canvas.height
        return feats

    def macro_feature_tensor(self, device: torch.device) -> torch.Tensor:
        return torch.tensor(self.macro_feature_matrix(), dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# JSON loader (our format)
# ---------------------------------------------------------------------------

def load_json(filepath: str) -> Netlist:
    """Load a netlist from our JSON format."""
    with open(filepath) as f:
        data = json.load(f)

    canvas = Canvas(
        width=data["canvas"]["width"],
        height=data["canvas"]["height"],
        columns=data["canvas"]["columns"],
        rows=data["canvas"]["rows"],
    )

    macros = [
        Macro(
            id=m["id"],
            name=m["name"],
            width=m["width"],
            height=m["height"],
            connections=m.get("connections", []),
        )
        for m in data["macros"]
    ]

    nets = [
        Net(weight=n["weight"], pins=n["pins"])
        for n in data["nets"]
    ]

    return Netlist(name=data.get("name", "unnamed"), canvas=canvas, macros=macros, nets=nets)


# ---------------------------------------------------------------------------
# circuit_training protobuf text loader (subset)
# ---------------------------------------------------------------------------

def load_prototext(filepath: str) -> Netlist:
    """
    Load a circuit_training netlist.pb.txt file.
    Parses a subset of the proto: MACRO nodes for macro placement.
    """
    with open(filepath, "r") as f:
        content = f.read()

    # Parse canvas metadata
    canvas_width, canvas_height = 356.592, 356.640
    columns, rows = 35, 33

    metadata_block = re.search(r'name\s*:\s*"__metadata__".*?(?=\nnode\s*\{|\Z)', content, re.DOTALL)
    if metadata_block:
        blk = metadata_block.group(0)
        cw = re.search(r'key\s*:\s*"canvas_width".*?f\s*:\s*([\d.]+)', blk, re.DOTALL)
        ch = re.search(r'key\s*:\s*"canvas_height".*?f\s*:\s*([\d.]+)', blk, re.DOTALL)
        cl = re.search(r'key\s*:\s*"columns".*?i\s*:\s*(\d+)', blk, re.DOTALL)
        rw = re.search(r'key\s*:\s*"rows".*?i\s*:\s*(\d+)', blk, re.DOTALL)
        if cw: canvas_width = float(cw.group(1))
        if ch: canvas_height = float(ch.group(1))
        if cl: columns = int(cl.group(1))
        if rw: rows = int(rw.group(1))

    canvas = Canvas(width=canvas_width, height=canvas_height, columns=columns, rows=rows)

    # Split into node blocks
    blocks = re.split(r'\nnode\s*\{', content)
    macros: List[Macro] = []
    macro_id = 0

    for blk in blocks[1:]:
        name_m = re.search(r'name\s*:\s*"([^"]*)"', blk)
        type_m = re.search(r'type\s*:\s*(\S+)', blk)
        if not name_m or not type_m:
            continue
        name = name_m.group(1)
        ntype = type_m.group(1)

        if name == '__metadata__':
            continue
        if ntype not in ('MACRO', 'macro'):
            continue

        w_m = re.search(r'key\s*:\s*"width".*?f\s*:\s*([\d.e+-]+)', blk, re.DOTALL)
        h_m = re.search(r'key\s*:\s*"height".*?f\s*:\s*([\d.e+-]+)', blk, re.DOTALL)
        width = float(w_m.group(1)) if w_m else 10.0
        height = float(h_m.group(1)) if h_m else 10.0

        macros.append(Macro(id=macro_id, name=name, width=width, height=height))
        macro_id += 1

    # Build very basic connectivity: any two macros share a net if connected via a shared pin
    # (simplified — full parsing would need MACRO_PIN → net traversal)
    nets: List[Net] = []
    # For the prototext we build a trivial fully-connected net to allow the environment to run
    if len(macros) <= 20:
        for i in range(len(macros)):
            for j in range(i + 1, len(macros)):
                nets.append(Net(weight=1.0, pins=[i, j]))

    return Netlist(name=Path(filepath).stem, canvas=canvas, macros=macros, nets=nets)


# ---------------------------------------------------------------------------
# Auto-detect format and load
# ---------------------------------------------------------------------------

def load(filepath: str) -> Netlist:
    """Auto-detect format and load a netlist."""
    p = Path(filepath)
    if p.suffix == ".json":
        return load_json(filepath)
    elif p.suffix in (".txt", ".pb"):
        return load_prototext(filepath)
    else:
        # try JSON first, then proto
        try:
            return load_json(filepath)
        except Exception:
            return load_prototext(filepath)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    fp = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "toy_netlist.json")
    nl = load(fp)
    print(f"Loaded netlist: {nl.name}")
    print(f"  Canvas: {nl.canvas.width:.1f} x {nl.canvas.height:.1f}  ({nl.canvas.columns}x{nl.canvas.rows} grid)")
    print(f"  Macros: {nl.num_macros}")
    print(f"  Nets:   {len(nl.nets)}")
    adj = nl.adjacency_matrix()
    print(f"  Adj matrix shape: {adj.shape}, max weight: {adj.max():.1f}")
