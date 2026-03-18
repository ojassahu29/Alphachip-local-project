"""
AlphaChip-Lite: Logger
========================
Lightweight training logger with console + file output.
"""

import json
import os
import time
from typing import Dict, Optional


class Logger:
    """Simple logger that writes to console and a JSON-lines file."""

    def __init__(self, log_dir: str, name: str = "train"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{name}.jsonl")
        self.start_time = time.time()
        self._data = []

    def log(self, step: int, metrics: Dict[str, float], prefix: str = ""):
        elapsed = time.time() - self.start_time
        entry = {"step": step, "time": round(elapsed, 2), **metrics}
        self._data.append(entry)

        # Console
        parts = [f"{prefix}[{step:>5d}]" if prefix else f"[{step:>5d}]"]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:>8.4f}")
            else:
                parts.append(f"{k}={v}")
        parts.append(f"({elapsed:.0f}s)")
        print("  ".join(parts))

        # File
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_metric(self, key: str):
        return [d[key] for d in self._data if key in d]
