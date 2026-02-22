
from __future__ import annotations

from typing import Dict

def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    w = {k: float(v) for k, v in weights.items()}
    s = sum(max(0.0, v) for v in w.values())
    if s <= 0:
        # even split
        n = len(w)
        return {k: 1.0/n for k in w}
    return {k: max(0.0, v)/s for k, v in w.items()}
