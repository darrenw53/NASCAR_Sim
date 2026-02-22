
from __future__ import annotations

import pandas as pd

def blend_sim_with_fanduel(sim_summary: pd.DataFrame, fd_pool: pd.DataFrame, blend: float) -> pd.DataFrame:
    """Return a projection table with final_proj = blend*sim + (1-blend)*fd_fppg."""
    blend = float(blend)
    blend = max(0.0, min(1.0, blend))

    out = sim_summary.merge(fd_pool, on="driver_name", how="left")
    out["fd_fppg"] = pd.to_numeric(out["fd_fppg"], errors="coerce")
    out["salary"] = pd.to_numeric(out["salary"], errors="coerce")
    out["final_proj"] = blend*out["sim_fd_points"] + (1.0-blend)*out["fd_fppg"].fillna(out["sim_fd_points"])
    # simple value
    out["value_per_1k"] = out["final_proj"] / (out["salary"] / 1000.0)
    return out
