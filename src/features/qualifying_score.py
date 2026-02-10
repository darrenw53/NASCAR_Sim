
from __future__ import annotations

import pandas as pd
from .normalize import percentile_rank

def build_qualifying_score(qual_df: pd.DataFrame) -> pd.DataFrame:
    if qual_df is None or qual_df.empty:
        return pd.DataFrame(columns=["driver_name","qual_score","qual_pos","qual_speed","qual_lap_time"])
    pos = percentile_rank(qual_df["qual_pos"], higher_is_better=False)
    speed = percentile_rank(qual_df["qual_speed"], higher_is_better=True)
    lap_time = percentile_rank(qual_df["qual_lap_time"], higher_is_better=False)
    q = qual_df.copy()
    q["qual_score"] = (0.55*pos.fillna(0.5) + 0.30*speed.fillna(0.5) + 0.15*lap_time.fillna(0.5)).clip(0,1)
    return q[["driver_name","qual_score","qual_pos","qual_speed","qual_lap_time"]]
