
from __future__ import annotations

import pandas as pd
from .normalize import percentile_rank

def build_practice_score(practice_df: pd.DataFrame) -> pd.DataFrame:
    if practice_df is None or practice_df.empty:
        return pd.DataFrame(columns=["driver_name","practice_score","practice_pos","practice_speed","practice_lap_time"])
    speed = percentile_rank(practice_df["practice_speed"], higher_is_better=True)
    lap_time = percentile_rank(practice_df["practice_lap_time"], higher_is_better=False)
    pos = percentile_rank(practice_df["practice_pos"], higher_is_better=False)
    df = practice_df.copy()
    df["practice_score"] = (0.55*speed.fillna(0.5) + 0.30*lap_time.fillna(0.5) + 0.15*pos.fillna(0.5)).clip(0,1)
    return df[["driver_name","practice_score","practice_pos","practice_speed","practice_lap_time"]]
