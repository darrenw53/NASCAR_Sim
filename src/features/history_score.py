
from __future__ import annotations

import pandas as pd
from .normalize import percentile_rank

def build_track_history_score(results_all_years: pd.DataFrame) -> pd.DataFrame:
    """Aggregate historical results at this track into a 0..1 score (higher is better)."""
    if results_all_years is None or results_all_years.empty:
        return pd.DataFrame(columns=["driver_name","history_score","history_avg_finish","history_starts"])
    g = results_all_years.groupby(["driver_name"], dropna=False)
    agg = g.agg(
        history_avg_finish=("finish_pos","mean"),
        history_avg_start=("start_pos","mean"),
        history_avg_rating=("driver_rating","mean"),
        history_laps_led=("laps_led","mean"),
        history_starts=("finish_pos","count"),
    ).reset_index()
    finish_component = percentile_rank(agg["history_avg_finish"], higher_is_better=False)
    rating_component = percentile_rank(agg["history_avg_rating"], higher_is_better=True).fillna(0.5)
    led_component = percentile_rank(agg["history_laps_led"], higher_is_better=True).fillna(0.5)
    agg["history_score"] = (0.65*finish_component + 0.25*rating_component + 0.10*led_component).clip(0,1)
    return agg[["driver_name","history_score","history_avg_finish","history_starts"]]
