
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
from .normalize import percentile_rank

def load_season_baseline(driver_stats_path: str | Path) -> pd.DataFrame:
    data = json.loads(Path(driver_stats_path).read_text(encoding="utf-8", errors="ignore"))
    rows=[]
    for d in data.get("drivers", []) or []:
        race_splits = d.get("race_splits", []) or []
        starts = 0.0
        finish_sum = 0.0
        top10 = 0.0
        wins = 0.0
        laps_led = 0.0
        for rs in race_splits:
            st = rs.get("starts") or 0
            af = rs.get("avg_finish_position")
            if af is None:
                continue
            starts += st
            finish_sum += st * float(af)
            top10 += float(rs.get("top_10") or 0)
            wins += float(rs.get("wins") or 0)
            laps_led += float(rs.get("laps_led") or 0)

        avg_finish = (finish_sum / starts) if starts > 0 else None
        rows.append({
            "driver_name": d.get("full_name") or f"{d.get('first_name','').strip()} {d.get('last_name','').strip()}".strip(),
            "season_starts": starts,
            "season_avg_finish": avg_finish,
            "season_wins": wins,
            "season_top10": top10,
            "season_laps_led": laps_led,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["driver_name","baseline_score","season_avg_finish"])
    finish_component = percentile_rank(df["season_avg_finish"], higher_is_better=False)
    wins_component = percentile_rank(df["season_wins"], higher_is_better=True).fillna(0.5)
    top10_component = percentile_rank(df["season_top10"], higher_is_better=True).fillna(0.5)
    led_component = percentile_rank(df["season_laps_led"], higher_is_better=True).fillna(0.5)
    df["baseline_score"] = (0.60*finish_component + 0.20*wins_component + 0.10*top10_component + 0.10*led_component).clip(0,1)
    return df[["driver_name","baseline_score","season_avg_finish","season_starts"]]
