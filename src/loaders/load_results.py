
from __future__ import annotations

from pathlib import Path
import pandas as pd
from ._common import load_json_with_optional_header

def load_race_results(path: str | Path) -> pd.DataFrame:
    """Load a NASCAR race result JSON (SportsRadar-style).
    Returns one row per driver with at least:
      driver_id, driver_name, finish_pos, start_pos, laps_led, fastest_laps, driver_rating, laps_completed
    """
    data = load_json_with_optional_header(path)
    rows = []
    for r in data.get("results", []):
        d = r.get("driver", {}) or {}
        rows.append({
            "driver_id": d.get("id"),
            "driver_name": d.get("full_name") or f"{d.get('first_name','').strip()} {d.get('last_name','').strip()}".strip(),
            "finish_pos": r.get("position"),
            "start_pos": r.get("start_position"),
            "status": r.get("status"),
            "driver_rating": r.get("driver_rating"),
            "laps_led": r.get("laps_led", 0),
            "fastest_laps": r.get("fastest_laps", 0),
            "laps_completed": r.get("laps_completed"),
        })
    df = pd.DataFrame(rows)
    # ensure numeric
    for c in ["finish_pos","start_pos","driver_rating","laps_led","fastest_laps","laps_completed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_results_folder(folder: str | Path) -> pd.DataFrame:
    """Load all json files in a race_results folder and tag with season (from filename)."""
    folder = Path(folder)
    all_dfs = []
    for p in sorted(folder.glob("*.json")):
        season = p.stem
        df = load_race_results(p)
        df["season"] = season
        all_dfs.append(df)
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)
