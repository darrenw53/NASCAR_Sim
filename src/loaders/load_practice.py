
from __future__ import annotations

from pathlib import Path
import pandas as pd
from ._common import load_json_with_optional_header

def load_practice(path: str | Path, session: int = -1) -> pd.DataFrame:
    """Load practice results. session=-1 uses latest practice entry."""
    data = load_json_with_optional_header(path)
    practices = data.get("practices", []) or []
    if not practices:
        return pd.DataFrame()
    prac = practices[session]
    rows=[]
    for r in prac.get("results", []) or []:
        d = r.get("driver", {}) or {}
        rows.append({
            "driver_id": d.get("id"),
            "driver_name": d.get("full_name") or f"{d.get('first_name','').strip()} {d.get('last_name','').strip()}".strip(),
            "practice_pos": r.get("position"),
            "practice_speed": r.get("speed"),
            "practice_lap_time": r.get("lap_time"),
            "laps_completed": r.get("laps_completed"),
            "status": r.get("status"),
        })
    df=pd.DataFrame(rows)
    for c in ["practice_pos","practice_speed","practice_lap_time","laps_completed"]:
        if c in df.columns:
            df[c]=pd.to_numeric(df[c], errors="coerce")
    return df
