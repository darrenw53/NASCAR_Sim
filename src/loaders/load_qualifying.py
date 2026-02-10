
from __future__ import annotations

from pathlib import Path
import pandas as pd
from ._common import load_json_with_optional_header

def load_qualifying(path: str | Path) -> pd.DataFrame:
    data = load_json_with_optional_header(path)
    rows=[]
    for r in data.get("qualifying", []) or []:
        d = r.get("driver", {}) or {}
        rows.append({
            "driver_id": d.get("id"),
            "driver_name": d.get("full_name") or f"{d.get('first_name','').strip()} {d.get('last_name','').strip()}".strip(),
            "qual_pos": r.get("position"),
            "qual_speed": r.get("speed"),
            "qual_lap_time": r.get("lap_time"),
            "status": r.get("status"),
        })
    df=pd.DataFrame(rows)
    for c in ["qual_pos","qual_speed","qual_lap_time"]:
        if c in df.columns:
            df[c]=pd.to_numeric(df[c], errors="coerce")
    return df
