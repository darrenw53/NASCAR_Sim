from __future__ import annotations

from pathlib import Path
import pandas as pd
from ._common import load_json_with_optional_header


def _pick_session_index(practices: list[dict], session: int) -> int:
    """
    session = -1 means: pick the most recent practice that actually has results.
    If none have results, fall back to the last entry.
    """
    if not practices:
        return -1

    if session != -1:
        # clamp to valid range
        if session < 0:
            session = len(practices) + session
        return max(0, min(session, len(practices) - 1))

    # session == -1 (default): find latest with non-empty results
    for i in range(len(practices) - 1, -1, -1):
        results = practices[i].get("results", []) or []
        if len(results) > 0:
            return i

    # fallback
    return len(practices) - 1


def load_practice(path: str | Path, session: int = -1) -> pd.DataFrame:
    """Load practice results. session=-1 uses latest practice entry that has results."""
    data = load_json_with_optional_header(path)
    practices = data.get("practices", []) or []
    if not practices:
        return pd.DataFrame()

    idx = _pick_session_index(practices, session)
    if idx < 0:
        return pd.DataFrame()

    prac = practices[idx]
    rows = []
    for r in prac.get("results", []) or []:
        d = r.get("driver", {}) or {}
        rows.append(
            {
                "driver_id": d.get("id"),
                "driver_name": d.get("full_name")
                or f"{(d.get('first_name') or '').strip()} {(d.get('last_name') or '').strip()}".strip(),
                "practice_pos": r.get("position"),
                "practice_speed": r.get("speed"),
                "practice_lap_time": r.get("lap_time"),
                "laps_completed": r.get("laps_completed"),
                "status": r.get("status"),
            }
        )

    df = pd.DataFrame(rows)
    for c in ["practice_pos", "practice_speed", "practice_lap_time", "laps_completed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
