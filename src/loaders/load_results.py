from __future__ import annotations

from pathlib import Path
import re
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
        rows.append(
            {
                "driver_id": d.get("id"),
                "driver_name": d.get("full_name")
                or f"{(d.get('first_name','') or '').strip()} {(d.get('last_name','') or '').strip()}".strip(),
                "finish_pos": r.get("position"),
                "start_pos": r.get("start_position"),
                "status": r.get("status"),
                "driver_rating": r.get("driver_rating"),
                "laps_led": r.get("laps_led", 0),
                "fastest_laps": r.get("fastest_laps", 0),
                "laps_completed": r.get("laps_completed"),
            }
        )

    df = pd.DataFrame(rows)

    # ensure numeric
    for c in ["finish_pos", "start_pos", "driver_rating", "laps_led", "fastest_laps", "laps_completed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _infer_season_from_path(p: Path) -> str:
    """
    Try to infer a 4-digit season/year from filename or any parent folder.
    Examples it will handle:
      race_results/2024.json
      race_results/2024/atlanta_1.json
      race_results/atlanta_2024_r1.json
    Fallback: file stem
    """
    year_re = re.compile(r"(19|20)\d{2}")

    # check filename first
    m = year_re.search(p.stem)
    if m:
        return m.group(0)

    # then check parent folder names (closest first)
    for part in [x.name for x in p.parents]:
        m = year_re.search(part)
        if m:
            return m.group(0)

    return p.stem


def load_results_folder(folder: str | Path) -> pd.DataFrame:
    """
    Load all result json files in a race_results folder (recursive) and tag with season.

    IMPORTANT: previously this only used folder.glob("*.json") (non-recursive),
    which breaks when you store results as race_results/2024/*.json, etc.
    """
    folder = Path(folder)
    if not folder.exists():
        return pd.DataFrame()

    all_dfs = []

    # recursive search to support year subfolders
    for p in sorted(folder.rglob("*.json")):
        try:
            df = load_race_results(p)

            # skip non-result jsons (or empty parses)
            if df is None or df.empty:
                continue

            df["season"] = _infer_season_from_path(p)
            df["source_file"] = p.name
            all_dfs.append(df)

        except Exception:
            # if one file is malformed, don't kill the entire week
            continue

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)
