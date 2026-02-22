from __future__ import annotations

import re
from pathlib import Path
import json
import itertools
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from config.sim_defaults import DEFAULTS
from config.tracks import TRACK_TYPE_PRESETS

from src.loaders.load_results import load_results_folder
from src.loaders.load_practice import load_practice
from src.loaders.load_qualifying import load_qualifying
from src.loaders.load_fanduel import load_fanduel_csv

from src.features.history_score import build_track_history_score
from src.features.practice_score import build_practice_score
from src.features.qualifying_score import build_qualifying_score
from src.features.baseline_score import load_season_baseline

from src.utils.helpers import normalize_weights
from src.simulation.race_simulator import SimConfig, simulate_race
from src.fantasy.projections import blend_sim_with_fanduel


APP_TITLE = "SignalAI NASCAR Simulator + FanDuel Optimizer (Starter)"

ROOT = Path(__file__).parent
DATA_WEEKLY = ROOT / "data" / "weekly"
DATA_DRIVERS = ROOT / "data" / "drivers"
DATA_FD = ROOT / "data" / "fan_duel"  # optional folder


# -----------------------------
# Helpers
# -----------------------------
def name_key(s: str) -> str:
    """Normalize driver names to match across sources (handles Jr/Sr/initials/punctuation)."""
    s = (s or "").lower().strip()
    s = re.sub(r"[^\w\s]", "", s)                 # remove punctuation
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s) # remove suffix tokens
    s = " ".join(s.split())
    parts = s.split()
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]} {parts[-1]}"              # first + last


@st.cache_data(show_spinner=False)
def list_races() -> list[str]:
    if not DATA_WEEKLY.exists():
        return []
    return sorted([p.name for p in DATA_WEEKLY.iterdir() if p.is_dir()])


@st.cache_data(show_spinner=False)
def load_week(race_slug: str):
    week_dir = DATA_WEEKLY / race_slug

    notes_path = week_dir / "notes.json"
    notes = json.loads(notes_path.read_text(encoding="utf-8")) if notes_path.exists() else {}

    results_dir = week_dir / "race_results"
    results_all = load_results_folder(results_dir) if results_dir.exists() else pd.DataFrame()

    practice_path = week_dir / "practice.json"
    qual_path = week_dir / "qualifying.json"

    practice_df = load_practice(practice_path) if practice_path.exists() else pd.DataFrame()
    qual_df = load_qualifying(qual_path) if qual_path.exists() else pd.DataFrame()

    return notes, results_all, practice_df, qual_df


@st.cache_data(show_spinner=False)
def load_baseline():
    # Default to 2025 season strength as baseline
    p = DATA_DRIVERS / "2025_driver_stats.json"
    if not p.exists():
        return pd.DataFrame(columns=["driver_id", "driver_name", "baseline_score"])
    return load_season_baseline(p)


def load_fd_pool_with_fallback() -> Tuple[pd.DataFrame, str]:
    """
    FanDuel salary slate load:
    1) Prefer local file data/weekly/week_players.csv (this exists in your repo zip)
    2) Else try data/fan_duel/week_players.csv
    3) Else allow upload in sidebar
    """
    weekly_default = DATA_WEEKLY / "week_players.csv"
    fd_default = DATA_FD / "week_players.csv"

    if weekly_default.exists():
        return load_fanduel_csv(weekly_default), f"Loaded: {weekly_default.as_posix()}"
    if fd_default.exists():
        return load_fanduel_csv(fd_default), f"Loaded: {fd_default.as_posix()}"

    return pd.DataFrame(), "No local FanDuel slate found"


def build_driver_table(
    notes,
    results_all,
    practice_df,
    qual_df,
    baseline_df,
    fd_df,
    weights: dict[str, float],
) -> pd.DataFrame:
    hist = build_track_history_score(results_all).copy()
    prac = build_practice_score(practice_df).copy()
    qual = build_qualifying_score(qual_df).copy()
    base = baseline_df.copy()

    # Start from FanDuel pool so optimizer always has salary data
    df = fd_df.copy()
    df["name_key"] = df["driver_name"].astype(str).apply(name_key)

    for t in (hist, prac, qual, base):
        if "driver_name" in t.columns:
            t["name_key"] = t["driver_name"].astype(str).apply(name_key)

    # Merge via normalized key (prevents minor name formatting mismatches)
    if "driver_name" in hist.columns:
        df = df.merge(hist.drop(columns=["driver_name"]), on="name_key", how="left")
    else:
        df = df.merge(hist, on="name_key", how="left")

    if "driver_name" in prac.columns:
        df = df.merge(prac.drop(columns=["driver_name"]), on="name_key", how="left", suffixes=("", "_p"))
    else:
        df = df.merge(prac, on="name_key", how="left", suffixes=("", "_p"))

    if "driver_name" in qual.columns:
        df = df.merge(qual.drop(columns=["driver_name"]), on="name_key", how="left", suffixes=("", "_q"))
    else:
        df = df.merge(qual, on="name_key", how="left", suffixes=("", "_q"))

    if "driver_name" in base.columns:
        df = df.merge(base.drop(columns=["driver_name"]), on="name_key", how="left", suffixes=("", "_b"))
    else:
        df = df.merge(base, on="name_key", how="left", suffixes=("", "_b"))

    df = df.drop(columns=["name_key"])

    # Neutral fill for missing scores
    for c in ["history_score", "practice_score", "qual_score", "baseline_score"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.5)

    w = normalize_weights(weights)
    df["composite_score"] = (
        w["past"] * df["history_score"]
        + w["practice"] * df["practice_score"]
        + w["qual"] * df["qual_score"]
        + w["baseline"] * df["baseline_score"]
    ).clip(0, 1)

    keep = [
        "fd_id",
        "driver_name",
        "salary",
        "fd_fppg",
        "composite_score",
        "history_score",
        "practice_score",
        "qual_score",
        "baseline_score",
        "history_avg_finish",
        "history_starts",
        "practice_pos",
        "practice_speed",
        "qual_pos",
        "qual_speed",
        "season_avg_finish",
        "season_starts",
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    df = df[keep].copy()

    return df.sort_values("composite_score", ascending=False).reset_index(drop=True)


def _objective_col(projections: pd.DataFrame) -> str:
    """Prefer upside, fallback to projection."""
    if "p90_fd_points" in projections.columns:
        return "p90_fd_points"
    return "final_proj"


def optimize_multiple_lineups(
    projections: pd.DataFrame,
    salary_cap: int,
    lineup_size: int,
    pool_size: int,
    lineup_count: int,
) -> pd.DataFrame:
    """
    Enumerate all combinations from a reduced pool and return top N lineups by objective
    (p90_fd_points preferred, else final_proj). All lineups are inherently unique (different combos).
    """
    obj_col = _objective_col(projections)

    pool = projections.dropna(subset=["salary", obj_col]).copy()
    pool["salary"] = pd.to_numeric(pool["salary"], errors="coerce")
    pool[obj_col] = pd.to_numeric(pool[obj_col], errors="coerce")
    pool = pool.dropna(subset=["salary", obj_col])

    pool = pool.sort_values(obj_col, ascending=False).head(int(pool_size)).reset_index(drop=True)

    if len(pool) < lineup_size:
        raise ValueError("Not enough drivers in optimizer pool after filters.")

    salaries = pool["salary"].to_numpy()
    scores = pool[obj_col].to_numpy()

    valid = []
    idxs = range(len(pool))

    for comb in itertools.combinations(idxs, int(lineup_size)):
        sal = salaries[list(comb)].sum()
        if sal > salary_cap:
            continue
        score = scores[list(comb)].sum()
        valid.append((score, sal, comb))

    if not valid:
        raise ValueError("No valid lineups found under salary cap. Increase pool size or raise cap.")

    valid.sort(key=lambda x: x[0], reverse=True)
    top = valid[: int(lineup_count)]

    # Build lineup rows
    rows = []
    for rank, (score, sal, comb) in enumerate(top, start=1):
        chosen = pool.loc[list(comb), :].copy()

        # helpful ordering
        if "avg_finish" in chosen.columns:
            chosen = chosen.sort_values(["avg_finish", obj_col], ascending=[True, False])
        else:
            chosen = chosen.sort_values(obj_col, ascending=False)

        rows.append({
            "rank": rank,
            "objective_col": obj_col,
            "total_upside": float(score),
            "total_salary": int(sal),
            "drivers": chosen["driver_name"].tolist(),
            "fd_ids": chosen["fd_id"].tolist(),
        })

    out = pd.DataFrame(rows)
    return out


def build_fd_upload_template_like_csv(lineups_df: pd.DataFrame, lineup_size: int) -> pd.DataFrame:
    """
    Create a CSV that matches the attached FanDuel template structure:
    Driver x lineup_size, then "" and "Instructions" columns, plus 4 instruction rows.
    """
    # Columns: Driver,Driver,Driver,Driver,Driver,"","Instructions"
    cols = ["Driver"] * int(lineup_size) + ["", "Instructions"]

    instruction_rows = [
        ([""] * lineup_size) + ["", "1) Create a lineup by inserting any player from this list into their appropriate position columns"],
        ([""] * lineup_size) + ["", "2) You can paste either the Player ID + Name OR ID from their respective columns"],
        ([""] * lineup_size) + ["", "3) Any information to the right of the last position column will be ignored in the upload"],
        ([""] * lineup_size) + ["", "4) Maximum of 250 lineups per upload"],
    ]

    data_rows = []
    for _, r in lineups_df.iterrows():
        # FanDuel accepts "ID + Name" or just "ID" — we'll use "ID:Name" like the template allows
        cells = [f"{int(fid)}:{name}" for fid, name in zip(r["fd_ids"], r["drivers"])]
        cells = cells[:lineup_size] + ["", ""]
        data_rows.append(cells)

    df = pd.DataFrame(instruction_rows + data_rows, columns=cols)
    return df


def main():
    st.set_page_config(page_title="NASCAR Simulator", layout="wide")
    st.title(APP_TITLE)

    races = list_races()
    if not races:
        st.error("No weekly races found in data/weekly. Add a folder per race.")
        return

    with st.sidebar:
        st.header("Week")
        race_slug = st.selectbox("Select race folder", races, index=0)

        st.header("Weights (slider-driven)")
        past_weight = st.slider("Past track results", 0.0, 1.0, float(DEFAULTS["past_weight"]), 0.01)
        practice_weight = st.slider("Practice", 0.0, 1.0, float(DEFAULTS["practice_weight"]), 0.01)
        qual_weight = st.slider("Qualifying", 0.0, 1.0, float(DEFAULTS["qual_weight"]), 0.01)
        baseline_weight = st.slider("Season baseline", 0.0, 1.0, float(DEFAULTS["baseline_weight"]), 0.01)

        st.header("Simulation")
        n_sims = st.number_input("Simulations", min_value=2000, max_value=200000, value=int(DEFAULTS["n_sims"]), step=1000)
        rng_seed = st.number_input("RNG seed (repeatability)", min_value=0, max_value=9999999, value=int(DEFAULTS["rng_seed"]), step=1)

        blend_sim_vs_fd = st.slider("Blend: Sim vs FanDuel FPPG", 0.0, 1.0, float(DEFAULTS["blend_sim_vs_fd"]), 0.01)
        performance_sd = st.slider("Performance randomness (SD)", 0.20, 1.50, float(DEFAULTS["performance_sd"]), 0.01)

        dnf_base = st.slider("Base DNF probability", 0.00, 0.35, float(DEFAULTS["dnf_base"]), 0.01)

        dominator_top_k = st.slider("Dominator pool (top K)", 3, 12, int(DEFAULTS["dominator_top_k"]), 1)
        dominator_strength = st.slider("Dominator concentration", 0.60, 2.50, float(DEFAULTS["dominator_strength"]), 0.05)

        st.header("FanDuel Slate")
        fd_df_local, fd_label = load_fd_pool_with_fallback()
        st.caption(fd_label)

        uploaded = st.file_uploader("Upload FanDuel week_players.csv (optional override)", type=["csv"])
        if uploaded is not None:
            # FanDuel loader expects a path; write to temp then read
            tmp = ROOT / "data" / "_tmp_uploaded_week_players.csv"
            tmp.write_bytes(uploaded.getvalue())
            try:
                fd_df_local = load_fanduel_csv(tmp)
                fd_label = f"Uploaded: {uploaded.name}"
            except Exception as e:
                st.error(f"Could not parse uploaded FanDuel CSV: {e}")
                fd_df_local = pd.DataFrame()

        st.header("Optimizer Settings")
        salary_cap = st.number_input("Salary cap", min_value=30000, max_value=70000, value=int(DEFAULTS["salary_cap"]), step=1000)
        lineup_size = st.number_input("Lineup size", min_value=4, max_value=6, value=int(DEFAULTS["lineup_size"]), step=1)
        optimizer_pool = st.slider("Optimizer player pool (top N by upside/proj)", 10, 44, int(DEFAULTS["optimizer_pool"]), 1)

        lineup_count = st.selectbox("Number of unique lineups", options=[1, 25, 50, 100, 150], index=1)

        run = st.button("Run simulation")

    notes, results_all, practice_df, qual_df = load_week(race_slug)
    baseline_df = load_baseline()
    fd_df = fd_df_local

    if fd_df is None or fd_df.empty:
        st.error("FanDuel slate not found. Upload week_players.csv in the sidebar to continue.")
        st.stop()

    track_type = (notes.get("track_type") or "").strip().lower()
    total_laps = int(notes.get("total_laps") or 200)

    if track_type in TRACK_TYPE_PRESETS:
        preset = TRACK_TYPE_PRESETS[track_type]
        st.info(
            f"Track preset detected: **{track_type}** "
            f"(suggested SD={preset['performance_sd']}, DNF boost={preset['dnf_boost']})"
        )

    weights = {"past": past_weight, "practice": practice_weight, "qual": qual_weight, "baseline": baseline_weight}

    driver_table = build_driver_table(notes, results_all, practice_df, qual_df, baseline_df, fd_df, weights)

    colA, colB = st.columns([1.2, 1.0], gap="large")
    with colA:
        st.subheader("Driver pool + component scores")
        st.caption("Scores are normalized 0–1. Missing data is treated as neutral (0.50).")
        st.dataframe(driver_table, use_container_width=True, height=520)

    with colB:
        st.subheader("Composite score leaderboard")
        chart_df = driver_table[["driver_name", "composite_score"]].head(20).set_index("driver_name")
        st.bar_chart(chart_df)

        st.subheader("Weight normalization (auto)")
        st.write(normalize_weights(weights))

    if not run:
        st.stop()

    # Sim config
    dnf_prob = float(dnf_base)
    if track_type in TRACK_TYPE_PRESETS:
        dnf_prob = min(0.60, dnf_prob + float(TRACK_TYPE_PRESETS[track_type]["dnf_boost"]))

    cfg = SimConfig(
        n_sims=int(n_sims),
        rng_seed=int(rng_seed),
        performance_sd=float(performance_sd),
        total_laps=int(total_laps),
        dnf_prob=float(dnf_prob),
        dominator_top_k=int(dominator_top_k),
        dominator_strength=float(dominator_strength),
    )

    st.divider()
    st.subheader("Simulation results")

    sim_input = driver_table[["driver_name", "composite_score"]].copy()
    if "qual_pos" in driver_table.columns:
        sim_input["qual_pos"] = pd.to_numeric(driver_table["qual_pos"], errors="coerce")
    sim_input["driver_id"] = np.arange(len(sim_input))  # internal id for this run

    with st.spinner("Running Monte Carlo..."):
        sim_summary, sims_long = simulate_race(sim_input, cfg)

    projections = (
        blend_sim_with_fanduel(sim_summary, fd_df, blend=float(blend_sim_vs_fd))
        .sort_values("final_proj", ascending=False)
        .reset_index(drop=True)
    )

    c1, c2 = st.columns([1.0, 1.0], gap="large")
    with c1:
        st.subheader("Top projections (Sim blended with FanDuel FPPG)")
        show_cols = [
            "fd_id",
            "driver_name",
            "salary",
            "fd_fppg",
            "sim_fd_points",
            "final_proj",
            "value_per_1k",
            "win_pct",
            "top5_pct",
            "top10_pct",
            "avg_finish",
            "dnf_pct",
            "avg_laps_led",
            "p90_fd_points",
        ]
        for c in show_cols:
            if c not in projections.columns:
                projections[c] = np.nan
        st.dataframe(projections[show_cols].head(40), use_container_width=True, height=520)

    with c2:
        st.subheader("Win% (Top 15)")
        win_chart = projections[["driver_name", "win_pct"]].head(15).set_index("driver_name")
        st.bar_chart(win_chart)

        st.subheader("Top-5% (Top 15)")
        top5_chart = projections[["driver_name", "top5_pct"]].head(15).set_index("driver_name")
        st.bar_chart(top5_chart)

    st.divider()
    st.subheader("FanDuel optimizer (multi-lineup)")

    # --- Include / Exclude driver controls ---
    st.caption("Use these to force lock/ban drivers from the optimizer pool.")
    all_names = projections["driver_name"].dropna().astype(str).tolist()

    colx, coly = st.columns(2)
    with colx:
        include_names = st.multiselect(
            "Include ONLY these drivers (optional)",
            options=all_names,
            default=[],
        )
    with coly:
        exclude_names = st.multiselect(
            "Exclude these drivers (optional)",
            options=all_names,
            default=[],
        )

    opt_df = projections.copy()
    if include_names:
        opt_df = opt_df[opt_df["driver_name"].isin(include_names)].copy()
    if exclude_names:
        opt_df = opt_df[~opt_df["driver_name"].isin(exclude_names)].copy()

    try:
        # Build many unique lineups, ranked by upside desc
        lineups_df = optimize_multiple_lineups(
            opt_df,
            salary_cap=int(salary_cap),
            lineup_size=int(lineup_size),
            pool_size=int(optimizer_pool),
            lineup_count=int(lineup_count),
        )

        st.caption(f"Objective: `{lineups_df['objective_col'].iloc[0]}` (higher = more upside).")
        st.dataframe(lineups_df[["rank", "total_upside", "total_salary", "drivers"]], use_container_width=True)

        if int(lineup_count) >= 25:
            upload_df = build_fd_upload_template_like_csv(lineups_df, lineup_size=int(lineup_size))
            st.download_button(
                "Download FanDuel Upload CSV (template format)",
                data=upload_df.to_csv(index=False),
                file_name="fanduel_lineup_upload_template_format.csv",
                mime="text/csv",
            )
        else:
            st.info("Select 25+ lineups to enable FanDuel upload CSV output (template format).")

    except Exception as e:
        st.warning(f"Optimizer did not return lineups: {e}")

    st.divider()
    st.subheader("Download artifacts")

    out_proj = ROOT / "outputs" / "projections" / f"{race_slug}_projections.csv"
    out_sims = ROOT / "outputs" / "sims" / f"{race_slug}_sims_long.csv"

    out_proj.parent.mkdir(parents=True, exist_ok=True)
    out_sims.parent.mkdir(parents=True, exist_ok=True)

    projections.to_csv(out_proj, index=False)
    sims_long.to_csv(out_sims, index=False)

    st.download_button(
        "Download projections CSV",
        data=out_proj.read_bytes(),
        file_name=out_proj.name,
        mime="text/csv",
    )
    st.download_button(
        "Download sim-long CSV",
        data=out_sims.read_bytes(),
        file_name=out_sims.name,
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
