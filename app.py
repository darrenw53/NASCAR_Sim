import os
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import streamlit as st

from config.sim_defaults import DEFAULTS
from config.tracks import TRACK_TYPE_PRESETS
from src.loaders.week_loader import load_week
from src.loaders.baseline_loader import load_baseline
from src.loaders.load_fanduel import load_fanduel_csv
from src.simulation.sim_engine import run_simulation
from src.scoring.fanduel_scoring import add_fanduel_points
from src.optimizer.fd_optimizer import optimize_lineups, format_upload_csv


APP_TITLE = "NASCAR Simulation + FanDuel Optimizer"


# -----------------------------
# FanDuel slate selection/upload
# -----------------------------
@st.cache_data(show_spinner=False)
def _read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(pd.io.common.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def _list_local_fd_csvs(fd_dir: Path) -> List[Path]:
    if not fd_dir.exists():
        return []
    return sorted([p for p in fd_dir.glob("*.csv")])


def choose_fd_pool() -> Tuple[pd.DataFrame, str]:
    """
    Returns (fd_df, label) where fd_df contains at least:
    - name (driver name)
    - salary
    - id (FanDuel player id)
    - fppg (optional but recommended)
    """

    st.caption("Upload the current FanDuel slate file OR select a CSV already in /data/fan_duel.")

    fd_dir = Path("data") / "fan_duel"
    local_files = _list_local_fd_csvs(fd_dir)

    mode = st.radio(
        "FanDuel slate source",
        options=["Upload CSV", "Use local CSV"],
        horizontal=True,
        index=0,
    )

    if mode == "Upload CSV":
        up = st.file_uploader(
            "Upload FanDuel CSV (salary slate export)",
            type=["csv"],
            accept_multiple_files=False,
        )
        if up is None:
            return pd.DataFrame(), "No slate uploaded"

        try:
            fd_df = _read_csv_bytes(up.getvalue())
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")
            return pd.DataFrame(), "Upload failed"

        try:
            fd_df = load_fanduel_csv(fd_df)  # supports df input
        except TypeError:
            # older loader signature expects a path; fallback: write temp then read
            tmp = Path("data") / "fan_duel" / "_uploaded_tmp.csv"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(up.getvalue())
            fd_df = load_fanduel_csv(tmp)

        return fd_df, f"Uploaded: {up.name}"

    # Use local CSV
    if not local_files:
        st.info("No local FanDuel CSV files found in data/fan_duel/.")
        return pd.DataFrame(), "No local slate found"

    labels = [p.name for p in local_files]
    choice = st.selectbox("Select local FanDuel CSV", labels, index=0)
    chosen_path = fd_dir / choice

    try:
        fd_df = load_fanduel_csv(chosen_path)
    except Exception as e:
        st.error(f"Could not load {chosen_path}: {e}")
        return pd.DataFrame(), "Local load failed"

    return fd_df, f"Local: {choice}"


# -----------------------------
# Week selection utilities
# -----------------------------
def list_week_folders(weekly_root: Path) -> List[str]:
    if not weekly_root.exists():
        return []
    return sorted([p.name for p in weekly_root.iterdir() if p.is_dir()])


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    weekly_root = Path("data") / "weekly"
    week_folders = list_week_folders(weekly_root)

    if not week_folders:
        st.error("No week folders found in data/weekly/. Add a week folder (e.g., 2026_daytona_500).")
        st.stop()

    race_slug = st.selectbox("Select week / race folder", week_folders, index=max(0, len(week_folders) - 1))

    # -----------------------------
    # Sidebar controls
    # -----------------------------
    with st.sidebar:
        st.header("Weights")

        w_history = st.slider("Past results weight", 0.0, 1.0, float(DEFAULTS["w_history"]), 0.01)
        w_practice = st.slider("Practice weight", 0.0, 1.0, float(DEFAULTS["w_practice"]), 0.01)
        w_qual = st.slider("Qualifying weight", 0.0, 1.0, float(DEFAULTS["w_qual"]), 0.01)
        w_season = st.slider("Season strength weight", 0.0, 1.0, float(DEFAULTS["w_season"]), 0.01)

        st.header("Simulation")

        n_sims = st.number_input("Simulations", min_value=2000, max_value=200000, value=int(DEFAULTS["n_sims"]), step=1000)
        rng_seed = st.number_input("RNG seed (repeatability)", min_value=0, max_value=9999999, value=int(DEFAULTS["rng_seed"]), step=1)

        blend_sim_vs_fd = st.slider("Blend: Sim vs FanDuel FPPG", 0.0, 1.0, float(DEFAULTS["blend_sim_vs_fd"]), 0.01)
        performance_sd = st.slider("Performance randomness (SD)", 0.20, 1.50, float(DEFAULTS["performance_sd"]), 0.01)

        dnf_base = st.slider("Base DNF probability", 0.00, 0.35, float(DEFAULTS["dnf_base"]), 0.01)

        dominator_top_k = st.slider("Dominator pool (top K)", 3, 12, int(DEFAULTS["dominator_top_k"]), 1)
        dominator_strength = st.slider("Dominator concentration", 0.60, 2.50, float(DEFAULTS["dominator_strength"]), 0.05)

        st.header("FanDuel")
        fd_df, fd_label = choose_fd_pool()
        if fd_df.empty:
            st.warning("Upload/select a FanDuel slate CSV to enable optimizer + salary data.")

        st.header("Optimizer")
        salary_cap = st.number_input("Salary cap", min_value=30000, max_value=70000, value=int(DEFAULTS["salary_cap"]), step=1000)
        lineup_size = st.number_input("Lineup size", min_value=4, max_value=6, value=int(DEFAULTS["lineup_size"]), step=1)
        optimizer_pool = st.slider("Optimizer player pool (top N by proj)", 10, 44, int(DEFAULTS["optimizer_pool"]), 1)

        lineup_count = st.selectbox(
            "Number of unique lineups",
            options=[1, 25, 50, 100, 150],
            index=1
        )

        run = st.button("Run simulation")

    # -----------------------------
    # Load week + baseline
    # -----------------------------
    notes, results_all, practice_df, qual_df = load_week(race_slug)
    baseline_df = load_baseline()

    if fd_df is None or fd_df.empty:
        st.error("FanDuel slate CSV is missing. Upload it in the sidebar (FanDuel section) to continue.")
        st.stop()

    track_type = (notes.get("track_type") or "").strip().lower()
    total_laps = int(notes.get("total_laps") or 200)

    if track_type in TRACK_TYPE_PRESETS:
        preset = TRACK_TYPE_PRESETS[track_type]
        st.info(
            f"Track preset detected: **{track_type}** "
            f"(suggested SD={preset['performance_sd']}, DNF boost={preset['dnf_boost']})"
        )

    # -----------------------------
    # Run simulation
    # -----------------------------
    if run:
        weights = {
            "history": w_history,
            "practice": w_practice,
            "qual": w_qual,
            "season": w_season,
        }

        sim_params = {
            "n_sims": int(n_sims),
            "rng_seed": int(rng_seed),
            "blend_sim_vs_fd": float(blend_sim_vs_fd),
            "performance_sd": float(performance_sd),
            "dnf_base": float(dnf_base),
            "dominator_top_k": int(dominator_top_k),
            "dominator_strength": float(dominator_strength),
            "track_type": track_type,
            "total_laps": total_laps,
        }

        with st.spinner("Running simulation..."):
            sim_df = run_simulation(
                notes=notes,
                results_all=results_all,
                practice_df=practice_df,
                qual_df=qual_df,
                baseline_df=baseline_df,
                fd_df=fd_df,
                weights=weights,
                sim_params=sim_params,
            )

        # Add FanDuel scoring columns (mean / p90 / etc if sim engine provides)
        sim_df = add_fanduel_points(sim_df)

        st.subheader("Simulation Results")
        st.dataframe(sim_df, use_container_width=True)

        # -----------------------------
        # Optimizer pool selection
        # -----------------------------
        # Use the top N by upside proxy for optimization pool
        pool_df = sim_df.copy()

        # Determine upside column
        upside_col = None
        for c in ["p90_fd_points", "fd_points_p90", "fd_p90", "p90"]:
            if c in pool_df.columns:
                upside_col = c
                break
        if upside_col is None:
            # fallback to projected / blended
            for c in ["fd_points_proj", "fd_points", "proj_fd_points", "projection"]:
                if c in pool_df.columns:
                    upside_col = c
                    break

        if upside_col is None:
            st.error("Could not find an upside/projection column to optimize on.")
            st.stop()

        pool_df = pool_df.sort_values(by=upside_col, ascending=False).head(int(optimizer_pool)).reset_index(drop=True)

        st.subheader("Optimizer Pool")
        st.caption(f"Using top {optimizer_pool} drivers by `{upside_col}` for optimization.")
        st.dataframe(pool_df, use_container_width=True)

        # -----------------------------
        # Generate multiple unique lineups
        # -----------------------------
        st.subheader("Optimized Lineups (sorted by upside)")

        lineups_df = optimize_lineups(
            player_pool=pool_df,
            lineup_count=int(lineup_count),
            salary_cap=int(salary_cap),
            lineup_size=int(lineup_size),
            objective_col=upside_col,    # maximize upside
            uniqueness_min_diff=1,       # ensure each lineup differs by >= 1 driver
        )

        if lineups_df.empty:
            st.warning("No feasible lineups found. Try increasing optimizer pool or raising salary cap.")
            st.stop()

        st.dataframe(lineups_df, use_container_width=True)

        # -----------------------------
        # CSV export for 25+ lineups
        # -----------------------------
        if int(lineup_count) >= 25:
            upload_df = format_upload_csv(lineups_df, lineup_size=int(lineup_size))

            st.download_button(
                "Download FanDuel Upload CSV",
                data=upload_df.to_csv(index=False),
                file_name="fanduel_lineup_upload.csv",
                mime="text/csv",
            )

            st.caption("Upload CSV format mirrors FanDuel template: repeated `Driver` columns; values are `fd_id:Name`.")

        else:
            st.info("Select 25+ lineups to enable FanDuel upload CSV output.")


if __name__ == "__main__":
    main()
