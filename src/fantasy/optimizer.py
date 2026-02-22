from __future__ import annotations

import itertools
import math
import random
from typing import Iterable, List, Tuple

import pandas as pd


def _pick_score_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Return (primary_upside_col, projection_col)."""
    # upside preference order
    for c in ("p90_fd_points", "p80_fd_points", "sim_fd_points", "final_proj"):
        if c in df.columns:
            upside = c
            break
    else:
        upside = "final_proj"

    # projection / mean preference order
    for c in ("final_proj", "sim_fd_points", "fd_fppg"):
        if c in df.columns:
            proj = c
            break
    else:
        proj = upside

    return upside, proj


def _clean_pool(df: pd.DataFrame, pool_size: int) -> pd.DataFrame:
    pool = df.dropna(subset=["salary"]).copy()
    pool["salary"] = pd.to_numeric(pool["salary"], errors="coerce")
    pool = pool.dropna(subset=["salary"])

    upside_col, proj_col = _pick_score_columns(pool)
    pool[upside_col] = pd.to_numeric(pool[upside_col], errors="coerce")
    pool[proj_col] = pd.to_numeric(pool[proj_col], errors="coerce")
    pool = pool.dropna(subset=[upside_col, proj_col])

    # Sort by upside first (ceiling), then by mean projection
    pool = pool.sort_values([upside_col, proj_col], ascending=[False, False]).head(int(pool_size)).reset_index(drop=True)
    pool.attrs["upside_col"] = upside_col
    pool.attrs["proj_col"] = proj_col
    return pool


def optimize_lineup(
    df: pd.DataFrame,
    salary_cap: int = 50000,
    lineup_size: int = 5,
    pool_size: int = 25
) -> pd.DataFrame:
    """Single best lineup (legacy).

    Expects at minimum: driver_name, salary, final_proj
    If present, also carries through: avg_finish, fd_fppg, sim_fd_points, value_per_1k, p90_fd_points

    Returns a DataFrame of chosen drivers with total_salary/total_proj repeated per row.
    """
    pool = _clean_pool(df, pool_size=pool_size)

    if len(pool) < lineup_size:
        raise ValueError("Not enough drivers in optimizer pool after filters.")

    upside_col, proj_col = pool.attrs["upside_col"], pool.attrs["proj_col"]

    best = None
    best_score = -1e18

    salaries = pool["salary"].to_numpy()
    scores = pool[proj_col].to_numpy()

    idxs = range(len(pool))
    for comb in itertools.combinations(idxs, lineup_size):
        sal = salaries[list(comb)].sum()
        if sal > salary_cap:
            continue
        score = scores[list(comb)].sum()
        if score > best_score:
            best_score = score
            best = comb

    if best is None:
        raise ValueError("No valid lineup found under salary cap. Try increasing pool size or check salaries.")

    # Include reference columns if they exist
    cols = ["driver_name", "salary", proj_col]
    optional = ["avg_finish", "fd_fppg", "sim_fd_points", "value_per_1k", upside_col, "fd_id"]
    for c in optional:
        if c in pool.columns and c not in cols:
            cols.append(c)

    chosen = pool.loc[list(best), cols].copy()
    chosen["total_salary"] = chosen["salary"].sum()
    chosen["total_proj"] = chosen[proj_col].sum()
    if upside_col in chosen.columns:
        chosen["total_upside"] = chosen[upside_col].sum()

    # nicer ordering if present
    if "avg_finish" in chosen.columns:
        chosen = chosen.sort_values(["avg_finish", proj_col], ascending=[True, False]).reset_index(drop=True)
    else:
        chosen = chosen.sort_values(proj_col, ascending=False).reset_index(drop=True)

    return chosen


def optimize_lineups(
    df: pd.DataFrame,
    n_lineups: int = 25,
    salary_cap: int = 50000,
    lineup_size: int = 5,
    pool_size: int = 25,
    max_exact_combos: int = 2_000_000,
    random_search_iters: int = 200_000,
    rng_seed: int | None = None,
) -> pd.DataFrame:
    """Generate multiple *unique* lineups ranked by upside (ceiling).

    - Uses exact brute force when the combination count is manageable.
    - Falls back to randomized search when the search space is too large.

    Returns one row per lineup with:
      driver_1..driver_k, fd_1..fd_k (if fd_id present), total_salary, total_proj, total_upside
    """
    if n_lineups <= 0:
        raise ValueError("n_lineups must be positive")

    pool = _clean_pool(df, pool_size=pool_size)
    if len(pool) < lineup_size:
        raise ValueError("Not enough drivers in optimizer pool after filters.")

    upside_col, proj_col = pool.attrs["upside_col"], pool.attrs["proj_col"]

    # arrays for speed
    salaries = pool["salary"].to_numpy(dtype=float)
    upside = pool[upside_col].to_numpy(dtype=float)
    proj = pool[proj_col].to_numpy(dtype=float)

    idxs = list(range(len(pool)))
    combo_count = math.comb(len(pool), lineup_size)

    # store unique lineups as frozenset of indices
    best_sets: List[Tuple[float, float, float, Tuple[int, ...]]] = []  # (upside, proj, salary, idx_tuple)

    def _add_candidate(comb_idx: Tuple[int, ...]):
        sal = float(salaries[list(comb_idx)].sum())
        if sal > salary_cap:
            return
        up = float(upside[list(comb_idx)].sum())
        pr = float(proj[list(comb_idx)].sum())
        best_sets.append((up, pr, sal, comb_idx))

    if rng_seed is not None:
        random.seed(rng_seed)

    if combo_count <= max_exact_combos:
        for comb_idx in itertools.combinations(idxs, lineup_size):
            _add_candidate(comb_idx)
    else:
        # Random search with mild bias toward high-upside drivers
        weights = upside - upside.min() + 1e-6
        weights = weights / weights.sum()
        for _ in range(int(random_search_iters)):
            # sample without replacement
            comb_idx = tuple(sorted(random.sample(idxs, lineup_size)))
            _add_candidate(comb_idx)

    if not best_sets:
        raise ValueError("No valid lineups found under salary cap. Try increasing pool size or check salaries.")

    # sort by upside then projection then cheaper salary (as tiebreaker)
    best_sets.sort(key=lambda x: (x[0], x[1], -x[2]), reverse=True)

    # de-duplicate
    seen = set()
    unique: List[Tuple[float, float, float, Tuple[int, ...]]] = []
    for up, pr, sal, comb_idx in best_sets:
        key = tuple(comb_idx)
        if key in seen:
            continue
        seen.add(key)
        unique.append((up, pr, sal, comb_idx))
        if len(unique) >= n_lineups:
            break

    # build output dataframe
    rows = []
    has_fd = "fd_id" in pool.columns
    for rank, (up, pr, sal, comb_idx) in enumerate(unique, start=1):
        picks = pool.loc[list(comb_idx)].copy()

        # sort by avg_finish if present else by upside/proj
        if "avg_finish" in picks.columns:
            picks = picks.sort_values(["avg_finish", upside_col, proj_col], ascending=[True, False, False])
        else:
            picks = picks.sort_values([upside_col, proj_col], ascending=[False, False])

        r = {"rank": rank, "total_salary": int(round(sal)), "total_proj": pr, "total_upside": up}
        for i, (_, row) in enumerate(picks.iterrows(), start=1):
            r[f"driver_{i}"] = str(row.get("driver_name", ""))
            if has_fd:
                r[f"fd_{i}"] = str(row.get("fd_id", ""))
            r[f"salary_{i}"] = float(row.get("salary", 0))
            r[f"proj_{i}"] = float(row.get(proj_col, 0))
            r[f"upside_{i}"] = float(row.get(upside_col, 0))
        rows.append(r)

    out = pd.DataFrame(rows)
    return out
