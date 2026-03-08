from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np
import pandas as pd


def _prepare_pool(
    df: pd.DataFrame,
    pool_size: int | None = None,
    value_weight: float = 0.20,
    p90_weight: float = 0.03,
) -> pd.DataFrame:
    pool = df.dropna(subset=["salary", "final_proj"]).copy()
    pool["salary"] = pd.to_numeric(pool["salary"], errors="coerce")
    pool["final_proj"] = pd.to_numeric(pool["final_proj"], errors="coerce")

    optional_numeric = [
        "avg_finish",
        "fd_fppg",
        "sim_fd_points",
        "value_per_1k",
        "p90_fd_points",
        "qual_pos",
    ]
    for col in optional_numeric:
        if col in pool.columns:
            pool[col] = pd.to_numeric(pool[col], errors="coerce")

    pool = pool.dropna(subset=["salary", "final_proj"]).copy()

    value_weight = float(max(0.0, value_weight))
    p90_weight = float(max(0.0, p90_weight))

    # Primary sort signal remains final_proj.
    # Value and p90 are only used as small, capped nudges so cheap drivers
    # do not get pushed too aggressively to the top of the optimizer pool.
    pool["optimizer_sort_score"] = pool["final_proj"].astype(float)

    if value_weight > 0.0 and "value_per_1k" in pool.columns:
        value_bonus = pd.to_numeric(pool["value_per_1k"], errors="coerce").fillna(0.0)

        # Center around the pool mean so this acts as a relative nudge only.
        value_bonus = value_bonus - value_bonus.mean()

        # Cap the impact so outlier cheap drivers do not dominate pool ordering.
        value_bonus = value_bonus.clip(lower=-0.75, upper=0.75)

        pool["optimizer_sort_score"] += value_weight * value_bonus

    if p90_weight > 0.0 and "p90_fd_points" in pool.columns:
        p90_bonus = pd.to_numeric(pool["p90_fd_points"], errors="coerce").fillna(0.0)

        # Center and cap ceiling influence as a mild secondary factor.
        p90_bonus = p90_bonus - p90_bonus.mean()
        p90_bonus = p90_bonus.clip(lower=-3.0, upper=3.0)

        pool["optimizer_sort_score"] += p90_weight * p90_bonus

    pool = pool.sort_values(
        ["optimizer_sort_score", "final_proj"],
        ascending=[False, False]
    ).reset_index(drop=True)

    if pool_size is not None:
        pool_size = int(pool_size)
        if pool_size > 0:
            pool = pool.head(pool_size).reset_index(drop=True)

    return pool


def _combo_arrays(pool_len: int, lineup_size: int) -> np.ndarray:
    return np.array(list(itertools.combinations(range(pool_len), int(lineup_size))), dtype=np.int16)


def _build_combo_table(
    pool: pd.DataFrame,
    salary_cap: int,
    lineup_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    combos = _combo_arrays(len(pool), int(lineup_size))
    if combos.size == 0:
        return combos, np.array([], dtype=float), np.array([], dtype=float)

    salaries = pool["salary"].to_numpy(dtype=float)
    projs = pool["final_proj"].to_numpy(dtype=float)

    combo_salaries = salaries[combos].sum(axis=1)
    combo_scores = projs[combos].sum(axis=1)

    valid_mask = combo_salaries <= float(salary_cap)
    return combos[valid_mask], combo_salaries[valid_mask], combo_scores[valid_mask]


def _lineup_overlap_count(a: Iterable[int], b: Iterable[int]) -> int:
    return len(set(a).intersection(set(b)))


def optimize_lineup(
    df: pd.DataFrame,
    salary_cap: int = 50000,
    lineup_size: int = 5,
    pool_size: int | None = None,
    value_weight: float = 0.20,
    p90_weight: float = 0.03,
) -> pd.DataFrame:
    """
    Return the single best lineup using the existing final_proj values.

    This does not change any simulation or projection calculations.
    It only changes optimizer pool ranking and lineup search.
    """
    pool = _prepare_pool(
        df,
        pool_size=pool_size,
        value_weight=value_weight,
        p90_weight=p90_weight,
    )

    if len(pool) < int(lineup_size):
        raise ValueError("Not enough drivers in optimizer pool after filters.")

    combos, combo_salaries, combo_scores = _build_combo_table(
        pool,
        salary_cap=int(salary_cap),
        lineup_size=int(lineup_size),
    )

    if len(combos) == 0:
        raise ValueError("No valid lineup found under salary cap. Try increasing pool size or check salaries.")

    best_idx = int(np.lexsort((combo_salaries, combo_scores))[-1])
    best = combos[best_idx]

    cols = ["driver_name", "salary", "final_proj"]
    optional = [
        "avg_finish",
        "fd_fppg",
        "sim_fd_points",
        "value_per_1k",
        "p90_fd_points",
        "qual_pos",
        "fd_id",
        "optimizer_sort_score",
    ]
    for c in optional:
        if c in pool.columns:
            cols.append(c)

    chosen = pool.loc[list(best), cols].copy()
    chosen["total_salary"] = int(round(float(combo_salaries[best_idx])))
    chosen["total_proj"] = float(combo_scores[best_idx])

    if "avg_finish" in chosen.columns:
        chosen = chosen.sort_values(["avg_finish", "final_proj"], ascending=[True, False]).reset_index(drop=True)
    else:
        chosen = chosen.sort_values("final_proj", ascending=False).reset_index(drop=True)

    return chosen


def optimize_top_n_lineups(
    df: pd.DataFrame,
    n: int = 150,
    salary_cap: int = 50000,
    lineup_size: int = 5,
    pool_size: int | None = None,
    max_exposure_pct: float = 1.0,
    min_unique_drivers: int = 1,
    max_dominators: int | None = None,
    dominator_pool_size: int = 8,
    value_weight: float = 0.20,
    p90_weight: float = 0.03,
) -> pd.DataFrame:
    """
    Generate top-N unique lineups without changing projection math.

    Lineups are ranked by total final_proj. Optional controls apply only to lineup construction:
    - max_exposure_pct: max share of exported lineups any one driver can appear in
    - min_unique_drivers: each new lineup must differ by at least this many drivers vs prior selected lineups
    - max_dominators: optional cap on number of dominator candidates (top projected drivers) in a lineup
    """
    target_n = max(1, min(int(n), 250))
    min_unique_drivers = max(1, int(min_unique_drivers))
    max_exposure_pct = float(max(0.0, min(1.0, max_exposure_pct)))

    pool = _prepare_pool(
        df,
        pool_size=pool_size,
        value_weight=value_weight,
        p90_weight=p90_weight,
    )

    if len(pool) < int(lineup_size):
        raise ValueError("Not enough drivers in optimizer pool after filters.")

    combos, combo_salaries, combo_scores = _build_combo_table(
        pool,
        salary_cap=int(salary_cap),
        lineup_size=int(lineup_size),
    )
    if len(combos) == 0:
        raise ValueError("No valid lineups found under salary cap. Try increasing pool size or check salaries.")

    if max_dominators is not None and int(max_dominators) > 0:
        dom_n = min(int(dominator_pool_size), len(pool))
        dominator_flags = np.zeros(len(pool), dtype=int)
        dominator_flags[:dom_n] = 1
        combo_dom_counts = dominator_flags[combos].sum(axis=1)

        dom_mask = combo_dom_counts <= int(max_dominators)
        combos = combos[dom_mask]
        combo_salaries = combo_salaries[dom_mask]
        combo_scores = combo_scores[dom_mask]

    if len(combos) == 0:
        raise ValueError("No valid lineups remained after dominator constraint.")

    order = np.lexsort((combo_salaries, combo_scores))[::-1]
    combos = combos[order]
    combo_salaries = combo_salaries[order]
    combo_scores = combo_scores[order]

    max_exposure_count = max(1, int(np.floor(target_n * max_exposure_pct + 1e-9)))
    exposure_counts = np.zeros(len(pool), dtype=int)

    selected_combos: list[tuple[int, ...]] = []
    selected_rows: list[dict] = []
    has_fd = "fd_id" in pool.columns

    for comb, sal, score in zip(combos, combo_salaries, combo_scores):
        comb_tuple = tuple(int(x) for x in comb.tolist())

        if max_exposure_pct < 1.0:
            if any(exposure_counts[idx] >= max_exposure_count for idx in comb_tuple):
                continue

        too_similar = False
        max_overlap = int(lineup_size) - int(min_unique_drivers)
        for prev in selected_combos:
            if _lineup_overlap_count(prev, comb_tuple) > max_overlap:
                too_similar = True
                break
        if too_similar:
            continue

        row: dict[str, object] = {
            "lineup_rank": len(selected_rows) + 1,
            "total_salary": int(round(float(sal))),
            "total_proj": float(score),
        }
        for i, idx in enumerate(comb_tuple, start=1):
            row[f"driver_{i}"] = str(pool.loc[idx, "driver_name"])
            if has_fd:
                row[f"fd_{i}"] = pool.loc[idx, "fd_id"]

        selected_rows.append(row)
        selected_combos.append(comb_tuple)

        for idx in comb_tuple:
            exposure_counts[idx] += 1

        if len(selected_rows) >= target_n:
            break

    if not selected_rows:
        raise ValueError("No lineups satisfied the selected exposure/uniqueness constraints.")

    return pd.DataFrame(selected_rows)
