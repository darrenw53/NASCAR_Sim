from __future__ import annotations

import itertools
import pandas as pd


def optimize_lineup(
    df: pd.DataFrame,
    salary_cap: int = 50000,
    lineup_size: int = 5,
    pool_size: int = 25
) -> pd.DataFrame:
    """Brute-force optimizer over a reduced pool for speed.

    Expects at minimum: driver_name, salary, final_proj
    If present, also carries through: avg_finish, fd_fppg, sim_fd_points, value_per_1k

    Returns a DataFrame of chosen drivers with total_salary/total_proj repeated per row.
    """
    pool = df.dropna(subset=["salary", "final_proj"]).copy()
    pool["salary"] = pd.to_numeric(pool["salary"], errors="coerce")
    pool["final_proj"] = pd.to_numeric(pool["final_proj"], errors="coerce")
    pool = pool.dropna(subset=["salary", "final_proj"])

    pool = pool.sort_values("final_proj", ascending=False).head(int(pool_size)).reset_index(drop=True)

    if len(pool) < lineup_size:
        raise ValueError("Not enough drivers in optimizer pool after filters.")

    best = None
    best_score = -1e18

    salaries = pool["salary"].to_numpy()
    projs = pool["final_proj"].to_numpy()

    idxs = range(len(pool))
    for comb in itertools.combinations(idxs, lineup_size):
        sal = salaries[list(comb)].sum()
        if sal > salary_cap:
            continue
        score = projs[list(comb)].sum()
        if score > best_score:
            best_score = score
            best = comb

    if best is None:
        raise ValueError("No valid lineup found under salary cap. Try increasing pool size or check salaries.")

    # Include reference columns if they exist
    cols = ["driver_name", "salary", "final_proj"]
    optional = ["avg_finish", "fd_fppg", "sim_fd_points", "value_per_1k"]
    for c in optional:
        if c in pool.columns:
            cols.append(c)

    chosen = pool.loc[list(best), cols].copy()
    chosen["total_salary"] = chosen["salary"].sum()
    chosen["total_proj"] = chosen["final_proj"].sum()

    # nicer ordering if present
    if "avg_finish" in chosen.columns:
        chosen = chosen.sort_values(["avg_finish", "final_proj"], ascending=[True, False]).reset_index(drop=True)
    else:
        chosen = chosen.sort_values("final_proj", ascending=False).reset_index(drop=True)

    return chosen

