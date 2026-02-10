
from __future__ import annotations

import itertools
import pandas as pd
import numpy as np

def optimize_lineup(df: pd.DataFrame, salary_cap: int = 50000, lineup_size: int = 5, pool_size: int = 25) -> pd.DataFrame:
    """Brute-force optimizer over a reduced pool for speed.
    Expects columns: driver_name, salary, final_proj
    Returns a 1-row DataFrame lineup with total_salary/total_proj.
    """
    pool = df.dropna(subset=["salary","final_proj"]).copy()
    pool["salary"] = pd.to_numeric(pool["salary"], errors="coerce")
    pool["final_proj"] = pd.to_numeric(pool["final_proj"], errors="coerce")
    pool = pool.sort_values("final_proj", ascending=False).head(int(pool_size)).reset_index(drop=True)

    if len(pool) < lineup_size:
        raise ValueError("Not enough drivers in optimizer pool.")

    best = None
    best_score = -1e18

    salaries = pool["salary"].to_numpy()
    projs = pool["final_proj"].to_numpy()
    names = pool["driver_name"].to_numpy()

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

    chosen = pool.loc[list(best), ["driver_name","salary","final_proj"]].copy()
    chosen["total_salary"] = chosen["salary"].sum()
    chosen["total_proj"] = chosen["final_proj"].sum()
    return chosen
