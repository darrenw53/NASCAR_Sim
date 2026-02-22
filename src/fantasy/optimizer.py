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




def optimize_top_n_lineups(
    df: pd.DataFrame,
    n: int = 150,
    salary_cap: int = 50000,
    lineup_size: int = 5,
    pool_size: int = 25
) -> pd.DataFrame:
    """
    Generate the top-N unique lineups (by total final_proj) under the salary cap
    using the same reduced-pool brute-force approach as optimize_lineup().

    Returns one row per lineup with:
      - lineup_rank (1..N)
      - total_salary
      - total_proj
      - driver_1..driver_5 (driver_name)
      - fd_1..fd_5 (fd_id if present)
    """
    n = int(n)
    n = max(1, min(n, 250))  # FD max per upload is 250; keep sane.
    pool = df.dropna(subset=["salary", "final_proj"]).copy()
    pool["salary"] = pd.to_numeric(pool["salary"], errors="coerce")
    pool["final_proj"] = pd.to_numeric(pool["final_proj"], errors="coerce")
    pool = pool.dropna(subset=["salary", "final_proj"])

    pool = pool.sort_values("final_proj", ascending=False).head(int(pool_size)).reset_index(drop=True)

    if len(pool) < lineup_size:
        raise ValueError("Not enough drivers in optimizer pool after filters.")

    salaries = pool["salary"].to_numpy()
    projs = pool["final_proj"].to_numpy()

    # Enumerate all combinations; keep those under cap.
    combos = []
    idxs = range(len(pool))
    for comb in itertools.combinations(idxs, lineup_size):
        sal = float(salaries[list(comb)].sum())
        if sal > salary_cap:
            continue
        score = float(projs[list(comb)].sum())
        combos.append((score, sal, comb))

    if not combos:
        raise ValueError("No valid lineups found under salary cap. Try increasing pool size or check salaries.")

    # Sort by projection desc, salary desc as tiebreak (optional)
    combos.sort(key=lambda x: (x[0], x[1]), reverse=True)
    combos = combos[:n]

    rows = []
    has_fd = "fd_id" in pool.columns
    for rank, (score, sal, comb) in enumerate(combos, start=1):
        row = {
            "lineup_rank": rank,
            "total_salary": int(round(sal)),
            "total_proj": score,
        }
        for i, idx in enumerate(comb, start=1):
            row[f"driver_{i}"] = str(pool.loc[idx, "driver_name"])
            if has_fd:
                row[f"fd_{i}"] = pool.loc[idx, "fd_id"]
        rows.append(row)

    out = pd.DataFrame(rows)
    return out

