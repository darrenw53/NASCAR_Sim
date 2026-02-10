
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from config.scoring import (
    LAP_COMPLETED_PTS, LAP_LED_PTS, PLACE_DIFF_PTS, finish_points
)

@dataclass
class SimConfig:
    n_sims: int = 20000
    rng_seed: int | None = 42
    performance_sd: float = 0.65
    total_laps: int = 200
    dnf_prob: float = 0.10
    dominator_top_k: int = 6
    dominator_strength: float = 1.35

def simulate_race(drivers: pd.DataFrame, cfg: SimConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a race n_sims times.
    drivers must contain: driver_id, driver_name, composite_score, qual_pos (optional)
    Returns:
      summary_df: one row per driver (win%, top5%, avg_finish, avg_fd_points, etc.)
      sims_df: long format (sim, driver_id, finish_pos, laps_led, dnf, fd_points)
    """
    df = drivers.copy()
    df = df.dropna(subset=["driver_name"]).reset_index(drop=True)
    n = len(df)
    if n < 5:
        raise ValueError("Need at least 5 drivers to simulate.")
    rng = np.random.default_rng(cfg.rng_seed)

    base = df["composite_score"].fillna(df["composite_score"].median()).to_numpy(dtype=float)

    qual_pos = df.get("qual_pos", pd.Series([np.nan]*n)).to_numpy(dtype=float)
    # If qual_pos missing, treat as mid-pack for place-diff
    qual_pos_filled = np.where(np.isfinite(qual_pos), qual_pos, np.nanmedian(qual_pos) if np.isfinite(np.nanmedian(qual_pos)) else (n+1)/2)

    sims_rows = []
    # Precompute dominator probabilities from base strength
    dom_logits = np.exp(base * cfg.dominator_strength)
    dom_probs = dom_logits / dom_logits.sum()

    for s in range(cfg.n_sims):
        noise = rng.normal(0.0, cfg.performance_sd, size=n)
        perf = base + noise
        order = np.argsort(-perf)  # descending = best first
        finish_pos = np.empty(n, dtype=int)
        finish_pos[order] = np.arange(1, n+1)

        # DNFs: random drivers, but reduce probability for stronger drivers
        # (Still can DNF on superspeedways)
        strength_factor = (base - base.min()) / (base.max() - base.min() + 1e-9)  # 0..1
        dnf_p = np.clip(cfg.dnf_prob * (1.15 - 0.5*strength_factor), 0.01, 0.45)
        dnf = rng.random(n) < dnf_p
        # If DNF, push them toward back (keep relative order among DNFs)
        if dnf.any():
            non = np.where(~dnf)[0]
            dn = np.where(dnf)[0]
            # reorder: non-DNF keep their order; DNF placed after, by perf
            non_order = non[np.argsort(finish_pos[non])]
            dn_order = dn[np.argsort(-perf[dn])]
            new_order = np.concatenate([non_order, dn_order])
            finish_pos[new_order] = np.arange(1, n+1)

        # Laps led distribution: allocate total laps among top_k by a Dirichlet draw
        top_k = min(cfg.dominator_top_k, n)
        leaders = order[:top_k]
        # Use dom_probs among leaders
        alpha = (dom_probs[leaders] * top_k) + 0.25
        share = rng.dirichlet(alpha)
        laps_led = np.zeros(n, dtype=int)
        laps_led[leaders] = np.floor(share * cfg.total_laps).astype(int)
        # fix rounding leftover
        leftover = cfg.total_laps - laps_led.sum()
        if leftover > 0:
            add_to = rng.choice(leaders, size=leftover, replace=True)
            for i in add_to:
                laps_led[i] += 1

        laps_completed = np.where(dnf, rng.integers(low=int(cfg.total_laps*0.35), high=cfg.total_laps, size=n), cfg.total_laps)

        # FanDuel points
        fp_finish = np.array([finish_points(p) for p in finish_pos], dtype=float)
        place_diff = (qual_pos_filled - finish_pos) * PLACE_DIFF_PTS
        fp = fp_finish + place_diff + (laps_completed * LAP_COMPLETED_PTS) + (laps_led * LAP_LED_PTS)

        for i in range(n):
            sims_rows.append({
                "sim": s,
                "driver_id": df.loc[i, "driver_id"],
                "driver_name": df.loc[i, "driver_name"],
                "finish_pos": int(finish_pos[i]),
                "qual_pos": float(qual_pos_filled[i]),
                "laps_led": int(laps_led[i]),
                "laps_completed": int(laps_completed[i]),
                "dnf": bool(dnf[i]),
                "fd_points": float(fp[i]),
            })

    sims = pd.DataFrame(sims_rows)

    # Summary
    summ = sims.groupby(["driver_id","driver_name"], dropna=False).agg(
        win_pct=("finish_pos", lambda x: (x==1).mean()),
        top5_pct=("finish_pos", lambda x: (x<=5).mean()),
        top10_pct=("finish_pos", lambda x: (x<=10).mean()),
        avg_finish=("finish_pos","mean"),
        avg_laps_led=("laps_led","mean"),
        dnf_pct=("dnf","mean"),
        sim_fd_points=("fd_points","mean"),
        p90_fd_points=("fd_points", lambda x: float(np.percentile(x,90))),
    ).reset_index()

    return summ.sort_values(["sim_fd_points"], ascending=False), sims
