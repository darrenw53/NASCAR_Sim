
from __future__ import annotations

DEFAULTS = {
    "n_sims": 20000,
    "rng_seed": 42,
    "past_weight": 0.40,
    "practice_weight": 0.25,
    "qual_weight": 0.20,
    "baseline_weight": 0.15,  # season-long driver strength
    "blend_sim_vs_fd": 0.60,  # final proj = blend*sim + (1-blend)*FD_FPPG
    "performance_sd": 0.65,   # bigger = more randomness in finishing order
    "dnf_base": 0.07,         # baseline DNF probability
    "dnf_superspeedway_boost": 0.08,  # additional DNF risk for superspeedways
    "dominator_top_k": 6,
    "dominator_strength": 1.35,  # higher -> laps led more concentrated among top drivers
    "optimizer_pool": 25,
    "salary_cap": 50000,
    "lineup_size": 5
}
