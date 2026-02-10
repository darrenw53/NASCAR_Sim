
from __future__ import annotations

# Very light metadata; you can extend this as you add track-specific tuning.
TRACK_TYPE_PRESETS = {
    "superspeedway": {
        "performance_sd": 0.90,
        "dnf_boost": 0.10,
    },
    "short_track": {
        "performance_sd": 0.70,
        "dnf_boost": 0.04,
    },
    "intermediate": {
        "performance_sd": 0.60,
        "dnf_boost": 0.05,
    },
    "road_course": {
        "performance_sd": 0.75,
        "dnf_boost": 0.05,
    }
}
