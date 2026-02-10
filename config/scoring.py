
from __future__ import annotations

# FanDuel NASCAR scoring (see FanDuel rules / NASCAR training guide)
# - Laps completed: 0.1
# - Laps led: 0.1
# - Place differential: +/- 0.5 per position gained/lost (qualifying -> finish)
# - Finishing position points: 1st 43, 2nd 40, 3rd 38, 4th 37, 5th 36, ... 40th 1

LAP_COMPLETED_PTS = 0.1
LAP_LED_PTS = 0.1
PLACE_DIFF_PTS = 0.5

# Finishing points for positions 1..40 (FanDuel lists through 40th)
FINISH_POINTS = {
    1: 43, 2: 40, 3: 38, 4: 37, 5: 36, 6: 35, 7: 34, 8: 33, 9: 32, 10: 31,
    11: 30, 12: 29, 13: 28, 14: 27, 15: 26, 16: 25, 17: 24, 18: 23, 19: 22, 20: 21,
    21: 20, 22: 19, 23: 18, 24: 17, 25: 16, 26: 15, 27: 14, 28: 13, 29: 12, 30: 11,
    31: 10, 32: 9, 33: 8, 34: 7, 35: 6, 36: 5, 37: 4, 38: 3, 39: 2, 40: 1
}

def finish_points(position: int) -> float:
    # If field is >40, clamp to 40th for FanDuel scoring table
    pos = int(position)
    if pos <= 1:
        return float(FINISH_POINTS[1])
    if pos >= 40:
        return float(FINISH_POINTS[40])
    return float(FINISH_POINTS[pos])
