
from __future__ import annotations

import numpy as np
import pandas as pd

def percentile_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Return percentile in [0,1]. If higher_is_better=False, lower values become higher percentile."""
    s = pd.to_numeric(series, errors="coerce")
    if not higher_is_better:
        s = -s
    # rank(pct=True) yields 0..1, ties get average rank
    return s.rank(pct=True)

def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if sd == 0 or np.isnan(sd):
        return pd.Series([0.0]*len(s), index=s.index)
    return (s - mu) / sd
