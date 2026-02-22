
from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd

def load_fanduel_csv(path_or_file: Any) -> pd.DataFrame:
    """Load FanDuel player pool CSV.

    Accepts a filesystem path (str/Path) or an uploaded file-like object (Streamlit uploader).
    """
    df = pd.read_csv(path_or_file)
    # clean extra unnamed columns
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False, regex=True)]
    df["driver_name"] = (df["First Name"].astype(str).str.strip() + " " + df["Last Name"].astype(str).str.strip()).str.strip()
    df["salary"] = pd.to_numeric(df["Salary"], errors="coerce")
    df["fd_fppg"] = pd.to_numeric(df["FPPG"], errors="coerce")
    return df[["Id","driver_name","salary","fd_fppg"]].rename(columns={"Id":"fd_id"})
