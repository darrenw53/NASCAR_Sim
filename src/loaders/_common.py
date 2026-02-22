
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

def load_json_with_optional_header(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    txt = p.read_text(encoding="utf-8", errors="ignore")
    i = txt.find("{")
    if i == -1:
        raise ValueError(f"No JSON object found in {p}")
    return json.loads(txt[i:])
