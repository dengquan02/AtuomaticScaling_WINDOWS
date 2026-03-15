"""Project persistence helpers."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union


PROJECT_VERSION = 1


def save_project(meta: Dict[str, Any], out_path: Union[str, Path]) -> None:
    payload = {
        "version": PROJECT_VERSION,
        "created_at": datetime.utcnow().isoformat() + "Z",
        **meta,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def load_project(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data
