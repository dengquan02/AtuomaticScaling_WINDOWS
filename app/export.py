"""Export helpers for CSV/JSON summaries."""

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Union


def to_csv(rows: Iterable[Mapping[str, object]], out_path: Union[str, Path]) -> None:
    rows = list(rows)
    if not rows:
        return
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def to_json(payload: Mapping[str, object], out_path: Union[str, Path]) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
