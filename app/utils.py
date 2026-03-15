"""Helper utilities for resource resolution (works inside PyInstaller bundles)."""

import os
import sys
from pathlib import Path
from typing import Optional, List


def get_base_dir() -> Path:
    """Return the directory that contains bundled resources."""

    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        return Path(bundle_root)
    # app/ directory lives one level below repo root
    return Path(__file__).resolve().parents[1]


def resolve_relative_path(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        return path
    base = get_base_dir()
    candidate = base / relative
    if candidate.exists():
        return candidate
    return path


def resolve_config_path(config_argument: Optional[str]) -> Path:
    if config_argument:
        return resolve_relative_path(config_argument)
    default = get_base_dir() / "my_config.yaml"
    return default


def _candidate_matplotlib_paths() -> List[Path]:
    base = get_base_dir()
    if getattr(sys, "frozen", False):
        exec_dir = Path(sys.executable).resolve().parent
    else:
        exec_dir = base
    return [
        base / "matplotlib" / "mpl-data",
        base / "mpl-data",
        exec_dir / "matplotlib" / "mpl-data",
        exec_dir / "mpl-data",
        resolve_relative_path("matplotlib/mpl-data"),
    ]


def ensure_matplotlib_data_path() -> Optional[Path]:
    """Ensure MATPLOTLIBDATA points to a valid directory; return the path used."""

    current = os.environ.get("MATPLOTLIBDATA")
    if current and Path(current).exists():
        return Path(current)
    for path in _candidate_matplotlib_paths():
        if path.exists():
            os.environ["MATPLOTLIBDATA"] = str(path)
            return path
    return None
