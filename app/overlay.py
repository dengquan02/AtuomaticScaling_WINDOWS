"""Visualization helpers for overlaying predictions on raw ionograms."""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from app.utils import ensure_matplotlib_data_path

ensure_matplotlib_data_path()
import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt


LOGGER = logging.getLogger(__name__)

COLORS: Dict[str, str] = {"E": "#1f77b4", "F1": "#2ca02c", "F2": "#d62728"}


def overlay_traces(
    image: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    out_path: Union[str, Path],
    title: Optional[str] = None,
    max_points: int = 4000,
) -> None:
    """Save an overlay plot of detected traces on top of the raw image."""

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    ax.imshow(image)
    h, w = image.shape[:2]
    layer_names = ("E", "F1", "F2")
    for idx, layer in enumerate(layer_names):
        mask = probabilities[0, :h, :w, idx]
        ys, xs = np.where(mask > threshold)
        if ys.size == 0:
            continue
        if ys.size > max_points:
            choice = np.random.choice(ys.size, max_points, replace=False)
            ys = ys[choice]
            xs = xs[choice]
        ax.scatter(xs, ys, s=5, c=COLORS[layer], alpha=0.8, label=f"{layer} trace")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xlabel("Frequency pixel")
    ax.set_ylabel("Height pixel")
    ax.legend(loc="upper right")
    if title:
        ax.set_title(title)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    LOGGER.info("Saved overlay to %s", out_path)
