"""Parameter extraction utilities built on top of DIAS helpers."""

from typing import Dict, Optional

import numpy as np

from dias.dataIO.dataPostProcess import get_minH_maxF


LAYERS = ("E", "F1", "F2")


def extract_params(probabilities: np.ndarray, threshold: float) -> Dict[str, Dict[str, Optional[float]]]:
    """Return per-layer min height (h') and max frequency (f0) pixel indices."""

    stats = get_minH_maxF(probabilities, th=threshold)
    results = {}
    for idx, layer in enumerate(LAYERS):
        min_h, max_f = stats[idx]
        hp = None if min_h >= 9999 else float(min_h)
        f0 = None if max_f <= 0 else float(max_f)
        results[layer] = {"hp": hp, "f0": f0}
    return results


def estimate_uncertainty(probabilities: np.ndarray, threshold: float, delta: float = 0.05) -> Dict[str, Dict[str, float]]:
    """Derive crude uncertainty spans by perturbing the detection threshold."""

    lower = get_minH_maxF(probabilities, th=max(0.0, threshold - delta))
    upper = get_minH_maxF(probabilities, th=min(1.0, threshold + delta))
    h, w = probabilities.shape[1:3]
    res: Dict[str, Dict[str, float]] = {}
    for idx, layer in enumerate(LAYERS):
        hp_span = abs(upper[idx, 0] - lower[idx, 0])
        f_span = abs(upper[idx, 1] - lower[idx, 1])
        hp_conf = 1.0 - min(1.0, hp_span / max(1.0, h))
        f_conf = 1.0 - min(1.0, f_span / max(1.0, w))
        res[layer] = {
            "hp_span": float(hp_span),
            "f_span": float(f_span),
            "hp_conf": round(max(0.0, hp_conf), 4),
            "f_conf": round(max(0.0, f_conf), 4),
        }
    return res
