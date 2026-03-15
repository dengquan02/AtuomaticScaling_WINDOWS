"""Input decoding utilities for the local ionogram application."""

from pathlib import Path
import pickle
from typing import Tuple, Union

import numpy as np

try:  # Pillow is the most convenient reader for common image formats.
    from PIL import Image
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError(
        "Pillow is required for reading image inputs. Please install it via 'pip install Pillow'."
    ) from exc


ImageArray = np.ndarray


def load_image(path: Union[str, Path]) -> ImageArray:
    """Load .png/.jpg image and normalize to float32 RGB in [0, 1]."""

    img_path = Path(path)
    with Image.open(img_path) as img:
        rgb = img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    return arr


def load_legacy(path: Union[str, Path]) -> ImageArray:
    """Load legacy pickle data used by DIAS training datasets."""

    with open(path, "rb") as fh:
        data = pickle.load(fh)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:  # expand grayscale to channel axis
        arr = np.expand_dims(arr, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    return arr


def _ensure_three_channels(arr: ImageArray) -> ImageArray:
    if arr.ndim != 3:
        raise ValueError(f"Expected HxWxC array, got shape {arr.shape}")
    if arr.shape[2] == 3:
        return arr
    if arr.shape[2] == 1:
        return np.repeat(arr, 3, axis=2)
    if arr.shape[2] > 3:
        return arr[:, :, :3]
    raise ValueError(f"Unsupported channel count: {arr.shape[2]}")


def to_model_input(arr: ImageArray, pad_height: int = 512, pad_width: int = 512) -> Tuple[ImageArray, ImageArray]:
    """Pad/crop image to model input size and return (model_batch, original)."""

    arr = _ensure_three_channels(arr).astype(np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    h, w, _ = arr.shape
    canvas = np.zeros((pad_height, pad_width, 3), dtype=np.float32)
    copy_h = min(h, pad_height)
    copy_w = min(w, pad_width)
    canvas[:copy_h, :copy_w, :] = arr[:copy_h, :copy_w, :]
    batch = np.expand_dims(canvas, 0)
    return batch, arr
