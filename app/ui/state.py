"""State objects shared between GUI widgets."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


LayerPoints = List[Tuple[float, float]]


@dataclass
class TraceData:
    layer: str
    points: LayerPoints = field(default_factory=list)
    source: str = "auto"


@dataclass
class Document:
    file_path: Path
    image: np.ndarray
    traces: Dict[str, TraceData]
    params: Dict[str, Dict[str, Optional[float]]]
    uncertainty: Dict[str, Dict[str, float]]
    probabilities: Optional[np.ndarray] = None
    threshold: float = 0.3

    def to_project_item(self) -> Dict[str, object]:
        return {
            "file": str(self.file_path),
            "traces": {layer: trace.points for layer, trace in self.traces.items()},
            "params": self.params,
            "uncertainty": self.uncertainty,
            "threshold": self.threshold,
        }

    # --- trace manipulation helpers -------------------------------------------------
    def insert_point(self, layer: str, point: Tuple[float, float], index: Optional[int] = None) -> int:
        trace = self.traces[layer]
        pts = trace.points
        if index is None or index >= len(pts):
            pts.append(point)
            idx = len(pts) - 1
        else:
            pts.insert(index, point)
            idx = index
        trace.source = "edited"
        self._update_params_from_points(layer)
        return idx

    def move_point(self, layer: str, index: int, point: Tuple[float, float]) -> None:
        pts = self.traces[layer].points
        if 0 <= index < len(pts):
            pts[index] = point
            self.traces[layer].source = "edited"
            self._update_params_from_points(layer)

    def delete_point(self, layer: str, index: int) -> Optional[Tuple[float, float]]:
        pts = self.traces[layer].points
        if 0 <= index < len(pts):
            removed = pts.pop(index)
            self.traces[layer].source = "edited"
            self._update_params_from_points(layer)
            return removed
        return None

    def find_point_index(self, layer: str, point: Tuple[float, float], tolerance: float = 8.0) -> Optional[int]:
        pts = self.traces[layer].points
        if not pts:
            return None
        px, py = point
        best_idx = None
        best_dist = tolerance
        for idx, (x, y) in enumerate(pts):
            dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
            if dist <= best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    def update_auto_traces(self, layer_points: Dict[str, LayerPoints], params, uncertainty, probabilities, threshold):
        self.threshold = threshold
        self.probabilities = probabilities
        self.params = params
        self.uncertainty = uncertainty
        for layer, points in layer_points.items():
            self.traces[layer].points = list(points)
            self.traces[layer].source = "auto"

    def _update_params_from_points(self, layer: str) -> None:
        pts = self.traces[layer].points
        if not pts:
            self.params[layer] = {"hp": None, "f0": None}
            return
        xs, ys = zip(*pts)
        self.params[layer] = {"hp": float(min(ys)), "f0": float(max(xs))}


@dataclass
class ApplicationState:
    documents: List[Document] = field(default_factory=list)
    current_index: int = -1
    threshold: float = 0.3

    def add_document(self, doc: Document) -> None:
        self.documents.append(doc)
        self.current_index = len(self.documents) - 1

    def set_current(self, index: int) -> None:
        if 0 <= index < len(self.documents):
            self.current_index = index

    @property
    def current_document(self) -> Optional[Document]:
        if 0 <= self.current_index < len(self.documents):
            return self.documents[self.current_index]
        return None


def default_params(layers: Sequence[str]) -> Dict[str, Dict[str, Optional[float]]]:
    return {layer: {"hp": None, "f0": None} for layer in layers}


def default_uncertainty(layers: Sequence[str]) -> Dict[str, Dict[str, float]]:
    return {layer: {"hp_span": 0.0, "f_span": 0.0, "hp_conf": 0.0, "f_conf": 0.0} for layer in layers}


def create_document(path: Path, image: np.ndarray, layers: Sequence[str]) -> Document:
    traces = {layer: TraceData(layer=layer) for layer in layers}
    return Document(path, image, traces, default_params(layers), default_uncertainty(layers))


def mask_to_points(probabilities: np.ndarray, layer_index: int, threshold: float, max_points: int = 1024) -> LayerPoints:
    mask = probabilities[0, :, :, layer_index]
    h, w = mask.shape
    pts: LayerPoints = []
    for x in range(w):
        column = mask[:, x]
        y = int(np.argmax(column))
        if column[y] >= threshold:
            pts.append((float(x), float(y)))
    if len(pts) > max_points:
        step = max(1, len(pts) // max_points)
        pts = pts[::step]
    return pts
