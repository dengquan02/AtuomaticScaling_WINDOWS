"""Tkinter canvas wrapper for ionogram visualization."""

from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

from .tools import ToolMode


class TraceCanvas(tk.Canvas):
    def __init__(self, master, controller, width=520, height=520):
        super().__init__(master, width=width, height=height, background="black", highlightthickness=0)
        self.controller = controller
        self._photo = None
        self.doc = None
        self.scale = 1.0
        self.mode = ToolMode.SELECT
        self.current_layer = "F2"
        self._dragging = False
        self.bind("<Button-1>", self._on_left_click)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Button-3>", self._on_right_click)

    def set_document(self, doc) -> None:
        self.doc = doc
        self._render()

    def set_mode(self, mode: ToolMode) -> None:
        self.mode = mode

    def set_layer(self, layer: str) -> None:
        self.current_layer = layer

    def _render(self) -> None:
        self.delete("all")
        if self.doc is None:
            return
        img = (self.doc.image * 255.0).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        self._photo = ImageTk.PhotoImage(pil_img)
        self.create_image(0, 0, image=self._photo, anchor="nw", tags="bg")
        for layer, trace in self.doc.traces.items():
            if not trace.points:
                continue
            color = {"E": "#1f77b4", "F1": "#2ca02c", "F2": "#d62728"}.get(layer, "yellow")
            flat = [coord for pt in trace.points for coord in pt]
            self.create_line(*flat, fill=color, width=2, smooth=True, tags=f"trace-{layer}")
            for idx, (x, y) in enumerate(trace.points):
                r = 4
                self.create_oval(x - r, y - r, x + r, y + r, fill=color, outline="white", width=1, tags=f"pt-{layer}-{idx}")

    def _canvas_to_image(self, event) -> Tuple[float, float]:
        return float(self.canvasx(event.x)), float(self.canvasy(event.y))

    def _on_left_click(self, event) -> None:
        if self.doc is None:
            return
        x, y = self._canvas_to_image(event)
        if self.mode == ToolMode.ADD:
            self.controller.add_point(self.current_layer, (x, y))
            self._render()
            return
        # selection mode
        if self.controller.begin_move(self.current_layer, (x, y)):
            self._dragging = True

    def _on_drag(self, event) -> None:
        if not self._dragging:
            return
        x, y = self._canvas_to_image(event)
        if self.controller.move_point((x, y)):
            self._render()

    def _on_release(self, event) -> None:
        if not self._dragging:
            return
        self._dragging = False
        x, y = self._canvas_to_image(event)
        if self.controller.end_move((x, y)):
            self._render()

    def _on_right_click(self, event) -> None:
        if self.doc is None:
            return
        x, y = self._canvas_to_image(event)
        if self.controller.delete_point(self.current_layer, (x, y)):
            self._render()
