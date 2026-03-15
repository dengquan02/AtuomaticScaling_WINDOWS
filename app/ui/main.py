"""Tkinter-based GUI for local ionogram interpretation."""

import argparse
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from app.decoder import load_image, load_legacy, to_model_input
from app.model_service import ModelService
from app.params import LAYERS, estimate_uncertainty, extract_params
from app import project, export
from app.overlay import overlay_traces
from app.cli import IMG_EXTS, LEGACY_EXTS
from app.utils import resolve_config_path
from app.ui.canvas import TraceCanvas
from app.ui.history import AddPointCommand, DeletePointCommand, HistoryManager, MovePointCommand
from app.ui.state import ApplicationState, Document, create_document, mask_to_points
from app.ui.tools import Layer, ToolMode


LOGGER = logging.getLogger(__name__)


def _load_image_from_path(path: Path) -> np.ndarray:
    if path.suffix.lower() in IMG_EXTS:
        return load_image(path)
    if path.suffix.lower() in LEGACY_EXTS:
        return load_legacy(path)
    raise ValueError(f"Unsupported file: {path}")


class MainWindow:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.state = ApplicationState()
        self.history = HistoryManager()
        self.model_service: Optional[ModelService] = None
        self.config_path = resolve_config_path(args.config)
        self.current_layer = Layer.F2.value
        self.current_drag_index: Optional[int] = None
        self.drag_layer: Optional[str] = None
        self.drag_start_point: Optional[Tuple[float, float]] = None
        self.root = tk.Tk()
        self.root.title("Ionogram Interpreter")
        self.threshold_var = tk.DoubleVar(value=args.threshold)
        self.mode_var = tk.StringVar(value=ToolMode.SELECT.value)
        self.layer_var = tk.StringVar(value=self.current_layer)
        self.status_var = tk.StringVar(value="Ready")
        self._build_layout()
        if args.input:
            self._load_prefill(Path(args.input))

    # ------------------------------------------------------------------ UI assembly
    def _build_layout(self) -> None:
        self.root.geometry("1100x700")
        # toolbar
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(toolbar, text="Open", command=self.open_files).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Run Inference", command=self.run_inference).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Save Project", command=self.save_project).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Load Project", command=self.load_project).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Export CSV", command=self.export_csv).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Undo", command=self.undo).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Redo", command=self.redo).pack(side=tk.LEFT)

        ttk.Label(toolbar, text="Threshold").pack(side=tk.LEFT, padx=(10, 2))
        thresh_scale = ttk.Scale(toolbar, from_=0.1, to=0.9, variable=self.threshold_var, command=self._threshold_changed)
        thresh_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        # main split
        body = ttk.Frame(self.root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left_panel = ttk.Frame(body, width=200)
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(left_panel, text="Files").pack(anchor="w")
        self.file_list = tk.Listbox(left_panel, exportselection=False)
        self.file_list.pack(fill=tk.BOTH, expand=True)
        self.file_list.bind("<<ListboxSelect>>", self._on_file_select)

        right_panel = ttk.Frame(body)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = TraceCanvas(right_panel, controller=self)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        side_panel = ttk.Frame(right_panel, width=220)
        side_panel.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(side_panel, text="Layer").pack(anchor="w")
        for layer in Layer:
            ttk.Radiobutton(side_panel, text=layer.value, variable=self.layer_var, value=layer.value, command=self._layer_changed).pack(anchor="w")

        ttk.Label(side_panel, text="Mode").pack(anchor="w", pady=(10, 0))
        for mode in ToolMode:
            ttk.Radiobutton(side_panel, text=mode.value.title(), variable=self.mode_var, value=mode.value, command=self._mode_changed).pack(anchor="w")

        ttk.Label(side_panel, text="Parameters").pack(anchor="w", pady=(10, 0))
        self.param_tree = ttk.Treeview(side_panel, columns=("hp", "f0", "conf", "src"), show="headings", height=6)
        for col, width in zip(("hp", "f0", "conf", "src"), (60, 60, 60, 70)):
            self.param_tree.heading(col, text=col.upper())
            self.param_tree.column(col, width=width, anchor="center")
        self.param_tree.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(side_panel, text="Overlay PNG", command=self.export_overlay).pack(fill=tk.X, pady=(5, 0))

        status_bar = ttk.Label(self.root, textvariable=self.status_var, anchor="w")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ---------------------------------------------------------------------- helpers
    def _load_prefill(self, path: Path) -> None:
        if path.is_dir():
            files = sorted([p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS | LEGACY_EXTS])
        else:
            files = [path]
        for file_path in files[: self.args.max_files or 5]:
            self._add_document_from_path(file_path)

    def _add_document_from_path(self, path: Path) -> None:
        try:
            image = _load_image_from_path(path)
        except Exception as exc:  # pragma: no cover - UI feedback
            messagebox.showerror("Load error", f"Failed to load {path}: {exc}")
            return
        doc = create_document(path, image, LAYERS)
        self.state.add_document(doc)
        self.file_list.insert(tk.END, path.name)
        self.file_list.selection_clear(0, tk.END)
        self.file_list.selection_set(tk.END)
        self.file_list.activate(tk.END)
        self.canvas.set_document(doc)
        self._refresh_params()

    def _on_file_select(self, _event) -> None:
        if not self.file_list.curselection():
            return
        idx = self.file_list.curselection()[0]
        self.state.set_current(idx)
        doc = self.state.current_document
        self.canvas.set_document(doc)
        self._refresh_params()

    def _layer_changed(self) -> None:
        self.current_layer = self.layer_var.get()
        self.canvas.set_layer(self.current_layer)

    def _mode_changed(self) -> None:
        self.canvas.set_mode(ToolMode(self.mode_var.get()))

    def _threshold_changed(self, _val) -> None:
        self.state.threshold = self.threshold_var.get()
        self.status_var.set(f"Threshold: {self.state.threshold:.2f}")

    # ------------------------------------------------------------------- file ops
    def open_files(self) -> None:
        paths = filedialog.askopenfilenames(filetypes=[("Ionogram", "*.png *.jpg *.jpeg *.pickle *.pkl")])
        for name in paths:
            self._add_document_from_path(Path(name))

    def save_project(self) -> None:
        if not self.state.documents:
            messagebox.showinfo("Save", "No documents to save.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".ionproj", filetypes=[("Ionogram Project", "*.ionproj")])
        if not path:
            return
        payload = {
            "settings": {"threshold": self.state.threshold},
            "documents": [doc.to_project_item() for doc in self.state.documents],
        }
        project.save_project(payload, path)
        self.status_var.set(f"Project saved: {path}")

    def load_project(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Ionogram Project", "*.ionproj")])
        if not path:
            return
        data = project.load_project(path)
        self.state = ApplicationState()
        self.file_list.delete(0, tk.END)
        documents = data.get("documents", data.get("items", []))
        for item in documents:
            file_path = Path(item["file"])
            if not file_path.exists():
                continue
            image = _load_image_from_path(file_path)
            doc = create_document(file_path, image, LAYERS)
            doc.params = item.get("params", doc.params)
            doc.uncertainty = item.get("uncertainty", doc.uncertainty)
            doc.threshold = item.get("threshold", data.get("settings", {}).get("threshold", self.state.threshold))
            traces = item.get("traces", {})
            for layer, pts in traces.items():
                doc.traces[layer].points = [tuple(pt) for pt in pts]
                doc.traces[layer].source = "edited"
            self.state.add_document(doc)
            self.file_list.insert(tk.END, file_path.name)
        if self.state.documents:
            self.file_list.selection_set(0)
            self._on_file_select(None)
        self.status_var.set(f"Loaded project: {path}")

    def export_csv(self) -> None:
        if not self.state.documents:
            messagebox.showinfo("Export", "Nothing to export.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        rows = []
        for doc in self.state.documents:
            row = {"file": str(doc.file_path)}
            for layer in LAYERS:
                row[f"hp_{layer}"] = doc.params[layer]["hp"]
                row[f"f0_{layer}"] = doc.params[layer]["f0"]
                row[f"confidence_{layer}"] = doc.uncertainty[layer].get("hp_conf", 0.0)
            rows.append(row)
        export.to_csv(rows, path)
        self.status_var.set(f"Exported CSV: {path}")

    def export_overlay(self) -> None:
        doc = self.state.current_document
        if doc is None or doc.probabilities is None:
            messagebox.showinfo("Overlay", "Run inference first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if not path:
            return
        overlay_traces(doc.image, doc.probabilities, self.state.threshold, path, title=doc.file_path.name)
        self.status_var.set(f"Overlay saved: {path}")

    # ---------------------------------------------------------------- inference
    def ensure_model(self) -> ModelService:
        if self.model_service is None:
            self.model_service = ModelService(self.config_path, gpu_id=self.args.gpu_id)
        return self.model_service

    def run_inference(self) -> None:
        doc = self.state.current_document
        if doc is None:
            messagebox.showinfo("Inference", "Select a document first.")
            return

        def worker():
            try:
                self.status_var.set("Running inference...")
                ms = self.ensure_model()
                batch, _ = to_model_input(doc.image)
                probs = ms.predict(batch)
                params = extract_params(probs, self.state.threshold)
                unc = estimate_uncertainty(probs, self.state.threshold)
                layer_points = {
                    layer: mask_to_points(probs, idx, self.state.threshold)
                    for idx, layer in enumerate(LAYERS)
                }
                self.root.after(
                    0,
                    lambda: self._apply_inference(doc, layer_points, params, unc, probs),
                )
            except Exception as exc:
                LOGGER.exception("Inference failed")
                self.status_var.set("Inference failed!")
                # 注意：直接在闭包里捕获 exc 会在 Python 3 中被清理掉，导致 NameError。
                # 这里先把消息格式化好，再通过默认参数传入 lambda，避免悬空引用。
                msg = f"Failed: {exc}"
                self.root.after(0, lambda m=msg: messagebox.showerror("Inference", m))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_inference(self, doc, layer_points, params, unc, probs):
        doc.update_auto_traces(layer_points, params, unc, probs, self.state.threshold)
        self.canvas.set_document(doc)
        self._refresh_params()
        self.status_var.set("Inference completed")

    # ----------------------------------------------------------------- history ops
    def undo(self) -> None:
        desc = self.history.undo()
        if desc:
            self.canvas._render()  # noqa: SLF001
            self._refresh_params()
            self.status_var.set(f"Undo: {desc}")

    def redo(self) -> None:
        desc = self.history.redo()
        if desc:
            self.canvas._render()
            self._refresh_params()
            self.status_var.set(f"Redo: {desc}")

    # ------------------------------------------------------------- canvas actions
    def add_point(self, layer: str, point: Tuple[float, float]) -> None:
        doc = self.state.current_document
        if doc is None:
            return
        self.history.run(AddPointCommand(doc, layer, point))
        self._refresh_params()

    def begin_move(self, layer: str, point: Tuple[float, float]) -> bool:
        doc = self.state.current_document
        if doc is None:
            return False
        idx = doc.find_point_index(layer, point)
        if idx is None:
            return False
        self.current_drag_index = idx
        self.drag_layer = layer
        self.drag_start_point = doc.traces[layer].points[idx]
        return True

    def move_point(self, point: Tuple[float, float]) -> bool:
        doc = self.state.current_document
        if doc is None or self.current_drag_index is None or self.drag_layer is None:
            return False
        doc.move_point(self.drag_layer, self.current_drag_index, point)
        return True

    def end_move(self, point: Tuple[float, float]) -> bool:
        doc = self.state.current_document
        if doc is None or self.current_drag_index is None or self.drag_layer is None:
            return False
        old_pt = self.drag_start_point or point
        # revert to original before pushing command
        doc.move_point(self.drag_layer, self.current_drag_index, old_pt)
        cmd = MovePointCommand(doc, self.drag_layer, self.current_drag_index, point, old_pt)
        self.history.run(cmd)
        self.current_drag_index = None
        self.drag_layer = None
        self.drag_start_point = None
        self.canvas.set_document(doc)
        self._refresh_params()
        return True

    def delete_point(self, layer: str, point: Tuple[float, float]) -> bool:
        doc = self.state.current_document
        if doc is None:
            return False
        idx = doc.find_point_index(layer, point)
        if idx is None:
            return False
        self.history.run(DeletePointCommand(doc, layer, idx))
        self._refresh_params()
        return True

    # ------------------------------------------------------------------ utilities
    def _refresh_params(self) -> None:
        self.param_tree.delete(*self.param_tree.get_children())
        doc = self.state.current_document
        if doc is None:
            return
        for layer in LAYERS:
            hp = doc.params[layer]["hp"]
            f0 = doc.params[layer]["f0"]
            conf = doc.uncertainty[layer].get("hp_conf", 0.0)
            source = doc.traces[layer].source
            self.param_tree.insert("", tk.END, values=(fmt(hp), fmt(f0), f"{conf:.2f}", source))

    def run(self) -> None:
        self.root.mainloop()


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}"


def gather_inputs(path: Path, max_files: Optional[int]) -> List[Path]:
    if path.is_file():
        return [path]
    files: List[Path] = []
    for candidate in sorted(path.rglob("*")):
        if candidate.suffix.lower() in IMG_EXTS | LEGACY_EXTS:
            files.append(candidate)
            if max_files and len(files) >= max_files:
                break
    return files


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ionogram GUI")
    parser.add_argument("--input", help="Optional file or directory to preload", default=None)
    parser.add_argument("--config", default=None, help="Config file (defaults to bundled config)")
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--check", action="store_true", help="Run import checks without launching GUI")
    return parser.parse_args(argv)


def headless_check(args: argparse.Namespace) -> None:
    LOGGER.info("Running headless check")
    if not args.input:
        LOGGER.info("No input provided; check complete")
        return
    files = gather_inputs(Path(args.input), args.max_files or 3)
    for path in files:
        image = _load_image_from_path(path)
        _ = create_document(path, image, LAYERS)
    LOGGER.info("Loaded %d files", len(files))


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    if args.check:
        headless_check(args)
        return
    app = MainWindow(args)
    app.run()


if __name__ == "__main__":  # pragma: no cover
    main()
