"""Command-line interface for local ionogram inference."""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.decoder import load_image, load_legacy, to_model_input
from app.model_service import ModelService
from app.params import LAYERS, estimate_uncertainty, extract_params
from app.overlay import overlay_traces
from app import export, project
from app.utils import resolve_config_path


LOGGER = logging.getLogger("app.cli")

IMG_EXTS = {".png", ".jpg", ".jpeg"}
LEGACY_EXTS = {".pickle", ".pkl"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local ionogram interpreter")
    parser.add_argument("--input", required=True, help="File or directory with ionogram images")
    parser.add_argument("--config", default=None, help="YAML config for the model (defaults to bundled config)")
    parser.add_argument("--out", default="result/local", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.3, help="Detection threshold")
    parser.add_argument("--gpu-id", type=int, default=None, help="GPU id to use (optional)")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files processed")
    parser.add_argument("--no-overlays", action="store_true", help="Skip overlay image generation")
    parser.add_argument("--no-project", action="store_true", help="Skip project file generation")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def gather_inputs(path: Path, max_files: Optional[int]) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    candidates: List[Path] = []
    for p in sorted(path.rglob("*")):
        if p.suffix.lower() in IMG_EXTS | LEGACY_EXTS:
            candidates.append(p)
            if max_files and len(candidates) >= max_files:
                break
    return candidates


def load_sample(path: Path) -> np.ndarray:
    if path.suffix.lower() in IMG_EXTS:
        return load_image(path)
    if path.suffix.lower() in LEGACY_EXTS:
        return load_legacy(path)
    raise ValueError(f"Unsupported file type: {path}")


def flatten_params(file_path: Path, params, unc) -> Dict[str, object]:
    row: Dict[str, object] = {"file": str(file_path)}
    for layer in LAYERS:
        prefix = layer.lower()
        hp = params[layer]["hp"]
        f0 = params[layer]["f0"]
        row[f"hp_{prefix}"] = "" if hp is None else round(hp, 2)
        row[f"f0_{prefix}"] = "" if f0 is None else round(f0, 2)
        conf = min(unc[layer]["hp_conf"], unc[layer]["f_conf"])
        row[f"confidence_{prefix}"] = round(conf, 4)
    return row


def run_infer(args: argparse.Namespace) -> None:
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")
    input_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = gather_inputs(input_path, args.max_files)
    if not files:
        LOGGER.warning("No matching files under %s", input_path)
        return
    LOGGER.info("Found %d files", len(files))
    config_path = resolve_config_path(args.config)
    args.config = str(config_path)
    ms = ModelService(config_path, gpu_id=args.gpu_id)
    rows = []
    summaries = []
    errors = []
    start = time.time()
    for idx, path in enumerate(files, start=1):
        t0 = time.time()
        try:
            arr = load_sample(path)
            batch, original = to_model_input(arr)
            preds = ms.predict(batch)
            params = extract_params(preds, args.threshold)
            unc = estimate_uncertainty(preds, args.threshold)
            rows.append(flatten_params(path, params, unc))
            overlay_path = out_dir / "overlays" / f"{path.stem}.png"
            if not args.no_overlays:
                overlay_traces(original, preds, args.threshold, overlay_path, title=path.name)
            summaries.append(
                {
                    "file": str(path),
                    "params": params,
                    "uncertainty": unc,
                    "overlay": str(overlay_path.relative_to(out_dir)) if not args.no_overlays else None,
                    "elapsed_sec": round(time.time() - t0, 3),
                }
            )
            LOGGER.info("[%d/%d] Processed %s", idx, len(files), path.name)
        except Exception as exc:  # pragma: no cover - runtime guard
            LOGGER.exception("Failed to process %s", path)
            errors.append({"file": str(path), "error": str(exc)})
    csv_path = out_dir / "params.csv"
    export.to_csv(rows, csv_path)
    summary_payload = {
        "total": len(files),
        "processed": len(rows),
        "failed": len(errors),
        "threshold": args.threshold,
        "config": args.config,
        "duration_sec": round(time.time() - start, 2),
        "items": summaries,
        "errors": errors,
    }
    export.to_json(summary_payload, out_dir / "summary.json")
    if not args.no_project:
        project.save_project(
            {
                "settings": {
                    "threshold": args.threshold,
                    "config": args.config,
                },
                "items": summaries,
                "errors": errors,
            },
            out_dir / "project.ionproj",
        )
    LOGGER.info("Done. Results saved to %s", out_dir)


def main() -> None:
    args = parse_args()
    run_infer(args)


if __name__ == "__main__":
    main()
