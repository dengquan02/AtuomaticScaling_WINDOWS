"""TensorFlow model loading and inference helpers."""

import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

import yaml

import tensorflow as tf

from app.utils import resolve_relative_path


LOGGER = logging.getLogger(__name__)


class ModelService:
    """Thin wrapper around the DIAS TensorFlow model for local inference."""

    def __init__(self, config_path: Union[str, Path], gpu_id: Optional[int] = None) -> None:
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as fh:
            self.cfgs = yaml.safe_load(fh)
        self.gpu_id = gpu_id
        self._prepare_devices()
        self.model = self._load_model()

    def _prepare_devices(self) -> None:
        if self.gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
            LOGGER.info("Using GPU id %s", self.gpu_id)

    def _load_model(self):
        model_path = resolve_relative_path(self.cfgs["Test"]["ModelPath"])
        LOGGER.info("Loading model from %s", model_path)
        try:
            # 优先使用常规 Keras 接口加载（兼容旧环境）
            return tf.keras.models.load_model(model_path, compile=False)
        except tf.errors.ResourceExhaustedError as err:
            LOGGER.warning("GPU memory exhausted (%s). Falling back to CPU.", err)
            self._force_cpu()
            return tf.keras.models.load_model(model_path, compile=False)
        except ValueError as err:
            # Keras 3 不再支持直接 load SavedModel，回退到 TFSMLayer 以仅做推理
            msg = str(err)
            if "File format not supported" in msg and ".model" in msg:
                LOGGER.warning(
                    "Keras 3 cannot load SavedModel via load_model; "
                    "falling back to keras.layers.TFSMLayer for inference only."
                )
                try:
                    from keras.layers import TFSMLayer  # type: ignore[import]

                    layer = TFSMLayer(str(model_path), call_endpoint="serving_default")
                    return layer
                except Exception as inner_err:  # pragma: no cover - 环境相关
                    LOGGER.error("Failed to load SavedModel via TFSMLayer: %s", inner_err)
                    raise
            raise

    def _force_cpu(self) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:  # pragma: no cover - depends on runtime
            pass

    def predict(self, batch_input) -> Any:
        """Run inference and return numpy probabilities.

        在 Keras 2.x 场景下，self.model 通常是 tf.keras.Model；
        在 Keras 3 + SavedModel 场景下，可能是 TFSMLayer。
        这里统一通过可调用接口获取输出并转为 numpy。
        """

        model = self.model
        if hasattr(model, "predict"):
            preds = model.predict(batch_input, verbose=0)  # type: ignore[call-arg]
        else:
            preds = model(batch_input)

        # TFSMLayer 可能返回 dict，需要取第一个 tensor
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  if isinstance(preds, dict):
            preds = next(iter(preds.values()))

        if hasattr(preds, "numpy"):
            return preds.numpy()
        return preds

    def describe(self) -> dict:
        return {
            "config_path": str(self.config_path),
            "model_path": self.cfgs["Test"]["ModelPath"],
        }


def dump_model_summary(model_service: ModelService) -> str:
    buffer = []
    model_service.model.summary(print_fn=buffer.append)
    return "\n".join(buffer)
