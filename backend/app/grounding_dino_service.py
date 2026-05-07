from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import threading
import urllib.request
import urllib.error
import importlib.resources as resources
from typing import Any, List

import numpy as np

_DEFAULT_CONFIG_URL = (
    "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/"
    "groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
_DEFAULT_WEIGHTS_URL = (
    "https://huggingface.co/IDEA-Research/grounding-dino-base/resolve/main/"
    "pytorch_model.bin"
)


@dataclass(frozen=True)
class DetectionConfig:
    prompt: str
    box_threshold: float
    text_threshold: float


class GroundingDinoService:
    def __init__(self) -> None:
        self._model = None
        self._transform = None
        self._device = None
        self._load_error: Exception | None = None
        self._model_lock = threading.Lock()
        self._infer_lock = threading.Lock()

        self._box_threshold = float(os.getenv("GDINO_BOX_THRESHOLD", "0.35"))
        self._text_threshold = float(os.getenv("GDINO_TEXT_THRESHOLD", "0.25"))

    def _resolve_device(self) -> str:
        device = os.getenv("GDINO_DEVICE", "auto").lower()
        if device == "auto":
            try:
                import torch
            except Exception:
                return "cpu"
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _resolve_file(self, env_key: str, default_name: str, url: str) -> Path:
        env_path = os.getenv(env_key)
        if env_path:
            return Path(env_path)

        if default_name == "GroundingDINO_SwinT_OGC.py":
            package_path = self._find_packaged_config(default_name)
            if package_path is not None:
                return package_path

        cache_dir = Path.home() / ".cache" / "groundingdino"
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / default_name
        if not target.exists():
            self._download_file(url, target)
        return target

    @staticmethod
    def _find_packaged_config(filename: str) -> Path | None:
        try:
            config_path = resources.files("groundingdino") / "config" / filename
        except Exception:
            return None
        if config_path.is_file():
            return Path(config_path)
        return None

    @staticmethod
    def _download_file(url: str, target: Path) -> None:
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        headers = {"User-Agent": "hexy-backend"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request) as response, target.open("wb") as handle:
                handle.write(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise RuntimeError(
                    "Failed to download Grounding DINO weights (auth required). "
                    "Set HF_TOKEN/HUGGINGFACE_HUB_TOKEN or GDINO_WEIGHTS_PATH."
                ) from exc
            raise

    def _ensure_model(self) -> None:
        if self._model is not None or self._load_error is not None:
            return

        with self._model_lock:
            if self._model is not None or self._load_error is not None:
                return
            try:
                import torch
                import groundingdino.datasets.transforms as T
                from groundingdino.util import box_ops
                from groundingdino.util.inference import load_model, predict
            except Exception as exc:
                self._load_error = exc
                return

            self._device = self._resolve_device()
            if self._device != "cpu":
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.set_float32_matmul_precision("high")

            config_path = self._resolve_file(
                "GDINO_CONFIG_PATH",
                "GroundingDINO_SwinT_OGC.py",
                _DEFAULT_CONFIG_URL,
            )
            weights_path = self._resolve_file(
                "GDINO_WEIGHTS_PATH",
                "pytorch_model.bin",
                _DEFAULT_WEIGHTS_URL,
            )

            self._model = load_model(
                str(config_path),
                str(weights_path),
                device=self._device,
            )
            self._model.eval()

            self._transform = T.Compose(
                [
                    T.RandomResize([800], max_size=1333),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            self._box_ops = box_ops
            self._predict = predict
            self._torch = torch

    def detect(self, image: np.ndarray, config: DetectionConfig) -> List[dict[str, Any]]:
        self._ensure_model()
        if self._load_error is not None:
            raise RuntimeError(
                f"Grounding DINO unavailable: {self._load_error}"
            ) from self._load_error
        if self._model is None or self._transform is None:
            raise RuntimeError("Grounding DINO model was not initialized.")

        image_source = image
        height, width = image_source.shape[:2]

        pil_image = self._to_pil(image_source)
        image_tensor, _ = self._transform(pil_image, None)
        image_tensor = image_tensor.to(self._device)

        with self._infer_lock, self._torch.inference_mode():
            boxes, logits, phrases = self._predict(
                model=self._model,
                image=image_tensor,
                caption=config.prompt,
                box_threshold=config.box_threshold,
                text_threshold=config.text_threshold,
                device=self._device,
            )

        if boxes is None or len(boxes) == 0:
            return []

        boxes_xyxy = self._box_ops.box_cxcywh_to_xyxy(boxes)
        scale = self._torch.tensor([width, height, width, height], device=boxes_xyxy.device)
        boxes_xyxy = (boxes_xyxy * scale).cpu().numpy()

        scores = logits
        if scores is None:
            scores = self._torch.zeros((boxes_xyxy.shape[0],), device=self._torch.device("cpu"))
        else:
            scores = scores.squeeze(-1) if scores.ndim > 1 else scores
            scores = scores.detach().cpu()
            if scores.max().item() > 1.0 or scores.min().item() < 0.0:
                scores = self._torch.sigmoid(scores)

        detections: List[dict[str, Any]] = []
        for idx, box in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = [float(v) for v in box]
            x1 = max(0.0, min(x1, width))
            y1 = max(0.0, min(y1, height))
            x2 = max(0.0, min(x2, width))
            y2 = max(0.0, min(y2, height))

            label = phrases[idx] if phrases and idx < len(phrases) else config.prompt
            confidence = float(scores[idx].item()) if idx < len(scores) else 0.0

            detections.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                }
            )

        return detections

    @staticmethod
    def _to_pil(image: np.ndarray):
        from PIL import Image

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return Image.fromarray(image)

    def build_config(self, prompt: str | None = None) -> DetectionConfig:
        prompt_value = prompt or os.getenv("GDINO_PROMPT_DEFAULT", "person")
        return DetectionConfig(
            prompt=prompt_value,
            box_threshold=self._box_threshold,
            text_threshold=self._text_threshold,
        )
