from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import time
import urllib.request
from typing import Any, Iterable, List

import numpy as np

_DEFAULT_CONFIG_URL = (
    "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/"
    "groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
_DEFAULT_WEIGHTS_URL = (
    "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0/"
    "groundingdino_swint_ogc.pth"
)


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: List[float]


class GroundingDinoInferencer:
    def __init__(
        self,
        config_path: str | None = None,
        weights_path: str | None = None,
        device: str | None = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> None:
        self._box_threshold = box_threshold
        self._text_threshold = text_threshold
        self._device = device or self._resolve_device()

        self._config_path = Path(
            config_path
            or os.getenv("GDINO_CONFIG_PATH")
            or self._resolve_file(
                "GroundingDINO_SwinT_OGC.py", _DEFAULT_CONFIG_URL
            )
        )
        self._weights_path = Path(
            weights_path
            or os.getenv("GDINO_WEIGHTS_PATH")
            or self._resolve_file(
                "groundingdino_swint_ogc.pth", _DEFAULT_WEIGHTS_URL
            )
        )

        self._model = None
        self._transform = None
        self._box_ops = None
        self._predict = None
        self._torch = None

        self._load_model()

    def _resolve_device(self) -> str:
        device = os.getenv("GDINO_DEVICE", "auto").lower()
        if device == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _resolve_file(self, name: str, url: str) -> Path:
        cache_dir = Path.home() / ".cache" / "groundingdino"
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / name
        if not target.exists():
            urllib.request.urlretrieve(url, target)  # noqa: S310 - trusted source
        return target

    def _load_model(self) -> None:
        import torch
        import groundingdino.datasets.transforms as T
        from groundingdino.util import box_ops
        from groundingdino.util.inference import load_model, predict

        if self._device != "cpu":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        self._model = load_model(
            str(self._config_path),
            str(self._weights_path),
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

    def infer(self, image: np.ndarray, prompt: str) -> List[Detection]:
        if self._model is None:
            raise RuntimeError("Grounding DINO model is not loaded.")

        height, width = image.shape[:2]
        pil_image = self._to_pil(image)
        image_tensor, _ = self._transform(pil_image, None)
        image_tensor = image_tensor.to(self._device)

        with self._torch.inference_mode():
            boxes, logits, phrases = self._predict(
                model=self._model,
                image=image_tensor,
                caption=prompt,
                box_threshold=self._box_threshold,
                text_threshold=self._text_threshold,
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

        detections: List[Detection] = []
        for idx, box in enumerate(boxes_xyxy):
            x1, y1, x2, y2 = [float(v) for v in box]
            x1 = max(0.0, min(x1, width))
            y1 = max(0.0, min(y1, height))
            x2 = max(0.0, min(x2, width))
            y2 = max(0.0, min(y2, height))

            label = phrases[idx] if phrases and idx < len(phrases) else prompt
            confidence = float(scores[idx].item()) if idx < len(scores) else 0.0

            detections.append(
                Detection(label=label, confidence=confidence, bbox=[x1, y1, x2, y2])
            )

        return detections

    def draw_detections(
        self, image: np.ndarray, detections: Iterable[Detection]
    ) -> np.ndarray:
        from PIL import ImageDraw, ImageFont

        canvas = self._to_pil(image).copy()
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            draw.rectangle((x1, y1, x2, y2), outline=(255, 64, 64), width=2)
            label = f"{det.label} {det.confidence:.2f}"
            draw.text((x1 + 4, y1 + 4), label, fill=(255, 255, 255), font=font)

        return np.array(canvas)

    @staticmethod
    def _to_pil(image: np.ndarray):
        from PIL import Image

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return Image.fromarray(image)


def detections_to_json(detections: Iterable[Detection]) -> str:
    payload = {
        "detections": [
            {"label": d.label, "confidence": d.confidence, "bbox": d.bbox}
            for d in detections
        ]
    }
    return json.dumps(payload)


def run_demo(image: np.ndarray, prompts: list[str]) -> None:
    inferencer = GroundingDinoInferencer()

    for prompt in prompts:
        start = time.perf_counter()
        detections = inferencer.infer(image, prompt)
        latency_ms = (time.perf_counter() - start) * 1000
        print(f"Prompt: {prompt} | detections={len(detections)} | {latency_ms:.1f}ms")
        print(detections_to_json(detections))


if __name__ == "__main__":
    # Replace this with a MuJoCo-rendered frame or any RGB numpy array.
    sample = np.zeros((720, 1280, 3), dtype=np.uint8)
    run_demo(
        sample,
        ["person", "trapped injured person", "person lying in cave"],
    )
