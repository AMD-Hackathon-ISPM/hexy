from __future__ import annotations

from dataclasses import dataclass
import os
import threading
from typing import Any, List

import numpy as np


@dataclass(frozen=True)
class DetectionConfig:
    prompt: str
    box_threshold: float
    text_threshold: float


class GroundingDinoService:
    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device = None
        self._load_error: Exception | None = None
        self._model_lock = threading.Lock()
        self._infer_lock = threading.Lock()

        self._box_threshold = float(os.getenv("GDINO_BOX_THRESHOLD", "0.35"))
        self._text_threshold = float(os.getenv("GDINO_TEXT_THRESHOLD", "0.25"))
        self._hf_model = os.getenv("GDINO_HF_MODEL", "LeBabyOx/dino-cave-survivor")
        self._hf_subfolder = os.getenv("GDINO_HF_SUBFOLDER", "checkpoint_epoch_8")

    def _resolve_device(self) -> str:
        device = os.getenv("GDINO_DEVICE", "auto").lower()
        if device == "auto":
            try:
                import torch
            except Exception:
                return "cpu"
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _ensure_model(self) -> None:
        if self._model is not None or self._load_error is not None:
            return

        with self._model_lock:
            if self._model is not None or self._load_error is not None:
                return
            try:
                import torch
                from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

                self._device = self._resolve_device()

                self._processor = AutoProcessor.from_pretrained(
                    self._hf_model,
                    subfolder=self._hf_subfolder,
                )
                self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                    self._hf_model,
                    subfolder=self._hf_subfolder,
                ).to(self._device)
                self._model.eval()
                self._torch = torch
            except Exception as exc:
                self._load_error = exc

    def detect(self, image: np.ndarray, config: DetectionConfig) -> List[dict[str, Any]]:
        self._ensure_model()
        if self._load_error is not None:
            raise RuntimeError(
                f"Grounding DINO unavailable: {self._load_error}"
            ) from self._load_error
        if self._model is None or self._processor is None:
            raise RuntimeError("Grounding DINO model was not initialized.")

        height, width = image.shape[:2]
        pil_image = self._to_pil(image)

        inputs = self._processor(
            images=pil_image,
            text=config.prompt,
            return_tensors="pt",
        ).to(self._device)

        with self._infer_lock, self._torch.inference_mode():
            outputs = self._model(**inputs)

        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=config.box_threshold,
            target_sizes=[(height, width)],
        )

        result = results[0]
        boxes = result.get("boxes")
        scores = result.get("scores")
        labels = result.get("labels", [])

        if boxes is None or len(boxes) == 0:
            return []

        detections: List[dict[str, Any]] = []
        for idx, box in enumerate(boxes.cpu().numpy()):
            x1, y1, x2, y2 = [float(v) for v in box]
            x1 = max(0.0, min(x1, width))
            y1 = max(0.0, min(y1, height))
            x2 = max(0.0, min(x2, width))
            y2 = max(0.0, min(y2, height))

            label = labels[idx] if idx < len(labels) else config.prompt
            confidence = float(scores[idx].item()) if scores is not None and idx < len(scores) else 0.0

            detections.append({
                "label": label,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
            })

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
