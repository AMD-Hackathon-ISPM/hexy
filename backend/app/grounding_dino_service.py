from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import threading
from typing import Any, List

import numpy as np

logger = logging.getLogger("uvicorn.error")


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
        self._white_min_channel = int(os.getenv("GDINO_WHITE_MIN_CHANNEL", "190"))
        self._white_max_spread = int(os.getenv("GDINO_WHITE_MAX_SPREAD", "35"))
        self._white_min_ratio = float(os.getenv("GDINO_WHITE_MIN_RATIO", "0.06"))

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

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> List[int]:
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep: List[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        return keep

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
        # transformers >=4.51 renamed "labels" (int IDs) to "text_labels" (strings)
        labels = result.get("text_labels") or result.get("labels", [])

        n = len(boxes) if boxes is not None else 0
        logger.info("[dino] raw %d detection(s) prompt=%r threshold=%.2f", n, config.prompt, config.box_threshold)

        if boxes is None or len(boxes) == 0:
            return []

        min_box_area = 0.005 * width * height  # reject boxes smaller than 0.5% of frame
        frame_area = width * height

        raw_boxes: List[np.ndarray] = []
        raw_scores: List[float] = []
        raw_labels: List[str] = []

        for idx, box in enumerate(boxes.cpu().numpy()):
            x1, y1, x2, y2 = [float(v) for v in box]
            x1 = max(0.0, min(x1, width))
            y1 = max(0.0, min(y1, height))
            x2 = max(0.0, min(x2, width))
            y2 = max(0.0, min(y2, height))

            box_w = x2 - x1
            box_h = y2 - y1
            box_area = box_w * box_h

            # drop tiny boxes (noise)
            if box_area < min_box_area:
                continue

            # drop boxes that cover more than 60% of the frame (hallucinated backgrounds)
            if box_area > 0.6 * frame_area:
                continue

            # drop very wide boxes — humanoids standing are taller than wide
            if box_w > 0 and box_h / box_w < 0.35:
                continue

            raw_label = labels[idx] if idx < len(labels) else config.prompt
            label = str(raw_label) if not isinstance(raw_label, str) else raw_label
            confidence = float(scores[idx].item()) if scores is not None and idx < len(scores) else 0.0

            # Color gate: humanoids are white/bright; cave walls are brown/dark.
            # Keep boxes where enough pixels are white-ish: high min channel and low channel spread.
            crop = image[int(y1):int(y2), int(x1):int(x2)]
            if crop.size > 0:
                min_ch = crop.min(axis=2)
                max_ch = crop.max(axis=2)
                whiteish = (min_ch >= self._white_min_channel) & ((max_ch - min_ch) <= self._white_max_spread)
                if whiteish.mean() < self._white_min_ratio:
                    continue

            raw_boxes.append(np.array([x1, y1, x2, y2]))
            raw_scores.append(confidence)
            raw_labels.append(label)

        if not raw_boxes:
            return []

        # NMS: collapse overlapping boxes, keep highest-confidence one
        boxes_arr = np.array(raw_boxes)
        scores_arr = np.array(raw_scores)
        keep = self._nms(boxes_arr, scores_arr, iou_threshold=0.5)

        detections: List[dict[str, Any]] = []
        for i in keep:
            x1, y1, x2, y2 = raw_boxes[i]
            detections.append({
                "label": raw_labels[i],
                "confidence": raw_scores[i],
                "bbox": [x1, y1, x2, y2],
            })

        logger.info("[dino] %d detection(s) after filtering+NMS", len(detections))
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
