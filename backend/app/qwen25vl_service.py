from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import json
import os
import threading
from typing import Any

# Disable CUDA graph capture — prevents GGML_ASSERT node-count mismatch
# when llm.reset() is called between create_chat_completion() calls.
os.environ.setdefault("GGML_CUDA_NO_CAPTURE", "1")

logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class Qwen25VlRequest:
    system_prompt: str
    user_prompt: str
    image_b64: str | None = None
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9


class Qwen25VlService:
    """Qwen 2.5 inference via llama-cpp-python using a local GGUF model."""

    def __init__(self) -> None:
        self._llm = None
        self._load_error: Exception | None = None
        self._model_lock = threading.Lock()
        self._loading = False
        self._load_thread: threading.Thread | None = None

        self._local_dir = Path(
            os.getenv(
                "QWEN25VL_LOCAL_DIR",
                str(Path(__file__).resolve().parents[2] / "modelSetUp" / "qwen25vl"),
            )
        )
        # QWEN_GGUF_PATH is resolved by entrypoint.sh based on available VRAM.
        # Fallback: pick the first .gguf found in local_dir.
        self._gguf_path: str | None = os.getenv("QWEN_GGUF_PATH")
        self._n_gpu_layers = int(os.getenv("QWEN_N_GPU_LAYERS", "-1"))  # -1 = all
        self._n_ctx = int(os.getenv("QWEN_N_CTX", "4096"))

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _resolve_gguf(self) -> Path:
        """Return the GGUF file to load."""
        if self._gguf_path:
            p = Path(self._gguf_path)
            if p.exists():
                return p

        # Auto-discover: prefer Q6_K over IQ4_XS
        candidates = sorted(self._local_dir.glob("*.gguf"), key=lambda p: p.name)
        if not candidates:
            raise FileNotFoundError(
                f"No .gguf files found in {self._local_dir}. "
                "Set QWEN_GGUF_PATH or place GGUF files in the model dir."
            )
        # Prefer higher quality if multiple exist
        for c in reversed(candidates):
            if "q6" in c.name.lower() or "q5" in c.name.lower() or "q8" in c.name.lower():
                return c
        return candidates[0]

    def _load_model(self) -> None:
        if self._llm is not None or self._load_error is not None:
            return

        with self._model_lock:
            if self._llm is not None or self._load_error is not None:
                return
            try:
                self._loading = True
                from llama_cpp import Llama

                gguf_path = self._resolve_gguf()
                logger.info(
                    "[qwen] loading GGUF %s  n_gpu_layers=%d  n_ctx=%d",
                    gguf_path.name,
                    self._n_gpu_layers,
                    self._n_ctx,
                )

                self._llm = Llama(
                    model_path=str(gguf_path),
                    n_gpu_layers=self._n_gpu_layers,
                    n_ctx=self._n_ctx,
                    verbose=False,
                )
                logger.info("[qwen] model ready (%s)", gguf_path.name)
            except Exception as exc:
                logger.exception("[qwen] load FAILED: %s", exc)
                self._load_error = exc
            finally:
                self._loading = False

    def ensure_model_async(self) -> None:
        if self._llm is not None or self._load_error is not None:
            return
        if self._loading:
            return
        if self._load_thread is not None and self._load_thread.is_alive():
            return
        self._load_thread = threading.Thread(target=self._load_model, daemon=True)
        self._load_thread.start()

    @property
    def is_ready(self) -> bool:
        return self._llm is not None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, payload: Qwen25VlRequest) -> dict[str, Any]:
        if self._load_error is not None:
            raise RuntimeError(f"Qwen2.5 unavailable: {self._load_error}") from self._load_error
        if not self.is_ready:
            self.ensure_model_async()
            raise RuntimeError("Qwen2.5 is still loading. Try again shortly.")

        # Build chat messages (Qwen chat template: system + user)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": payload.system_prompt},
            {"role": "user", "content": payload.user_prompt},
        ]

        # Note: image_b64 is ignored for text-only GGUF models.
        # The user prompt already contains structured JSON with
        # detection + audio data, so vision is not required.

        # Reset KV cache between calls to prevent llama_decode -1 (context overflow).
        self._llm.reset()
        result = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=payload.max_new_tokens,
            temperature=max(payload.temperature, 1e-8),
            top_p=payload.top_p,
        )

        text = ""
        if result and "choices" in result and result["choices"]:
            msg = result["choices"][0].get("message", {})
            text = msg.get("content", "")

        json_payload = self._extract_json(text)
        return {"text": text, "json": json_payload}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < 0 or end <= start:
            return None
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return None
