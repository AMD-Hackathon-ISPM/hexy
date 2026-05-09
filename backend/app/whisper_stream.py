from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import asyncio
import io
import os
import random
import threading
import time
import wave
from typing import Deque, Iterable, List, Optional

import numpy as np


@dataclass(frozen=True)
class TranscriptChunk:
    text: str
    timestamp: float
    direction: str
    pan: float
    distance_m: float
    rms: float
    source_id: str


class TranscriptMemory:
    def __init__(self, max_chunks: int = 30) -> None:
        self._chunks: Deque[TranscriptChunk] = deque(maxlen=max_chunks)
        self._lock = threading.Lock()

    def add(self, chunk: TranscriptChunk) -> None:
        with self._lock:
            self._chunks.append(chunk)

    def combined_text(self) -> str:
        with self._lock:
            return " ".join(chunk.text for chunk in self._chunks)


class KeywordDetector:
    def __init__(self, keywords: Iterable[str]) -> None:
        self._keywords = [kw.strip().lower() for kw in keywords if kw.strip()]

    def find_keyword(self, text: str) -> Optional[str]:
        lowered = text.lower()
        for keyword in self._keywords:
            if keyword in lowered:
                return keyword
        return None


class WhisperModel:
    def __init__(self) -> None:
        self._model = None
        self._device = None
        self._lock = threading.Lock()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from faster_whisper import WhisperModel as FasterWhisperModel

            device = os.getenv("WHISPER_DEVICE", "auto").lower()
            if device == "auto":
                try:
                    import torch
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    device = "cpu"

            model_name = os.getenv("WHISPER_MODEL", "base")
            model_path = os.getenv("WHISPER_MODEL_PATH")
            compute_type = "float16" if device == "cuda" else "int8"

            self._model = FasterWhisperModel(
                model_path or model_name,
                device=device,
                compute_type=compute_type,
            )
            self._device = device

    def transcribe(self, audio: np.ndarray) -> str:
        self._ensure_model()
        if self._model is None:
            raise RuntimeError("Whisper model failed to load.")

        segments, _ = self._model.transcribe(
            audio,
            language=os.getenv("WHISPER_LANGUAGE", "en"),
        )
        return "".join(segment.text for segment in segments).strip()


class SyntheticAudioSource:
    def __init__(
        self,
        phrases: List[str],
        sample_rate: int = 16000,
        source_ids: Iterable[str] | None = None,
    ) -> None:
        self._phrases = phrases
        self._sample_rate = sample_rate
        self._cache: dict[str, np.ndarray] = {}
        self._cache_lock = threading.Lock()
        self._tts_lock = threading.Lock()
        configured_source_ids = [
            source_id.strip()
            for source_id in (
                source_ids
                or os.getenv(
                    "WHISPER_SOURCE_IDS",
                    "survivor_0,survivor_1,survivor_2,survivor_3",
                ).split(",")
            )
            if source_id.strip()
        ]
        self._source_ids = configured_source_ids or ["survivor_0"]

    def generate_phrase(self) -> tuple[np.ndarray, str, float, float, str, float]:
        phrase = random.choice(self._phrases)
        mono = self._get_or_synthesize(phrase)

        pan = random.uniform(-1.0, 1.0)
        distance_m = random.uniform(
            float(os.getenv("WHISPER_MIN_DISTANCE_M", "2.0")),
            float(os.getenv("WHISPER_MAX_DISTANCE_M", "18.0")),
        )
        source_id = random.choice(self._source_ids)

        # Inverse-distance style attenuation with a soft floor to prevent silence.
        ref_dist = float(os.getenv("WHISPER_DISTANCE_REF_M", "2.0"))
        rolloff = float(os.getenv("WHISPER_DISTANCE_ROLLOFF", "1.0"))
        attenuation = (ref_dist / max(distance_m, 0.5)) ** rolloff
        attenuation = max(0.05, min(1.0, attenuation))
        mono = mono * attenuation
        left = mono * (1.0 - max(0.0, pan))
        right = mono * (1.0 + min(0.0, pan))
        stereo = np.stack([left, right], axis=1)
        mono_mix = stereo.mean(axis=1)

        rms = float(np.sqrt(np.mean(np.square(mono_mix))))
        return mono_mix.astype(np.float32), phrase, pan, distance_m, source_id, rms

    def _get_or_synthesize(self, phrase: str) -> np.ndarray:
        with self._cache_lock:
            cached = self._cache.get(phrase)
        if cached is not None:
            return cached

        with self._tts_lock:
            audio = self._synthesize_tts(phrase)

        with self._cache_lock:
            self._cache[phrase] = audio
        return audio

    def _synthesize_tts(self, phrase: str) -> np.ndarray:
        try:
            import pyttsx3
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("pyttsx3 is required for synthetic TTS") from exc

        engine = pyttsx3.init()
        engine.setProperty("rate", int(os.getenv("TTS_RATE", "165")))
        temp_path = f"/tmp/tts_{int(time.time() * 1000)}.wav"
        engine.save_to_file(phrase, temp_path)
        engine.runAndWait()

        audio, sample_rate = self._load_wav(temp_path)
        if sample_rate != self._sample_rate:
            audio = self._resample(audio, sample_rate, self._sample_rate)
        return audio

    @staticmethod
    def _load_wav(path: str) -> tuple[np.ndarray, int]:
        with wave.open(path, "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        return audio, sample_rate

    @staticmethod
    def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if src_rate == dst_rate:
            return audio
        duration = audio.shape[0] / src_rate
        target_len = int(duration * dst_rate)
        x_old = np.linspace(0, duration, num=audio.shape[0], endpoint=False)
        x_new = np.linspace(0, duration, num=target_len, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)


def estimate_direction(pan: float) -> str:
    if pan > 0.15:
        return "right"
    if pan < -0.15:
        return "left"
    return "center"


async def stream_transcriptions(
    send_json,
    interval_sec: float = 3.0,
    source_ids: Iterable[str] | None = None,
) -> None:
    phrases = [
        "help",
        "save me",
        "over here",
    ]
    source = SyntheticAudioSource(phrases, source_ids=source_ids)
    whisper_model = WhisperModel()
    memory = TranscriptMemory()
    detector = KeywordDetector(phrases)

    emit_prob = float(os.getenv("WHISPER_EMIT_PROB", "0.75"))

    def get_delay() -> float:
        raw_min = os.getenv("WHISPER_MIN_INTERVAL_SEC")
        raw_max = os.getenv("WHISPER_MAX_INTERVAL_SEC")
        if raw_min or raw_max:
            min_delay = float(raw_min or interval_sec * 0.6)
            max_delay = float(raw_max or interval_sec * 1.6)
        else:
            min_delay = max(0.8, interval_sec * 0.6)
            max_delay = max(min_delay + 0.1, interval_sec * 1.6)
        return random.uniform(min_delay, max_delay)

    rms_threshold = float(os.getenv("WHISPER_AUDIO_RMS_THRESHOLD", "0.03"))
    rms_max = float(os.getenv("WHISPER_AUDIO_RMS_MAX", "0.7"))

    while True:
        if random.random() <= emit_prob:
            audio, phrase, pan, distance_m, source_id, rms = await asyncio.to_thread(
                source.generate_phrase
            )
            if rms < rms_threshold or rms > rms_max:
                await asyncio.sleep(get_delay())
                continue
            await send_json(
                {
                    "audio_emit": True,
                    "phrase": phrase,
                    "timestamp": time.time(),
                    "source_id": source_id,
                }
            )
            transcript = await asyncio.to_thread(whisper_model.transcribe, audio)
            timestamp = time.time()
            direction = estimate_direction(pan)

            if transcript:
                chunk = TranscriptChunk(
                    text=transcript,
                    timestamp=timestamp,
                    direction=direction,
                    pan=pan,
                    distance_m=distance_m,
                    rms=rms,
                    source_id=source_id,
                )
                memory.add(chunk)
                await send_json(
                    {
                        "transcript": transcript,
                        "timestamp": timestamp,
                        "direction": direction,
                        "pan": round(pan, 2),
                        "distance_m": round(distance_m, 2),
                        "rms": round(rms, 4),
                        "source_id": source_id,
                    }
                )

            keyword = detector.find_keyword(transcript)
            if keyword:
                await send_json({"audio_alert": True, "keyword": keyword})

        await asyncio.sleep(get_delay())
