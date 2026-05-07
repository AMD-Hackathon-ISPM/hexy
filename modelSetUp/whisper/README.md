# Whisper server-side streaming (ROCm/CUDA-ready)

This setup generates synthetic speech on the server side ("help", "save me", "over here"), runs Whisper continuously, and emits transcript chunks plus audio alert events.

## Backend endpoint

WebSocket:
```
ws://localhost:8000/audio/whisper?interval_sec=3
```

Messages:
- Transcript chunk:
```
{"transcript":"help","timestamp":1710000000.0,"direction":"left","pan":-0.4}
```
- Audio alert:
```
{"audio_alert":true,"keyword":"help"}
```

## GPU setup (ROCm/CUDA)

Install `torch` at container startup (already handled by the backend entrypoint). You can override:
```
TORCH_ACCELERATOR=cuda|rocm|cpu|auto
```

## Local run (without Docker)

1) Install dependencies:
```
pip install -r backend/requirements.txt
```

2) Ensure TTS engine is available:
- Linux: `espeak-ng`
- Windows: built-in SAPI voices

3) Start backend:
```
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Notes

- Direction is estimated from synthetic stereo pan (left/right/center).
- Keywords: help, save me, over here.
- Interval controls how often a synthetic phrase is generated.
- Random emission controls:
	- WHISPER_EMIT_PROB (0-1)
	- WHISPER_MIN_INTERVAL_SEC / WHISPER_MAX_INTERVAL_SEC
