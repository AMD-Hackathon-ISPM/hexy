from __future__ import annotations

from pathlib import Path
import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .grounding_dino_service import GroundingDinoService
from .mujoco_sim import MujocoSimulator, MujocoState
from .whisper_stream import stream_transcriptions


class ErrorResponse(BaseModel):
    detail: str


def _parse_cors_origins(raw: str) -> list[str]:
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _resolve_model_path() -> Path:
    env_model = os.getenv("MODEL_PATH")
    if env_model:
        return Path(env_model)

    cave_root = os.getenv("CAVE_ROOT")
    if cave_root:
        candidate = Path(cave_root) / "cave_hexapod.xml"
        if candidate.exists():
            return candidate

    rl_root = os.getenv("RL_ROOT")
    if rl_root:
        candidate = Path(rl_root) / "models" / "hexapod_static.xml"
        return candidate

    # Fallback: sibling folder when running from monorepo layout.
    return Path(__file__).resolve().parents[2] / "learning" / "models" / "hexapod_static.xml"


def _resolve_assets_root() -> Path:
    rl_root = os.getenv("RL_ROOT")
    if rl_root:
        return Path(rl_root)

    return Path(__file__).resolve().parents[2] / "learning"


def _resolve_cave_root() -> Path:
    cave_root = os.getenv("CAVE_ROOT")
    if cave_root:
        return Path(cave_root)

    return Path(__file__).resolve().parents[2] / "cave_env"


_simulator: MujocoSimulator | None = None
_simulator_error: Exception | None = None
_dino_service: GroundingDinoService | None = None
_dino_error: Exception | None = None
_render_executor = ThreadPoolExecutor(max_workers=1)


def _get_simulator() -> MujocoSimulator:
    global _simulator, _simulator_error
    if _simulator is not None:
        return _simulator
    if _simulator_error is not None:
        raise _simulator_error

    model_path = _resolve_model_path()
    if not model_path.exists():
        _simulator_error = FileNotFoundError(
            f"Model file not found at {model_path}. Set RL_ROOT or MODEL_PATH."
        )
        raise _simulator_error

    resolved_model_path = model_path
    try:
        raw_model = model_path.read_text(encoding="utf-8")
        patched_model = raw_model
        if "file=\"../rl/STLFILES/" in patched_model:
            patched_model = patched_model.replace(
                'file="../rl/STLFILES/', 'file="/vendor_rl/STLFILES/'
            )

        if "file=\"meshes/" in patched_model or "file=\"./meshes/" in patched_model:
            cave_root = _resolve_cave_root()
            cave_root_str = str(cave_root)
            patched_model = patched_model.replace(
                'file="meshes/', f'file="{cave_root_str}/meshes/'
            )
            patched_model = patched_model.replace(
                'file="./meshes/', f'file="{cave_root_str}/meshes/'
            )

        if patched_model != raw_model:
            tmp_path = Path("/tmp/hexy_cave_hexapod.xml")
            tmp_path.write_text(patched_model, encoding="utf-8")
            resolved_model_path = tmp_path
    except Exception:
        resolved_model_path = model_path

    try:
        _simulator = MujocoSimulator(resolved_model_path)
    except Exception as exc:  # pragma: no cover - runtime dependency
        _simulator_error = exc
        raise
    return _simulator


def _get_dino_service() -> GroundingDinoService:
    global _dino_service, _dino_error
    if _dino_service is not None:
        return _dino_service
    if _dino_error is not None:
        raise _dino_error

    try:
        _dino_service = GroundingDinoService()
    except Exception as exc:  # pragma: no cover - runtime dependency
        _dino_error = exc
        raise
    return _dino_service


app = FastAPI(title="Hexy Backend", version="0.1.0")
logger = logging.getLogger("uvicorn.error")

assets_root = _resolve_assets_root()
if assets_root.exists():
    app.mount("/assets/rl", StaticFiles(directory=str(assets_root)), name="assets-rl")

cave_root = _resolve_cave_root()
if cave_root.exists():
    app.mount("/assets/cave", StaticFiles(directory=str(cave_root)), name="assets-cave")

cors_origins = _parse_cors_origins(
    os.getenv("CORS_ORIGINS", "http://localhost:5173")
)
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/mujoco/state", response_model=MujocoState, responses={503: {"model": ErrorResponse}})
def mujoco_state() -> MujocoState:
    try:
        simulator = _get_simulator()
        return simulator.state()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/mujoco/reset", response_model=MujocoState, responses={503: {"model": ErrorResponse}})
def mujoco_reset() -> MujocoState:
    try:
        simulator = _get_simulator()
        return simulator.reset()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.websocket("/mujoco/stream")
async def mujoco_stream(websocket: WebSocket) -> None:
    await websocket.accept()

    raw_interval = websocket.query_params.get("interval_ms", "50")
    try:
        interval_ms = int(raw_interval)
    except ValueError:
        await websocket.send_json({"detail": "interval_ms must be an integer"})
        await websocket.close(code=1003)
        return

    interval_ms = max(10, min(interval_ms, 1000))
    logger.info("mujoco stream accepted interval_ms=%s", interval_ms)

    try:
        simulator = _get_simulator()
    except Exception as exc:
        logger.exception("mujoco stream init failed")
        await websocket.send_json({"detail": str(exc)})
        await websocket.close(code=1011, reason=str(exc)[:120])
        return

    try:
        while True:
            try:
                state = simulator.state().model_dump()
                await websocket.send_json(state)
            except Exception as exc:
                logger.exception("mujoco stream send failed: %s", exc)
                try:
                    await websocket.send_json({"detail": str(exc)})
                except Exception:
                    pass
                try:
                    await websocket.close(code=1011, reason=str(exc)[:120])
                except Exception:
                    pass
                return
            await asyncio.sleep(interval_ms / 1000)
    except WebSocketDisconnect as exc:
        logger.info("mujoco stream disconnected code=%s", getattr(exc, "code", None))
        return
    except Exception as exc:
        logger.exception("mujoco stream crashed: %s", exc)
        try:
            await websocket.close(code=1011, reason=str(exc)[:120])
        except Exception:
            pass


@app.websocket("/mujoco/detections")
async def mujoco_detections(websocket: WebSocket) -> None:
    await websocket.accept()

    raw_interval = websocket.query_params.get("interval_ms", "100")
    raw_width = websocket.query_params.get("width", "640")
    raw_height = websocket.query_params.get("height", "480")
    prompt = websocket.query_params.get("prompt")
    camera_name = websocket.query_params.get("camera") or os.getenv(
        "GDINO_CAMERA_DEFAULT"
    )

    try:
        interval_ms = int(raw_interval)
        width = int(raw_width)
        height = int(raw_height)
    except ValueError:
        await websocket.send_json({"detail": "interval_ms/width/height must be integers"})
        await websocket.close(code=1003)
        return

    interval_ms = max(50, min(interval_ms, 2000))
    width = max(160, min(width, 1920))
    height = max(120, min(height, 1080))
    logger.info(
        "mujoco detections accepted interval_ms=%s width=%s height=%s",
        interval_ms,
        width,
        height,
    )

    try:
        simulator = _get_simulator()
        dino_service = _get_dino_service()
    except Exception as exc:
        logger.exception("mujoco detections init failed")
        await websocket.send_json({"detail": str(exc)})
        await websocket.close(code=1011, reason=str(exc)[:120])
        return

    detection_config = dino_service.build_config(prompt)

    try:
        while True:
            loop = asyncio.get_running_loop()
            frame = await loop.run_in_executor(
                _render_executor,
                simulator.render_rgb,
                width,
                height,
                camera_name,
            )
            detections = await asyncio.to_thread(
                dino_service.detect, frame, detection_config
            )
            await websocket.send_json({"detections": detections})
            await asyncio.sleep(interval_ms / 1000)
    except WebSocketDisconnect as exc:
        logger.info("mujoco detections disconnected code=%s", getattr(exc, "code", None))
        return
    except Exception as exc:
        logger.exception("mujoco detections crashed: %s", exc)
        try:
            await websocket.close(code=1011, reason=str(exc)[:120])
        except Exception:
            pass


@app.websocket("/audio/whisper")
async def audio_whisper(websocket: WebSocket) -> None:
    await websocket.accept()

    raw_interval = websocket.query_params.get("interval_sec", "3")
    try:
        interval_sec = float(raw_interval)
    except ValueError:
        await websocket.send_json({"detail": "interval_sec must be a number"})
        await websocket.close(code=1003)
        return

    interval_sec = max(1.0, min(interval_sec, 10.0))
    logger.info("audio whisper accepted interval_sec=%s", interval_sec)

    try:
        await stream_transcriptions(websocket.send_json, interval_sec=interval_sec)
    except WebSocketDisconnect as exc:
        logger.info("audio whisper disconnected code=%s", getattr(exc, "code", None))
        return
    except Exception as exc:
        logger.exception("audio whisper crashed: %s", exc)
        try:
            await websocket.close(code=1011, reason=str(exc)[:120])
        except Exception:
            pass
