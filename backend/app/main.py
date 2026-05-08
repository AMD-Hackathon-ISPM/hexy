from __future__ import annotations

from pathlib import Path
import asyncio
import logging
import math
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
    return _resolve_static_model_path()


def _resolve_static_model_path() -> Path:
    rl_root = os.getenv("RL_ROOT")
    if rl_root:
        return Path(rl_root) / "models" / "hexapod_static.xml"

    return (
        Path(__file__).resolve().parents[2]
        / "learning"
        / "models"
        / "hexapod_static.xml"
    )


def _is_auto_cave_model(model_path: Path) -> bool:
    if os.getenv("MODEL_PATH"):
        return False
    return model_path == _resolve_cave_root() / "cave_hexapod.xml"


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


def _ensure_hexapod_freejoint(model_xml: str) -> str:
    body_start = model_xml.find('<body name="hexapod"')
    if body_start < 0:
        return model_xml

    body_tag_end = model_xml.find(">", body_start)
    if body_tag_end < 0:
        return model_xml

    body_children_start = body_tag_end + 1
    first_children = model_xml[body_children_start : body_children_start + 300]
    if "<freejoint" in first_children or 'type="free"' in first_children:
        return model_xml

    line_start = model_xml.rfind("\n", 0, body_start) + 1
    body_indent = model_xml[line_start:body_start]
    joint_indent = f"{body_indent}  "
    return (
        model_xml[:body_children_start]
        + f'\n{joint_indent}<freejoint name="hexapod_root"/>'
        + model_xml[body_children_start:]
    )


def _prepare_model_path(model_path: Path) -> Path:
    resolved_model_path = model_path
    raw_model = model_path.read_text(encoding="utf-8")
    patched_model = _ensure_hexapod_freejoint(raw_model)
    assets_root = _resolve_assets_root()
    stl_root = assets_root / "STLFILES"
    if "file=\"../rl/STLFILES/" in patched_model:
        patched_model = patched_model.replace(
            'file="../rl/STLFILES/', f'file="{stl_root}/'
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

    # Fix meshdir path when creating temp file.
    if 'meshdir="../STLFILES"' in patched_model:
        patched_model = patched_model.replace(
            'meshdir="../STLFILES"', f'meshdir="{stl_root}"'
        )

    if patched_model != raw_model:
        tmp_name = f"hexy_{model_path.stem.replace(' ', '_')}.xml"
        tmp_path = Path("/tmp") / tmp_name
        tmp_path.write_text(patched_model, encoding="utf-8")
        resolved_model_path = tmp_path

    return resolved_model_path


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

    model_paths = [model_path]
    if _is_auto_cave_model(model_path):
        static_model_path = _resolve_static_model_path()
        if static_model_path.exists() and static_model_path != model_path:
            model_paths.append(static_model_path)

    last_error: Exception | None = None
    for candidate_path in model_paths:
        try:
            resolved_model_path = _prepare_model_path(candidate_path)
            _simulator = MujocoSimulator(resolved_model_path)
            if candidate_path != model_path:
                logger.warning(
                    "loaded fallback MuJoCo model after cave scene failure: %s",
                    candidate_path,
                )
            return _simulator
        except Exception as exc:  # pragma: no cover - runtime dependency
            last_error = exc
            if candidate_path == model_path and len(model_paths) > 1:
                logger.exception(
                    "auto-detected cave MuJoCo model failed; falling back to static model"
                )
                continue
            _simulator_error = exc
            raise

    if last_error is not None:
        _simulator_error = last_error
        raise last_error
    return _simulator


def _get_dino_service() -> GroundingDinoService:
    global _dino_service, _dino_error
    if _dino_service is not None:
        return _dino_service
    if _dino_error is not None:
        raise _dino_error

    try:
        _dino_service = GroundingDinoService()
    except ImportError as exc:
        _dino_error = RuntimeError("GroundingDINO not available - object detection disabled")
        raise _dino_error
    except Exception as exc:  # pragma: no cover - runtime dependency
        _dino_error = exc
        raise
    return _dino_service


app = FastAPI(title="Hexy Backend", version="0.1.0")
logger = logging.getLogger("uvicorn.error")
WASD_GAIT_PHASE_RATE = 4.8

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


class MujocoStepRequest(BaseModel):
    ctrl: list[float] | None = None
    n_steps: int = 1
    key: str | None = None


def _build_wasd_control(
    simulator: MujocoSimulator,
    key: str,
    sim_time: float | None = None,
) -> list[float]:
    key = key.lower().strip()
    if key not in {"w", "a", "s", "d"}:
        raise ValueError("key must be one of: w, a, s, d")

    num_ctrl = int(simulator.model.nu)
    if num_ctrl == 18:
        ctrl = [0.0] * 18
        leg_controls = (
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (9, 10, 11),
            (12, 13, 14),
            (15, 16, 17),
        )
        tripod_offsets = (0.0, 0.5, 0.5, 0.0, 0.0, 0.5)
        side_signs = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)
        t = sim_time if sim_time is not None else 0.0

        for leg_index, (coxa_idx, femur_idx, tibia_idx) in enumerate(leg_controls):
            phase = (t * WASD_GAIT_PHASE_RATE + tripod_offsets[leg_index]) % 1.0
            swing = phase < 0.5
            lift = math.sin(math.pi * min(phase * 2.0, 1.0)) if swing else 0.0
            stride = 0.75 if swing else -0.45

            if key in {"w", "s"}:
                direction = -1.0 if key == "w" else 1.0
                ctrl[coxa_idx] = direction * stride
            else:
                direction = 1.0 if key == "a" else -1.0
                ctrl[coxa_idx] = direction * side_signs[leg_index] * stride

            ctrl[femur_idx] = 0.65 + 0.25 * lift if swing else -0.25
            ctrl[tibia_idx] = -0.55 if swing else 0.25

        return ctrl

    if key == "w":
        return [-0.7] * num_ctrl
    if key == "s":
        return [0.7] * num_ctrl
    if key == "a":
        return [0.4] * num_ctrl
    return [-0.4] * num_ctrl


@app.post("/mujoco/step", response_model=MujocoState, responses={503: {"model": ErrorResponse}, 400: {"model": ErrorResponse}})
def mujoco_step(request: MujocoStepRequest) -> MujocoState:
    try:
        simulator = _get_simulator()
        if request.key is not None:
            key = request.key.lower().strip()
            n_steps = max(
                max(1, request.n_steps),
                min(50, round(0.05 / float(simulator.model.opt.timestep))),
            )
            return simulator.drive_key(
                key,
                lambda sim_time: _build_wasd_control(simulator, key, sim_time),
                n_steps=n_steps,
            )
        else:
            ctrl = request.ctrl
        return simulator.step(ctrl, n_steps=max(1, request.n_steps))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
