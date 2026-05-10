from __future__ import annotations

from pathlib import Path
import asyncio
import logging
import math
import os
import tempfile
import json
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import numpy as np

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .autonomous_agent import AutonomousAgent
from .grounding_dino_service import GroundingDinoService
from .agent_prompt import get_agent_prompt
from .mujoco_sim import MujocoSimulator, MujocoState
from .qwen25vl_service import Qwen25VlRequest, Qwen25VlService
from .whisper_stream import stream_transcriptions


class ErrorResponse(BaseModel):
    detail: str


class DebugCaptureRequest(BaseModel):
    count: int = 1
    prefix: str | None = None


class DebugCaptureResponse(BaseModel):
    remaining: int
    directory: str


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

    generated_cave_model = _resolve_cave_root() / "cave_hexapod.xml"
    if generated_cave_model.exists() and not os.getenv("RL_ROOT"):
        return generated_cave_model

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

    repo_root = Path(__file__).resolve().parents[2]
    generated_cave_root = repo_root / "cave-gen" / "cave_env"
    if generated_cave_root.exists():
        return generated_cave_root
    return repo_root / "cave_env"


_simulator: MujocoSimulator | None = None
_simulator_error: Exception | None = None
_dino_service: GroundingDinoService | None = None
_dino_error: Exception | None = None
_render_executor = ThreadPoolExecutor(max_workers=1)
_qwen_service: Qwen25VlService | None = None
_qwen_error: Exception | None = None

# Protects all simulator state mutations (step / drive_key).
_sim_lock = threading.Lock()

# Autonomous agent singleton
_autonomous_agent: AutonomousAgent = AutonomousAgent()

# Latest audio context from active whisper WebSocket connections
_latest_audio: dict[str, object] = {}
_latest_audio_lock = threading.Lock()


class DinoDebugCapture:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._remaining = 0
        self._prefix = "dino"
        self._directory = Path(os.getenv("GDINO_DEBUG_DIR", "/tmp/dino_debug"))

    def request(self, count: int, prefix: str | None) -> DebugCaptureResponse:
        safe_count = max(1, min(int(count), 25))
        with self._lock:
            self._remaining = safe_count
            if prefix:
                self._prefix = "".join(ch for ch in prefix if ch.isalnum() or ch in "-_")[:32]
        return DebugCaptureResponse(
            remaining=self.remaining(),
            directory=str(self._directory),
        )

    def remaining(self) -> int:
        with self._lock:
            return self._remaining

    def capture(self, frame, camera_label: str, width: int, height: int) -> str | None:
        with self._lock:
            if self._remaining <= 0:
                return None
            self._remaining -= 1
            index = self._remaining
            prefix = self._prefix

        self._directory.mkdir(parents=True, exist_ok=True)
        timestamp_ms = int(time.time() * 1000)
        safe_camera = "".join(
            ch for ch in (camera_label or "default") if ch.isalnum() or ch in "-_"
        )
        filename = f"{prefix}_{safe_camera}_{width}x{height}_{timestamp_ms}_{index}.png"
        filepath = self._directory / filename
        from PIL import Image

        if frame.dtype != np.uint8:
            frame = frame.clip(0, 255).astype("uint8")
        Image.fromarray(frame).save(filepath)
        return str(filepath)


_dino_debug_capture = DinoDebugCapture()


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
        tmp_path = Path(tempfile.gettempdir()) / tmp_name
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


def _get_qwen_service() -> Qwen25VlService:
    global _qwen_service, _qwen_error
    if _qwen_service is not None:
        return _qwen_service
    if _qwen_error is not None:
        raise _qwen_error

    try:
        _qwen_service = Qwen25VlService()
    except Exception as exc:  # pragma: no cover - runtime dependency
        _qwen_error = exc
        raise
    return _qwen_service


app = FastAPI(title="Hexy Backend", version="0.1.0")
logger = logging.getLogger("uvicorn.error")
WASD_GAIT_PHASE_RATE = 4.8
WASD_STEP_LENGTH = 0.055
WASD_TURN_STEP_LENGTH = 0.045
WASD_SWING_HEIGHT = 0.035
WASD_STANCE_PRESS = -0.004
COXA_KP = 1.6
LEG_KP = 2.2
JOINT_KD = 0.06
LEG_CTRL_LIMIT = 0.9
JOINT_LIMITS_RAD = {
    "coxa": (math.radians(-45.0), math.radians(45.0)),
    "femur": (math.radians(-70.0), math.radians(70.0)),
    "tibia": (math.radians(-120.0), math.radians(20.0)),
}

WASD_LEG_RAW_SPECS = (
    {
        "name": "front_left",
        "actuators": (0, 1, 2),
        "axis_sign": 1.0,
        "phase_offset": 0.0,
        "base": (0.030114, -0.037986, -0.012402),
        "femur": (0.009459, -0.002609, 0.007674),
        "tibia": (0.044345, -0.044318, -0.004287),
        "foot": (0.006977, -0.014958, -0.055788),
    },
    {
        "name": "front_right",
        "actuators": (3, 4, 5),
        "axis_sign": -1.0,
        "phase_offset": 0.5,
        "base": (-0.030114, -0.037986, -0.012402),
        "femur": (-0.009459, -0.002609, 0.007674),
        "tibia": (-0.044345, -0.044318, -0.004287),
        "foot": (-0.006977, -0.014958, -0.055788),
    },
    {
        "name": "center_left",
        "actuators": (6, 7, 8),
        "axis_sign": 1.0,
        "phase_offset": 0.5,
        "base": (0.035859, 0.001752, 0.014379),
        "femur": (0.011493, 0.010205, -0.019107),
        "tibia": (0.062694, 0.000019, -0.004287),
        "foot": (0.015510, -0.005644, -0.055788),
    },
    {
        "name": "center_right",
        "actuators": (9, 10, 11),
        "axis_sign": -1.0,
        "phase_offset": 0.0,
        "base": (-0.035859, 0.001752, 0.014379),
        "femur": (-0.011493, 0.010205, -0.019107),
        "tibia": (-0.062694, 0.000019, -0.004287),
        "foot": (-0.015510, -0.005644, -0.055788),
    },
    {
        "name": "back_left",
        "actuators": (12, 13, 14),
        "axis_sign": 1.0,
        "phase_offset": 0.0,
        "base": (0.015530, 0.043299, -0.012404),
        "femur": (0.005152, 0.012775, 0.007676),
        "tibia": (0.044318, 0.044345, -0.004287),
        "foot": (0.014958, 0.006977, -0.055788),
    },
    {
        "name": "back_right",
        "actuators": (15, 16, 17),
        "axis_sign": -1.0,
        "phase_offset": 0.5,
        "base": (-0.015530, 0.043299, -0.012404),
        "femur": (-0.005152, 0.012775, 0.007676),
        "tibia": (-0.044318, 0.044345, -0.004287),
        "foot": (-0.014958, 0.006977, -0.055788),
    },
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _wrap_angle(value: float) -> float:
    while value > math.pi:
        value -= math.tau
    while value < -math.pi:
        value += math.tau
    return value


def _smoothstep(value: float) -> float:
    value = _clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _v_add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _v_sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot_xy(a: tuple[float, float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _rotate_z(vec: tuple[float, float, float], angle: float) -> tuple[float, float, float]:
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return (
        vec[0] * cos_a - vec[1] * sin_a,
        vec[0] * sin_a + vec[1] * cos_a,
        vec[2],
    )


def _prepare_leg_spec(raw: dict[str, object]) -> dict[str, object]:
    femur = raw["femur"]
    tibia = raw["tibia"]
    foot = raw["foot"]
    assert isinstance(femur, tuple)
    assert isinstance(tibia, tuple)
    assert isinstance(foot, tuple)

    rest_rel = _v_add(_v_add(femur, tibia), foot)
    rest_xy_len = max(math.hypot(rest_rel[0], rest_rel[1]), 1e-6)
    plane_dir = (rest_rel[0] / rest_xy_len, rest_rel[1] / rest_xy_len)
    femur_plane_x = _dot_xy(femur, plane_dir)
    tibia_plane_x = _dot_xy(tibia, plane_dir)
    foot_plane_x = _dot_xy(foot, plane_dir)
    upper_len = max(math.hypot(tibia_plane_x, tibia[2]), 1e-6)
    lower_len = max(math.hypot(foot_plane_x, foot[2]), 1e-6)
    rest_upper_angle = math.atan2(tibia[2], tibia_plane_x)
    rest_lower_angle = math.atan2(foot[2], foot_plane_x)
    rest_relative_lower = _wrap_angle(rest_lower_angle - rest_upper_angle)

    rest_target_x = tibia_plane_x + foot_plane_x
    rest_target_z = tibia[2] + foot[2]
    rest_dist = _clamp(
        math.hypot(rest_target_x, rest_target_z),
        abs(upper_len - lower_len) + 1e-5,
        upper_len + lower_len - 1e-5,
    )
    rest_target_angle = math.atan2(rest_target_z, rest_target_x)
    cos_shoulder = _clamp(
        (upper_len * upper_len + rest_dist * rest_dist - lower_len * lower_len)
        / (2.0 * upper_len * rest_dist),
        -1.0,
        1.0,
    )
    shoulder_offset = math.acos(cos_shoulder)
    plus_error = abs(_wrap_angle(rest_target_angle + shoulder_offset - rest_upper_angle))
    minus_error = abs(_wrap_angle(rest_target_angle - shoulder_offset - rest_upper_angle))
    elbow_sign = 1.0 if plus_error <= minus_error else -1.0

    spec = dict(raw)
    spec.update(
        {
            "rest_rel": rest_rel,
            "rest_foot": _v_add(raw["base"], rest_rel),  # type: ignore[arg-type]
            "rest_yaw": math.atan2(rest_rel[1], rest_rel[0]),
            "plane_dir": plane_dir,
            "femur_plane_x": femur_plane_x,
            "femur_z": femur[2],
            "upper_len": upper_len,
            "lower_len": lower_len,
            "rest_upper_angle": rest_upper_angle,
            "rest_relative_lower": rest_relative_lower,
            "elbow_sign": elbow_sign,
        }
    )
    return spec


WASD_LEG_SPECS = tuple(_prepare_leg_spec(raw) for raw in WASD_LEG_RAW_SPECS)

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


@app.on_event("startup")
def warm_models() -> None:
    try:
        _get_dino_service().ensure_model_async()
    except Exception:
        logger.exception("[dino] warmup failed")
    try:
        _get_qwen_service().ensure_model_async()
    except Exception:
        logger.exception("[qwen25vl] warmup failed")
    # Start autonomous agent — the three loops handle unready services gracefully
    try:
        sim = _get_simulator()
        _autonomous_agent.start(
            simulator=sim,
            dino_service=_get_dino_service(),
            qwen_service=_get_qwen_service(),
            sim_lock=_sim_lock,
            build_control_fn=_build_wasd_control,
            render_fn=lambda w, h, cam_id: _render_executor.submit(
                sim.render_rgb_with_camera_id, w, h, cam_id
            ).result(),
        )
    except Exception:
        logger.exception("[agent] autonomous agent warmup failed")


@app.get("/agent/prompt")
def agent_prompt() -> dict[str, str]:
    return {"prompt": get_agent_prompt()}


@app.get("/agent/status")
def agent_status() -> dict[str, object]:
    try:
        qwen = _get_qwen_service()
        if qwen._load_error is not None:  # type: ignore[attr-defined]
            return {"status": "error", "detail": str(qwen._load_error)}
        status = "ready" if qwen.is_ready else "loading"
        return {"status": status}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


@app.post(
    "/mujoco/detections/debug",
    response_model=DebugCaptureResponse,
    responses={400: {"model": ErrorResponse}},
)
def request_dino_debug_capture(payload: DebugCaptureRequest) -> DebugCaptureResponse:
    try:
        return _dino_debug_capture.request(payload.count, payload.prefix)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/mujoco/detections/frame", responses={503: {"model": ErrorResponse}})
def dino_debug_frame(
    width: int = 640,
    height: int = 360,
    camera: str | None = None,
) -> Response:
    try:
        simulator = _get_simulator()
        camera_id, resolved_camera, fallback = simulator.resolve_camera(camera)
        if fallback:
            logger.info(
                "[dino] frame camera fallback requested=%r resolved=%s",
                camera,
                resolved_camera,
            )
        frame = _render_executor.submit(
            simulator.render_rgb_with_camera_id,
            width,
            height,
            camera_id,
        ).result()
        from PIL import Image

        buf = BytesIO()
        Image.fromarray(frame).save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


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


class AgentRespondRequest(BaseModel):
    instruction: str
    detections: list[dict[str, object]] | None = None
    audio_transcript: str | None = None
    audio_direction: str | None = None
    audio_distance_m: float | None = None
    image_base64: str | None = None
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None


def _foot_target_for_key(
    spec: dict[str, object],
    key: str,
    sim_time: float,
) -> tuple[float, float, float]:
    rest_foot = spec["rest_foot"]
    assert isinstance(rest_foot, tuple)
    phase_offset = float(spec["phase_offset"])
    phase = (sim_time * WASD_GAIT_PHASE_RATE + phase_offset) % 1.0

    if key in {"w", "s"}:
        direction = 1.0 if key == "w" else -1.0
        step_vec = (0.0, direction * WASD_STEP_LENGTH, 0.0)
    else:
        turn_direction = 1.0 if key == "a" else -1.0
        tangent = (-rest_foot[1], rest_foot[0])
        tangent_len = max(math.hypot(tangent[0], tangent[1]), 1e-6)
        step_vec = (
            turn_direction * tangent[0] / tangent_len * WASD_TURN_STEP_LENGTH,
            turn_direction * tangent[1] / tangent_len * WASD_TURN_STEP_LENGTH,
            0.0,
        )

    if phase < 0.5:
        swing_progress = _smoothstep(phase * 2.0)
        travel = swing_progress - 0.5
        lift = math.sin(math.pi * phase * 2.0) * WASD_SWING_HEIGHT
        z = rest_foot[2] + lift
    else:
        stance_progress = _smoothstep((phase - 0.5) * 2.0)
        travel = 0.5 - stance_progress
        z = rest_foot[2] + WASD_STANCE_PRESS

    return (
        rest_foot[0] + step_vec[0] * travel,
        rest_foot[1] + step_vec[1] * travel,
        z,
    )


def _solve_leg_ik(
    spec: dict[str, object],
    target_foot: tuple[float, float, float],
) -> tuple[float, float, float]:
    base = spec["base"]
    plane_dir = spec["plane_dir"]
    assert isinstance(base, tuple)
    assert isinstance(plane_dir, tuple)

    target_rel = _v_sub(target_foot, base)
    rest_yaw = float(spec["rest_yaw"])
    axis_sign = float(spec["axis_sign"])
    physical_coxa = _wrap_angle(math.atan2(target_rel[1], target_rel[0]) - rest_yaw)
    physical_coxa = _clamp(physical_coxa, *JOINT_LIMITS_RAD["coxa"])

    rotated_rel = _rotate_z(target_rel, -physical_coxa)
    target_x = _dot_xy(rotated_rel, plane_dir) - float(spec["femur_plane_x"])
    target_z = rotated_rel[2] - float(spec["femur_z"])

    upper_len = float(spec["upper_len"])
    lower_len = float(spec["lower_len"])
    min_dist = abs(upper_len - lower_len) + 1e-5
    max_dist = upper_len + lower_len - 1e-5
    target_dist = math.hypot(target_x, target_z)
    if target_dist < 1e-6:
        target_x = min_dist
        target_z = 0.0
        target_dist = min_dist
    elif target_dist < min_dist or target_dist > max_dist:
        clamped_dist = _clamp(target_dist, min_dist, max_dist)
        scale = clamped_dist / target_dist
        target_x *= scale
        target_z *= scale
        target_dist = clamped_dist

    target_angle = math.atan2(target_z, target_x)
    cos_shoulder = _clamp(
        (upper_len * upper_len + target_dist * target_dist - lower_len * lower_len)
        / (2.0 * upper_len * target_dist),
        -1.0,
        1.0,
    )
    upper_angle = target_angle + float(spec["elbow_sign"]) * math.acos(cos_shoulder)
    knee_x = upper_len * math.cos(upper_angle)
    knee_z = upper_len * math.sin(upper_angle)
    lower_angle = math.atan2(target_z - knee_z, target_x - knee_x)

    femur_physical = _wrap_angle(upper_angle - float(spec["rest_upper_angle"]))
    tibia_physical = _wrap_angle(
        _wrap_angle(lower_angle - upper_angle) - float(spec["rest_relative_lower"])
    )

    return (
        _clamp(physical_coxa / axis_sign, *JOINT_LIMITS_RAD["coxa"]),
        _clamp(femur_physical / axis_sign, *JOINT_LIMITS_RAD["femur"]),
        _clamp(tibia_physical / axis_sign, *JOINT_LIMITS_RAD["tibia"]),
    )


def _actuator_joint_state(
    simulator: MujocoSimulator,
    actuator_index: int,
) -> tuple[float, float]:
    joint_id = int(simulator.model.actuator_trnid[actuator_index, 0])
    qpos_index = int(simulator.model.jnt_qposadr[joint_id])
    qvel_index = int(simulator.model.jnt_dofadr[joint_id])
    return float(simulator.data.qpos[qpos_index]), float(simulator.data.qvel[qvel_index])


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
        t = sim_time if sim_time is not None else 0.0

        for spec in WASD_LEG_SPECS:
            actuators = spec["actuators"]
            assert isinstance(actuators, tuple)
            target_foot = _foot_target_for_key(spec, key, t)
            joint_targets = _solve_leg_ik(spec, target_foot)

            for joint_index, actuator_index in enumerate(actuators):
                current_angle, joint_velocity = _actuator_joint_state(
                    simulator, int(actuator_index)
                )
                kp = COXA_KP if joint_index == 0 else LEG_KP
                angle_error = joint_targets[joint_index] - current_angle
                ctrl[int(actuator_index)] = _clamp(
                    kp * angle_error - JOINT_KD * joint_velocity,
                    -LEG_CTRL_LIMIT,
                    LEG_CTRL_LIMIT,
                )

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
        with _sim_lock:
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


@app.get("/agent/autonomous/status")
def agent_autonomous_status() -> dict[str, object]:
    return {"running": _autonomous_agent.running}


@app.post("/agent/autonomous/stop")
def agent_autonomous_stop() -> dict[str, str]:
    _autonomous_agent.stop()
    return {"status": "stopped"}


@app.get("/agent/state")
def agent_state() -> dict[str, object]:
    """Current autonomous agent state — polled by frontend for live reasoning display."""
    return _autonomous_agent.get_state()


@app.post("/agent/respond", responses={503: {"model": ErrorResponse}, 400: {"model": ErrorResponse}})
def agent_respond(request: AgentRespondRequest) -> dict[str, object]:
    try:
        logger.info("[agent] respond instruction_len=%d", len(request.instruction))

        # Intercept stop/resume keywords before touching the LLM
        cmd = request.instruction.strip().lower()
        if cmd in {"stop", "halt", "freeze", "pause"}:
            _autonomous_agent.pause()
            return {"text": "Agent paused. Say 'start again' to resume.", "json": {"action": "stop", "reasoning": "Operator commanded stop."}}
        if cmd in {"start again", "resume", "go", "start", "continue"}:
            _autonomous_agent.resume()
            return {"text": "Agent resumed.", "json": {"action": "resume", "reasoning": "Operator commanded resume."}}

        # Route operator instruction to the autonomous agent's LLM queue so it
        # drives the robot. We also do an immediate synchronous inference here
        # and return it so the frontend can display the reasoning.
        _autonomous_agent.push_instruction(request.instruction)

        qwen = _get_qwen_service()

        # Auto-capture DINO detections if none provided
        detections = request.detections or []
        if not detections:
            try:
                sim = _get_simulator()
                dino = _get_dino_service()
                cam_id = sim.resolve_camera(os.getenv("GDINO_CAMERA_DEFAULT"))[0]
                frame = _render_executor.submit(
                    sim.render_rgb_with_camera_id, 640, 360, cam_id
                ).result(timeout=2.0)
                detections = dino.detect(frame, dino.build_config())
            except Exception as dino_exc:
                logger.warning("[agent] auto-DINO skipped: %s", dino_exc)

        # Fall back to cached whisper audio if caller didn't supply it
        audio_transcript = request.audio_transcript
        audio_direction = request.audio_direction
        audio_distance_m = request.audio_distance_m
        if audio_transcript is None:
            with _latest_audio_lock:
                if _latest_audio:
                    audio_transcript = str(_latest_audio.get("transcript", "")) or None
                    audio_direction = audio_direction or str(_latest_audio.get("direction", ""))
                    raw_dist = _latest_audio.get("distance_m")
                    audio_distance_m = audio_distance_m or (float(raw_dist) if raw_dist is not None else None)

        system_prompt = get_agent_prompt()
        user_payload = {
            "instruction": request.instruction,
            "detections": detections,
            "audio": {
                "transcript": audio_transcript,
                "direction": audio_direction,
                "distance_m": audio_distance_m,
            },
        }
        result = qwen.infer(
            Qwen25VlRequest(
                system_prompt=system_prompt,
                user_prompt=json.dumps(user_payload, ensure_ascii=True),
                max_new_tokens=request.max_new_tokens or 256,
                temperature=request.temperature or 0.2,
                top_p=request.top_p or 0.9,
            )
        )
        return result
    except RuntimeError as exc:
        if "loading" in str(exc).lower():
            return {"text": "Model loading... try again in a moment.", "json": None, "status": "loading"}
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("[agent] respond failed")
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
    camera_id, resolved_camera, fallback = simulator.resolve_camera(camera_name)
    logger.info(
        "[dino] camera requested=%r resolved=%s fallback=%s",
        camera_name,
        resolved_camera,
        fallback,
    )

    try:
        while True:
            loop = asyncio.get_running_loop()
            frame = await loop.run_in_executor(
                _render_executor,
                simulator.render_rgb_with_camera_id,
                width,
                height,
                camera_id,
            )
            if _dino_debug_capture.remaining() > 0:
                await asyncio.to_thread(
                    _dino_debug_capture.capture,
                    frame,
                    resolved_camera,
                    width,
                    height,
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

    survivor_source_ids: list[str] | None = None
    try:
        simulator = _get_simulator()
        survivor_source_ids = simulator.body_names_with_prefix("survivor_")
    except Exception as exc:
        logger.warning("audio whisper using default survivor source ids: %s", exc)

    async def _send_and_cache(msg: dict[str, object]) -> None:
        if "transcript" in msg:
            with _latest_audio_lock:
                _latest_audio.update(msg)
            _autonomous_agent.update_audio(dict(msg))
        await websocket.send_json(msg)

    try:
        await stream_transcriptions(
            _send_and_cache,
            interval_sec=interval_sec,
            source_ids=survivor_source_ids,
        )
    except WebSocketDisconnect as exc:
        logger.info("audio whisper disconnected code=%s", getattr(exc, "code", None))
        return
    except Exception as exc:
        logger.exception("audio whisper crashed: %s", exc)
        try:
            await websocket.close(code=1011, reason=str(exc)[:120])
        except Exception:
            pass
