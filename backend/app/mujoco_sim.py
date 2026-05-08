from __future__ import annotations

from pathlib import Path
import math
import threading
import time
from typing import Callable, Iterable, List

import numpy as np
from pydantic import BaseModel

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - handled by caller
    raise RuntimeError(
        "MuJoCo is not installed. Install the `mujoco` Python package first."
    ) from exc


class MujocoState(BaseModel):
    time: float
    qpos: List[float]
    qvel: List[float]
    ctrl: List[float]
    ncon: int
    nu: int
    njnt: int
    updated_at: float


class MujocoSimulator:
    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        self._lock = threading.Lock()
        self._renderer = None
        self._render_width = 0
        self._render_height = 0

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def reset(self) -> MujocoState:
        with self._lock:
            mujoco.mj_resetData(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            return self._state_locked()

    def step(self, ctrl: List[float] | None, n_steps: int = 1) -> MujocoState:
        with self._lock:
            if ctrl is not None:
                if len(ctrl) != self.model.nu:
                    raise ValueError(
                        f"Expected {self.model.nu} control values, got {len(ctrl)}"
                    )
                self.data.ctrl[:] = np.array(ctrl, dtype=np.float64)
            else:
                self.data.ctrl[:] = 0.0

            for _ in range(max(1, n_steps)):
                mujoco.mj_step(self.model, self.data)
            return self._state_locked()

    def drive_key(
        self,
        key: str,
        ctrl_builder: Callable[[float], List[float]],
        n_steps: int = 1,
    ) -> MujocoState:
        with self._lock:
            for _ in range(max(1, n_steps)):
                ctrl = ctrl_builder(float(self.data.time))
                if len(ctrl) != self.model.nu:
                    raise ValueError(
                        f"Expected {self.model.nu} control values, got {len(ctrl)}"
                    )
                self.data.ctrl[:] = np.array(ctrl, dtype=np.float64)
                self._stabilize_base_pose_locked()
                self._apply_base_key_velocity_locked(key)
                mujoco.mj_step(self.model, self.data)
                self._stabilize_base_pose_locked()
            return self._state_locked()

    def state(self) -> MujocoState:
        with self._lock:
            return self._state_locked()

    def _apply_base_key_velocity_locked(self, key: str) -> None:
        root_addresses = self._root_freejoint_addresses_locked()
        if root_addresses is None:
            return
        qpos_adr, qvel_adr = root_addresses

        quat = self.data.qpos[qpos_adr + 3 : qpos_adr + 7]
        yaw = self._yaw_from_quat(quat)

        drive_speed = 0.35
        turn_speed = 1.2
        key = key.lower()

        if key in {"w", "s"}:
            direction = -1.0 if key == "w" else 1.0
            self.data.qvel[qvel_adr] = direction * drive_speed * math.sin(yaw)
            self.data.qvel[qvel_adr + 1] = -direction * drive_speed * math.cos(yaw)
            self.data.qvel[qvel_adr + 5] *= 0.4
            return

        if key in {"a", "d"}:
            direction = 1.0 if key == "a" else -1.0
            self.data.qvel[qvel_adr] *= 0.4
            self.data.qvel[qvel_adr + 1] *= 0.4
            self.data.qvel[qvel_adr + 5] = direction * turn_speed

    def _root_freejoint_addresses_locked(self) -> tuple[int, int] | None:
        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "hexapod_root"
        )
        if joint_id < 0:
            return None
        if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            return None

        qpos_adr = int(self.model.jnt_qposadr[joint_id])
        qvel_adr = int(self.model.jnt_dofadr[joint_id])
        if qpos_adr + 7 > self.model.nq or qvel_adr + 6 > self.model.nv:
            return None
        return qpos_adr, qvel_adr

    def _stabilize_base_pose_locked(self) -> None:
        root_addresses = self._root_freejoint_addresses_locked()
        if root_addresses is None:
            return
        qpos_adr, qvel_adr = root_addresses

        quat = self.data.qpos[qpos_adr + 3 : qpos_adr + 7]
        yaw = self._yaw_from_quat(quat)
        half_yaw = 0.5 * yaw
        self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.array(
            [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
            dtype=np.float64,
        )
        self.data.qvel[qvel_adr + 2] = min(max(self.data.qvel[qvel_adr + 2], -0.15), 0.15)
        self.data.qvel[qvel_adr + 3] = 0.0
        self.data.qvel[qvel_adr + 4] = 0.0
        mujoco.mj_forward(self.model, self.data)

    @staticmethod
    def _yaw_from_quat(quat: np.ndarray) -> float:
        return math.atan2(
            2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
            1.0 - 2.0 * (quat[2] * quat[2] + quat[3] * quat[3]),
        )

    def _state_locked(self) -> MujocoState:
        def _sanitize(values: Iterable[float]) -> List[float]:
            return [float(v) if math.isfinite(v) else 0.0 for v in values]

        return MujocoState(
            time=float(self.data.time) if math.isfinite(self.data.time) else 0.0,
            qpos=_sanitize(self.data.qpos),
            qvel=_sanitize(self.data.qvel),
            ctrl=_sanitize(self.data.ctrl),
            ncon=int(self.data.ncon),
            nu=int(self.model.nu),
            njnt=int(self.model.njnt),
            updated_at=time.time(),
        )

    def render_rgb(
        self,
        width: int = 640,
        height: int = 480,
        camera_name: str | None = None,
    ) -> np.ndarray:
        width = max(64, int(width))
        height = max(64, int(height))

        with self._lock:
            if (
                self._renderer is None
                or self._render_width != width
                or self._render_height != height
            ):
                self._renderer = mujoco.Renderer(self.model, height=height, width=width)
                self._render_width = width
                self._render_height = height

            camera_id = None
            if camera_name:
                candidate = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
                )
                if candidate >= 0:
                    camera_id = candidate

            if camera_id is None:
                self._renderer.update_scene(self.data)
            else:
                self._renderer.update_scene(self.data, camera=camera_id)

            frame = self._renderer.render()
            return frame.copy()
