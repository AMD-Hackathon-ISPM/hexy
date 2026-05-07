from __future__ import annotations

from pathlib import Path
import math
import threading
import time
from typing import Iterable, List

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

    def state(self) -> MujocoState:
        with self._lock:
            return self._state_locked()

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
