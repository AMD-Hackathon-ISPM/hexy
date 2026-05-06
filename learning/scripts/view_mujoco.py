from __future__ import annotations

from pathlib import Path
import math
import time


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "hexapod_static.xml"


def main() -> int:
    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        print("MuJoCo viewer is not installed. Install the `mujoco` Python package first.")
        return 1

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    target_a = 1 if model.nu > 1 else 0
    target_b = 4 if model.nu > 4 else target_a

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Keep the camera centered on the robot root with a comfortable
        # distance for the rescaled assembled STL model.
        root_body_id = 1
        viewer.cam.lookat[:] = data.xpos[root_body_id]
        viewer.cam.distance = 0.35
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20

        start = time.time()
        while viewer.is_running():
            t = time.time() - start

            data.ctrl[:] = 0.0
            data.ctrl[target_a] = 0.35 * math.sin(2.0 * t)
            data.ctrl[target_b] = 0.20 * math.sin(2.0 * t + math.pi / 2.0)

            mujoco.mj_step(model, data)
            viewer.cam.lookat[:] = data.xpos[root_body_id]
            viewer.sync()
            time.sleep(model.opt.timestep)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
