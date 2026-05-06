from __future__ import annotations

from pathlib import Path
import math

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "hexapod_static.xml"


def _joint_summary(model, data) -> None:
    joint_names = []
    joint_angles = []
    for joint_id in range(model.njnt):
        name = model.joint(joint_id).name
        qpos_adr = model.jnt_qposadr[joint_id]
        joint_names.append(name)
        joint_angles.append(float(data.qpos[qpos_adr]))

    print("Joint snapshot:")
    for name, angle in zip(joint_names[:6], joint_angles[:6]):
        print(f"  {name}: {angle:.4f} rad")


def main() -> int:
    try:
        import mujoco
    except ImportError:
        print("MuJoCo is not installed. Install the `mujoco` Python package first.")
        return 1

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    print("Loaded articulated model.")
    print(f"Actuators: {model.nu}")
    print(f"Joints: {model.njnt}")

    data.ctrl[:] = 0.0
    for _ in range(200):
        mujoco.mj_step(model, data)

    print("After zero-control settling:")
    print(f"  base z: {data.xpos[1][2]:.4f}")
    print(f"  contacts: {data.ncon}")
    _joint_summary(model, data)

    target_actuator = 1 if model.nu > 1 else 0
    for step in range(300):
        data.ctrl[:] = 0.0
        data.ctrl[target_actuator] = 0.35 * math.sin(step * 0.05)
        mujoco.mj_step(model, data)

    print("After single-joint sine test:")
    print(f"  tested actuator: {model.actuator(target_actuator).name}")
    print(f"  contacts: {data.ncon}")
    _joint_summary(model, data)
    print(f"  ctrl sample: {np.array2string(data.ctrl, precision=3)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
