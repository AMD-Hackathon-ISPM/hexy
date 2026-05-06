from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "hexapod_static.xml"


def main() -> int:
    try:
        import mujoco
    except ImportError:
        print("MuJoCo is not installed. Install the `mujoco` Python package first.")
        return 1

    try:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    except Exception as exc:  # pragma: no cover - smoke-test script
        print(f"Failed to load MuJoCo model: {exc}")
        return 1

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print("MuJoCo load succeeded.")
    print(f"Model path: {MODEL_PATH}")
    print(f"Bodies: {model.nbody}")
    print(f"Geoms: {model.ngeom}")
    print(f"Meshes: {model.nmesh}")
    print(f"Joints: {model.njnt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
