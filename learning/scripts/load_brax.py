from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "hexapod_static.xml"


def _get_dof(system: object) -> object:
    qd_size = getattr(system, "qd_size", None)
    if callable(qd_size):
        return qd_size()
    return qd_size if qd_size is not None else "unknown"


def _get_actuator_count(system: object) -> object:
    act_size = getattr(system, "act_size", None)
    if callable(act_size):
        return act_size()

    actuator = getattr(system, "actuator", None)
    q_id = getattr(actuator, "q_id", None)
    if q_id is not None:
        try:
            return len(q_id)
        except TypeError:
            return "unknown"

    return 0


def main() -> int:
    try:
        from brax.io import mjcf
    except ImportError:
        print("Brax is not installed. Install the `brax` Python package first.")
        return 1

    try:
        system = mjcf.load(str(MODEL_PATH))
    except Exception as exc:  # pragma: no cover - smoke-test script
        print(f"Failed to load Brax system from MJCF: {exc}")
        return 1

    print("Brax load succeeded.")
    print(f"Model path: {MODEL_PATH}")
    print(f"Links: {len(getattr(system, 'link_names', []))}")
    print(f"DOF: {_get_dof(system)}")
    print(f"Actuators: {_get_actuator_count(system)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
