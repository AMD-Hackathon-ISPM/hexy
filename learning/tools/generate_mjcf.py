from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
STL_DIR = ROOT / "STLFILES"
OUTPUT = ROOT / "models" / "hexapod_static.xml"

BODY_Z = 0.22
MESH_SCALE_VALUE = 0.0254
MESH_SCALE = f"{MESH_SCALE_VALUE} {MESH_SCALE_VALUE} {MESH_SCALE_VALUE}"
SEGMENT_DENSITY = 250.0
CONTACT_TYPE = "1"
CONTACT_AFFINITY = "1"
FOOT_RADIUS = "0.02"
BODY_Z = 0.065

SEGMENT_COLORS = {
    "Coxa": "0.85 0.35 0.25 1",
    "Femur": "0.25 0.55 0.85 1",
    "Tibia": "0.25 0.75 0.45 1",
}

SEGMENT_INERTIALS = {
    "Coxa": {"mass": "0.20", "diaginertia": "0.00025 0.00025 0.00025"},
    "Femur": {"mass": "0.15", "diaginertia": "0.00018 0.00018 0.00018"},
    "Tibia": {"mass": "0.10", "diaginertia": "0.00012 0.00012 0.00012"},
}

JOINT_RANGES = {
    "Coxa": "-45 45",
    "Femur": "-70 70",
    "Tibia": "-120 20",
}

JOINT_AXES = {
    "L": {"Coxa": "0 0 1", "Femur": "0 1 0", "Tibia": "0 1 0"},
    "R": {"Coxa": "0 0 -1", "Femur": "0 -1 0", "Tibia": "0 -1 0"},
}

BODY_CENTROID = (-7.261785287401571e-08, -0.07167244823189402, 2.5893180025540574)

JOINT_ANCHORS = {
    ("F", "L"): {
        "Coxa": (1.1856070160865784, -1.5671929121017456, 2.101041316986084),
        "Femur": (1.5580149292945862, -1.669890284538269, 2.403157353401184),
        "Tibia": (3.3038647174835205, -3.4147064685821533, 2.2343859672546387),
    },
    ("F", "R"): {
        "Coxa": (-1.1856070756912231, -1.5671928524971008, 2.101041555404663),
        "Femur": (-1.5580143332481384, -1.6698910593986511, 2.403157353401184),
        "Tibia": (-3.303863286972046, -3.414707899093628, 2.2343859672546387),
    },
    ("C", "L"): {
        "Coxa": (1.4117573499679565, -0.002706561703234911, 3.1554027795791626),
        "Femur": (1.8642438054084778, 0.3990747630596161, 2.403157353401184),
        "Tibia": (4.332517147064209, 0.3998056501150131, 2.2343862056732178),
    },
    ("C", "R"): {
        "Coxa": (-1.411755919456482, -0.0027065405156463385, 3.155403256416321),
        "Femur": (-1.8642408847808838, 0.3990747034549713, 2.403157353401184),
        "Tibia": (-4.33251428604126, 0.39980484545230865, 2.2343862056732178),
    },
    ("B", "L"): {
        "Coxa": (0.6114285886287689, 1.6330201625823975, 2.1009562015533447),
        "Femur": (0.8142586648464203, 2.135975480079651, 2.403157353401184),
        "Tibia": (2.5590744018554688, 3.8818256855010986, 2.234386682510376),
    },
    ("B", "R"): {
        "Coxa": (-0.6114285290241241, 1.6330201625823975, 2.100956439971924),
        "Femur": (-0.814258873462677, 2.1359752416610718, 2.403157353401184),
        "Tibia": (-2.559075355529785, 3.8818248510360718, 2.2343865633010864),
    },
}

FOOT_LOCAL_POS = {
    ("F", "L"): (0.0069765683862437365, -0.014958486807450011, -0.05578831807509711),
    ("F", "R"): (-0.006976571440489396, -0.014958478013328886, -0.05578831818041592),
    ("C", "L"): (0.01551043428503949, -0.00564406010573325, -0.05578831491553263),
    ("C", "R"): (-0.01551043291589488, -0.005644066372202788, -0.05578831491553263),
    ("B", "L"): (0.014958470746330593, 0.006976553588950113, -0.055788320971364574),
    ("B", "R"): (-0.014958466428259149, 0.00697655709080074, -0.05578832399928048),
}

LEG_ORDER = ["F", "C", "B"]
SIDE_ORDER = ["L", "R"]

MESH_RE = re.compile(r"^(Coxa|Femur|Tibia)\.([FCB])\.(L|R)\.stl$", re.IGNORECASE)


def mesh_name(filename: str) -> str:
    return filename.replace(".stl", "").replace(".", "_")


def xml_attrs(**attrs: object) -> str:
    return " ".join(f'{key}="{value}"' for key, value in attrs.items())


def scaled(values: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(value * MESH_SCALE_VALUE for value in values)


def sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(x - y for x, y in zip(a, b))


def neg(a: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(-x for x in a)


def vec_text(values: tuple[float, float, float]) -> str:
    return " ".join(f"{value:.6f}" for value in values)


def build_asset_block(mesh_files: list[Path]) -> list[str]:
    lines = ['  <asset>']
    for mesh_file in mesh_files:
        lines.append(
            "    <mesh "
            + xml_attrs(
                name=mesh_name(mesh_file.name),
                file=mesh_file.name,
                scale=MESH_SCALE,
            )
            + "/>"
        )
    lines.extend(
        [
            '    <texture name="grid" type="2d" builtin="checker" rgb1="0.18 0.18 0.20" rgb2="0.28 0.28 0.32" width="512" height="512"/>',
            '    <material name="grid" texture="grid" texrepeat="6 6" reflectance="0.1"/>',
            "  </asset>",
        ]
    )
    return lines


def leg_label(position: str, side: str) -> str:
    name_map = {"F": "front", "C": "center", "B": "back"}
    side_map = {"L": "left", "R": "right"}
    return f"{name_map[position]}_{side_map[side]}"


def build_leg_block(position: str, side: str) -> list[str]:
    label = leg_label(position, side)
    anchors = JOINT_ANCHORS[(position, side)]
    coxa_anchor = anchors["Coxa"]
    femur_anchor = anchors["Femur"]
    tibia_anchor = anchors["Tibia"]

    leg_pos = vec_text(scaled(sub(coxa_anchor, BODY_CENTROID)))
    coxa_mesh = f"Coxa_{position}_{side}"
    femur_mesh = f"Femur_{position}_{side}"
    tibia_mesh = f"Tibia_{position}_{side}"
    coxa_geom_pos = vec_text(scaled(neg(coxa_anchor)))
    femur_pos = vec_text(scaled(sub(femur_anchor, coxa_anchor)))
    femur_geom_pos = vec_text(scaled(neg(femur_anchor)))
    tibia_pos = vec_text(scaled(sub(tibia_anchor, femur_anchor)))
    tibia_geom_pos = vec_text(scaled(neg(tibia_anchor)))
    foot_pos = vec_text(FOOT_LOCAL_POS[(position, side)])

    return [
        f'      <body name="{label}" pos="{leg_pos}">',
        f'        <inertial pos="0 0 0" mass="{SEGMENT_INERTIALS["Coxa"]["mass"]}" diaginertia="{SEGMENT_INERTIALS["Coxa"]["diaginertia"]}"/>',
        f'        <joint name="{label}_coxa_joint" type="hinge" axis="{JOINT_AXES[side]["Coxa"]}" range="{JOINT_RANGES["Coxa"]}" damping="2" armature="0.01"/>',
        f'        <geom type="mesh" mesh="{coxa_mesh}" pos="{coxa_geom_pos}" rgba="{SEGMENT_COLORS["Coxa"]}" density="{SEGMENT_DENSITY}" contype="{CONTACT_TYPE}" conaffinity="{CONTACT_AFFINITY}" friction="0.8 0.1 0.1"/>',
        f'        <body name="{label}_femur" pos="{femur_pos}">',
        f'          <inertial pos="0 0 0" mass="{SEGMENT_INERTIALS["Femur"]["mass"]}" diaginertia="{SEGMENT_INERTIALS["Femur"]["diaginertia"]}"/>',
        f'          <joint name="{label}_femur_joint" type="hinge" axis="{JOINT_AXES[side]["Femur"]}" range="{JOINT_RANGES["Femur"]}" damping="2" armature="0.01"/>',
        f'          <geom type="mesh" mesh="{femur_mesh}" pos="{femur_geom_pos}" rgba="{SEGMENT_COLORS["Femur"]}" density="{SEGMENT_DENSITY}" contype="{CONTACT_TYPE}" conaffinity="{CONTACT_AFFINITY}" friction="0.8 0.1 0.1"/>',
        f'          <body name="{label}_tibia" pos="{tibia_pos}">',
        f'            <inertial pos="0 0 0" mass="{SEGMENT_INERTIALS["Tibia"]["mass"]}" diaginertia="{SEGMENT_INERTIALS["Tibia"]["diaginertia"]}"/>',
        f'            <joint name="{label}_tibia_joint" type="hinge" axis="{JOINT_AXES[side]["Tibia"]}" range="{JOINT_RANGES["Tibia"]}" damping="1" armature="0.01"/>',
        f'            <geom type="mesh" mesh="{tibia_mesh}" pos="{tibia_geom_pos}" rgba="{SEGMENT_COLORS["Tibia"]}" density="{SEGMENT_DENSITY}" contype="{CONTACT_TYPE}" conaffinity="{CONTACT_AFFINITY}" friction="1.2 0.2 0.2"/>',
        f'            <geom name="{label}_foot" type="sphere" pos="{foot_pos}" size="{FOOT_RADIUS}" rgba="0 0 0 0" contype="{CONTACT_TYPE}" conaffinity="{CONTACT_AFFINITY}" friction="1.8 0.2 0.2"/>',
        "          </body>",
        "        </body>",
        "      </body>",
    ]


def build_worldbody_block() -> list[str]:
    lines = [
        "  <worldbody>",
        '    <light pos="0 0 2.5" dir="0 0 -1"/>',
        '    <geom name="floor" type="plane" pos="0 0 0" size="3 3 0.1" material="grid"/>',
        "",
        f'    <body name="hexapod" pos="0 0 {BODY_Z}">',
        f'      <geom name="body_visual" type="mesh" mesh="BODY" pos="{vec_text(scaled(neg(BODY_CENTROID)))}" rgba="0.72 0.72 0.76 1" contype="{CONTACT_TYPE}" conaffinity="{CONTACT_AFFINITY}" friction="0.8 0.1 0.1"/>',
        "",
    ]

    for position in LEG_ORDER:
        for side in SIDE_ORDER:
            lines.extend(build_leg_block(position, side))
            lines.append("")

    if lines[-1] == "":
        lines.pop()

    lines.extend(["    </body>", "  </worldbody>"])
    return lines


def build_actuator_block() -> list[str]:
    lines = ["  <actuator>"]
    for position in LEG_ORDER:
        for side in SIDE_ORDER:
            label = leg_label(position, side)
            lines.append(
                f'    <motor name="{label}_coxa_motor" joint="{label}_coxa_joint" gear="60" ctrllimited="true" ctrlrange="-1 1"/>'
            )
            lines.append(
                f'    <motor name="{label}_femur_motor" joint="{label}_femur_joint" gear="80" ctrllimited="true" ctrlrange="-1 1"/>'
            )
            lines.append(
                f'    <motor name="{label}_tibia_motor" joint="{label}_tibia_joint" gear="80" ctrllimited="true" ctrlrange="-1 1"/>'
            )
    lines.append("  </actuator>")
    return lines


def validate_mesh_set(mesh_files: list[Path]) -> None:
    required = {"BODY.stl"}
    for position in LEG_ORDER:
        for side in SIDE_ORDER:
            for segment in ("Coxa", "Femur", "Tibia"):
                required.add(f"{segment}.{position}.{side}.stl")

    found = {path.name for path in mesh_files}
    missing = sorted(required - found)
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"Missing STL files: {missing_text}")


def main() -> None:
    mesh_files = sorted(STL_DIR.glob("*.stl"), key=lambda path: path.name)
    validate_mesh_set(mesh_files)

    extra = [
        mesh_file.name
        for mesh_file in mesh_files
        if mesh_file.name != "BODY.stl" and not MESH_RE.match(mesh_file.name)
    ]
    if extra:
        extra_text = ", ".join(extra)
        raise ValueError(f"Unexpected STL naming pattern: {extra_text}")

    lines = [
        '<mujoco model="vendor_agnostic_hexapod_static">',
        '  <compiler angle="degree" meshdir="../STLFILES"/>',
        '  <option timestep="0.01" gravity="0 0 -9.81"/>',
        "",
        "  <visual>",
        '    <headlight ambient="0.5 0.5 0.5" diffuse="0.7 0.7 0.7" specular="0.1 0.1 0.1"/>',
        '    <rgba haze="0.15 0.25 0.35 1"/>',
        "  </visual>",
        "",
    ]
    lines.extend(build_asset_block(mesh_files))
    lines.append("")
    lines.extend(build_worldbody_block())
    lines.append("")
    lines.extend(build_actuator_block())
    lines.append("</mujoco>")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
