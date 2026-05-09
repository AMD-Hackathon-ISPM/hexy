#!/usr/bin/env python3

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import xml.etree.ElementTree as ET


CAVE_OUTPUT_DIR = Path(os.environ.get("CAVE_OUTPUT_DIR", "cave_env"))
HEXAPOD_MODEL_PATH = Path(
    os.environ.get("HEXAPOD_MODEL_PATH", "/vendor_rl/models/hexapod_static.xml")
)
CAVE_XML_PATH = CAVE_OUTPUT_DIR / "cave_maze.xml"
OUTPUT_XML_PATH = CAVE_OUTPUT_DIR / "cave_hexapod.xml"
MUJOCO_DINO_CAMERA_HEIGHT = float(os.environ.get("MUJOCO_DINO_CAMERA_HEIGHT", "0.3"))


def require_child(parent: ET.Element, tag: str, source: Path) -> ET.Element:
    child = parent.find(tag)
    if child is None:
        raise ValueError(f"{source} is missing <{tag}>")
    return child


def remove_direct_child(parent: ET.Element, child: ET.Element | None) -> None:
    if child is not None:
        parent.remove(child)


def remove_world_body(worldbody: ET.Element, name: str) -> ET.Element:
    for body in list(worldbody.findall("body")):
        if body.get("name") == name:
            worldbody.remove(body)
            return body
    raise ValueError(f"cave scene is missing worldbody body {name!r}")


def parse_vec(raw: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not raw:
        return default
    values = [float(part) for part in raw.split()]
    if len(values) < 3:
        return default
    return values[0], values[1], values[2]


def fmt_vec(values: tuple[float, float, float]) -> str:
    return " ".join(f"{value:.6f}" for value in values)


def lowest_foot_bottom_z(hexapod_body: ET.Element) -> float | None:
    bottoms: list[float] = []

    def walk(node: ET.Element, offset: tuple[float, float, float]) -> None:
        next_offset = offset
        if node.tag == "body":
            local_pos = parse_vec(node.get("pos"), (0.0, 0.0, 0.0))
            next_offset = (
                offset[0] + local_pos[0],
                offset[1] + local_pos[1],
                offset[2] + local_pos[2],
            )

        if node.tag == "geom" and (node.get("name") or "").endswith("_foot"):
            local_pos = parse_vec(node.get("pos"), (0.0, 0.0, 0.0))
            size_raw = node.get("size") or "0"
            radius = float(size_raw.split()[0])
            bottoms.append(next_offset[2] + local_pos[2] - radius)

        for child in node:
            walk(child, next_offset)

    for child in hexapod_body:
        walk(child, (0.0, 0.0, 0.0))
    return min(bottoms) if bottoms else None


def resolve_hexapod_spawn(spawn_body: ET.Element, hexapod_body: ET.Element) -> tuple[float, float, float]:
    spawn_x, spawn_y, fallback_z = parse_vec(spawn_body.get("pos"), (0.0, 0.0, 0.065))
    foot_bottom = lowest_foot_bottom_z(hexapod_body)
    if foot_bottom is None:
        return spawn_x, spawn_y, fallback_z

    clearance = float(os.environ.get("HEXAPOD_FLOOR_CLEARANCE", "0.002"))
    return spawn_x, spawn_y, clearance - foot_bottom


def rewrite_cave_mesh_paths(asset: ET.Element) -> None:
    for mesh in asset.findall("mesh"):
        file_attr = mesh.get("file")
        if not file_attr:
            continue
        if "/" not in file_attr:
            mesh.set("file", f"meshes/{file_attr}")


def append_hexapod_mesh_assets(cave_asset: ET.Element, hexapod_asset: ET.Element) -> None:
    for mesh in hexapod_asset.findall("mesh"):
        file_attr = mesh.get("file")
        if not file_attr:
            raise ValueError(f"hexapod mesh {mesh.get('name')!r} is missing file")
        mesh.set("file", f"../rl/STLFILES/{Path(file_attr).name}")
        cave_asset.append(mesh)


def find_hexapod_body(hexapod_worldbody: ET.Element) -> ET.Element:
    for body in hexapod_worldbody.findall("body"):
        if body.get("name") == "hexapod":
            return body
    raise ValueError("hexapod model is missing worldbody body 'hexapod'")


def ensure_root_freejoint(hexapod_body: ET.Element) -> None:
    for child in list(hexapod_body):
        if child.tag == "freejoint":
            child.set("name", "hexapod_root")
            return
        if child.tag == "joint" and child.get("type") == "free":
            child.set("name", "hexapod_root")
            return

    hexapod_body.insert(0, ET.Element("freejoint", {"name": "hexapod_root"}))


def ensure_robot_pov_camera(hexapod_body: ET.Element) -> None:
    camera_attrs = {
        "name": "robot_pov",
        "mode": "fixed",
        "pos": f"0 0 {MUJOCO_DINO_CAMERA_HEIGHT:.2f}",
        "xyaxes": "1 0 0 0 0 1",
        "fovy": "70",
    }

    for child in hexapod_body.findall("camera"):
        if child.get("name") == "robot_pov":
            child.attrib.clear()
            child.attrib.update(camera_attrs)
            return

    insert_index = (
        1 if len(hexapod_body) > 0 and hexapod_body[0].tag == "freejoint" else 0
    )
    hexapod_body.insert(
        insert_index,
        ET.Element(
            "camera",
            camera_attrs,
        ),
    )


def ensure_robot_headlamp(hexapod_body: ET.Element) -> None:
    headlamp_attrs = {
        "name": "robot_headlamp",
        "pos": "0 0.18 0.62",
        "dir": "0 1 -0.08",
        "diffuse": "1.65 1.48 1.16",
        "specular": "0.12 0.12 0.10",
        "directional": "true",
        "castshadow": "false",
    }
    fill_attrs = {
        "name": "robot_soft_fill",
        "pos": "0 0.06 0.40",
        "diffuse": "0.45 0.42 0.36",
        "castshadow": "false",
    }

    wanted = {
        "robot_headlamp": headlamp_attrs,
        "robot_soft_fill": fill_attrs,
    }
    existing = {child.get("name"): child for child in hexapod_body.findall("light")}
    for name, attrs in wanted.items():
        light = existing.get(name)
        if light is None:
            hexapod_body.insert(1, ET.Element("light", attrs))
            continue
        light.attrib.clear()
        light.attrib.update(attrs)


def build() -> None:
    if not CAVE_XML_PATH.exists():
        raise FileNotFoundError(f"cave scene not found: {CAVE_XML_PATH}")
    if not HEXAPOD_MODEL_PATH.exists():
        raise FileNotFoundError(f"hexapod model not found: {HEXAPOD_MODEL_PATH}")

    cave_tree = ET.parse(CAVE_XML_PATH)
    cave_root = cave_tree.getroot()
    hexapod_root = ET.parse(HEXAPOD_MODEL_PATH).getroot()

    cave_compiler = require_child(cave_root, "compiler", CAVE_XML_PATH)
    cave_compiler.attrib.pop("meshdir", None)

    cave_asset = require_child(cave_root, "asset", CAVE_XML_PATH)
    cave_worldbody = require_child(cave_root, "worldbody", CAVE_XML_PATH)
    hexapod_asset = require_child(hexapod_root, "asset", HEXAPOD_MODEL_PATH)
    hexapod_worldbody = require_child(hexapod_root, "worldbody", HEXAPOD_MODEL_PATH)
    hexapod_actuator = require_child(hexapod_root, "actuator", HEXAPOD_MODEL_PATH)

    rewrite_cave_mesh_paths(cave_asset)
    spawn_body = remove_world_body(cave_worldbody, "robot")
    remove_direct_child(cave_root, cave_root.find("sensor"))
    remove_direct_child(cave_root, cave_root.find("keyframe"))

    append_hexapod_mesh_assets(cave_asset, hexapod_asset)

    hexapod_body = find_hexapod_body(hexapod_worldbody)
    ensure_root_freejoint(hexapod_body)
    ensure_robot_pov_camera(hexapod_body)
    ensure_robot_headlamp(hexapod_body)
    hexapod_body.set("pos", fmt_vec(resolve_hexapod_spawn(spawn_body, hexapod_body)))
    cave_worldbody.append(hexapod_body)
    cave_root.append(hexapod_actuator)

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cave_root.insert(
        0,
        ET.Comment(
            f" Generated by build_cave_hexapod.py at {generated_at}; "
            f"sources: {CAVE_XML_PATH.name} + {HEXAPOD_MODEL_PATH.name} "
        ),
    )

    ET.indent(cave_tree, space="  ")
    tmp_path = OUTPUT_XML_PATH.with_suffix(OUTPUT_XML_PATH.suffix + ".tmp")
    cave_tree.write(tmp_path, encoding="utf-8", xml_declaration=True)
    os.replace(tmp_path, OUTPUT_XML_PATH)
    print(f"  Merged cave hexapod XML: {OUTPUT_XML_PATH}")


if __name__ == "__main__":
    build()
