#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   MuJoCo Cave Search-and-Rescue Simulation System                           ║
║                                                                              ║
║   Module 1 — Cave Maze Generator                                             ║
║     • Catmull-Rom spline path through U-shaped maze waypoints               ║
║     • Fractal Brownian Motion (6-octave) wall/ceiling noise                 ║
║     • Stalactite field, cave "breathing" (width & height variation)         ║
║     • Flat navigable ground plane for humanoid locomotion                   ║
║     • Rock obstacle geoms scattered along path (with seed control)          ║
║     • Entrance & exit clearly widened with taper transitions                ║
║     • Binary STL inner (collision) + outer (visual shell) meshes            ║
║                                                                              ║
║   Module 2 — Human / Survivor Generator                                     ║
║     • Capsule-based humanoid bodies (head + torso + limbs)                  ║
║     • Three pose types: lying, sitting, slumped                             ║
║     • Random placement along cave path with clearance checks                ║
║     • Partial occlusion via nearby rock geometry                            ║
║     • Per-survivor metadata: position, visibility_level, occlusion_%        ║
║                                                                              ║
║   Module 3 — Synthetic Dataset Pipeline                                     ║
║     • mujoco.Renderer renders RGB images from robot-eye camera              ║
║     • Saves to dataset/images/  +  dataset/labels/ JSON                    ║
║     • Full metadata: person_present, position, lighting, occlusion          ║
║     • Per-episode randomisation of cave seed, survivors, lighting           ║
║     • Target ≥ 1000 samples — configurable via N_EPISODES                  ║
║                                                                              ║
║   Requirements : Python 3.8+, numpy, mujoco                                ║
║   Run          : python3 cave_simulation_system.py                          ║
║   View env     : python3 -m mujoco.viewer cave_env/cave_maze.xml            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import os
import struct
import math
import json
import random
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Union


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 0 — GLOBAL CONFIGURATION
#  Every tunable knob lives here.  Change seed for a new cave universe.
# ═══════════════════════════════════════════════════════════════════════════
global INNER_RADIUS
# ── Output paths ────────────────────────────────────────────────────────────
OUTPUT_DIR   = os.environ.get("CAVE_OUTPUT_DIR",  "cave_env")
MESH_DIR     = os.environ.get("CAVE_MESH_DIR",    os.path.join(OUTPUT_DIR, "meshes"))
DATASET_DIR  = os.environ.get("CAVE_DATASET_DIR", "dataset")
IMAGES_DIR   = f"{DATASET_DIR}/images"
LABELS_DIR   = f"{DATASET_DIR}/labels"

# ── Dataset generation ──────────────────────────────────────────────────────
N_EPISODES        = 1000    # Total samples to generate
RENDER_WIDTH      = 640     # Camera image width  (pixels)
RENDER_HEIGHT     = 480     # Camera image height (pixels)
CAMERA_FOV_DEG    = 80      # Horizontal field of view
RENDER_SEED_START = 1000    # Base RNG seed for episode randomisation
PERSON_PRESENT_RATE = 0.85  # Positive samples where survivors should be visible

# ── Base cave geometry ──────────────────────────────────────────────────────
REAL_TO_SIM_SCALE = 0.065 / 0.20
INNER_RADIUS   = 0.85   # Half-width of corridor (scaled metres)
CAVE_HEIGHT    = 1.55   # Floor-to-ceiling height at arch peak (scaled metres)
WALL_THICKNESS = 0.12   # Legacy outer-shell thickness for dataset mode
OUTER_LAYER_OFFSET = 0.16  # Visual-only outside shell for orbit/layout view
FLOOR_Z        = 0.0    # Z elevation of flat ground plane
CAMERA_CLEARANCE_HEIGHT = 1.15
HEXAPOD_FOOT_BOTTOM_REL_Z = -0.084803
HEXAPOD_FLOOR_CLEARANCE = 0.002
HEXAPOD_SPAWN_Z = FLOOR_Z + HEXAPOD_FLOOR_CLEARANCE - HEXAPOD_FOOT_BOTTOM_REL_Z
ROBOT_SPAWN_FRACTION = 0.12
CAVE_PATH_FRACTION = 0.42
MUJOCO_DINO_CAMERA_HEIGHT = 0.30
ROOM_MODE = False
ROOM_HALF_SIZE = 6.0
ROOM_HEIGHT = 2.0

# ── Organic wall noise ──────────────────────────────────────────────────────
NOISE_SEED       = 42
NOISE_AMP        = 0.16
STALACTITE_AMP   = 0.08
WALL_VARIATION   = 0.12
HEIGHT_VARIATION = 0.08

# ── Entrance / exit widening ────────────────────────────────────────────────
ENTRANCE_MULT     = 1.00
EXIT_MULT         = 1.00
TRANSITION_SLICES = 12

# ── Mesh resolution ─────────────────────────────────────────────────────────
N_PROFILE  = 12
SWEEP_STEP = 0.35
CAVE_WALL_SECTIONS = 5

# ── Rock obstacles ──────────────────────────────────────────────────────────
N_ROCKS          = 0        # Live scene keeps the path clear by default
ROCK_MIN_SIZE    = 0.15     # Minimum rock half-extent (metres)
ROCK_MAX_SIZE    = 0.35     # Maximum rock half-extent (metres)
ROCK_CLEAR_ZONE  = 0.80     # Clear zone fraction near path centre (no rocks)

# ── Survivor / human parameters ─────────────────────────────────────────────
MAX_SURVIVORS     = 12      # Maximum survivors per episode
SURVIVOR_SCALE    = 0.58    # Size scale for survivor geometry
MIN_SURVIVORS     = 2       # Minimum survivors when present
SURVIVOR_POSES    = ["lying", "sitting", "slumped"]
MIN_CLEARANCE     = 0.35    # Minimum metres from path centre for survivors
VISIBILITY_BANDS  = ["low", "medium", "high"]
SURVIVOR_VIEW_DISTANCE_MIN = 1.6
SURVIVOR_VIEW_DISTANCE_MAX = 3.2
SURVIVOR_VIEW_YAW_JITTER_DEG = 3.0

# ── Maze waypoints (top-down, metres) ───────────────────────────────────────
# Robot spawn (camera viewpoint)
ROBOT_SPAWN = (0.0, 2.5, HEXAPOD_SPAWN_Z)

STONE_MATERIALS = [
    ("cave_stone_dark", "0.22 0.23 0.22 1.0", "0.05", "0.015"),
    ("cave_stone_warm", "0.33 0.31 0.27 1.0", "0.06", "0.018"),
    ("cave_stone_green", "0.25 0.29 0.25 1.0", "0.04", "0.012"),
    ("cave_stone_slate", "0.28 0.30 0.32 1.0", "0.05", "0.014"),
    ("cave_stone_damp", "0.16 0.17 0.16 1.0", "0.03", "0.010"),
    ("cave_stone_olive", "0.29 0.28 0.21 1.0", "0.04", "0.012"),
]
WALL_MATERIAL_NAME = "cave_stone_dark"


@dataclass
class CavePathSpec:
    name: str
    path_points: list
    seed_offset: int = 0
    anchor_idx: Optional[int] = None
    side: int = 0


def robot_spawn_from_path(path_points: list) -> Tuple[float, float, float]:
    """Return an interior spawn on the cave centerline with feet above floor."""
    if not path_points:
        return ROBOT_SPAWN
    idx = int(np.clip(round((len(path_points) - 1) * ROBOT_SPAWN_FRACTION), 0, len(path_points) - 1))
    pt = path_points[idx]
    return (float(pt[0]), float(pt[1]), float(HEXAPOD_SPAWN_Z))


def shorten_path(path_points: list, keep_fraction: float = CAVE_PATH_FRACTION) -> list:
    """Keep the first part of the cave path for a shorter live scene."""
    if not path_points:
        return path_points
    keep_count = max(16, int(math.ceil(len(path_points) * keep_fraction)))
    return path_points[: min(keep_count, len(path_points))]


def path_length(path_points: list) -> float:
    if len(path_points) < 2:
        return 0.0
    return sum(
        float(np.linalg.norm(path_points[i + 1] - path_points[i]))
        for i in range(len(path_points) - 1)
    )


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 1 — NOISE FUNCTIONS  (pure numpy/math — no external libs)
# ═══════════════════════════════════════════════════════════════════════════

_FBM_FREQS = [
    (1.2731, 0.7853, 0.9173),
    (2.4789, 1.8317, 2.1053),
    (4.7771, 3.9631, 4.4271),
    (9.3529, 7.8179, 8.6173),
    (17.893, 15.712, 16.137),
    (35.312, 29.738, 31.592),
]


def fbm_noise(x: float, y: float, z: float = 0.0,
              seed: int = 0, octaves: int = 6) -> float:
    """6-octave FBM using cross-coupled sine waves.  Returns ≈ [−1, 1]."""
    s = seed * 0.17394
    val, amp, wsum = 0.0, 0.5, 0.0
    for i in range(min(octaves, len(_FBM_FREQS))):
        fx, fy, fz = _FBM_FREQS[i]
        px = s + i * 2.39996
        py = s * 1.618 + i * 1.61803
        pz = s * 2.718 + i * 0.95491
        n = (
            math.sin(x * fx + y * fy * 0.71 + z * fz * 0.53 + px)
            * math.cos(y * fy + z * fz * 0.63 + py)
            + math.sin(x * fx * 0.83 + z * fz + pz) * 0.41
        ) / 1.41
        val  += amp * n
        wsum += amp
        amp  *= 0.5
    return max(-1.0, min(1.0, val / wsum))


def stalactite_field(x: float, y: float, seed: int = 0) -> float:
    """Stalactite intensity ∈ [0, 1] — three scales (large / medium / fine)."""
    s  = seed * 0.13791
    n1 = (1.0 - abs(math.cos(x * 1.73 + y * 1.31 + s))) ** 2
    n2 = (1.0 - abs(math.cos(x * 3.41 + y * 2.89 + s * 1.6))) ** 2
    n3 = (1.0 - abs(math.cos(x * 6.91 + y * 5.37 + s * 2.8))) ** 2
    return 0.50 * n1 + 0.30 * n2 + 0.20 * n3


def low_freq_vary(x: float, y: float, seed: int = 0) -> float:
    """Smooth low-frequency variation ∈ [−1, 1] for global dimension swing."""
    s = seed * 0.27318
    return (
        0.60 * math.sin(x * 0.21 + y * 0.17 + s)
      + 0.30 * math.sin(x * 0.43 + y * 0.36 + s * 1.5)
      + 0.10 * math.sin(x * 0.87 + y * 0.71 + s * 2.1)
    )


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 2 — PATH UTILITIES
# ═══════════════════════════════════════════════════════════════════════════
def generate_maze_waypoints(seed: int, n_points=8):
    """Compact cave path bounded to a ~20x20 m square — no open-sky outside zone."""
    rng = np.random.default_rng(seed)
    BOUND = 6.0  # keep every waypoint within ±6 m from origin

    pts = []
    pos = np.array([0.0, 0.0])
    direction = np.array([1.0, 0.0])

    for _ in range(n_points):
        angle = rng.uniform(-0.7, 0.7)
        rot = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])
        direction = rot @ direction

        step = rng.uniform(1.5, 3.0)
        next_pos = pos + direction * step

        # Reflect off boundary so the cave stays inside the square
        for ax in range(2):
            if abs(next_pos[ax]) > BOUND:
                direction[ax] = -direction[ax]
                next_pos = pos + direction * step

        pos = next_pos.copy()
        pts.append((float(pos[0]), float(pos[1])))

    return pts

def _catmull_rom(p0, p1, p2, p3, t: float) -> np.ndarray:
    p0, p1, p2, p3 = (np.asarray(p, float) for p in (p0, p1, p2, p3))
    return 0.5 * (
        2.0 * p1
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t ** 2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t ** 3
    )


def build_smooth_path(waypoints_2d, step: float = SWEEP_STEP) -> list:
    """Densify 2-D waypoints into a smooth 3-D centre-line."""
    pts = [np.array([wx, wy, FLOOR_Z], float) for wx, wy in waypoints_2d]
    pts = [pts[0]] + pts + [pts[-1]]
    path = []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i-1], pts[i], pts[i+1], pts[i+2]
        seg = float(np.linalg.norm(p2 - p1))
        n   = max(2, int(seg / step))
        for j in range(n):
            path.append(_catmull_rom(p0, p1, p2, p3, j / n))
    path.append(pts[-2].copy())
    return path


def get_entrance_path_coordinates(path_points: list,
                                  n_points: int = TRANSITION_SLICES
                                  ) -> List[Tuple[float, float, float]]:
    """
    Return the cave entrance path coordinates.

    By default this returns the first widened transition segment of the path,
    which is the useful entrance corridor rather than only one point.
    """
    if not path_points:
        return []
    count = max(1, min(int(n_points), len(path_points)))
    return [
        (float(pt[0]), float(pt[1]), float(pt[2]))
        for pt in path_points[:count]
    ]


def get_entrance_coordinate(path_points: list) -> Tuple[float, float, float]:
    """Return the first centre-line coordinate of the cave entrance."""
    coords = get_entrance_path_coordinates(path_points, n_points=1)
    if not coords:
        raise ValueError("Cannot get entrance coordinate from an empty path")
    return coords[0]


def compute_frames(path_points: list) -> list:
    """Frenet (forward, right, up) frames at every path point."""
    n = len(path_points)
    up_world = np.array([0.0, 0.0, 1.0])
    frames   = []
    for i in range(n):
        fwd = (path_points[i+1] - path_points[i]) if i < n-1 \
              else (path_points[i] - path_points[i-1])
        fl  = np.linalg.norm(fwd)
        if fl < 1e-9:
            frames.append(frames[-1] if frames else
                          (np.array([0,1,0.]), np.array([1,0,0.]), up_world.copy()))
            continue
        fwd /= fl
        right = np.cross(fwd, up_world)
        rl    = np.linalg.norm(right)
        right = right / rl if rl > 1e-9 else np.array([1., 0., 0.])
        up    = np.cross(right, fwd)
        up   /= np.linalg.norm(up)
        frames.append((fwd.copy(), right.copy(), up.copy()))
    return frames


def frame_at_path_index(path_points: list, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a local (forward, right, up) frame for one path sample."""
    n_pts = len(path_points)
    up_world = np.array([0.0, 0.0, 1.0])
    if n_pts < 2:
        return np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), up_world

    idx = int(np.clip(idx, 0, n_pts - 1))
    if idx < n_pts - 1:
        fwd = path_points[idx + 1] - path_points[idx]
    else:
        fwd = path_points[idx] - path_points[idx - 1]

    fl = np.linalg.norm(fwd)
    if fl < 1e-9:
        fwd = np.array([0.0, 1.0, 0.0])
    else:
        fwd = fwd / fl

    right = np.cross(fwd, up_world)
    rl = np.linalg.norm(right)
    right = right / rl if rl > 1e-9 else np.array([1.0, 0.0, 0.0])
    up = np.cross(right, fwd)
    up = up / np.linalg.norm(up)
    return fwd.copy(), right.copy(), up.copy()


def _normalise_xy(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    out = np.asarray(vec, dtype=float)
    out = out[:2]
    length = np.linalg.norm(out)
    if length < 1e-9:
        out = np.asarray(fallback, dtype=float)[:2]
        length = np.linalg.norm(out)
    if length < 1e-9:
        return np.array([1.0, 0.0])
    return out / length


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 3 — CAVE CROSS-SECTION PROFILE
# ═══════════════════════════════════════════════════════════════════════════

def get_slice_dimensions(slice_idx: int, n_slices: int, pt: np.ndarray) -> tuple:
    px, py = float(pt[0]), float(pt[1])
    wf = 1.0 + WALL_VARIATION   * low_freq_vary(px * 0.07, py * 0.07, NOISE_SEED)
    hf = 1.0 + HEIGHT_VARIATION * low_freq_vary(px * 0.09, py * 0.09, NOISE_SEED + 17)
    if slice_idx < TRANSITION_SLICES:
        t  = slice_idx / TRANSITION_SLICES
        ef = ENTRANCE_MULT * (1 - t) + 1.0 * t
        wf *= ef
        hf *= 1.0 + (ef - 1.0) * 0.55
    tail = n_slices - 1 - slice_idx
    if tail < TRANSITION_SLICES:
        t  = tail / TRANSITION_SLICES
        ef = EXIT_MULT * (1 - t) + 1.0 * t
        wf *= ef
        hf *= 1.0 + (ef - 1.0) * 0.55
    return max(0.45, min(2.3, wf)), max(0.50, min(2.1, hf))


def arch_profile(world_pos: np.ndarray, seed: int,
                 width_factor: float = 1.0,
                 height_factor: float = 1.0) -> list:
    """Realistic cave arch cross-section with FBM roughness + stalactites."""
    section = []
    wx, wy  = float(world_pos[0]), float(world_pos[1])
    eff_r   = INNER_RADIUS * width_factor
    eff_h   = CAVE_HEIGHT  * height_factor
    asym_base = 1.0 + 0.13 * fbm_noise(wx * 0.08, wy * 0.08, 0.0, seed, octaves=2)

    for j in range(N_PROFILE):
        alpha  = math.pi * j / (N_PROFILE - 1)
        flare  = (1.0 + 0.20 * math.exp(-((alpha) ** 2) / 0.40)
                      + 0.20 * math.exp(-((alpha - math.pi) ** 2) / 0.40))
        base_r = math.cos(alpha) * eff_r * asym_base * flare
        base_h = math.sin(alpha) * eff_h
        ceil_lin  = max(0.0, math.sin(alpha))
        ceil_zone = ceil_lin ** 0.65
        wall_n    = fbm_noise(
            wx * 0.44 + alpha * 0.83,
            wy * 0.44,
            alpha * 1.71 + wx * 0.13,
            seed=seed, octaves=6
        )
        rough_amp    = NOISE_AMP * eff_r * (0.22 + 0.78 * ceil_zone)
        wall_perturb = wall_n * rough_amp
        stal_i  = stalactite_field(wx * 0.90 + alpha * 1.51, wy * 0.90, seed=seed)
        stal_drop = stal_i * STALACTITE_AMP * eff_h * (ceil_lin ** 2.2)
        r_len = math.hypot(base_r, base_h)
        if r_len > 1e-9:
            nr      = base_r / r_len
            nz      = base_h / r_len
            local_r = base_r + wall_perturb * nr
            local_h = base_h + wall_perturb * nz - stal_drop
        else:
            local_r = base_r
            local_h = base_h - stal_drop
        if ceil_lin > 0.72:
            local_h = max(local_h, CAMERA_CLEARANCE_HEIGHT)
        local_h = max(0.06, local_h)
        section.append((local_r, local_h))
    return section


def cave_slice_limits(path_points: list,
                      slice_idx: int,
                      seed_offset: int = 0) -> Tuple[float, float]:
    """
    Conservative usable half-width and ceiling height for one cave slice.

    The mesh profile is noisy and asymmetric, so placement uses the smaller
    side wall distance to keep objects fully inside the cave arch.
    """
    n_slices = len(path_points)
    pt = path_points[int(np.clip(slice_idx, 0, n_slices - 1))]
    wf, hf = get_slice_dimensions(slice_idx, n_slices, pt)
    profile = arch_profile(
        pt,
        seed=NOISE_SEED + seed_offset,
        width_factor=wf,
        height_factor=hf,
    )
    min_r = min(lr for lr, _ in profile)
    max_r = max(lr for lr, _ in profile)
    max_h = max(lu for _, lu in profile)
    return max(0.0, min(abs(min_r), abs(max_r))), max_h

# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 4 — MESH GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def _build_vertex_grid(path_points, frames,
                       extra_offset: float = 0.0,
                       seed_offset: int = 0) -> list:
    n_slices = len(path_points)
    verts    = []
    for idx, (pt, (fwd, right, up)) in enumerate(zip(path_points, frames)):
        center    = pt.copy()
        center[2] = FLOOR_Z
        wf, hf    = get_slice_dimensions(idx, n_slices, pt)
        profile   = arch_profile(pt, seed=NOISE_SEED + seed_offset,
                                 width_factor=wf, height_factor=hf)
        slice_verts = []
        for (lr, lu) in profile:
            if extra_offset > 0.0:
                r_len = math.hypot(lr, lu)
                if r_len > 1e-9:
                    lr += extra_offset * (lr / r_len)
                    lu += extra_offset * (lu / r_len)
                lu = max(0.06, lu)
            v = center + lr * right + lu * up
            slice_verts.append(v)
        verts.append(slice_verts)
    return verts


def generate_cave_mesh(path_points, frames,
                       extra_offset: float   = 0.0,
                       seed_offset:  int     = 0,
                       inward_normals: bool  = True,
                       open_entrance:  bool  = True,
                       open_exit:      bool  = True) -> list:
    verts     = _build_vertex_grid(path_points, frames, extra_offset, seed_offset)
    n_slices  = len(verts)
    n_profile = N_PROFILE
    tris      = []
    for i in range(n_slices - 1):
        for j in range(n_profile - 1):
            v00, v10 = verts[i][j],     verts[i+1][j]
            v11, v01 = verts[i+1][j+1], verts[i][j+1]
            if inward_normals:
                tris.append([v00, v10, v11])
                tris.append([v00, v11, v01])
            else:
                tris.append([v00, v11, v10])
                tris.append([v00, v01, v11])

    def make_cap(slice_verts, sign, add_cap):
        if not add_cap:
            return
        origin    = ((slice_verts[0] + slice_verts[-1]) * 0.5).copy()
        origin[2] = FLOOR_Z + CAVE_HEIGHT * 0.28
        for j in range(n_profile - 1):
            va, vb = slice_verts[j], slice_verts[j+1]
            if sign == -1:
                tris.append([origin, va, vb] if inward_normals else [origin, vb, va])
            else:
                tris.append([origin, vb, va] if inward_normals else [origin, va, vb])

    make_cap(verts[0],  -1, not open_entrance)
    make_cap(verts[-1], +1, not open_exit)

    valid = []
    for tri in tris:
        v0, v1, v2 = (np.asarray(v, np.float64) for v in tri)
        if 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0)) > 1e-8:
            valid.append(tri)
    return valid


@dataclass
class CaveMeshSpec:
    name: str
    filename: str
    material: str
    triangles: list
    contype: int = 1
    conaffinity: int = 1
    friction: str = "0.85 0.005 0.0001"


def _valid_triangles(tris: list) -> list:
    valid = []
    for tri in tris:
        v0, v1, v2 = (np.asarray(v, np.float64) for v in tri)
        if 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0)) > 1e-8:
            valid.append(tri)
    return valid


def _is_opening_segment(slice_idx: int, profile_segment_idx: int, openings: Optional[list]) -> bool:
    if not openings:
        return False
    for opening in openings:
        center_idx = int(opening["center_idx"])
        half_slices = int(opening["half_slices"])
        if abs(slice_idx - center_idx) > half_slices:
            continue

        side = int(opening["side"])
        profile_segments = int(opening["profile_segments"])
        if side >= 0 and profile_segment_idx < profile_segments:
            return True
        if side < 0 and profile_segment_idx >= (N_PROFILE - 1 - profile_segments):
            return True
    return False


def generate_cave_mesh_sections(path_points, frames,
                                section_count: int = CAVE_WALL_SECTIONS,
                                seed_offset: int = 0,
                                extra_offset: float = 0.0,
                                inward_normals: bool = True,
                                name_prefix: str = "cave_wall",
                                filename_prefix: str = "cave_wall",
                                material_offset: int = 0,
                                contype: int = 1,
                                conaffinity: int = 1,
                                openings: Optional[list] = None) -> List[CaveMeshSpec]:
    """Build low-poly cave wall sections with one shared wall material."""
    verts = _build_vertex_grid(path_points, frames, extra_offset, seed_offset)
    n_slices = len(verts)
    if n_slices < 2:
        return []

    section_count = max(1, min(int(section_count), n_slices - 1))
    sections: List[CaveMeshSpec] = []
    for section_idx in range(section_count):
        start = int(round(section_idx * (n_slices - 1) / section_count))
        end = int(round((section_idx + 1) * (n_slices - 1) / section_count))
        if end <= start:
            continue

        tris = []
        for i in range(start, end):
            for j in range(N_PROFILE - 1):
                if _is_opening_segment(i, j, openings):
                    continue
                v00, v10 = verts[i][j],     verts[i + 1][j]
                v11, v01 = verts[i + 1][j + 1], verts[i][j + 1]
                if inward_normals:
                    tris.append([v00, v11, v10])
                    tris.append([v00, v01, v11])
                else:
                    tris.append([v00, v10, v11])
                    tris.append([v00, v11, v01])

        sections.append(
            CaveMeshSpec(
                name=f"{name_prefix}_{section_idx:02d}",
                filename=f"{filename_prefix}_{section_idx:02d}.stl",
                material=WALL_MATERIAL_NAME,
                triangles=_valid_triangles(tris),
                contype=contype,
                conaffinity=conaffinity,
            )
        )
    return sections


def _section_count_for_path(path_points: list) -> int:
    """Keep chunks large enough that material variation is visible, not noisy."""
    if len(path_points) <= 36:
        return 3
    if len(path_points) <= 72:
        return 4
    return CAVE_WALL_SECTIONS


def generate_cave_wall_layers(paths: List[CavePathSpec]) -> List[CaveMeshSpec]:
    """Generate inner visible walls plus an outside visual shell for each tunnel."""
    meshes: List[CaveMeshSpec] = []

    for path_idx, path in enumerate(paths):
        frames = compute_frames(path.path_points)
        section_count = _section_count_for_path(path.path_points)
        name_safe = path.name.replace("-", "_")

        meshes.extend(
            generate_cave_mesh_sections(
                path.path_points,
                frames,
                section_count=section_count,
                seed_offset=path.seed_offset,
                extra_offset=0.0,
                inward_normals=True,
                name_prefix=f"cave_inner_{name_safe}",
                filename_prefix=f"cave_inner_{name_safe}",
                material_offset=path_idx,
                contype=1,
                conaffinity=1,
            )
        )
        meshes.extend(
            generate_cave_mesh_sections(
                path.path_points,
                frames,
                section_count=section_count,
                seed_offset=path.seed_offset,
                extra_offset=OUTER_LAYER_OFFSET,
                inward_normals=False,
                name_prefix=f"cave_outer_{name_safe}",
                filename_prefix=f"cave_outer_{name_safe}",
                material_offset=path_idx + 2,
                contype=0,
                conaffinity=0,
            )
        )

    return meshes


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION 5 — BINARY STL WRITER
# ═══════════════════════════════════════════════════════════════════════════

def _face_normal(v0, v1, v2) -> np.ndarray:
    n = np.cross(v1 - v0, v2 - v0)
    l = np.linalg.norm(n)
    return n / l if l > 1e-12 else np.array([0., 0., 1.])


def write_binary_stl(triangles: list, filepath: str) -> None:
    hdr = (b"MuJoCo Cave SAR Generator" + b"\x00" * 80)[:80]
    with open(filepath, "wb") as f:
        f.write(hdr)
        f.write(struct.pack("<I", len(triangles)))
        for tri in triangles:
            v0, v1, v2 = (np.asarray(v, np.float32) for v in tri)
            n = _face_normal(v0, v1, v2).astype(np.float32)
            f.write(struct.pack("<fff", *n))
            f.write(struct.pack("<fff", *v0))
            f.write(struct.pack("<fff", *v1))
            f.write(struct.pack("<fff", *v2))
            f.write(struct.pack("<H", 0))
    kb = os.path.getsize(filepath) / 1024
    print(f"  ✓  {len(triangles):>7,} tris  →  {filepath}  ({kb:.0f} KB)")


def generate_room_mesh(half_size: float, height: float) -> list:
    """Generate a simple square cavern mesh (walls + ceiling)."""
    x0, x1 = -half_size, half_size
    y0, y1 = -half_size, half_size
    z0, z1 = FLOOR_Z, FLOOR_Z + height

    # Each face is two triangles.
    faces = [
        # +Y wall
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1)],
        [(x0, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        # -Y wall
        [(x1, y0, z0), (x0, y0, z0), (x0, y0, z1)],
        [(x1, y0, z0), (x0, y0, z1), (x1, y0, z1)],
        # +X wall
        [(x1, y1, z0), (x1, y0, z0), (x1, y0, z1)],
        [(x1, y1, z0), (x1, y0, z1), (x1, y1, z1)],
        # -X wall
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z1), (x0, y0, z1)],
        # Ceiling
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1)],
        [(x0, y0, z1), (x1, y1, z1), (x0, y1, z1)],
    ]

    return [[np.array(v0), np.array(v1), np.array(v2)] for v0, v1, v2 in faces]


# ═══════════════════════════════════════════════════════════════════════════
#  MODULE 1 EXTENSION — ROCK OBSTACLE GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RockObstacle:
    """A single randomly-sized ellipsoid rock in the cave."""
    pos:   Tuple[float, float, float]   # (x, y, z) world position
    size:  Tuple[float, float, float]   # half-extents (a, b, c)
    euler: Tuple[float, float, float]   # rotation Euler angles (radians)
    name:  str
    


def generate_rocks(path_points: list, rng: np.random.Generator,
                   n_rocks: int = N_ROCKS,
                   seed_offset: int = 0) -> List[RockObstacle]:
    """
    Scatter rock obstacle geoms along the cave path.

    Strategy
    ─────────
    • Pick a random path index (avoiding first/last 5% for entry/exit clearance)
    • Offset laterally within the local cave arch so the path centre stays
      walkable and rocks remain inside the cave
    • Random height: sits on floor (z = half_c) to simulate real boulders
    • Random aspect-ratio ellipsoid size clamped to configured range
    • Random yaw rotation for visual variety
    """
    rocks   = []
    n_pts   = len(path_points)
    margin  = max(5, int(n_pts * 0.05))

    for k in range(n_rocks):
        # Random size
        sa = rng.uniform(ROCK_MIN_SIZE, ROCK_MAX_SIZE)
        sb = rng.uniform(ROCK_MIN_SIZE, ROCK_MAX_SIZE)
        sc = rng.uniform(ROCK_MIN_SIZE, ROCK_MAX_SIZE * 0.7)   # flatter

        # Choose a path sample with enough arch width for the rock.
        idx = int(rng.integers(margin, n_pts - margin))
        half_width, _ = cave_slice_limits(path_points, idx, seed_offset)
        rock_margin = max(sa, sb) + 0.10
        usable_half_width = max(0.05, half_width - rock_margin)

        side = rng.choice([-1.0, 1.0])
        lat_min = min(usable_half_width * 0.45, ROCK_CLEAR_ZONE * usable_half_width)
        lat_max = max(lat_min + 0.05, usable_half_width)
        lateral = side * rng.uniform(lat_min, lat_max)

        pt = path_points[idx]
        _, right_dir, _ = frame_at_path_index(path_points, idx)

        world_pos = pt[:3].copy()
        world_pos += lateral * right_dir
        world_pos[2] = FLOOR_Z + sc   # sits on floor

        euler = (0.0, 0.0, float(rng.uniform(0, math.pi)))

        rocks.append(RockObstacle(
            pos   = (float(world_pos[0]), float(world_pos[1]), float(world_pos[2])),
            size  = (float(sa), float(sb), float(sc)),
            euler = euler,
            name  = f"rock_{k:03d}",
        ))

    return rocks


def generate_rocks_room(rng: np.random.Generator, n_rocks: int = N_ROCKS) -> List[RockObstacle]:
    rocks = []
    if n_rocks <= 0:
        return rocks
    margin = 1.0
    attempts = 0
    while len(rocks) < n_rocks and attempts < n_rocks * 40:
        attempts += 1
        sa = rng.uniform(ROCK_MIN_SIZE, ROCK_MAX_SIZE)
        sb = rng.uniform(ROCK_MIN_SIZE, ROCK_MAX_SIZE)
        sc = rng.uniform(ROCK_MIN_SIZE, ROCK_MAX_SIZE * 0.7)
        px = rng.uniform(-ROOM_HALF_SIZE + margin, ROOM_HALF_SIZE - margin)
        py = rng.uniform(-ROOM_HALF_SIZE + margin, ROOM_HALF_SIZE - margin)
        pz = FLOOR_Z + sc
        euler = (0.0, 0.0, float(rng.uniform(0, math.pi)))
        candidate = RockObstacle(
            pos=(float(px), float(py), float(pz)),
            size=(float(sa), float(sb), float(sc)),
            euler=euler,
            name=f"rock_{len(rocks):03d}",
        )
        if any(
            math.hypot(candidate.pos[0] - rock.pos[0], candidate.pos[1] - rock.pos[1])
            < (max(candidate.size) + max(rock.size) + 0.6)
            for rock in rocks
        ):
            continue
        rocks.append(candidate)
    return rocks


# ═══════════════════════════════════════════════════════════════════════════
#  MODULE 2 — HUMAN / SURVIVOR GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SurvivorMetadata:
    """Full metadata record for one survivor placed in the scene."""
    survivor_id:        int
    survivor_position:  Tuple[float, float, float]
    pose:               str      # "lying" | "sitting" | "slumped"
    visibility_level:   str      # "low" | "medium" | "high"
    occlusion_percentage: float  # 0–100


@dataclass
class Survivor:
    """Complete survivor spec: geometry + metadata."""
    meta:   SurvivorMetadata
    geoms:  List[dict]           # list of MuJoCo geom kwargs dicts


# ── Pose definitions ────────────────────────────────────────────────────────
#
#  Each pose is a dict of named body segments → local offsets & orientations
#  relative to the survivor's root position.
#  Geoms: head (sphere), torso (capsule), upper_arm×2, lower_arm×2, thigh×2, shin×2

def _make_survivor_geoms(root: np.ndarray, pose: str,
                          sid: int, rng: np.random.Generator) -> List[dict]:
    """
    Build a list of MuJoCo geom dicts representing a simplified humanoid.

    Body proportions (metric):
      • head:      r = 0.11 m
      • torso:     r = 0.13 m, L = 0.55 m
      • upper arm: r = 0.05 m, L = 0.30 m
      • lower arm: r = 0.04 m, L = 0.25 m
      • thigh:     r = 0.07 m, L = 0.42 m
      • shin:      r = 0.05 m, L = 0.38 m

    Poses are approximated with Euler angle offsets:
      lying  — torso horizontal, arms along sides, legs straight
      sitting — torso ~75° up, legs bent ~90° at hip, lower legs vertical
      slumped — torso ~30° from floor, one arm extended, legs splayed
    """
    geoms: List[dict] = []
    rgba_skin = "0.95 0.95 0.95 1.0"
    prefix = f"surv{sid}"
    s = SURVIVOR_SCALE

    yaw_deg = float(rng.uniform(0.0, 360.0))   # random body facing direction
    yaw_rad = math.radians(yaw_deg)

    def rot_xy(dx, dy, angle):
        c, s = math.cos(angle), math.sin(angle)
        return c * dx - s * dy, s * dx + c * dy

    # ── Helper to append a geom ────────────────────────────────────────────
    def add(name, gtype, pos_local, size, euler_local=(0, 0, 0)):
        scaled_pos = tuple(value * SURVIVOR_SCALE for value in pos_local)
        scaled_size = tuple(value * SURVIVOR_SCALE for value in size)
        lx, ly = rot_xy(scaled_pos[0], scaled_pos[1], yaw_rad)
        geoms.append({
            "name":  f"{prefix}_{name}",
            "type":  gtype,
            "pos":   (lx, ly, scaled_pos[2]),
            "size":  scaled_size,
            "euler": (euler_local[0],
                      euler_local[1],
                      euler_local[2] + yaw_deg),
            "rgba":  rgba_skin,
        })

    # ── LYING POSE ─────────────────────────────────────────────────────────
    if pose == "lying":
        # Torso flat on ground (along Y-axis), capsule along fromto
        add("torso",      "capsule", ( 0.00,  0.00, 0.13), (0.13, 0.55), (90, 0, 0))
        add("head",       "sphere",  ( 0.00,  0.32, 0.18), (0.11,))
        add("upper_arm_L","capsule", ( 0.23,  0.00, 0.07), (0.05, 0.28), (90, 0, 0))
        add("upper_arm_R","capsule", (-0.23,  0.00, 0.07), (0.05, 0.28), (90, 0, 0))
        add("lower_arm_L","capsule", ( 0.23, -0.28, 0.07), (0.04, 0.24), (90, 0, 0))
        add("lower_arm_R","capsule", (-0.23, -0.28, 0.07), (0.04, 0.24), (90, 0, 0))
        add("thigh_L",    "capsule", ( 0.10, -0.35, 0.07), (0.07, 0.40), (90, 0, 0))
        add("thigh_R",    "capsule", (-0.10, -0.35, 0.07), (0.07, 0.40), (90, 0, 0))
        add("shin_L",     "capsule", ( 0.10, -0.78, 0.07), (0.05, 0.36), (90, 0, 0))
        add("shin_R",     "capsule", (-0.10, -0.78, 0.07), (0.05, 0.36), (90, 0, 0))

    # ── SITTING POSE ───────────────────────────────────────────────────────
    elif pose == "sitting":
        add("torso",      "capsule", ( 0.00,  0.00, 0.40), (0.13, 0.28), (0, 0, 0))
        add("head",       "sphere",  ( 0.00,  0.04, 0.82), (0.11,))
        add("upper_arm_L","capsule", ( 0.22,  0.00, 0.55), (0.05, 0.28), (0, 10, 0))
        add("upper_arm_R","capsule", (-0.22,  0.00, 0.55), (0.05, 0.28), (0, -10, 0))
        add("lower_arm_L","capsule", ( 0.28, -0.18, 0.30), (0.04, 0.22), (75, 0, 0))
        add("lower_arm_R","capsule", (-0.28, -0.18, 0.30), (0.04, 0.22), (75, 0, 0))
        add("thigh_L",    "capsule", ( 0.10,  0.20, 0.12), (0.07, 0.20), (90, 0, 0))
        add("thigh_R",    "capsule", (-0.10,  0.20, 0.12), (0.07, 0.20), (90, 0, 0))
        add("shin_L",     "capsule", ( 0.10,  0.20, 0.30), (0.05, 0.18), (10, 0, 0))
        add("shin_R",     "capsule", (-0.10,  0.20, 0.30), (0.05, 0.18), (10, 0, 0))

    # ── SLUMPED POSE ───────────────────────────────────────────────────────
    else:  # slumped
        add("torso",      "capsule", ( 0.00,  0.00, 0.22), (0.13, 0.28), (40, 0, 0))
        add("head",       "sphere",  (-0.05,  0.32, 0.45), (0.11,))
        add("upper_arm_L","capsule", ( 0.28,  0.18, 0.38), (0.05, 0.28), (60, 15, 0))
        add("upper_arm_R","capsule", (-0.25, -0.05, 0.20), (0.05, 0.28), (80, -5, 0))
        add("lower_arm_L","capsule", ( 0.42,  0.38, 0.14), (0.04, 0.22), (90, 10, 0))
        add("lower_arm_R","capsule", (-0.22, -0.30, 0.08), (0.04, 0.22), (90, -8, 0))
        add("thigh_L",    "capsule", ( 0.12, -0.20, 0.10), (0.07, 0.38), (80, 8, 0))
        add("thigh_R",    "capsule", (-0.12, -0.10, 0.10), (0.07, 0.38), (75, -5, 0))
        add("shin_L",     "capsule", ( 0.20, -0.60, 0.08), (0.05, 0.34), (90, 0, 0))
        add("shin_R",     "capsule", (-0.08, -0.50, 0.08), (0.05, 0.34), (90, 0, 0))

    return geoms


SURVIVOR_SCALE = REAL_TO_SIM_SCALE
SURVIVOR_ROOT_Z = {
    "lying":  FLOOR_Z + 0.05 * SURVIVOR_SCALE,
    "sitting": FLOOR_Z + 0.35 * SURVIVOR_SCALE,
    "slumped": FLOOR_Z + 0.15 * SURVIVOR_SCALE,
}

SURVIVOR_FOOTPRINT_RADIUS = {
    "lying":  1.05 * SURVIVOR_SCALE,
    "sitting": 0.55 * SURVIVOR_SCALE,
    "slumped": 0.85 * SURVIVOR_SCALE,
}

SURVIVOR_POSE_HEIGHT = {
    "lying":  0.35 * SURVIVOR_SCALE,
    "sitting": 0.95 * SURVIVOR_SCALE,
    "slumped": 0.60 * SURVIVOR_SCALE,
}

SURVIVOR_WALL_MARGIN = 0.18 * SURVIVOR_SCALE
SURVIVOR_ROCK_GAP = 0.15 * SURVIVOR_SCALE


def _nearest_path_index_2d(path_points: list, xy: np.ndarray) -> int:
    """Index of the closest path sample in the top-down plane."""
    best_idx = 0
    best_dist = float("inf")
    for i, pt in enumerate(path_points):
        dx = float(pt[0] - xy[0])
        dy = float(pt[1] - xy[1])
        d2 = dx * dx + dy * dy
        if d2 < best_dist:
            best_idx = i
            best_dist = d2
    return best_idx


def _visibility_from_occlusion(occ_pct: float, difficulty: str) -> str:
    """Map an occlusion percentage to a configured visibility band."""
    thresholds = {
        "easy": (40.0, 70.0),
        "medium": (25.0, 60.0),
        "hard": (15.0, 50.0),
    }
    high_max, medium_max = thresholds.get(difficulty, thresholds["medium"])
    if occ_pct < high_max:
        return "high"
    if occ_pct < medium_max:
        return "medium"
    return "low"


def _survivor_inside_cave(path_points: list,
                          idx: int,
                          lateral: float,
                          pose: str,
                          seed_offset: int = 0) -> bool:
    """Validate that a survivor pose fits inside the local cave arch."""
    half_width, ceiling = cave_slice_limits(path_points, idx, seed_offset)
    footprint = SURVIVOR_FOOTPRINT_RADIUS[pose] + SURVIVOR_WALL_MARGIN
    pose_top = SURVIVOR_ROOT_Z[pose] + SURVIVOR_POSE_HEIGHT[pose]
    return abs(lateral) + footprint <= half_width and pose_top <= ceiling - 0.10


def _root_from_lateral(path_points: list,
                       idx: int,
                       lateral: float,
                       pose: str) -> np.ndarray:
    """Build a survivor root position from a path slice and lateral offset."""
    _, right_dir, _ = frame_at_path_index(path_points, idx)
    root = path_points[idx][:3].copy()
    root += lateral * right_dir
    root[2] = SURVIVOR_ROOT_Z[pose]
    return root


def _rock_overlap_clear(root: np.ndarray,
                        pose: str,
                        rocks: List[RockObstacle]) -> bool:
    """Keep survivors inside the cave while allowing nearby partial occluders."""
    for rock in rocks:
        dx = rock.pos[0] - root[0]
        dy = rock.pos[1] - root[1]
        dist_2d = math.hypot(dx, dy)
        min_gap = (
            max(rock.size[0], rock.size[1])
            + SURVIVOR_FOOTPRINT_RADIUS[pose]
            + SURVIVOR_ROCK_GAP
        )
        if dist_2d < min_gap:
            return False
    return True


def _sample_rock_shadow_root(path_points: list,
                             rocks: List[RockObstacle],
                             rng: np.random.Generator,
                             pose: str,
                             camera_pos: np.ndarray,
                             seed_offset: int = 0) -> Optional[Tuple[np.ndarray, int, float]]:
    """
    Try to place a survivor behind a rock relative to the camera.

    Returns (root, path_idx, lateral) when the candidate still fits inside
    the cave, otherwise None.
    """
    if not rocks:
        return None

    for _ in range(24):
        rock = rocks[int(rng.integers(0, len(rocks)))]
        rock_xy = np.array([rock.pos[0], rock.pos[1]], dtype=float)
        cam_xy = np.array([camera_pos[0], camera_pos[1]], dtype=float)
        away = rock_xy - cam_xy
        away_len = np.linalg.norm(away)
        if away_len < 1e-6:
            continue
        away /= away_len

        radius = max(rock.size[0], rock.size[1])
        candidate_xy = rock_xy + away * rng.uniform(radius + 0.35, radius + 1.10)
        idx = _nearest_path_index_2d(path_points, candidate_xy)
        _, right_dir, _ = frame_at_path_index(path_points, idx)
        delta = np.array([candidate_xy[0] - path_points[idx][0],
                          candidate_xy[1] - path_points[idx][1],
                          0.0])
        lateral = float(np.dot(delta, right_dir))
        if not _survivor_inside_cave(path_points, idx, lateral, pose, seed_offset):
            continue
        half_width, _ = cave_slice_limits(path_points, idx, seed_offset)
        visible_lateral_limit = (
            half_width - SURVIVOR_FOOTPRINT_RADIUS[pose] - SURVIVOR_WALL_MARGIN
        ) * 0.55
        if abs(lateral) > max(0.25, visible_lateral_limit):
            continue

        root = _root_from_lateral(path_points, idx, lateral, pose)
        if _rock_overlap_clear(root, pose, rocks):
            return root, idx, lateral

    return None


def compute_visibility(camera_pos, survivor_pos, rocks):
    """
    Compute occlusion percentage based on rocks between camera and survivor.
    
    Uses distance-based heuristic: rocks closer to survivor and between camera
    and survivor increase occlusion.
    """
    cam_vec = np.array([camera_pos[0], camera_pos[1]])
    surv_vec = np.array([survivor_pos[0], survivor_pos[1]])
    
    # Vector from camera to survivor
    to_surv = surv_vec - cam_vec
    dist_to_surv = np.linalg.norm(to_surv)
    
    if dist_to_surv < 1e-6:
        return 0.0
    
    to_surv_norm = to_surv / dist_to_surv
    
    occ = 0.0
    for rock in rocks:
        rock_vec = np.array([rock.pos[0], rock.pos[1]])
        to_rock = rock_vec - cam_vec
        
        # Project rock onto camera-to-survivor line
        proj_len = np.dot(to_rock, to_surv_norm)
        
        # Rock is relevant only if it's between camera and survivor
        if proj_len > 0.0 and proj_len < dist_to_surv:
            # Distance from rock to the line
            perp_dist = np.linalg.norm(to_rock - proj_len * to_surv_norm)
            rock_radius = max(rock.size) * 0.6  # effective blocking radius
            
            if perp_dist < rock_radius:
                # Rock blocks some view
                # Scale by how close it is to survivor (closer = more occlusion)
                dist_to_survivor = dist_to_surv - proj_len
                blocking_factor = 1.0 - np.clip(perp_dist / rock_radius, 0.0, 1.0)
                distance_factor = 1.0 / (1.0 + dist_to_survivor * 0.5)
                occ += blocking_factor * distance_factor * 0.25  # up to 25% per rock
    
    return min(1.0, occ)


def place_survivors(path_points: list,
                    rocks: List[RockObstacle],
                    rng: np.random.Generator,
                    n_survivors: int,
                    difficulty: str = "medium",
                    camera_pos: Optional[np.ndarray] = None,
                    seed_offset: int = 0,
                    start_id: int = 0) -> List[Survivor]:
    """
    Randomly place n_survivors inside the cave and measure rock occlusion.
    
    Survivors are constrained against the generated cave cross-section:
    - lateral footprint must fit within the local noisy arch
    - pose height must fit below the local ceiling
    - hard/medium difficulty preferentially tries rock-shadow candidates
    
    Args:
        path_points: Cave center line points
        rocks: List of rock obstacles
        rng: Random number generator
        n_survivors: Number of survivors to place
        difficulty: "easy" (high visibility) to "hard" (low visibility)
        camera_pos: Camera position for occlusion calculation (optional)
        seed_offset: Mesh noise offset used for the cave profile
    """
    survivors: List[Survivor] = []
    n_pts  = len(path_points)
    margin = max(5, int(n_pts * 0.08))
    difficulty = difficulty if difficulty in {"easy", "medium", "hard"} else "medium"

    lateral_ranges = {
        "easy": (0.00, 0.20),
        "medium": (0.05, 0.30),
        "hard": (0.20, 0.50),
    }
    occlusion_bias = {
        "easy": 0.00,
        "medium": 0.15,
        "hard": 0.35,
    }
    occ_multiplier = {
        "easy": 0.70,
        "medium": 1.00,
        "hard": 1.35,
    }
    camera_ref = np.asarray(camera_pos if camera_pos is not None else ROBOT_SPAWN, dtype=float)

    for local_sid in range(n_survivors):
        sid = start_id + local_sid
        root = None
        pose = str(rng.choice(SURVIVOR_POSES))

        for _ in range(120):
            use_shadow = rng.random() < occlusion_bias[difficulty]
            shadow = None
            if use_shadow:
                shadow = _sample_rock_shadow_root(
                    path_points, rocks, rng, pose, camera_ref, seed_offset
                )

            if shadow is not None:
                candidate_root, idx, lateral = shadow
            else:
                idx = int(rng.integers(margin, n_pts - margin))
                half_width, _ = cave_slice_limits(path_points, idx, seed_offset)
                usable_half_width = half_width - (
                    SURVIVOR_FOOTPRINT_RADIUS[pose] + SURVIVOR_WALL_MARGIN
                )
                if usable_half_width <= 0.05:
                    pose = str(rng.choice(SURVIVOR_POSES))
                    continue

                lo, hi = lateral_ranges[difficulty]
                lat_min = min(usable_half_width * lo, usable_half_width)
                lat_max = max(lat_min + 0.05, usable_half_width * hi)
                lateral = float(rng.choice([-1.0, 1.0]) * rng.uniform(lat_min, lat_max))
                candidate_root = _root_from_lateral(path_points, idx, lateral, pose)

            if not _survivor_inside_cave(path_points, idx, lateral, pose, seed_offset):
                pose = str(rng.choice(SURVIVOR_POSES))
                continue
            if not _rock_overlap_clear(candidate_root, pose, rocks):
                continue

            root = candidate_root
            break

        if root is None:
            print(f"       ! Could not fit survivor {sid}; skipping placement")
            continue

        # ── Occlusion estimate ─────────────────────────────────────────────
        # Combine line-of-sight blocking with nearby-rock partial cover.
        occ = compute_visibility(camera_ref, root, rocks)
        nearby_occ = 0.0
        for rock in rocks:
            dx = rock.pos[0] - root[0]
            dy = rock.pos[1] - root[1]
            dz = rock.pos[2] - root[2]
            dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
            rock_size = max(rock.size)
            if dist_3d < rock_size * 3.5:
                nearby_occ += max(0.0, 1.0 - (dist_3d / (rock_size * 3.5))) * 0.12
        occ += nearby_occ

        # Add pose-based partial self-occlusion
        pose_occ = {"lying": 0.10, "sitting": 0.05, "slumped": 0.15}[pose]
        occ_pct = float(min(100.0, (occ * occ_multiplier[difficulty] + pose_occ) * 100.0))

        # ── Visibility level ───────────────────────────────────────────────
        vis = _visibility_from_occlusion(occ_pct, difficulty)

        meta  = SurvivorMetadata(
            survivor_id          = sid,
            survivor_position    = (float(root[0]), float(root[1]), float(root[2])),
            pose                 = pose,
            visibility_level     = vis,
            occlusion_percentage = round(occ_pct, 1),
        )
        geoms = _make_survivor_geoms(root, pose, sid, rng)
        survivors.append(Survivor(meta=meta, geoms=geoms))

    return survivors


def place_survivors_room(rng: np.random.Generator,
                         rocks: List[RockObstacle],
                         n_survivors: int) -> List[Survivor]:
    survivors: List[Survivor] = []
    wall_margin = 1.0
    for sid in range(n_survivors):
        pose = str(rng.choice(SURVIVOR_POSES))
        root = None
        for _ in range(120):
            px = rng.uniform(-ROOM_HALF_SIZE + wall_margin, ROOM_HALF_SIZE - wall_margin)
            py = rng.uniform(-ROOM_HALF_SIZE + wall_margin, ROOM_HALF_SIZE - wall_margin)
            pz = SURVIVOR_ROOT_Z[pose]
            candidate = np.array([px, py, pz], dtype=float)
            if not _rock_overlap_clear(candidate, pose, rocks):
                continue
            root = candidate
            break
        if root is None:
            continue

        meta = SurvivorMetadata(
            survivor_id=sid,
            survivor_position=(float(root[0]), float(root[1]), float(root[2])),
            pose=pose,
            visibility_level="high",
            occlusion_percentage=0.0,
        )
        geoms = _make_survivor_geoms(root, pose, sid, rng)
        survivors.append(Survivor(meta=meta, geoms=geoms))
    return survivors


# ═══════════════════════════════════════════════════════════════════════════
#  MODULE 1 — MUJOCO XML WRITER  (extended with rocks + survivors)
# ═══════════════════════════════════════════════════════════════════════════

def _geom_xml(g: dict) -> str:
    """Render a geom dict to an XML <geom .../> string."""
    pos_s  = " ".join(f"{v:.4f}" for v in g["pos"])
    size_s = " ".join(f"{v:.4f}" for v in g["size"])
    euler_s = " ".join(f"{v:.4f}" for v in g["euler"])
    return (
        f'    <geom name="{g["name"]}" type="{g["type"]}" '
        f'pos="{pos_s}" size="{size_s}" euler="{euler_s}" '
        f'rgba="{g["rgba"]}" contype="0" conaffinity="0"/>'
    )


def _camera_xyaxes(camera_pos: np.ndarray,
                   target_pos: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """Build MuJoCo camera xyaxes that look from camera_pos to target_pos."""
    forward = np.asarray(target_pos, dtype=float) - np.asarray(camera_pos, dtype=float)
    fl = np.linalg.norm(forward)
    if fl < 1e-9:
        forward = np.array([1.0, 0.0, 0.0])
    else:
        forward = forward / fl

    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    rl = np.linalg.norm(right)
    if rl < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / rl

    z_axis = -forward
    up = np.cross(z_axis, right)
    up = up / np.linalg.norm(up)
    return (
        float(right[0]), float(right[1]), float(right[2]),
        float(up[0]), float(up[1]), float(up[2]),
    )


def _normalise_cave_meshes(
    cave_meshes: Union[str, List[CaveMeshSpec]],
    outer_stl: Optional[str],
) -> Tuple[str, str]:
    if isinstance(cave_meshes, list):
        asset_lines = [
            f'    <mesh name="{mesh.name}" file="{mesh.filename}" scale="1 1 1"/>'
            for mesh in cave_meshes
        ]
        geom_lines = [
            f'    <geom name="{mesh.name}" type="mesh" mesh="{mesh.name}" '
            f'material="{mesh.material}" contype="{mesh.contype}" '
            f'conaffinity="{mesh.conaffinity}" friction="{mesh.friction}"/>'
            for mesh in cave_meshes
        ]
        return "\n".join(asset_lines), "\n".join(geom_lines)

    asset_lines = [f'    <mesh name="cave_inner" file="{cave_meshes}" scale="1 1 1"/>']
    geom_lines = [
        '    <geom name="cave_wall_inner" type="mesh" mesh="cave_inner"',
        '      material="cave_stone_dark"',
        '      contype="1" conaffinity="1"',
        '      friction="0.85 0.005 0.0001"/>',
    ]
    if outer_stl:
        asset_lines.append(f'    <mesh name="cave_outer" file="{outer_stl}" scale="1 1 1"/>')
        geom_lines.append(
            '    <geom name="cave_wall_outer" type="mesh" mesh="cave_outer" '
            f'material="{WALL_MATERIAL_NAME}" contype="0" conaffinity="0"/>'
        )
    return "\n".join(asset_lines), "\n".join(geom_lines)


def _stone_material_xml() -> str:
    lines = []
    for name, rgba, specular, shininess in STONE_MATERIALS:
        lines.append(
            f'    <material name="{name}" rgba="{rgba}" '
            f'specular="{specular}" shininess="{shininess}" reflectance="0.01"/>'
        )
    return "\n".join(lines)


def write_mujoco_xml(cave_meshes: Union[str, List[CaveMeshSpec]],
                     outer_stl: Optional[str], xml_path: str,
                     path_points: list,
                     rocks: Optional[List[RockObstacle]] = None,
                     survivors: Optional[List[Survivor]] = None,
                     light_seed: int = 0, difficulty : str ="medium",
                     dataset_camera_pos: Optional[np.ndarray] = None,
                     dataset_camera_target: Optional[np.ndarray] = None) -> None:
    """
    Write a complete MuJoCo scene XML with cave meshes, rocks,
    survivors, and a free-flying camera body for dataset rendering.
    """
    rocks     = rocks     or []
    survivors = survivors or []
    sx, sy, sz = robot_spawn_from_path(path_points)
    ex, ey, _ = get_entrance_coordinate(path_points)
    xx, xy  = float(path_points[-1][0]), float(path_points[-1][1])
    cave_mesh_asset_xml, cave_wall_geom_xml = _normalise_cave_meshes(cave_meshes, outer_stl)
    stone_material_xml = _stone_material_xml()

    n = len(path_points)
    def lp(frac):
        return path_points[min(int(frac * n), n - 1)]
    l1, l2, l3 = lp(0.25), lp(0.50), lp(0.75)
    lh = CAVE_HEIGHT * 0.80

    # ── Per-episode lighting variation ────────────────────────────────────
    rng_l  = np.random.default_rng(light_seed)
    base_d = float(rng_l.uniform(0.70, 1.05))   # overall brightness
    if difficulty == "hard":
        base_d *= 0.90
    elif difficulty == "easy":
        base_d *= 1.10

    warm   = float(rng_l.uniform(0.80, 1.00))   # warm tint
    cool   = float(rng_l.uniform(0.75, 0.95))   # cool tint

    # Occasionally add a dark area (simulates deep shadow)
    shadow_mult = float(rng_l.uniform(0.80, 1.0))

    room_wall_xml = ""
    if ROOM_MODE:
        wall_t = 0.20
        half = ROOM_HALF_SIZE
        wall_h = ROOM_HEIGHT * 0.5
        zc = FLOOR_Z + wall_h
        y_edge = half - wall_t * 0.5
        x_edge = half - wall_t * 0.5
        ceil_z = FLOOR_Z + ROOM_HEIGHT - wall_t * 0.5

        room_wall_xml = f"""
        <!-- ── Room collision walls ─────────────────────────────── -->
        <geom name=\"room_wall_y_pos\" type=\"box\" pos=\"0 {y_edge:.3f} {zc:.3f}\"
            size=\"{half:.3f} {wall_t * 0.5:.3f} {wall_h:.3f}\" material=\"cave_rock\"
            contype=\"1\" conaffinity=\"1\"/>
        ...
    """

    # ── Rock geom XML ─────────────────────────────────────────────────────
    rock_xml_lines = []
    for rock in rocks:
        px, py, pz   = rock.pos
        sa, sb, sc   = rock.size
        ex0, ey0, ez0 = rock.euler
        rock_xml_lines.append(
            f'    <geom name="{rock.name}" type="ellipsoid" '
            f'pos="{px:.3f} {py:.3f} {pz:.3f}" '
            f'size="{sa:.3f} {sb:.3f} {sc:.3f}" '
            f'euler="{math.degrees(ex0):.1f} {math.degrees(ey0):.1f} {math.degrees(ez0):.1f}" '
            f'material="cave_stone_damp" contype="1" conaffinity="1"/>'
        )
    rock_xml = "\n".join(rock_xml_lines)

    # ── Survivor body XML ─────────────────────────────────────────────────
    surv_xml_lines = []
    for surv in survivors:
        bx, by, bz = surv.meta.survivor_position
        surv_xml_lines.append(
            f'  <!-- Survivor {surv.meta.survivor_id}: '
            f'{surv.meta.pose}, vis={surv.meta.visibility_level}, '
            f'occ={surv.meta.occlusion_percentage:.0f}% -->'
        )
        surv_xml_lines.append(
            f'  <body name="survivor_{surv.meta.survivor_id}" '
            f'pos="{bx:.3f} {by:.3f} {bz:.3f}">'
        )
        for g in surv.geoms:
            surv_xml_lines.append(_geom_xml(g))
        surv_xml_lines.append("  </body>")
    surv_xml = "\n".join(surv_xml_lines)

    dataset_camera_xml = ""
    if dataset_camera_pos is not None and dataset_camera_target is not None:
        dc_pos = np.asarray(dataset_camera_pos, dtype=float)
        dc_target = np.asarray(dataset_camera_target, dtype=float)
        xyaxes = _camera_xyaxes(dc_pos, dc_target)
        forward = dc_target - dc_pos
        fl = np.linalg.norm(forward)
        forward = forward / fl if fl > 1e-9 else np.array([1.0, 0.0, 0.0])
        dataset_camera_xml = f"""
    <!-- ── Dataset camera aimed at visible survivor/cave corridor ──── -->
    <camera name="dataset_cam"
      pos="{dc_pos[0]:.3f} {dc_pos[1]:.3f} {dc_pos[2]:.3f}"
      xyaxes="{xyaxes[0]:.6f} {xyaxes[1]:.6f} {xyaxes[2]:.6f} {xyaxes[3]:.6f} {xyaxes[4]:.6f} {xyaxes[5]:.6f}"
      fovy="{CAMERA_FOV_DEG}"/>
    <light name="dataset_headlamp"
      pos="{dc_pos[0]:.3f} {dc_pos[1]:.3f} {dc_pos[2]:.3f}"
      dir="{forward[0]:.4f} {forward[1]:.4f} {forward[2]:.4f}"
      diffuse="1.45 1.28 0.98"
      specular="0.10 0.10 0.08"
      cutoff="65"
      exponent="6"
      castshadow="false"/>
"""

    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<!--
  Cave Search-and-Rescue Environment  — MuJoCo SAR Training Scene
  Generated by cave_simulation_system.py
  Entry ({ex:.1f}, {ey:.1f}) → Exit ({xx:.1f}, {xy:.1f})
-->
<mujoco model="cave_sar">

  <compiler angle="degree" meshdir="meshes/" autolimits="true"/>

  <option gravity="0 0 -9.81" timestep="0.002" integrator="RK4" cone="pyramidal"/>

  <visual>
    <headlight ambient="0.18 0.18 0.17" diffuse="0.48 0.45 0.38" specular="0.05 0.05 0.05"/>
    <rgba haze="0.06 0.05 0.04 1"/>
    <quality shadowsize="4096"/>
    <map stiffness="100" shadowscale="0.4"/>
    <global fovy="65"/>
  </visual>

  <asset>
{cave_mesh_asset_xml}

{stone_material_xml}
    <material name="cave_floor"
      rgba="0.20 0.21 0.20 1.0" specular="0.025" shininess="0.006"/>

    <texture type="skybox" builtin="flat"
      rgb1="0.03 0.03 0.05" rgb2="0.01 0.01 0.02"
      width="512" height="512"/>
  </asset>

  <worldbody>

    <!-- ── Cave interior lighting ────────────────────────────────── -->
    <light name="fill_0"
      pos="{l1[0]:.1f} {l1[1]:.1f} {lh:.1f}"
      diffuse="{base_d * 1.05 * shadow_mult:.3f} {base_d * 0.72 * shadow_mult:.3f} {base_d * 0.60 * shadow_mult:.3f}"
      castshadow="false"/>
    <light name="fill_1"
      pos="{l2[0]:.1f} {l2[1]:.1f} {lh:.1f}"
      diffuse="{base_d * 0.95:.3f} {base_d * 0.65:.3f} {base_d * 0.54:.3f}"
      castshadow="false"/>
    <light name="fill_2"
      pos="{l3[0]:.1f} {l3[1]:.1f} {lh:.1f}"
      diffuse="{base_d * 0.88:.3f} {base_d * 0.60:.3f} {base_d * 0.50:.3f}"
      castshadow="false"/>
    <light name="fill_3"
      pos="{sx:.1f} {sy:.1f} {lh:.1f}"
      diffuse="{base_d * 1.10:.3f} {base_d * 0.75:.3f} {base_d * 0.62:.3f}"
      castshadow="false"/>

    <!-- ── Ground plane ─────────────────────────────────────────── -->
        <geom name="ground" type="plane"
            pos="0 0 {FLOOR_Z:.3f}" size="{ROOM_HALF_SIZE * 2:.1f} {ROOM_HALF_SIZE * 2:.1f} 0.1"
      material="cave_floor"
      contype="1" conaffinity="1"
      friction="1.0 0.005 0.0001"/>

    <!-- ── Cave wall sections (collision + solid dark-grey visual) ─── -->
{cave_wall_geom_xml}

    <!-- ── Rock obstacles ───────────────────────────────────────── -->
{rock_xml}

    <!-- ── Survivors ────────────────────────────────────────────── -->
{surv_xml}
{dataset_camera_xml}

    <!-- ── Robot / camera body ──────────────────────────────────── -->
    <!--  Replace with your full humanoid MJCF.                       -->
    <body name="robot" pos="{sx:.3f} {sy:.3f} {sz:.3f}">
      <freejoint name="robot_root"/>
      <geom name="robot_body" type="box"
        size="0.28 0.38 0.14" rgba="0.20 0.50 0.80 1.0"
        mass="5.0" contype="1" conaffinity="1"/>
      <site name="imu_site"   pos="0 0  0.15"  size="0.01"/>
      <site name="lidar_site" pos="0 0.37 0.04" size="0.01"/>
      <light name="robot_headlamp"
        pos="0 0.42 0.58"
        dir="0 1 -0.18"
        diffuse="1.35 1.20 0.92"
        specular="0.10 0.10 0.08"
        directional="true"
        castshadow="false"/>
      <light name="robot_soft_fill"
        pos="0 0.15 0.85"
        diffuse="0.50 0.44 0.34"
        castshadow="false"/>
      <!-- Robot-eye camera — used by dataset renderer -->
      <camera name="robot_cam"
        pos="0 0.35 {MUJOCO_DINO_CAMERA_HEIGHT:.2f}"
        xyaxes="1 0 0 0 0 1"
        fovy="{CAMERA_FOV_DEG}"/>
    </body>

  </worldbody>

  <sensor>
    <accelerometer name="accel"      site="imu_site"/>
    <gyro          name="gyro"       site="imu_site"/>
    <magnetometer  name="mag"        site="imu_site"/>
    <rangefinder   name="range_front" site="lidar_site"/>
    <framepos      name="pos_sensor" objtype="site" objname="imu_site"/>
  </sensor>

  <keyframe>
    <key name="spawn_entry"
         qpos="{sx:.3f} {sy:.3f} {sz:.3f}  1 0 0 0"/>
  </keyframe>

</mujoco>
"""
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"  ✓  MuJoCo XML  →  {xml_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  MODULE 3 — SYNTHETIC DATASET PIPELINE
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class EpisodeLabel:
    """JSON label record for one rendered frame."""
    episode_id:         int
    image_file:         str
    person_present:     bool
    n_survivors:        int
    survivors:          List[dict]    # list of SurvivorMetadata-dicts
    lighting_level:     str           # "dark" | "dim" | "normal" | "bright"
    cave_seed:          int
    entrance_path_coordinates: List[Tuple[float, float, float]]
    robot_position:     Tuple[float, float, float]
    camera_yaw_deg:     float

def sample_difficulty(rng):
    return rng.choice(["easy", "medium", "hard"], p=[0.45, 0.40, 0.15])

def _lighting_level(light_seed: int, difficulty: str = "medium") -> str:
    """Map a light_seed to a categorical label for ML labelling."""
    rng  = np.random.default_rng(light_seed)
    base = float(rng.uniform(0.70, 1.05))
    if difficulty == "hard":
        base *= 0.90
    elif difficulty == "easy":
        base *= 1.10
    if base < 0.65:  return "dark"
    if base < 0.80:  return "dim"
    if base < 1.00:  return "normal"
    return "bright"


def _sample_camera_pose(path_points: list,
                         rng: np.random.Generator
                         ) -> Tuple[np.ndarray, float]:
    """
    Sample a robot camera position: random point along path,
    slightly randomised height and facing direction.

    Returns (position_xyz, yaw_degrees).
    """
    n_pts  = len(path_points)
    margin = max(5, int(n_pts * 0.05))
    idx    = int(rng.integers(margin, n_pts - margin))
    pt     = path_points[idx].copy()
    pt[2]  = FLOOR_Z + float(rng.uniform(0.40, 0.65))   # eye height

    # Forward along path with ±30° random jitter
    nxt   = path_points[min(idx + 5, n_pts - 1)]
    fwd2d = nxt[:2] - pt[:2]
    base_yaw = math.degrees(math.atan2(fwd2d[0], fwd2d[1]))
    yaw   = base_yaw + float(rng.uniform(-30.0, 30.0))
    return pt, yaw


def _sample_camera_pose_room(rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    px = rng.uniform(-ROOM_HALF_SIZE * 0.6, ROOM_HALF_SIZE * 0.6)
    py = rng.uniform(-ROOM_HALF_SIZE * 0.6, ROOM_HALF_SIZE * 0.6)
    pz = FLOOR_Z + float(rng.uniform(0.55, 0.85))
    yaw = float(rng.uniform(0.0, 360.0))
    return np.array([px, py, pz], dtype=float), yaw


def _yaw_towards(src: np.ndarray, dst: np.ndarray) -> float:
    """Yaw convention used by the robot freejoint camera."""
    delta = np.asarray(dst, dtype=float) - np.asarray(src, dtype=float)
    return math.degrees(math.atan2(delta[1], delta[0])) + 180.0


def _path_distance_prefix(path_points: list) -> np.ndarray:
    """Cumulative path distance at each path sample."""
    dists = np.zeros(len(path_points), dtype=float)
    for i in range(1, len(path_points)):
        dists[i] = dists[i - 1] + float(np.linalg.norm(path_points[i] - path_points[i - 1]))
    return dists


def _path_index_at_distance(prefix: np.ndarray, target_distance: float) -> int:
    """Closest path index for a cumulative distance."""
    if len(prefix) == 0:
        return 0
    idx = int(np.searchsorted(prefix, target_distance, side="left"))
    return int(np.clip(idx, 0, len(prefix) - 1))


def _sample_camera_pose_for_survivors(path_points: list,
                                      survivors: List[Survivor],
                                      rng: np.random.Generator,
                                      seed_offset: int = 0
                                      ) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Place the robot camera inside the cave and aim it at a survivor.

    This produces dataset frames where the cave interior and humanoid targets
    are visible, instead of random views that may miss the survivor entirely.
    """
    if not survivors:
        cam_pos, cam_yaw = _sample_camera_pose(path_points, rng)
        yaw_rad = math.radians(cam_yaw - 180.0)
        target = cam_pos + np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0]) * 4.0
        target[2] = FLOOR_Z + 0.55
        return cam_pos, cam_yaw, target

    target = np.array(survivors[int(rng.integers(0, len(survivors)))].meta.survivor_position)
    target_idx = _nearest_path_index_2d(path_points, target[:2])
    view_distance = float(rng.uniform(SURVIVOR_VIEW_DISTANCE_MIN, SURVIVOR_VIEW_DISTANCE_MAX))

    fwd_dir, right_dir, _ = frame_at_path_index(path_points, target_idx)
    direction = -1.0 if rng.random() < 0.75 else 1.0
    cam_pos = path_points[target_idx].copy()
    cam_pos += fwd_dir * direction * view_distance

    target_lateral = float(np.dot(target - path_points[target_idx], right_dir))
    half_width, _ = cave_slice_limits(path_points, target_idx, seed_offset)
    camera_lateral = float(np.clip(
        target_lateral * 0.65 + rng.uniform(-0.08, 0.08),
        -max(0.0, half_width - 0.55),
        max(0.0, half_width - 0.55),
    ))
    cam_pos += right_dir * camera_lateral
    cam_pos[2] = FLOOR_Z + float(rng.uniform(0.55, 0.75))

    aim_point = target.copy()
    aim_point[2] = FLOOR_Z + 0.45
    yaw = _yaw_towards(cam_pos, aim_point)
    yaw += float(rng.uniform(-SURVIVOR_VIEW_YAW_JITTER_DEG, SURVIVOR_VIEW_YAW_JITTER_DEG))
    return cam_pos, yaw, aim_point


def _sample_camera_pose_for_survivors_room(survivors: List[Survivor],
                                           rng: np.random.Generator) -> Tuple[np.ndarray, float, np.ndarray]:
    if not survivors:
        cam_pos, cam_yaw = _sample_camera_pose_room(rng)
        yaw_rad = math.radians(cam_yaw - 180.0)
        target = cam_pos + np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0]) * 4.0
        target[2] = FLOOR_Z + 0.55
        return cam_pos, cam_yaw, target

    target = np.array(survivors[int(rng.integers(0, len(survivors)))].meta.survivor_position)
    cam_pos, _ = _sample_camera_pose_room(rng)
    cam_pos[2] = FLOOR_Z + float(rng.uniform(0.55, 0.85))
    aim_point = target.copy()
    aim_point[2] = FLOOR_Z + 0.45
    yaw = _yaw_towards(cam_pos, aim_point)
    yaw += float(rng.uniform(-SURVIVOR_VIEW_YAW_JITTER_DEG, SURVIVOR_VIEW_YAW_JITTER_DEG))
    return cam_pos, yaw, aim_point


def _estimate_survivor_bbox(pos: Tuple[float, float, float],
                            pose: str,
                            cam_pos: np.ndarray,
                            cam_yaw: float) -> Optional[List[int]]:
    """
    Lightweight top-down projection for labels.

    It is intentionally conservative: it returns None for survivors behind the
    camera or outside the horizontal FOV, instead of producing misleading boxes.
    """
    target = np.array(pos, dtype=float)
    rel = target - cam_pos
    yaw_rad = math.radians(cam_yaw - 180.0)
    forward_axis = np.array([math.cos(yaw_rad), math.sin(yaw_rad)])
    right_axis = np.array([-math.sin(yaw_rad), math.cos(yaw_rad)])

    lateral = float(np.dot(rel[:2], right_axis))
    depth = float(np.dot(rel[:2], forward_axis))
    if depth <= 0.35:
        return None

    half_fov = math.radians(CAMERA_FOV_DEG * 0.5)
    angle = math.atan2(lateral, depth)
    if abs(angle) > half_fov * 1.10:
        return None

    x = int((0.5 + angle / (2.0 * half_fov)) * RENDER_WIDTH)
    vertical_center = {
        "lying": 0.70,
        "sitting": 0.58,
        "slumped": 0.63,
    }.get(pose, 0.62)
    y = int(RENDER_HEIGHT * vertical_center)

    scale = max(24, int(180.0 / max(depth, 1.0)))
    pose_width = {"lying": 1.8, "sitting": 0.9, "slumped": 1.2}.get(pose, 1.0)
    pose_height = {"lying": 0.7, "sitting": 1.5, "slumped": 1.1}.get(pose, 1.0)
    half_w = int(scale * pose_width * 0.5)
    half_h = int(scale * pose_height * 0.5)

    x0 = max(0, x - half_w)
    y0 = max(0, y - half_h)
    x1 = min(RENDER_WIDTH - 1, x + half_w)
    y1 = min(RENDER_HEIGHT - 1, y + half_h)
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def run_dataset_pipeline(n_episodes: int = N_EPISODES,
                          seed_start:  int = RENDER_SEED_START) -> None:
    """
    Main dataset generation loop.

    For each episode:
      1. Use a fixed cave mesh for geometric consistency
      2. Randomise rocks, survivors, lighting, camera pose, difficulty
      3. Build + write MuJoCo XML for this episode
      4. Load with mujoco.MjModel, render via mujoco.Renderer
      5. Save RGB PNG + JSON label
    """
    try:
        import mujoco                    # type: ignore
    except ImportError:
        print("  ✗  mujoco not installed.  Run: pip install mujoco")
        return

    # Pre-build the base mesh (expensive — done once) ──────────────────────
    print("\n[Dataset]  Building base cave path & meshes (once)…")
    base_seed = seed_start  # Use seed_start for consistent base mesh
    waypoints = generate_maze_waypoints(base_seed)
    base_path_points = shorten_path(build_smooth_path(waypoints, step=SWEEP_STEP))
    base_frames = compute_frames(base_path_points)

    inner_stl = f"cave_inner_seed_{base_seed}.stl"
    outer_stl = f"cave_outer_seed_{base_seed}.stl"
    inner_path = os.path.join(MESH_DIR, inner_stl)
    outer_path = os.path.join(MESH_DIR, outer_stl)

    if not (os.path.exists(inner_path) and os.path.exists(outer_path)):
        print("         Generating cave STL meshes…")
        if ROOM_MODE:
            inner_tris = generate_room_mesh(ROOM_HALF_SIZE, ROOM_HEIGHT)
            outer_tris = None
        else:
            inner_tris = generate_cave_mesh(
                base_path_points, base_frames,
                extra_offset=0.0,
                seed_offset=base_seed,
                inward_normals=True,
                open_entrance=False,
                open_exit=False
            )

            outer_tris = generate_cave_mesh(
                base_path_points, base_frames,
                extra_offset=WALL_THICKNESS,
                seed_offset=base_seed + 999,
                inward_normals=False,
                open_entrance=False,
                open_exit=False
            )

        write_binary_stl(inner_tris, inner_path)
        if not ROOM_MODE and outer_tris is not None:
            write_binary_stl(outer_tris, outer_path)
    else:
        print("         Reusing existing STL meshes.")

    # ── Episode loop ───────────────────────────────────────────────────────
    xml_path = os.path.join(OUTPUT_DIR, "cave_episode.xml")
    t0       = time.time()
    n_ok     = 0

    print(f"\n[Dataset]  Rendering {n_episodes} episodes…")
    print(f"           Output  →  {IMAGES_DIR}/  +  {LABELS_DIR}/\n")

    for ep in range(n_episodes):
        ep_seed    = seed_start + ep
        rng        = np.random.default_rng(ep_seed)
        difficulty = str(sample_difficulty(rng))
        path_points = base_path_points
        light_seed = int(rng.integers(0, 99999))

        # ── Randomise rocks ────────────────────────────────────────────────
        n_rocks = 0
        rocks = generate_rocks_room(rng, n_rocks) if ROOM_MODE else generate_rocks(path_points, rng, n_rocks=n_rocks, seed_offset=base_seed)
        # ── Initial camera pose for survivor placement heuristics ──────────
        cam_pos, cam_yaw = _sample_camera_pose_room(rng) if ROOM_MODE else _sample_camera_pose(path_points, rng)
        yaw_rad = math.radians(cam_yaw - 180.0)
        cam_target = cam_pos + np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0]) * 4.0
        cam_target[2] = FLOOR_Z + 0.55

        # ── Randomise survivors ────────────────────────────────────────────
        person_present = bool(rng.random() < PERSON_PRESENT_RATE)
        n_surv   = int(rng.integers(MIN_SURVIVORS, MAX_SURVIVORS + 1)) if person_present else 0
        if ROOM_MODE:
            survivors = place_survivors_room(rng, rocks, n_surv)
        else:
            survivors = place_survivors(path_points, rocks, rng, n_surv,
                                        difficulty=difficulty, camera_pos=cam_pos,
                                        seed_offset=base_seed)
        n_surv = len(survivors)
        person_present = n_surv > 0
        if person_present:
            if ROOM_MODE:
                cam_pos, cam_yaw, cam_target = _sample_camera_pose_for_survivors_room(
                    survivors, rng
                )
            else:
                cam_pos, cam_yaw, cam_target = _sample_camera_pose_for_survivors(
                    path_points, survivors, rng, seed_offset=base_seed
                )

        # ── Compute visibility (Point C) — refine based on actual camera pose ────
        for surv in survivors:
            root = np.array(surv.meta.survivor_position)

            # Recompute occlusion from actual camera position
            occ = compute_visibility(cam_pos, root, rocks)
            pose_occ = {
                "lying": 0.10,
                "sitting": 0.05,
                "slumped": 0.15
            }[surv.meta.pose]

            occ_multiplier = {"easy": 0.70, "medium": 1.00, "hard": 1.35}[difficulty]
            occ_pct = float(min(100.0, (occ * occ_multiplier + pose_occ) * 100.0))
            surv.meta.occlusion_percentage = round(occ_pct, 1)
            surv.meta.visibility_level = _visibility_from_occlusion(occ_pct, difficulty)
        # ── Write episode XML ──────────────────────────────────────────────
        write_mujoco_xml(
            inner_stl, outer_stl, xml_path,
            path_points,
            rocks=rocks,
            survivors=survivors,
            light_seed=light_seed,
            difficulty=difficulty,
            dataset_camera_pos=cam_pos,
            dataset_camera_target=cam_target
        )

        # ── Load model & inject camera pose ───────────────────────────────
        try:
            model = mujoco.MjModel.from_xml_path(xml_path)
            data  = mujoco.MjData(model)

            # Move robot body to camera position
            robot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
            if robot_id >= 0:
                # qpos for freejoint: [x, y, z, qw, qx, qy, qz]
                cy = math.cos(math.radians(cam_yaw) * 0.5)
                sy_q = math.sin(math.radians(cam_yaw) * 0.5)
                # Find freejoint qpos start
                jnt_adr  = model.body_jntadr[robot_id]
                qpos_adr = model.jnt_qposadr[jnt_adr]
                data.qpos[qpos_adr    ] = float(cam_pos[0])
                data.qpos[qpos_adr + 1] = float(cam_pos[1])
                data.qpos[qpos_adr + 2] = float(cam_pos[2])
                data.qpos[qpos_adr + 3] = cy    # qw
                data.qpos[qpos_adr + 4] = 0.0   # qx
                data.qpos[qpos_adr + 5] = 0.0   # qy
                data.qpos[qpos_adr + 6] = sy_q  # qz
                mujoco.mj_forward(model, data)

            # ── Render ─────────────────────────────────────────────────────
            renderer = mujoco.Renderer(model, height=RENDER_HEIGHT, width=RENDER_WIDTH)
            renderer.update_scene(data, camera="dataset_cam")
            rgb = renderer.render()   # shape (H, W, 3) uint8
            renderer.close()

        except Exception as exc:
            print(f"  ✗  Episode {ep:04d}: MuJoCo error — {exc}")
            continue

        # ── Save image ─────────────────────────────────────────────────────
        img_fname   = f"ep_{ep:06d}.png"
        img_path    = os.path.join(IMAGES_DIR, img_fname)
        _save_png(rgb, img_path)

        # ── Build survivor labels with bbox (Point D) ──────
        survivor_labels = []

        for surv in survivors:
            bbox = _estimate_survivor_bbox(
                surv.meta.survivor_position,
                surv.meta.pose,
                cam_pos,
                cam_yaw
            )
            if bbox is None:
                continue

            sdict = asdict(surv.meta)
            sdict["bbox"] = bbox

            survivor_labels.append(sdict)

        n_surv = len(survivor_labels)
        person_present = n_surv > 0

        # ── Save label ─────────────────────────────────────────────────────
        label = EpisodeLabel(
            episode_id       = ep,
            image_file       = img_fname,
            person_present   = person_present,
            n_survivors      = n_surv,
            survivors        = survivor_labels,
            lighting_level   = _lighting_level(light_seed, difficulty),
            cave_seed        = base_seed,
            entrance_path_coordinates = get_entrance_path_coordinates(path_points),
            robot_position   = (float(cam_pos[0]),
                                float(cam_pos[1]),
                                float(cam_pos[2])),
            camera_yaw_deg   = round(cam_yaw, 2),
        )
        lbl_path = os.path.join(LABELS_DIR, f"ep_{ep:06d}.json")
        with open(lbl_path, "w") as f:
            json.dump(asdict(label), f, indent=2)

        n_ok += 1
        if ep % 50 == 0 or ep == n_episodes - 1:
            elapsed = time.time() - t0
            rate    = n_ok / max(elapsed, 1e-6)
            eta     = (n_episodes - ep - 1) / max(rate, 1e-6)
            print(f"  [{ep+1:>5}/{n_episodes}]  "
                  f"{rate:.1f} ep/s  |  ETA {eta/60:.1f} min  |  "
                  f"saved {img_path}")

    elapsed = time.time() - t0
    print(f"\n  ✓  Dataset complete — {n_ok}/{n_episodes} episodes  "
          f"in {elapsed:.1f} s  ({n_ok/elapsed:.1f} ep/s)")


# ── Minimal PNG writer (no Pillow required) ──────────────────────────────────
import zlib

def _save_png(rgb_array: np.ndarray, filepath: str) -> None:
    """
    Save an (H, W, 3) uint8 numpy array as a valid PNG file.
    Pure Python implementation — no Pillow / imageio dependency.
    """
    h, w = rgb_array.shape[:2]

    def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
        length  = struct.pack(">I", len(data))
        crc     = struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        return length + chunk_type + data + crc

    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    ihdr      = png_chunk(b"IHDR", ihdr_data)

    # IDAT — filter byte 0 (none) before each row
    raw_rows = b""
    for row in rgb_array:
        raw_rows += b"\x00" + row.astype(np.uint8).tobytes()
    idat = png_chunk(b"IDAT", zlib.compress(raw_rows, 6))

    # IEND
    iend = png_chunk(b"IEND", b"")

    png_sig = b"\x89PNG\r\n\x1a\n"
    with open(filepath, "wb") as f:
        f.write(png_sig + ihdr + idat + iend)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║    MuJoCo Cave Search-and-Rescue Simulation System                  ║
║    Module 1: Cave  |  Module 2: Survivors  |  Module 3: Dataset     ║
╚══════════════════════════════════════════════════════════════════════╝""")

    # ── Directories ───────────────────────────────────────────────────────
    print("\n[1/7]  Creating output directories…")
    for d in [OUTPUT_DIR, MESH_DIR, IMAGES_DIR, LABELS_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"       {d}/")

    # ── Path + frames ─────────────────────────────────────────────────────
    print("\n[2/7]  Building Catmull-Rom cave path…")
    seed = 0  # fixed seed for preview
    waypoints = generate_maze_waypoints(seed)
    path_points = shorten_path(build_smooth_path(waypoints, SWEEP_STEP))
    cave_paths = [CavePathSpec(name="main", path_points=path_points, seed_offset=0)]
    total_len = path_length(path_points)
    entrance_xyz = get_entrance_coordinate(path_points)
    print(f"       {len(waypoints)} waypoints → "
          f"{len(path_points)} samples  (≈ {total_len:.1f} m)")
    print(f"       entrance path starts at "
          f"({entrance_xyz[0]:.2f}, {entrance_xyz[1]:.2f}, {entrance_xyz[2]:.2f})")

    print("\n[3/7]  Computing Frenet frames…")
    frames = compute_frames(path_points)
    print(f"       {len(frames)} frames")

    # ── Cave meshes ─────────────────────── ────────────────────────────────
    print("\n[4/7]  Generating low-poly cave wall layers…")
    for pattern in (
        "cave_wall_*.stl",
        "cave_inner_main_*.stl",
        "cave_outer_main_*.stl",
        "cave_inner_branch_*.stl",
        "cave_outer_branch_*.stl",
    ):
        for stale in Path(MESH_DIR).glob(pattern):
            stale.unlink()
    wall_meshes = generate_cave_wall_layers(cave_paths)
    for mesh in wall_meshes:
        write_binary_stl(mesh.triangles, os.path.join(MESH_DIR, mesh.filename))
    total_tris = sum(len(mesh.triangles) for mesh in wall_meshes)
    inner_tris = sum(len(mesh.triangles) for mesh in wall_meshes if mesh.name.startswith("cave_inner_"))
    outer_tris = total_tris - inner_tris
    print(f"       {len(wall_meshes)} wall sections → {total_tris} triangles "
          f"({inner_tris} inner, {outer_tris} outer)")

    # ── Rock obstacles (demo scene) ───────────────────────────────────────
    print("\n[5/7]  Generating rock obstacles…")
    demo_rng = np.random.default_rng(NOISE_SEED)
    if ROOM_MODE:
        rocks = generate_rocks_room(demo_rng, N_ROCKS)
    else:
        rocks = generate_rocks(path_points, demo_rng, N_ROCKS)
    print(f"       {len(rocks)} rock geoms placed along path")

    # ── Survivors (demo scene) ────────────────────────────────────────────
    print("\n[6/7]  Placing demo survivors…")
    robot_spawn = robot_spawn_from_path(path_points)
    survivors = place_survivors(path_points, rocks, demo_rng, 3,
                                difficulty="medium",
                                camera_pos=np.asarray(robot_spawn, dtype=float))
    for s in survivors:
        m = s.meta
        print(f"       Survivor {m.survivor_id}: pose={m.pose:8s}  "
              f"vis={m.visibility_level:6s}  occ={m.occlusion_percentage:.0f}%  "
              f"pos=({m.survivor_position[0]:.1f}, "
              f"{m.survivor_position[1]:.1f}, "
              f"{m.survivor_position[2]:.1f})")

    # ── Write base scene XML ──────────────────────────────────────────────
    print("\n[7/7]  Writing MuJoCo scene XML…")
    xml_path = os.path.join(OUTPUT_DIR, "cave_maze.xml")
    write_mujoco_xml(wall_meshes, None, xml_path,
                     path_points, rocks=rocks, survivors=survivors,
                     light_seed=NOISE_SEED)

    # ── Optional MuJoCo validation ────────────────────────────────────────
    print("\n       Validating with MuJoCo…")
    try:
        import mujoco                                    # type: ignore
        model = mujoco.MjModel.from_xml_path(xml_path)
        print(f"  ✓  MuJoCo OK — "
              f"{model.ngeom} geoms, {model.nmesh} meshes, "
              f"{model.nbody} bodies")
    except ImportError:
        print("       (mujoco not installed — skipping.  pip install mujoco)")
    except Exception as exc:
        print(f"  ✗  Validation error: {exc}")

    # ── Dataset generation ────────────────────────────────────────────────
    generate_dataset = os.environ.get("CAVE_GENERATE_DATASET", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if generate_dataset:
        print(f"\n{'─'*70}")
        print(f"  MODULE 3 — Synthetic Dataset Pipeline")
        print(f"  Target: {N_EPISODES} episodes  →  {IMAGES_DIR}/  +  {LABELS_DIR}/")
        print(f"{'─'*70}")
        run_dataset_pipeline(n_episodes=N_EPISODES, seed_start=RENDER_SEED_START)
    else:
        print("\n[Dataset]  Skipped. Set CAVE_GENERATE_DATASET=1 to render episodes.")

    # ── Summary ───────────────────────────────────────────────────────────
    mesh_summary = f"{len(wall_meshes)} wall STL sections, {total_tris} triangles"
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                           DONE  ✓                                   ║
╠══════════════════════════════════════════════════════════════════════╣
║  Static scene files:                                                ║
║    {xml_path:<68}║
║    {MESH_DIR:<68}║
║    {mesh_summary:<68}║
╠══════════════════════════════════════════════════════════════════════╣
║  Dataset:                                                           ║
║    {'disabled by default':<68}║
╠══════════════════════════════════════════════════════════════════════╣
║  View scene:                                                        ║
║    python3 -m mujoco.viewer {xml_path:<43}║
╚══════════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
