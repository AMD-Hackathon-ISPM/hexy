# Hexy

Monorepo for the Hexy hexapod project: frontend, backend, and learning/MuJoCo simulation in one tree.

```
.
├── frontend/   React + Vite SPA (Three.js + mujoco-react)
├── backend/    FastAPI with in-process MuJoCo simulation
└── learning/   MuJoCo + Brax/JAX simulation scripts and robot model
```

## Prerequisites

- Docker (with Compose v2)
- For GPU training: NVIDIA Container Toolkit

## Stack up (default)

```bash
cp .env.example .env          # only if you want to override the defaults
docker compose up --build
```

Open http://localhost:8080. The frontend proxies REST and WebSocket traffic
(`/health`, `/mujoco/*`, `/assets/*`) to the backend over the compose network,
so there is no CORS configuration to manage.

The backend is also reachable directly at http://localhost:8000 for manual API
checks during development.

### What scene loads?

The backend auto-detects `cave-gen/cave_env/cave_hexapod.xml` at startup:

- **File present** → cave scene loads (~10–15 s compile in the browser).
- **File missing** → falls back to the static hexapod scene (instant).

So if cave assets are already on disk (the usual case after a one-time
generation), `docker compose up --build` is all you need — **no regeneration
required**. The 30–60 min `cave-gen` job below is only for the very first
generation or when you want a brand-new procedural cave.

## Optional: training

The simulation/training service does not start by default — it is gated behind a
compose profile so the default `up` is fast and works on any machine.

```bash
# CPU (works anywhere)
docker compose --profile training run --rm learning python scripts/test_mujoco_control.py

# GPU (NVIDIA Container Toolkit required)
docker compose --profile training-gpu run --rm learning-gpu python scripts/test_mujoco_control.py
```

## Optional: cave environment

`cave-gen` procedurally generates a cave scene with the real hexapod spliced
in. The 1000-episode synthetic survivor dataset is opt-in via
`CAVE_GENERATE_DATASET=1`. It runs as a one-shot job under a compose profile, so
the default `up` is unaffected.

There are three speeds, pick the slowest one you actually need:

```bash
# 1. FASTEST (~seconds): use whatever cave assets are already on disk.
#    No cave-gen run at all — the backend reads cave-gen/cave_env/cave_hexapod.xml
#    directly. This is the normal day-to-day workflow.
docker compose up --build

# 2. FAST RE-MERGE (~seconds): re-stitch the hexapod into the existing cave
#    after editing cave_maze.xml or the hexapod model. Skips the slow mesh gen.
docker compose --profile data-gen run --rm --entrypoint python cave-gen build_cave_hexapod.py
docker compose restart backend

# 3. FULL REGEN (~seconds for the live scene): rebuild the lightweight cave
#    meshes from scratch. Only needed for the first ever generation or when you
#    want a brand-new procedural layout.
docker compose --profile data-gen run --rm cave-gen
docker compose restart backend

# Optional dataset render (~30-60 min on CPU):
CAVE_GENERATE_DATASET=1 docker compose --profile data-gen run --rm cave-gen
```

To revert to the static hexapod scene:

```bash
rm cave-gen/cave_env/cave_hexapod.xml
docker compose restart backend
```

The generated `cave_hexapod.xml` is intended for the Docker-backed app. For
local host viewing with `mujoco.viewer`, temporarily symlink `learning/STLFILES`
to `rl/STLFILES` so the browser-friendly `../rl/STLFILES/*` mesh paths resolve.

## Local development (without Docker)

Run the backend and frontend directly against your checked-out tree.

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload   # serves on http://localhost:8000

# Frontend (in another terminal)
cd frontend
npm install
npm run dev                     # serves on http://localhost:5173
```

The frontend's Vite dev server proxies `/health`, `/mujoco`, and `/assets/*` to
`localhost:8000`, so no `.env.local` is required for local development.
