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

## Optional: training

The simulation/training service does not start by default — it is gated behind a
compose profile so the default `up` is fast and works on any machine.

```bash
# CPU (works anywhere)
docker compose --profile training run --rm learning python scripts/test_mujoco_control.py

# GPU (NVIDIA Container Toolkit required)
docker compose --profile training-gpu run --rm learning-gpu python scripts/test_mujoco_control.py
```

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
