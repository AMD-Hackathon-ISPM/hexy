Hexy backend (FastAPI) bridging the frontend to the MuJoCo simulation.

## Local run

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set the MuJoCo model location:

```bash
set RL_ROOT=D:\Research\VendorAgnosticRL
```

4. Start the API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Docker

1. Copy .env.example to .env and set the host paths for VendorAgnosticRL and cave-gen.
2. Build and run:

```bash
docker compose up --build
```

The API will be available at http://localhost:8000.

### GPU autodetect (CUDA/ROCm)

The backend container installs `torch`/`torchvision` at startup based on detected devices:

- ROCm: `/dev/kfd` or `/opt/rocm` present
- CUDA: `/dev/nvidia0` or `/proc/driver/nvidia/version` present
- CPU: fallback

You can override with:

```
TORCH_ACCELERATOR=cuda|rocm|cpu|auto
```

Or skip the install entirely:

```
SKIP_TORCH_INSTALL=1
```

## API overview

- GET /health
- GET /mujoco/state
- POST /mujoco/reset
- POST /mujoco/step
  - body: { ctrl?: number[]; n_steps?: number; key?: 'w'|'a'|'s'|'d' }
- WS /mujoco/stream?interval_ms=50
- WS /mujoco/detections?interval_ms=100&width=640&height=480&prompt=person&camera=
	- Default camera: GDINO_CAMERA_DEFAULT (robot_pov)
- WS /audio/whisper?interval_sec=3
	- Emits transcript chunks and audio alert events
- Static assets:
	- /assets/rl/models/hexapod_static.xml
	- /assets/rl/STLFILES/*
	- /assets/cave/cave_hexapod.xml
	- /assets/cave/meshes/*
