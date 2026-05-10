# 🦎 HexySAR — AI-Powered Autonomous Hexapod for Cave Search & Rescue

> **Send AI first. Reduce human risk. Save humans faster.**

HexySAR is an autonomous hexapod robot designed for search-and-rescue missions in dangerous cave environments. Instead of sending human rescuers into unstable, dark, and unknown terrain first, HexySAR deploys an AI-powered robot that can explore, detect survivors, respond to voice cues, and report back — all autonomously.

The system combines **MuJoCo physics simulation**, **multimodal AI** (vision + audio + reasoning), and a **real-time web control interface** — with adaptive model quantization that enables deployment from AMD MI300X workstations all the way down to Orange Pi SBCs.

<p align="center">
  <a href="https://drive.google.com/file/d/1gs_yOKZyv6_t3wxFjLplpX3eHqoROJ5X/view?usp=sharing"><strong>📺 Watch Video Presentation</strong></a> ·
  <a href="https://huggingface.co/LeBabyOx/dino-cave-survivor"><strong>🤗 DINO Model on HuggingFace</strong></a> ·
  <a href="https://huggingface.co/spaces/pBillyy/Hexy"><strong>🚀 Live Demo on HF Spaces</strong></a>
</p>

---

## 🎯 The Problem

Cave rescue operations are among the most dangerous emergency responses:

- **Time-critical** — search operations take hours or days; every minute matters for survivor survival rates
- **Extreme hazards** — oxygen depletion, terrain collapse, and limited visibility in unknown environments
- **Rescuers become victims** — specialized cave skills required; teams face the same dangers as trapped survivors
- **Resource-intensive** — costly operations with limited personnel who can safely enter

Current approaches require humans to enter first. HexySAR changes that equation.

## 💡 Our Solution

HexySAR sends an **autonomous AI-powered hexapod robot** into the cave first:

| Step | What Happens |
|------|-------------|
| **1. Command** | Operator gives high-level instructions via chat ("search the left corridor") |
| **2. Explore** | Robot autonomously navigates unknown cave terrain |
| **3. Detect** | Vision AI spots survivors; Audio AI picks up distress calls |
| **4. Investigate** | Robot moves toward detected survivors, fusing visual + audio cues |
| **5. Identify** | Confirms survivor location and reports back to rescue team |

The operator can observe, override, or instruct the robot through natural language — but the robot makes its own decisions when left alone.

![How HexySAR Operates](docs/images/slide_04.png)

---

## 🏗️ System Architecture

HexySAR's design has three layers working together seamlessly:

| Layer | What It Does |
|-------|-------------|
| **Control Center** (Frontend) | Chat with the robot in plain English, see a live 3D view of the cave, watch the robot's cameras, get real-time updates |
| **AI Brain** (Backend) | Three AI systems: Vision AI (recognizes people), Hearing AI (understands speech/sounds), Decision AI (plans actions and routes) |
| **Hardware** (Simulation) | 6-leg hexapod body with cameras and microphones, running in MuJoCo physics — works on any computer |

![High Level Architecture](docs/images/slide_10.png)

---

## 🤖 AI Pipeline

HexySAR uses a **multimodal sense → think → act** loop running as three concurrent threads:

### 👁️ Vision — Grounding DINO (Fine-tuned)

- **Model**: [`LeBabyOx/dino-cave-survivor`](https://huggingface.co/LeBabyOx/dino-cave-survivor) — lightweight synthetic cave-domain adaptation
- **Training data**: MuJoCo synthetic cave dataset with low-light augmentations, blur/noise/fog randomization
- **Performance**: Precision 0.227, Recall 0.488 — learned to identify survivors in extreme cave conditions where standard models failed
- **Detection prompts**: `person . human . survivor . human silhouette`
- **Post-processing**: Area filters, aspect ratio checks, colour gating (white survivors in dark caves), per-crop luma analysis, NMS
- **Optimized for**: Cave survivor detection, multimodal SAR robotics, AMD ROCm / MI300X

### 🎙️ Audio — Whisper Spatial STT

- **Model**: faster-whisper (base), ~0.15 GB VRAM
- **Pipeline**: Synthetic spatial TTS → stereo panning → distance attenuation → Whisper transcription → direction/distance estimation
- **Keywords**: "help", "save me", "over here"
- **Spatial audio**: Pan-based direction (left/center/right), inverse-distance rolloff for realistic audibility

### 🧠 Reasoning — Qwen 2.5-VL (Quantized)

- **Model**: Qwen 2.5-VL via llama-cpp-python (GGUF format), ~1.7–2.5 GB VRAM
- **Input**: JSON with detections + audio context + operator instructions
- **Output**: Structured `{"action": "w", "reasoning": "..."}` commands
- **Quantization**: Adaptive GGUF selection (IQ4_XS / Q4_K_M / Q6_K) based on available VRAM at startup

### 🦿 Locomotion — 6-Leg Inverse Kinematics

- **Controller**: Hand-crafted tripod gait with smoothstep swing/stance phases
- **Per-leg IK**: Full kinematic chain solving (coxa → femur → tibia) with PD control
- **Physics**: MuJoCo simulation with heading stabilisation and joint limits enforcement
- **Commands**: Forward / backward / turn-left / turn-right / stop

### 🔄 Autonomous Agent — 3-Thread Architecture

| Thread | Role | Cycle |
|--------|------|-------|
| **Physics** (Spinal Cord) | Executes IK gait commands from the action queue | ~52 ms batches |
| **Sensors** (Autonomic) | Polls DINO + cached Whisper audio, auto-pauses when survivor is near | Every 2 s |
| **Brain** (LLM) | Qwen inference on combined sensor context, outputs navigation commands | ~3 s per decision |

---

## 🏋️ Training Strategy with MI300X

Off-the-shelf models were not enough for cave disaster scenarios. HexySAR uses a **tripartite training pipeline** designed to run simultaneously on AMD MI300X + ROCm 7.2:

![Training Strategy with MI300X](docs/images/slide_07.png)

| Pipeline | Framework | Task |
|----------|-----------|------|
| **1. Locomotion Policy** | JAX + Brax | Train hexapod RL locomotion directly inside MuJoCo |
| **2. Vision Detection** | Grounding DINO | Fine-tune on procedural synthetic cave data for survivors + rubble |
| **3. Action Reasoning** | Qwen 2.5-VL | Align using COCO + ExDark + synthetic data to output deterministic Action JSONs |

After training: **Quantize → Compress → Deploy** as an edge-inference container for the robot.

### Synthetic Training Pipeline

- Procedural cave dataset generation with terrain randomization (1000+ episodes)
- Low-light and fog augmentation for extreme conditions
- Survivor detection adaptation from synthetic to real-world scenarios

![Trained in Procedural Cave Environments](docs/images/slide_06.png)

---

## 🔧 AMD GPU & Edge Deployment

### Adaptive Quantization: One Robot, Many Computers

HexySAR automatically adapts its AI models to fit your computer's capabilities. The **same Docker image** runs on all of them — no configuration needed.

![Adaptive Quantization](docs/images/slide_15.png)

**Smart Startup Flow:**

1. **Detect Hardware** — NVIDIA GPU? AMD GPU? Apple Silicon? CPU? (< 1 second)
2. **Measure Available Memory** — `nvidia-smi`, `rocm-smi`, or system commands
3. **Pick the Best AI Model** — automatically selects the optimal quantization level
4. **Start the Robot** — loads and runs, all before you see the interface (10–30 seconds)

| VRAM Available | Quantization | Quality | Use Case |
|---------------|--------------|---------|----------|
| ≥ 6 GB | Q6_K | 99% | Best quality |
| ≥ 3 GB | Q4_K_M | 98% | Good quality |
| < 3 GB | IQ4_XS | 97% | Maximum stability |
| CPU / SBC | IQ4_XS | 97% | Minimum viable |

### Hardware Platform Support

| Platform | Detection | PyTorch Backend | llama-cpp Backend |
|----------|-----------|----------------|-------------------|
| **AMD GPU** | `/dev/kfd` · `/opt/rocm` | rocm6.1 wheel | ROCm (`-DGGML_HIPBLAS=on`) |
| **NVIDIA GPU** | `nvidia-smi` · `/dev/nvidia0` | cu121/cu124 wheel | CUDA (`-DGGML_CUDA=on`) |
| **Apple Silicon** | `uname -m == arm64 + Darwin` | MPS (`torch.backends.mps`) | Metal (`-DGGML_METAL=on`) |
| **CPU / SBC** | fallback (RPi, Jetson, x86) | CPU wheel | CPU (`-DGGML_BLAS=on`) |

### Real-World Performance

![Performance Benchmarks](docs/images/slide_16.png)

| Hardware | GPU | Quantization | Speed | Quality | Use Case |
|----------|-----|-------------|-------|---------|----------|
| RTX 3070 (Gaming PC) | NVIDIA RTX 3070 | Q6_K | 15 tok/s | 99% | — |
| **RX 7900 XTX (Workstation)** | **AMD RX 7900 XTX** | **Q6_K** | **15 tok/s** | **99%** | **High-performance deployments** |
| M3 Max (MacBook Pro) | Apple M3 Max | Q6_K | 18 tok/s | 99% | Portable rescue command center |
| Ryzen 9 7950X (CPU Only) | None | IQ4_XS | 4.2 tok/s | 97% | Budget deployments |
| Jetson Orin NX (Edge) | NVIDIA Jetson | IQ4_XS | 18 tok/s | 97% | Embedded rescue robots |
| Orange Pi 5 Ultra + Coral TPU | NPU + TPU | IQ4_XS | 28 s/response | 97% | Professional rescue operations |

> Even on an Orange Pi, HexySAR works — the robot can still make decisions every 30 seconds, which is fast enough for cave rescue. **One codebase, infinite hardware compatibility.**

### Memory Budget

| AI Component | VRAM Required |
|-------------|--------------|
| Vision AI (GroundingDINO) | ~1.2 GB |
| Hearing AI (Whisper) | ~0.15 GB |
| Thinking AI (Qwen 2.5) | ~1.7–2.5 GB |
| **Total** | **~3–4 GB** |

### ROCm Docker Build

```bash
# Build the backend with the AMD Dockerfile
docker compose build backend --build-arg DOCKERFILE=Dockerfile.amd

# Or set the accelerator explicitly
TORCH_ACCELERATOR=rocm docker compose up --build
```

---

## 🛠️ Tech Stack

### Backend

| Technology | Role |
|-----------|------|
| Grounding DINO | Survivor detection (precision 0.227, recall 0.488) |
| Whisper | Audio understanding for noise detection |
| Qwen 2.5-VL | Multimodal reasoning engine |
| MuJoCo + Brax | Physics simulation for training |
| Docker + Docker Compose | Deployment orchestration |

### Frontend

| Technology | Role |
|-----------|------|
| React + TypeScript + Vite | Chat interface and control panel |
| Tailwind | Utility-first CSS styling |
| Three.js + React Three Fiber | Real-time 3D visualization |
| mujoco-react | Browser-based simulation viewer |

---

## 🚀 Quick Start

### Prerequisites

- Docker (with Compose v2)
- For GPU acceleration: NVIDIA Container Toolkit **or** AMD ROCm
- Required model bundle (before first build):
  1. Download: https://drive.google.com/file/d/1AOns_bZGLbrnlYZFbRv-Sieed4Aipxe6/view?usp=sharing
  2. Extract the ZIP in the repository root (this project folder).
  3. Ensure extraction creates `modelSetUp/` in root, including:
     - `modelSetUp/groundingDino/weights`
     - `modelSetUp/qwen25vl`

### Run

```bash
docker compose up --build
```

> `modelSetUp/` must already exist in the repo root before running the build command.

All environment variables have sensible defaults, so **no `.env` file is required**.
If you need to customise ports or set a Hugging Face token:

```bash
cp .env.example .env   # then edit .env as needed
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
required**.

---

## 🗺️ Procedural Cave Environment

`cave-gen` procedurally generates a cave scene with the real hexapod spliced
in. The 1000-episode synthetic survivor dataset is opt-in via
`CAVE_GENERATE_DATASET=1`. It runs as a one-shot job under a compose profile, so
the default `up` is unaffected.

There are three speeds, pick the slowest one you actually need:

```bash
# 1. FASTEST (~seconds): use whatever cave assets are already on disk.
docker compose up --build

# 2. FAST RE-MERGE (~seconds): re-stitch the hexapod into the existing cave.
docker compose --profile data-gen run --rm --entrypoint python cave-gen build_cave_hexapod.py
docker compose restart backend

# 3. FULL REGEN: rebuild the cave meshes from scratch.
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

---

## 🏋️ Optional: Training

The simulation/training service does not start by default — it is gated behind a
compose profile so the default `up` is fast and works on any machine.

```bash
# CPU (works anywhere)
docker compose --profile training run --rm learning python scripts/test_mujoco_control.py

# GPU (NVIDIA Container Toolkit required)
docker compose --profile training-gpu run --rm learning-gpu python scripts/test_mujoco_control.py
```

---

## 💻 Local Development (without Docker)

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

---

## 📁 Repository Structure

```
.
├── frontend/      React + Vite SPA (Three.js + mujoco-react)
├── backend/       FastAPI with in-process MuJoCo simulation + AI services
├── learning/      MuJoCo + Brax/JAX simulation scripts and robot model
├── cave-gen/      Procedural cave maze generator + synthetic dataset pipeline
└── modelSetUp/    Pre-trained model weights (DINO, Qwen, Whisper)
```

---

## 🔌 API Overview

| Endpoint | Type | Description |
|----------|------|-------------|
| `GET /mujoco/state` | REST | Current simulation state |
| `POST /mujoco/step` | REST | Step simulation with controls or WASD key |
| `POST /mujoco/reset` | REST | Reset simulation |
| `WS /mujoco/stream` | WebSocket | Real-time simulation state stream (50 ms) |
| `WS /mujoco/detections` | WebSocket | Live Grounding DINO survivor detections |
| `WS /audio/whisper` | WebSocket | Spatial audio transcripts + alerts |
| `POST /agent/respond` | REST | Send operator instruction to Qwen agent |
| `GET /agent/state` | REST | Current autonomous agent state + reasoning |
| `GET /agent/autonomous/status` | REST | Agent running/paused status |
| `POST /agent/autonomous/stop` | REST | Stop autonomous agent |

---

## 🔗 Project Resources

| Resource | Link |
|----------|------|
| 📺 Video Presentation | [Watch on Google Drive](https://drive.google.com/file/d/1gs_yOKZyv6_t3wxFjLplpX3eHqoROJ5X/view?usp=sharing) |
| 📊 Presentation Slides | [View PDF](https://storage.googleapis.com/lablab-static-eu/presentations/submissions/oq9dyyryejwqmb9ehco67emb/oq9dyyryejwqmb9ehco67emb-1778432969348_adsnxslx0ullw5jjrnl2ieqs.pdf) |
| 🤗 DINO Model | [LeBabyOx/dino-cave-survivor](https://huggingface.co/LeBabyOx/dino-cave-survivor) |
| 🚀 Live Demo | [pBillyy/Hexy on HF Spaces](https://huggingface.co/spaces/pBillyy/Hexy) |
| 📦 Model Weights Bundle | [Download from Google Drive](https://drive.google.com/file/d/1AOns_bZGLbrnlYZFbRv-Sieed4Aipxe6/view?usp=sharing) |

---

## 👥 Team SINGKONG

Built for the AMD Hackathon.

HexySAR helps rescuers save lives without risking more lives.

---

## 📄 License

See [learning/LICENSE](learning/LICENSE) for the hexapod model license.
