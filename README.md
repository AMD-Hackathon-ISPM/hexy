# HexySAR: Autonomous Hexapod for Cave Search and Rescue

**Send the robot first. Reduce rescuer risk. Find survivors faster.**

HexySAR is an AI-powered hexapod robot system for cave search-and-rescue scenarios. It combines MuJoCo simulation, survivor detection, spatial audio, multimodal reasoning, and a web control interface so an operator can send high-level instructions while the robot explores hazardous cave terrain.

The project is built for local execution. The Hugging Face Space is a landing page only, not a hosted live demo, because the full system needs local model weights and several GB of RAM/VRAM for MuJoCo, Grounding DINO, Whisper, and Qwen inference.

## Links

| Resource | Link |
|----------|------|
| Video presentation | [Watch on Google Drive](https://drive.google.com/file/d/1gs_yOKZyv6_t3wxFjLplpX3eHqoROJ5X/view?usp=sharing) |
| Landing page | [Hugging Face Space](https://huggingface.co/spaces/pBillyy/Hexy) |
| Fine-tuned DINO model | [LeBabyOx/dino-cave-survivor](https://huggingface.co/LeBabyOx/dino-cave-survivor) |
| Model weights bundle | [Download from Google Drive](https://drive.google.com/file/d/1AOns_bZGLbrnlYZFbRv-Sieed4Aipxe6/view?usp=sharing) |
| Presentation slides | [View PDF](https://storage.googleapis.com/lablab-static-eu/presentations/submissions/oq9dyyryejwqmb9ehco67emb/oq9dyyryejwqmb9ehco67emb-1778432969348_adsnxslx0ullw5jjrnl2ieqs.pdf) |

## What It Does

HexySAR sends an autonomous robot into unsafe cave environments before human rescuers enter.

| Step | Behavior |
|------|----------|
| Command | Operator gives plain-English instructions such as "search the left corridor" |
| Explore | Robot navigates the cave simulation using hexapod locomotion |
| Detect | Vision AI identifies possible survivors and audio AI listens for distress calls |
| Reason | Qwen combines detections, audio context, and operator intent |
| Report | Robot returns status, survivor cues, and next actions through the control interface |

## System Architecture

![High-level architecture](docs/images/slide_10.png)

| Layer | Role |
|-------|------|
| Frontend | React control center with chat, robot telemetry, camera views, and 3D simulation |
| Backend | FastAPI service coordinating MuJoCo, WebSockets, AI inference, and agent state |
| AI stack | Grounding DINO for vision, faster-whisper for audio, Qwen 2.5-VL for reasoning |
| Simulation | MuJoCo hexapod with inverse kinematics, tripod gait, cameras, and cave assets |

The autonomous loop runs across three main threads:

| Thread | Role | Cycle |
|--------|------|-------|
| Physics | Executes queued locomotion commands through IK gait control | ~52 ms batches |
| Sensors | Polls survivor detections and cached spatial audio context | Every 2 s |
| Brain | Uses Qwen to choose structured navigation actions | ~3 s per decision |

## AI Pipeline

### Vision: Grounding DINO

- Fine-tuned model: [`LeBabyOx/dino-cave-survivor`](https://huggingface.co/LeBabyOx/dino-cave-survivor)
- Training data: synthetic MuJoCo cave scenes with low-light, blur, noise, and fog augmentation
- Detection prompts: `person . human . survivor . human silhouette`
- Reported performance: precision 0.227, recall 0.488 on the cave-survivor task
- Post-processing: area filters, aspect-ratio checks, color gating, luma analysis, and NMS

### Audio: Whisper Spatial STT

- Model: faster-whisper base
- Input: synthetic stereo distress calls with panning and distance attenuation
- Keywords: "help", "save me", "over here"
- Output: transcript plus approximate direction and distance cues

### Reasoning: Qwen 2.5-VL

- Runtime: llama-cpp-python with GGUF models
- Input: JSON containing detections, audio context, and operator instructions
- Output: structured movement commands such as `{"action": "w", "reasoning": "..."}`
- Quantization: selects IQ4_XS, Q4_K_M, or Q6_K based on available memory

### Locomotion

- Six-leg inverse kinematics controller
- Hand-crafted tripod gait with smoothstep swing and stance phases
- MuJoCo physics with heading stabilization and joint-limit enforcement
- Commands: forward, backward, turn left, turn right, stop

## Training and Data

![Procedural cave training environment](docs/images/slide_06.png)

HexySAR uses synthetic cave generation so perception and reasoning can be tested under low-light and noisy conditions.

| Pipeline | Framework | Purpose |
|----------|-----------|---------|
| Locomotion | MuJoCo + Brax/JAX | Train and test hexapod movement |
| Vision | Grounding DINO | Adapt survivor detection to cave-like scenes |
| Reasoning | Qwen 2.5-VL | Convert multimodal context into deterministic action JSON |

The dataset generation path is opt-in. The default Docker startup uses existing assets and does not regenerate the cave dataset.

## Hardware and Runtime Requirements

The full system is intended to run locally because inference and simulation are resource-heavy.

| Component | Approximate memory need |
|-----------|-------------------------|
| Grounding DINO vision | ~1.2 GB VRAM |
| faster-whisper audio | ~0.15 GB VRAM |
| Qwen 2.5 reasoning | ~1.7-2.5 GB VRAM |
| Full AI stack | ~3-4 GB VRAM, plus system RAM for simulation and services |

HexySAR detects available hardware at startup and selects a suitable backend where possible:

| Platform | Detection | Backend |
|----------|-----------|---------|
| AMD GPU | `/dev/kfd`, `/opt/rocm` | ROCm / HIPBLAS |
| NVIDIA GPU | `nvidia-smi`, `/dev/nvidia0` | CUDA |
| Apple Silicon | `arm64` on Darwin | MPS / Metal |
| CPU or SBC | fallback | CPU / BLAS |

## Quick Start

### Prerequisites

- Docker with Compose v2
- For GPU acceleration: NVIDIA Container Toolkit or AMD ROCm
- Model weights bundle from the link above

Before running, extract the model bundle into the repository root so these paths exist:

```text
modelSetUp/groundingDino/weights
modelSetUp/qwen25vl
```

Start the system:

```bash
docker compose up --build
```

Open the frontend:

```text
http://localhost:8080
```

The backend is available at `http://localhost:8000` for direct API checks. No `.env` file is required for the default local setup.

## Cave Assets

At startup, the backend checks for `cave-gen/cave_env/cave_hexapod.xml`.

| State | Result |
|-------|--------|
| File exists | Cave scene loads in the browser |
| File missing | Static hexapod scene loads instead |

To regenerate cave assets:

```bash
docker compose --profile data-gen run --rm cave-gen
docker compose restart backend
```

To generate the optional synthetic dataset:

```bash
CAVE_GENERATE_DATASET=1 docker compose --profile data-gen run --rm cave-gen
```

## Local Development

Backend:

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

The Vite dev server proxies `/health`, `/mujoco`, and `/assets/*` to `localhost:8000`.

## Repository Structure

```text
.
|-- frontend/      React + Vite control interface
|-- backend/       FastAPI, MuJoCo runtime, AI services, and WebSockets
|-- learning/      MuJoCo + Brax/JAX simulation scripts and robot model
|-- cave-gen/      Procedural cave generator and synthetic dataset pipeline
`-- modelSetUp/    Local model weights for DINO, Qwen, and Whisper
```

## Team

Built by Team SINGKONG for the AMD Hackathon.

## License

See [learning/LICENSE](learning/LICENSE) for the hexapod model license.
