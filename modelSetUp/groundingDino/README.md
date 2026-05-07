# Grounding DINO (ROCm/CUDA-ready)

This folder contains a low-latency Grounding DINO inference helper that can run on ROCm, CUDA, or CPU.

## 1) Install PyTorch (GPU)

ROCm (Linux only):
```
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1
```

CUDA:
```
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

CPU fallback:
```
pip install --upgrade pip
pip install torch torchvision
```

## 2) Install Grounding DINO helpers

```
pip install -r requirements.txt
```

## 3) Run a quick demo

```
python inference.py
```

The demo runs the test prompts:
- person
- trapped injured person
- person lying in cave

## 4) Environment variables

- GDINO_DEVICE: auto | cuda | cpu
- GDINO_CONFIG_PATH: path to config file
- GDINO_WEIGHTS_PATH: path to model weights
- GDINO_BOX_THRESHOLD: default 0.35
- GDINO_TEXT_THRESHOLD: default 0.25

## 5) MuJoCo frames

In your MuJoCo pipeline, pass the RGB frame (H,W,3 uint8) to:
```
from inference import GroundingDinoInferencer

inferencer = GroundingDinoInferencer()
frames = simulator.render_rgb(width=640, height=480)
results = inferencer.infer(frames, "person")
```

The helper returns JSON-friendly metadata:
```
{
  "detections": [
    {"label": "person", "confidence": 0.91, "bbox": [x1, y1, x2, y2]}
  ]
}
```
