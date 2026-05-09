#!/usr/bin/env sh
set -eu

install_torch() {
  accel="${TORCH_ACCELERATOR:-}"
  torch_version="${TORCH_VERSION:-2.3.1}"
  torchvision_version="${TORCHVISION_VERSION:-0.18.1}"
  if [ -z "$accel" ] || [ "$accel" = "auto" ]; then
    if [ -e /dev/kfd ] || [ -d /opt/rocm ]; then
      accel="rocm"
    elif command -v nvidia-smi >/dev/null 2>&1; then
      accel="cuda"
    elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${CUDA_VISIBLE_DEVICES}" != "-1" ]; then
      accel="cuda"
    elif [ -n "${NVIDIA_VISIBLE_DEVICES:-}" ] && [ "${NVIDIA_VISIBLE_DEVICES}" != "void" ] && [ "${NVIDIA_VISIBLE_DEVICES}" != "none" ]; then
      accel="cuda"
    elif [ -e /dev/nvidia0 ] || [ -e /proc/driver/nvidia/version ]; then
      accel="cuda"
    else
      accel="cpu"
    fi
  fi

  echo "[torch] accelerator=${accel}"

  case "$accel" in
    rocm)
      python -m pip install --no-cache-dir \
        "torch==${torch_version}" \
        "torchvision==${torchvision_version}" \
        --index-url https://download.pytorch.org/whl/rocm6.1
      ;;
    cuda)
      python -m pip install --no-cache-dir \
        "torch==${torch_version}" \
        "torchvision==${torchvision_version}" \
        --index-url https://download.pytorch.org/whl/cu121
      ;;
    cpu)
      python -m pip install --no-cache-dir \
        "torch==${torch_version}" \
        "torchvision==${torchvision_version}"
      ;;
    *)
      echo "[torch] Unknown TORCH_ACCELERATOR=${accel}. Use cuda|rocm|cpu|auto."
      exit 2
      ;;
  esac
}

if [ "${SKIP_TORCH_INSTALL:-0}" != "1" ]; then
  if python - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("torch") else 1)
PY
  then
    echo "[torch] Already installed."
  else
    install_torch
  fi
else
  echo "[torch] Install skipped (SKIP_TORCH_INSTALL=1)."
fi

# ── Dynamic GGUF selection based on free VRAM ──────────────────────
QWEN_MODEL_DIR="${QWEN25VL_LOCAL_DIR:-/modelSetUp/qwen25vl}"

if command -v nvidia-smi >/dev/null 2>&1; then
  FREE_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo "0")
  echo "[qwen] Detected free VRAM: ${FREE_VRAM}MB"

  if [ "$FREE_VRAM" -lt 3000 ] 2>/dev/null; then
    echo "[qwen] Low VRAM (<3GB). Loading IQ4_XS for maximum stability..."
    GGUF_PATTERN="IQ4_XS"
  else
    echo "[qwen] Sufficient VRAM. Loading Q6_K for maximum fidelity..."
    GGUF_PATTERN="Q6_K"
  fi

  # Find matching GGUF file
  GGUF_FILE=""
  for f in "${QWEN_MODEL_DIR}"/*.gguf; do
    case "$f" in
      *${GGUF_PATTERN}*) GGUF_FILE="$f"; break ;;
    esac
  done

  # Fallback: pick any .gguf
  if [ -z "$GGUF_FILE" ]; then
    for f in "${QWEN_MODEL_DIR}"/*.gguf; do
      if [ -f "$f" ]; then
        GGUF_FILE="$f"
        break
      fi
    done
  fi

  if [ -n "$GGUF_FILE" ]; then
    export QWEN_GGUF_PATH="$GGUF_FILE"
    echo "[qwen] Selected: ${GGUF_FILE}"
  else
    echo "[qwen] WARNING: No .gguf files found in ${QWEN_MODEL_DIR}"
  fi
else
  echo "[qwen] nvidia-smi not available, GGUF selection deferred to Python"
fi

exec "$@"
