#!/usr/bin/env sh
set -eu

install_torch() {
  accel="${TORCH_ACCELERATOR:-}"
  if [ -z "$accel" ] || [ "$accel" = "auto" ]; then
    if [ -e /dev/kfd ] || [ -d /opt/rocm ]; then
      accel="rocm"
    elif [ -e /dev/nvidia0 ] || [ -e /proc/driver/nvidia/version ]; then
      accel="cuda"
    else
      accel="cpu"
    fi
  fi

  echo "[torch] accelerator=${accel}"

  case "$accel" in
    rocm)
      python -m pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/rocm6.1
      ;;
    cuda)
      python -m pip install --no-cache-dir torch torchvision \
        --index-url https://download.pytorch.org/whl/cu121
      ;;
    cpu)
      python -m pip install --no-cache-dir torch torchvision
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

exec "$@"
