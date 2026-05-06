FROM python:3.11-slim-bookworm

ARG JAX_ACCELERATOR=cpu

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    XLA_PYTHON_CLIENT_PREALLOCATE=false

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    git \
    libegl1 \
    libgl1 \
    libgles2 \
    libglib2.0-0 \
    libosmesa6 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt
RUN if [ "$JAX_ACCELERATOR" = "cpu" ]; then \
      python -m pip install --no-cache-dir --upgrade "jax"; \
    elif [ "$JAX_ACCELERATOR" = "cuda12" ]; then \
      python -m pip install --no-cache-dir --upgrade "jax[cuda12]"; \
    else \
      echo "Unsupported JAX_ACCELERATOR=$JAX_ACCELERATOR" && exit 1; \
    fi

COPY . /workspace

CMD ["bash"]
