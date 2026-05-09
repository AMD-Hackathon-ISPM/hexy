FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        build-essential \
        g++-12 \
        libstdc++6 \
        libgl1 \
        libglib2.0-0 \
        libegl1 \
        libgles2 \
        libosmesa6 \
        espeak-ng \
        cmake \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        python-is-python3 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt ./

RUN grep -v '^torch==' requirements.txt > requirements.docker.txt \
    && python -m pip install --no-cache-dir -r requirements.docker.txt

# Upgrade cmake via pip so 'all-major' CUDA arch keyword works (requires 3.23+;
# Ubuntu 22.04 ships only 3.22). This takes ~5s and avoids apt source changes.
RUN pip install --no-cache-dir "cmake>=3.25"

# Build llama-cpp-python from source with CUDA support.
# LLAMA_CPP_CUDA_ARCH=all-major builds for all common GPU families (requires cmake>=3.23).
# Override to a single value like 86 or 89 for a smaller/faster per-machine build.
ARG LLAMA_CPP_VERSION=0.3.22
ARG LLAMA_CPP_CUDA_ARCH=86
# The build container has no GPU driver — only CUDA toolkit stubs.
# Create the .so.1 symlink the linker expects and point LDFLAGS at it.
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && CMAKE_ARGS="-DGGML_CUDA=on -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=${LLAMA_CPP_CUDA_ARCH}" \
       LDFLAGS="-L/usr/local/cuda/lib64/stubs" \
       LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}" \
       python -m pip install --no-cache-dir --no-binary=llama-cpp-python \
       "llama-cpp-python==${LLAMA_CPP_VERSION}"

COPY app ./app
COPY entrypoint.sh ./entrypoint.sh

RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
