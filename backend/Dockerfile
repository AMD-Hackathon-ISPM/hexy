FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ENV CMAKE_BUILD_PARALLEL_LEVEL=16 \
    MAX_JOBS=16

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        ffmpeg \
        build-essential \
        ninja-build \
        libgl1 \
        libglib2.0-0 \
        libegl1 \
        libgles2 \
        libosmesa6 \
        espeak-ng \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN grep -v '^torch' requirements.txt > requirements.docker.txt \
    && pip install --no-cache-dir -r requirements.docker.txt

# Upgrade cmake so the 'all-major' CUDA arch keyword works (requires 3.23+).
RUN pip install --no-cache-dir "cmake>=3.25"

# Build llama-cpp-python from source with CUDA support.
ARG LLAMA_CPP_VERSION=0.3.22
ARG LLAMA_CPP_CUDA_ARCH=86
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
    && CMAKE_ARGS="-GNinja -DGGML_CUDA=on -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=${LLAMA_CPP_CUDA_ARCH}" \
       LDFLAGS="-L/usr/local/cuda/lib64/stubs" \
       LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}" \
       pip install --no-cache-dir --no-binary=llama-cpp-python \
       "llama-cpp-python==${LLAMA_CPP_VERSION}"

COPY app ./app
COPY entrypoint.sh ./entrypoint.sh

RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
