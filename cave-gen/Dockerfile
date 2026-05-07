FROM python:3.11-slim

WORKDIR /work

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libegl1 libgles2 libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY cave-map.py build_cave_hexapod.py ./

CMD ["sh", "-c", "python cave-map.py && python build_cave_hexapod.py"]
