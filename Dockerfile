FROM pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/workspace
ENV MUJOCO_GL=egl
ENV TORCHDYNAMO_DISABLE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 \
    libgl1 \
    libglfw3 \
    libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

# Triton provoque le segmentation fault observé.
RUN pip uninstall -y triton pytorch-triton || true \
    && rm -rf \
        /usr/local/lib/python*/dist-packages/triton \
        /usr/local/lib/python*/dist-packages/triton-*.dist-info \
        /usr/local/lib/python*/site-packages/triton \
        /usr/local/lib/python*/site-packages/triton-*.dist-info

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt

RUN pip install --break-system-packages --no-cache-dir -r /tmp/requirements.txt

RUN mkdir -p /data/input /data/output

COPY src /workspace/src
COPY data/input /data/input

CMD ["python", "-u", "-X", "faulthandler", "-m", "src.train"]