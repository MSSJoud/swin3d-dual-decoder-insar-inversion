FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /workspace/
COPY src /workspace/src
COPY configs /workspace/configs

RUN pip install --upgrade pip && pip install .

ENTRYPOINT ["swin3d-insar-inversion"]
CMD ["--help"]

