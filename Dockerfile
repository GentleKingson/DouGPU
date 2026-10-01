# Official Python tag; no unverified image digest is asserted here.
# The host provides the NVIDIA driver through the NVIDIA Container Toolkit.
FROM python:3.12.14-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-cpu.txt requirements-gpu.txt constraints-cuda13.txt ./
RUN python -m pip install --no-cache-dir --only-binary=:all: -r requirements-gpu.txt \
    && python -m pip check

# Explicit copies exclude local virtual environments, runs, and cached weights.
COPY dougpu/ ./dougpu/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY upstream_cache/ ./upstream_cache/
COPY vendor/ ./vendor/
COPY LICENSE THIRD_PARTY_NOTICES.md ./
COPY bootstrap.py fork_run.py run_local.py tune_local.py ./

ENTRYPOINT ["python", "run_local.py"]
CMD ["--help"]
