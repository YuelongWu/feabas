FROM python:3.12-slim

# runtime libs for opencv-python
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/feabas

# Install dependencies first so this layer is cached across code-only changes
COPY setup.py README.md ./
RUN mkdir feabas && touch feabas/__init__.py \
    && pip install --no-cache-dir . \
    && pip uninstall -y feabas \
    && rm -rf feabas build *.egg-info

# Now the actual code
COPY . .
RUN pip install --no-cache-dir --no-deps -e .

ARG GIT_SHA=unknown
ENV FEABAS_GIT_SHA=${GIT_SHA} \
    PYTHONUNBUFFERED=1

# Scripts must run from the repo root so configs/ is found
CMD ["python", "scripts/stitch_main.py", "--help"]
