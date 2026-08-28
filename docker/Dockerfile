# syntax=docker/dockerfile:1.7
#
# Earth-2 Lab: the single attendee image for the TEMPO x Earth-2 hackathon.
#
# Design constraints this image is built to satisfy (see
# tempo_earth2_jupyterhub_gke_design.md):
#   * runs as a non-root user, no privileged mode;
#   * starts correctly under KubeSpawner as jupyterhub-singleuser;
#   * contains every runtime dependency - nothing is installed at spawn;
#   * ships an immutable reference copy of the notebooks, seeded into the
#     attendee home on first start and never overwritten afterwards;
#   * keeps the Earth2Studio model cache out of the per-user home directory;
#   * reports its own versions, CUDA visibility and data access on request.
#
# Build (on a GPU host or Cloud Build; the base image is amd64/arm64 but the
# CUDA stack only matters on amd64 GPU nodes):
#
#   docker build -f docker/Dockerfile -t earth2-lab:dev .
#
# The base image is ~20 GB. That is deliberate - it is NVIDIA's tested
# PyTorch/CUDA combination, and the design already requires pre-warming event
# nodes. Do not swap it for something smaller without re-running the full
# environment-check notebook on the target node image.

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.04-py3
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="Earth-2 Lab" \
      org.opencontainers.image.description="Earth2Studio + Glue-Jupyter + NASA TEMPO workshop environment" \
      org.opencontainers.image.licenses="Apache-2.0"

# ---------------------------------------------------------------------------
# Pins. Every version the workshop depends on is declared here so the release
# tag and the resolved digest describe the same software.
# ---------------------------------------------------------------------------
ARG EARTH2STUDIO_VERSION=0.17.0
ARG EARTH2STUDIO_EXTRAS=fcn,data,utils
ARG UV_VERSION=0.11.26
ARG NB_USER=jovyan
ARG NB_UID=1000
ARG NB_GID=100

# Bake the model checkpoint into the image layer. Off by default because each
# model has its own license and must be reviewed before redistribution; turn it
# on for the event image once that review is done.
ARG PREFETCH_MODEL=false
ARG PREFETCH_MODEL_NAME=FCN

ENV DEBIAN_FRONTEND=noninteractive \
    SHELL=/bin/bash \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1

# ---------------------------------------------------------------------------
# System packages. eccodes is required by cfgrib/pygrib for GFS initial
# conditions; geos/proj back cartopy; the rest are build prerequisites for
# packages that still ship source distributions.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        libeccodes-dev \
        libeccodes-tools \
        libgeos-dev \
        libproj-dev \
        make \
        proj-bin \
        python3-dev \
        rsync \
        tini \
        unzip \
        vim \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# uv, and removal of the NGC pip constraint file. The PyTorch container pins
# packages via PIP_CONSTRAINT; leaving it in place makes the Earth2Studio
# resolution fail in confusing ways. NVIDIA's own install guide unsets it.
#
# This install unavoidably replaces the base image's NGC-native torch: the
# `utils` extra depends on earth2grid, which is published only for
# torch>=2.13,<2.14, while Earth2Studio 0.17.0 itself pins the base image's
# exact torch build - so torch==2.12...(NGC) and earth2grid's requirement
# cannot both be satisfied, and only earth2grid's constraint is satisfiable
# from a public index. (Verified: constraining torch to the base build via
# UV_CONSTRAINT makes this install fail with "No solution found", not the
# torch we'd want it to pick.) The resolver settles on a PyPI torch build
# instead, which bundles its own cuBLAS - see the transformer_engine note
# below for the ABI consequence.
# ---------------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:0.11.26 /uv /uvx /bin/
ENV PIP_CONSTRAINT= \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_LINK_MODE=copy

# ---------------------------------------------------------------------------
# Earth2Studio. Installed from the tagged GitHub release, which is the form
# NVIDIA documents and tests.
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/opt/uv-cache,sharing=locked \
    uv pip install --system --break-system-packages \
        "earth2studio[${EARTH2STUDIO_EXTRAS}] @ git+https://github.com/NVIDIA/earth2studio.git@${EARTH2STUDIO_VERSION}"

# ---------------------------------------------------------------------------
# JupyterHub single-user server, Glue, and the geoscience stack.
# Installed after Earth2Studio so a resolver conflict shows up as a build
# failure here rather than as a silent downgrade of torch.
# ---------------------------------------------------------------------------
COPY docker/requirements-lab.txt /tmp/requirements-lab.txt
RUN --mount=type=cache,target=/opt/uv-cache,sharing=locked \
    uv pip install --system --break-system-packages -r /tmp/requirements-lab.txt \
    && rm /tmp/requirements-lab.txt

# ---------------------------------------------------------------------------
# The Earth2Studio install above (see note) always moves torch off the base
# image's NGC-native build. The base image's prebuilt transformer_engine is
# compiled against that NGC build's system CUDA toolkit, not the replacement
# torch's bundled cuBLAS, so importing it now raises an ABI "undefined
# symbol" OSError. physicsnemo (an Earth2Studio dependency) imports
# transformer_engine.pytorch unconditionally whenever any of its models are
# touched - including FCN's import path, even though FCN itself never uses
# it - and only guards against the package being *absent* (ImportError), not
# present-but-broken (OSError), so the broken copy takes down every
# physicsnemo model import, not just the ones that want acceleration.
# physicsnemo's own fallback (plain nn.LayerNorm) is correct and sufficient
# here - FCN does not benefit from transformer_engine - so remove the
# ABI-mismatched copy rather than carry a landmine into the event image.
# ---------------------------------------------------------------------------
RUN if ! python -c "import transformer_engine.pytorch" 2>/tmp/te-check.log; then \
        echo "transformer_engine is ABI-incompatible with the resolved torch build; removing it:"; \
        cat /tmp/te-check.log; \
        pip uninstall -y transformer_engine transformer_engine_cu13 transformer_engine_torch; \
    fi; \
    rm -f /tmp/te-check.log

# Fail the build rather than the event if the two installs disagreed.
# earth2studio.models.px.FCN is deliberately included, not just `import
# earth2studio`: it is what actually walks the physicsnemo model registry and
# exercises the torch/cuBLAS ABI pairing (see the transformer_engine note
# above), and a bare package import does not touch that path.
RUN python -c "import torch, earth2studio, jupyterhub, jupyterlab, glue_jupyter, xarray, cartopy; \
from earth2studio.models.px import FCN; \
print('earth2studio', earth2studio.__version__); \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
print('jupyterhub', jupyterhub.__version__)"

# ---------------------------------------------------------------------------
# Workshop layout.
#
#   /opt/earth2/notebooks   immutable reference copy of the released notebooks
#   /opt/earth2/cache/models  shared Earth2Studio model cache (pre-populated)
#   /opt/earth2/precomputed   pre-baked outputs for the CPU fallback path
#   /opt/earth2/bin           seed / restore / reset helper commands
#   /home/jovyan/work         attendee working copy, on the persistent volume
# ---------------------------------------------------------------------------
ENV EARTH2_ROOT=/opt/earth2 \
    WORKSHOP_REFERENCE_DIR=/opt/earth2/notebooks \
    WORKSHOP_PRECOMPUTED=/opt/earth2/precomputed \
    WORKSHOP_WORK_DIR=/home/${NB_USER}/work \
    EARTH2STUDIO_MODEL_CACHE=/opt/earth2/cache/models \
    EARTH2STUDIO_DATA_CACHE=/home/${NB_USER}/.cache/earth2studio/data \
    EARTH2STUDIO_PACKAGE_TIMEOUT=900 \
    PYTHONPATH=/opt/earth2/lib \
    PATH=/opt/earth2/bin:${PATH} \
    MPLCONFIGDIR=/tmp/matplotlib

RUN mkdir -p /opt/earth2/notebooks \
             /opt/earth2/cache/models \
             /opt/earth2/precomputed \
             /opt/earth2/bin \
             /opt/earth2/lib \
             /opt/earth2/scripts

COPY src/tempo_earth2 /opt/earth2/lib/tempo_earth2
# Only the rendered notebooks, not the percent-format sources they build from.
COPY notebooks/*.ipynb /opt/earth2/notebooks/
COPY scripts/ /opt/earth2/scripts/
COPY docker/bin/ /opt/earth2/bin/

RUN chmod -R a+rx /opt/earth2/bin \
    && python -c "import tempo_earth2; print('tempo_earth2', tempo_earth2.__version__)"

# A JupyterLab terminal starts a *login* shell, and /etc/profile rebuilds PATH
# from scratch - discarding the ENV PATH set above. Without this, the recovery
# commands are "command not found" at exactly the moment an attendee has been
# told to run one. Verified by `make verify-shell`.
RUN printf '%s\n' \
      '# Earth-2 Lab: keep the workshop commands reachable in login shells.' \
      'export PATH="/opt/earth2/bin:$HOME/.local/bin:$PATH"' \
      > /etc/profile.d/earth2-lab.sh \
    && chmod 0644 /etc/profile.d/earth2-lab.sh

# ---------------------------------------------------------------------------
# Optional: bake the model checkpoint. Turning this on removes the event-day
# dependency on Hugging Face and NGC entirely, at the cost of image size.
# ---------------------------------------------------------------------------
RUN if [ "${PREFETCH_MODEL}" = "true" ]; then \
        python /opt/earth2/scripts/prefetch_models.py --model "${PREFETCH_MODEL_NAME}" \
        && chmod -R a+rX /opt/earth2/cache; \
    fi

# ---------------------------------------------------------------------------
# Non-root attendee user.
# ---------------------------------------------------------------------------
RUN if ! id -u ${NB_UID} >/dev/null 2>&1; then \
        useradd --create-home --uid ${NB_UID} --gid ${NB_GID} --shell /bin/bash ${NB_USER}; \
    else \
        usermod -l ${NB_USER} -d /home/${NB_USER} -m "$(id -un ${NB_UID})" || true; \
    fi \
    && mkdir -p /home/${NB_USER}/work /home/${NB_USER}/.cache/earth2studio/data \
    && chown -R ${NB_UID}:${NB_GID} /home/${NB_USER} \
    && chmod -R a+rX /opt/earth2 \
    && chown -R ${NB_UID}:${NB_GID} /opt/earth2/cache/models
# The rest of /opt/earth2 is intentionally read-only. The model cache is the
# exception: unless PREFETCH_MODEL baked a checkpoint above, this directory is
# still empty here, and Earth2Studio downloads into it on first model load. In
# the real deployment this path gets shadowed by a pre-populated shared
# volume mount (see the design doc's cache options) so attendees never
# actually write here; without that mount - i.e. every local/single-container
# run - jovyan owning it is what makes a plain, non-baked image usable at all.

COPY docker/entrypoint.sh /usr/local/bin/earth2-entrypoint
RUN chmod +x /usr/local/bin/earth2-entrypoint

USER ${NB_UID}:${NB_GID}
WORKDIR /home/${NB_USER}
ENV HOME=/home/${NB_USER}

EXPOSE 8888

ENTRYPOINT ["tini", "-g", "--", "/usr/local/bin/earth2-entrypoint"]
CMD ["jupyterhub-singleuser"]
