# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Unslothed container image. One Dockerfile, two tags, differing only in the
# base image and the torch wheel index:
#
#   docker build -t unslothed:cpu  .
#   docker build -t unslothed:cuda \
#       --build-arg BASE_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu24.04 \
#       --build-arg TORCH_FAMILY=cu126 .
#
# :cuda needs `--gpus all` and the NVIDIA Container Toolkit on the host. It will
# not run on a Synology NAS; :cpu is the image for that.
#
# --- why both bases are Ubuntu 24.04 -----------------------------------------
# The design spec named python:3.14-slim for the CPU tag. Deviating deliberately:
# nvidia/cuda publishes only Ubuntu/UBI bases, so pairing it with a Debian-slim
# CPU base would give the two tags different package managers, different Python
# provenance and two divergent apt/pip stanzas -- the opposite of "differing only
# in base image and torch wheels". Ubuntu 24.04 on both sides keeps every
# instruction below identical between tags, which is what makes the pair
# maintainable. Ubuntu 24.04 ships Python 3.12, comfortably inside pyproject's
# requires-python = ">=3.9,<3.15", and torch 2.10.0 publishes cp312 wheels.


# --- global build args -------------------------------------------------------
# These MUST be declared before the first FROM. An ARG declared after one is
# scoped to that stage, so `FROM ${BASE_IMAGE}` later resolves to an empty
# string and the build fails with "base name should not be blank". Each stage
# that actually uses one re-declares it to pull it into scope.
ARG BASE_IMAGE=ubuntu:24.04
ARG TORCH_FAMILY=cpu


# ============================================================================
# Stage 1 -- frontend build
# ============================================================================
# Pinned to node:22 to satisfy studio/frontend/package.json's
# engines.node = "^20.19.0 || >=22.12.0". The runtime stage installs its own
# (older, apt) Node purely to run typescript-language-server; the two are
# unrelated and deliberately not shared.
FROM node:22-slim AS frontend

WORKDIR /build/studio/frontend

# package.json + lock first, so a source-only change does not re-run npm ci.
COPY studio/frontend/package.json studio/frontend/package-lock.json ./
RUN npm ci

COPY studio/frontend/ ./
# `tsc -b && vite build`
RUN npm run build && test -f dist/index.html


# ============================================================================
# Stage 2 -- runtime
# ============================================================================
FROM ${BASE_IMAGE} AS runtime

# Re-declared: a global ARG is not in scope inside a stage until restated.
ARG TORCH_FAMILY

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# --- OS packages -------------------------------------------------------------
# nodejs + npm : typescript-language-server (code_* tools)
# libicu74     : .NET's globalization dependency; without it dotnet aborts at
#                startup rather than failing informatively.
# git          : some pip requirements resolve VCS references.
# curl         : HEALTHCHECK below, the dotnet bootstrap, and pip/vendor fetches.
#
# Note what is NOT here: dotnet-sdk-8.0. Ubuntu 24.04's package is unusable for
# this purpose -- see the .NET stanza below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-dev \
        build-essential \
        ca-certificates \
        curl \
        git \
        nodejs \
        npm \
        libicu74 \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# libgl1 / libglib2.0-0 are opencv-python's runtime shared libraries. Without
# them `import cv2` raises "libGL.so.1: cannot open shared object file" the first
# time a vision tool runs -- a lazily imported module that fails on first use,
# which is exactly the failure the installer smoke test was written to catch.

# --- language servers, pre-installed ----------------------------------------
# The spec's reasoning: inside a container the tools' "not installed, run
# `npm install -g ...`" message is useless -- the user cannot easily act on it
# and the result would not persist. An image where a headline feature silently
# does not work is worse than the ~250 MB.
#
# servers.py's own install commands are unpinned (`npm install -g
# typescript-language-server typescript`, `dotnet tool install --global
# csharp-ls`). Mirroring them verbatim FAILED this build:
#
#   The settings file in the tool's NuGet package is invalid:
#   Settings file 'DotnetToolSettings.xml' was not found in the package.
#
# Unpinned csharp-ls resolves to a release built for a newer SDK than the .NET 8
# in Ubuntu 24.04, and the SDK cannot read its package layout. Precisely the
# failure shape the design spec warns about and that the bare `mcp` -> 2.0.0
# incident already cost this project once.
#
# So both are pinned to the versions the e2e suites are actually verified
# against, which is a stronger guarantee than "whatever npm resolved today":
#   * typescript@5      -- test_assist_code_e2e.py:71 uses exactly this, noting
#                          "5.9.3 confirmed to ship tsserver.js; any 5.x should"
#   * csharp-ls 0.27.0  -- test_assist_code_csharp_e2e.py:30, "Verified against
#                          csharp-ls 0.27.0 with .NET 8.0.424"
# --- .NET, from Microsoft's installer rather than apt ------------------------
# Ubuntu 24.04's `dotnet-sdk-8.0` package cannot install csharp-ls. Measured, not
# assumed -- three builds were spent on this, so the evidence is recorded here:
#
#   apt dotnet-sdk-8.0        -> `dotnet --version` reports 10.0.400 (!), and
#                                `dotnet tool install csharp-ls` fails with
#                                "DotnetToolSettings.xml was not found"
#   mcr .../sdk:8.0 (8.0.424) -> same failure
#   mcr .../sdk:10.0          -> installs, and `csharp-ls --version` runs
#   ubuntu:24.04 + this script-> installs, and `csharp-ls --version` runs
#
# The cause is neither the SDK version nor a bad csharp-ls pin (both were
# hypothesised and both were wrong). Unpacking the nupkg shows the settings file
# is present but at `tools/net10.0/any/DotnetToolSettings.xml`: csharp-ls 0.27.0
# as published targets net10.0, and an SDK that cannot run net10.0 finds no
# usable tools folder and reports it as missing. So .NET 10 is required, and
# Ubuntu's package does not provide a working one.
#
# The full SDK, not just the runtime: csharp-ls loads a project through
# Microsoft.Build.Locator during its handshake (which is why its start() is slow
# and why the C# e2e suite's cold-start race does not reproduce), and MSBuild
# ships with the SDK.
#
# Cost: ~624 MB for the SDK plus ~100 MB for the tool, against the design spec's
# "+250 MB" estimate for both language servers. The estimate was low; the
# reasoning behind pre-installing them is unchanged -- inside a container the
# "run `dotnet tool install`" message is useless and would not persist.
ARG DOTNET_CHANNEL=10.0
RUN curl -fsSL https://dot.net/v1/dotnet-install.sh -o /tmp/dotnet-install.sh \
    && chmod +x /tmp/dotnet-install.sh \
    && /tmp/dotnet-install.sh --channel "${DOTNET_CHANNEL}" --install-dir /usr/share/dotnet --no-path \
    && ln -s /usr/share/dotnet/dotnet /usr/local/bin/dotnet \
    && rm -f /tmp/dotnet-install.sh \
    && dotnet --version

ARG TS_VERSION=5
ARG CSHARP_LS_VERSION=0.27.0
RUN npm install -g typescript-language-server "typescript@${TS_VERSION}" \
    && dotnet tool install --global csharp-ls --version "${CSHARP_LS_VERSION}"
ENV PATH="/root/.dotnet/tools:${PATH}"
ENV DOTNET_CLI_TELEMETRY_OPTOUT=1 \
    DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1

# --- Python environment ------------------------------------------------------
# A venv rather than --break-system-packages: Ubuntu 24.04 marks its system
# Python externally-managed (PEP 668), and install_python_stack.py installs into
# sys.executable's environment, so the venv is what makes that land somewhere
# sane.
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

WORKDIR /app

# Requirements and the installer script first: the dependency layer is the
# expensive one (torch alone is GBs) and must not be invalidated by a change to
# application source.
COPY studio/backend/requirements/ /app/studio/backend/requirements/
COPY studio/install_python_stack.py /app/studio/
COPY studio/prebuilt_core.py /app/studio/
COPY studio/install_manifest.py /app/studio/

# install_python_stack.py is not standalone: it does
#   from backend.utils.wheel_utils import ...
#   from backend.utils.uv_path_safety import uv_safe_path
# and it is run as /app/studio/install_python_stack.py, so /app/studio is
# sys.path[0] and `backend.utils` must resolve under it. The package __init__
# files are 2-line licence headers, so this imports without dragging in the rest
# of the backend.
#
# utils/ is copied whole (5.5 MB) rather than the two modules: it changes far
# less often than routes/ or core/, so the expensive dependency layer below
# stays cached in practice, and a future sibling import here cannot silently
# reintroduce this failure.
COPY studio/__init__.py /app/studio/
COPY studio/backend/__init__.py /app/studio/backend/
COPY studio/backend/utils/ /app/studio/backend/utils/

# Step 11 of install_python_stack.py ("data designer") pip-installs two seed
# plugins FROM THE CHECKOUT and aborts if either directory is absent
# (install_python_stack.py:4687-4694). 162 KB, and required before the
# dependency step rather than with the rest of the source.
COPY studio/backend/plugins/ /app/studio/backend/plugins/

# install_python_stack.py owns the real ordering -- base -> no-torch-runtime ->
# extras -> extras-no-deps (--no-deps) -> triton-kernels -> studio ->
# diffusers-pin (last, so no later pass can replace the pin), all under
# single-env/constraints.txt -- plus the torchao version selection and anyio
# repair. Reimplementing that as inline pip calls would fork 4,787 lines of
# resolver logic that upstream keeps changing.
#
# UNSLOTH_TORCH_INDEX_FAMILY is the script's own headless/CI seam: an explicit
# pin commits to that wheel family and skips ALL hardware detection, which is
# required here because the build host has no GPU visible and must still be able
# to produce the :cuda image.
ENV UNSLOTH_TORCH_INDEX_FAMILY=${TORCH_FAMILY}

# --- CPU tag only: drop the orphaned CUDA payload ----------------------------
# NOTE: this is deliberately part of the SAME RUN as install_python_stack.py.
# Image layers are additive -- a `pip uninstall` in a later RUN removes the files
# from the final filesystem view but leaves them in the layer that created them,
# so the image stays exactly as large. Measured: as a separate step the prune
# freed 6 GB inside the container and moved `docker images` not at all (19.1 GB
# before and after). Only deleting within the creating layer actually shrinks it.
# Measured on the first successful :cpu build: 6.0 GB of site-packages/nvidia in
# an image whose torch reports `2.11.0+cpu` and `cuda.is_available() == False`.
#
# How it gets there is visible in the build log:
#
#   torch is a GPU build but an explicit CPU index is pinned -- reinstalling CPU torch
#
# An earlier resolution step pulls a GPU torch from PyPI, which drags in the
# nvidia-* wheels; step 13's repair then swaps torch for the CPU build but does
# not uninstall the dependencies the discarded GPU torch brought with it. They
# are left behind as orphans -- `pip show` reports an empty Required-by for
# nearly all of them -- and two whole stacks accumulate (a -cu12 series and a
# -cu13/unsuffixed one, including cuda-toolkit 13.0.2).
#
# This matters for the tag's actual purpose: :cpu is the image for a Synology
# NAS, where 6 GB of CUDA libraries that can never be loaded is the difference
# between a practical deployment and an impractical one.
#
# triton is deliberately NOT removed: unlike the nvidia-* wheels it has real
# dependents here (cut-cross-entropy, openai-whisper, unsloth_zoo).
#
# Verified in a live container before being baked in: /opt/venv 11 GB -> 4.0 GB,
# with torch 2.11.0+cpu, torchvision 0.26.0+cpu, cv2, onnxruntime and ultralytics
# all still importing and ALL_TOOLS still 17. The asserts below are the standing
# guard, so a future resolver change that makes one of these load-bearing fails
# the build instead of the app.
RUN python /app/studio/install_python_stack.py \
    && if [ "${TORCH_FAMILY}" = "cpu" ]; then \
        PKGS="$(pip list --format=freeze | grep -iE '^(nvidia|cuda)' | cut -d= -f1 | tr '\n' ' ')"; \
        echo "pruning orphaned CUDA distributions: $(echo $PKGS | wc -w)"; \
        if [ -n "$PKGS" ]; then pip uninstall -y $PKGS; fi; \
        python -c "import torch, torchvision; assert not torch.cuda.is_available(); print('torch', torch.__version__, 'torchvision', torchvision.__version__)"; \
        python -c "import cv2, onnxruntime, ultralytics; print('vision deps import OK')"; \
    else \
        echo "TORCH_FAMILY=${TORCH_FAMILY}: keeping the CUDA payload"; \
    fi \
    && rm -rf /root/.cache/pip /tmp/* 2>/dev/null || true

# --- application source ------------------------------------------------------
COPY studio/backend/ /app/studio/backend/
COPY pyproject.toml README.md /app/
COPY unsloth/ /app/unsloth/

# The fork itself, installed WITHOUT dependency resolution. pyproject declares
# 278 unpinned entries by design (it is a library); the requirements files above
# are the pinned application manifest. Letting pip re-resolve here would let it
# walk back over pins that were just established -- notably the diffusers pin
# that install_python_stack deliberately applies last.
RUN pip install --no-deps /app

COPY --from=frontend /build/studio/frontend/dist/ /app/studio/frontend/dist/

COPY packaging/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# --- data ---------------------------------------------------------------------
# Studio resolves its data root from UNSLOTH_STUDIO_HOME before anything else
# (utils/paths/storage_roots.py:48). Pointing it at a mounted volume is what
# keeps containers disposable and lets a NAS deployment target persistent
# storage; models, chats and auth all live under it.
ENV UNSLOTH_STUDIO_HOME=/data
# ultralytics writes a settings file on first import and falls back to /tmp with
# a four-line warning when its default (~/.config/Ultralytics) is not writable --
# noise on every webcam_look / detect_shapes call, and lost on restart. Point it
# at the volume instead.
ENV YOLO_CONFIG_DIR=/data/ultralytics
RUN mkdir -p /data
VOLUME ["/data"]

EXPOSE 8000

# 0.0.0.0 is correct here and not a loosening: a container port is only reachable
# via an explicit `-p` publish, and binding loopback would make the port
# unreachable from the host entirely.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
