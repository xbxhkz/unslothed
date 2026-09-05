#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Entrypoint for the Unslothed container images.
#
# Any arguments are passed straight through to the backend, so the image stays
# usable for one-off inspection without fighting the entrypoint:
#
#   docker run --rm unslothed:cpu --help
#   docker run --rm -p 8000:8000 -v unslothed-data:/data unslothed:cpu
#   docker run --rm --entrypoint sh unslothed:cpu -c 'csharp-ls --version'

set -euo pipefail

STUDIO_HOME="${UNSLOTH_STUDIO_HOME:-/data}"

# The volume is bind-mounted at runtime, so it is empty on first start no matter
# what the image did at build time. Create the root here rather than assuming.
mkdir -p "$STUDIO_HOME"

# run.py imports its siblings as top-level modules (`from core.inference import
# tools`, `from utils.paths import ...`), so studio/backend must be the working
# directory. This is the same requirement the PyInstaller build satisfies by
# putting BACKEND on sys.path.
cd /app/studio/backend

# --host 0.0.0.0: see the Dockerfile's HEALTHCHECK note -- a container port is
# reachable only through an explicit publish, and loopback would make -p useless.
exec python run.py \
    --host 0.0.0.0 \
    --port "${UNSLOTHED_PORT:-8000}" \
    --frontend /app/studio/frontend/dist \
    "$@"
