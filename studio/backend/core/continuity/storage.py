# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Atomic file I/O for the continuity engine. Every write in this package goes
through write_json_atomic; nothing writes a JSON file directly.

Pattern taken from core/inference/image_gallery.py:71-81 -- a dotted temp name
(never matches a real *.json listing), os.replace for the atomic swap, and the
temp file removed on any failure so a crash never leaves a stray partial write
sitting next to the real file.
"""

from __future__ import annotations

import json
import os
import uuid

from core.continuity.schemas import ContinuityError

_AI_DIR_NAME = ".ai"


def write_json_atomic(path: str, data: dict) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok = True)
    tmp_path = os.path.join(directory, f".{os.path.basename(path)}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp_path, "w", encoding = "utf-8") as f:
            json.dump(data, f, indent = 2)
            f.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def read_json(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding = "utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContinuityError(f"could not read {path}: {exc}") from exc


def ai_dir(project_dir: str) -> str:
    """The .ai/ directory under project_dir, created (with logs/ and
    checkpoints/) if it does not exist yet. Never raises for "not initialized
    yet" -- that is the normal starting state, not an error."""
    root = os.path.join(project_dir, _AI_DIR_NAME)
    os.makedirs(os.path.join(root, "logs"), exist_ok = True)
    os.makedirs(os.path.join(root, "checkpoints"), exist_ok = True)
    return root
