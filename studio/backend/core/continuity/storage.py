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
CURRENT_SCHEMA_VERSION = 1


def write_json_atomic(path: str, data: dict) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok = True)
    tmp_path = os.path.join(directory, f".{os.path.basename(path)}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp_path, "w", encoding = "utf-8") as f:
            json.dump(data, f, indent = 2)
            f.write("\n")
        os.replace(tmp_path, path)
    except Exception as exc:
        # Exception, not BaseException: a KeyboardInterrupt/SystemExit during
        # a write is a process-control signal, not a write failure, and must
        # propagate as itself rather than being masked as a ContinuityError.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise ContinuityError(f"could not write {path}: {exc}") from exc


def read_json(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding = "utf-8") as f:
            data = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContinuityError(f"could not read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ContinuityError(
            f"{path} contains valid JSON that is not an object "
            f"(got {type(data).__name__}) -- the file is corrupted or was overwritten "
            "by something else"
        )
    return data


def ai_dir_path(project_dir: str) -> str:
    """The .ai/ directory's path under project_dir. Never touches the
    filesystem -- safe to call from a read-only operation like validate or
    status, which must not create directories as a side effect of looking."""
    return os.path.join(project_dir, _AI_DIR_NAME)


def ai_dir(project_dir: str) -> str:
    """The .ai/ directory (with logs/ and checkpoints/), CREATED if it does
    not exist yet. Only call this from a write path -- use ai_dir_path for a
    read that must not have side effects. Never raises for "not initialized
    yet" -- that is the normal starting state -- but DOES raise
    ContinuityError if .ai exists as something other than a directory (a
    plain file, most likely from a hand-edited or corrupted tree), since
    silently trying to create logs/ under a file would otherwise surface as
    a confusing raw OSError three lines later."""
    root = ai_dir_path(project_dir)
    if os.path.exists(root) and not os.path.isdir(root):
        raise ContinuityError(f"{root} exists and is not a directory")
    try:
        os.makedirs(os.path.join(root, "logs"), exist_ok = True)
        os.makedirs(os.path.join(root, "checkpoints"), exist_ok = True)
    except OSError as exc:
        raise ContinuityError(f"could not create {root}: {exc}") from exc
    return root
