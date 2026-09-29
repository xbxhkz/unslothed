# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The continuity engine's public API.

Every function here operates on a project_dir the caller names explicitly.
Nothing in this module resolves a path on its own -- the app-side tool (added
in a later task) is what confines project_dir to a sandbox; this module trusts
whatever path it is given, the same way storage.py does.

The core RAISES ContinuityError rather than guessing at recovery. Never-raises
is each caller's own decision (the CLI catches and prints; the app tool catches
and returns a string) -- this module does not have a bare except anywhere
except inside record_error's fallback path, documented at that function.
"""

from __future__ import annotations

import dataclasses
import os

from core.continuity import storage
from core.continuity.schemas import ContinuityError, ProjectState

_STATE_FILENAME = "project_state.json"


def _state_path(project_dir: str) -> str:
    return os.path.join(storage.ai_dir(project_dir), _STATE_FILENAME)


def load_state(project_dir: str) -> ProjectState | None:
    raw = storage.read_json(_state_path(project_dir))
    if raw is None:
        return None
    return ProjectState.from_dict(raw)


def write_state(project_dir: str, state: ProjectState) -> None:
    storage.write_json_atomic(_state_path(project_dir), state.to_dict())


def update_state(project_dir: str, **fields) -> ProjectState:
    current = load_state(project_dir)
    if current is None:
        raise ContinuityError(
            f"no project_state.json at {project_dir!r} yet -- call write_state first"
        )
    valid_fields = {f.name for f in dataclasses.fields(ProjectState)}
    unknown = set(fields) - valid_fields
    if unknown:
        raise ContinuityError(f"unknown ProjectState field(s): {sorted(unknown)}")
    merged = dataclasses.replace(current, **fields)
    write_state(project_dir, merged)
    return merged
