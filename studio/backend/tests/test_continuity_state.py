# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""project_state.json: load, write, update. No test touches a model, a backend,
or a real tool -- this is file I/O against tmp_path."""

from __future__ import annotations

import pytest

from core.continuity import load_state, write_state, update_state
from core.continuity.schemas import ContinuityError, ProjectState


def _state(**over):
    base = dict(
        schema_version = 1, project = "demo", status = "active",
        current_phase = "phase-1", current_task = None, completion_percent = 0,
        last_checkpoint = None,
    )
    base.update(over)
    return ProjectState(**base)


def test_load_state_on_an_uninitialized_project_returns_none(tmp_path):
    assert load_state(str(tmp_path)) is None


def test_write_then_load_round_trips(tmp_path):
    write_state(str(tmp_path), _state(project = "demo", next_action = "do the thing"))
    loaded = load_state(str(tmp_path))
    assert loaded is not None
    assert loaded.project == "demo"
    assert loaded.next_action == "do the thing"


def test_update_state_requires_an_existing_state(tmp_path):
    with pytest.raises(ContinuityError):
        update_state(str(tmp_path), next_action = "x")


def test_update_state_merges_fields(tmp_path):
    write_state(str(tmp_path), _state(current_phase = "phase-1"))
    updated = update_state(str(tmp_path), current_phase = "phase-2", completion_percent = 40)
    assert updated.current_phase == "phase-2"
    assert updated.completion_percent == 40
    # untouched field survives the merge
    assert updated.project == "demo"
    assert load_state(str(tmp_path)).current_phase == "phase-2"


def test_update_state_rejects_an_unknown_field(tmp_path):
    write_state(str(tmp_path), _state())
    with pytest.raises(ContinuityError):
        update_state(str(tmp_path), not_a_real_field = "x")
