# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""checkpoints/<timestamp>.json: a snapshot of project_state.json plus a note,
pruned to the most recent N. Filesystem directory order is not creation order
on every OS, so pruning must sort on a value baked into each entry, not on
os.listdir's order.
"""

from __future__ import annotations

import json
import os

from core.continuity import checkpoint, load_state, write_state
from core.continuity.schemas import ProjectState


def _state(**over):
    base = dict(schema_version = 1, project = "demo", status = "active",
                current_phase = "p", current_task = None, completion_percent = 0,
                last_checkpoint = None)
    base.update(over)
    return ProjectState(**base)


def test_checkpoint_writes_a_readable_snapshot(tmp_path):
    write_state(str(tmp_path), _state(next_action = "step one"))
    name = checkpoint(str(tmp_path), "first checkpoint")
    path = os.path.join(str(tmp_path), ".ai", "checkpoints", name)
    with open(path, encoding = "utf-8") as f:
        data = json.load(f)
    assert data["note"] == "first checkpoint"
    assert data["state"]["next_action"] == "step one"


def test_checkpoint_prunes_the_oldest_first(tmp_path):
    write_state(str(tmp_path), _state())
    names = [checkpoint(str(tmp_path), f"note {i}", retain = 3) for i in range(5)]
    remaining = set(os.listdir(os.path.join(str(tmp_path), ".ai", "checkpoints")))
    # The three most recently WRITTEN (by call order, which this test controls),
    # not the three with the "largest" filesystem mtime or alphabetically-last name.
    assert remaining == set(names[-3:]), f"expected the 3 latest, got {remaining}"


def test_checkpoint_ignores_a_stray_non_checkpoint_file(tmp_path):
    """A file in checkpoints/ that this function never wrote (not this
    naming convention) must not crash every future checkpoint() call on
    int() of a non-numeric prefix."""
    write_state(str(tmp_path), _state())
    checkpoints_dir = os.path.join(str(tmp_path), ".ai", "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok = True)
    with open(os.path.join(checkpoints_dir, "notes.json"), "w", encoding = "utf-8") as f:
        f.write("{}")
    name = checkpoint(str(tmp_path), "still works")
    assert os.path.isfile(os.path.join(checkpoints_dir, name))


def test_checkpoint_sets_last_checkpoint_on_project_state(tmp_path):
    write_state(str(tmp_path), _state())
    name = checkpoint(str(tmp_path), "first checkpoint")
    assert load_state(str(tmp_path)).last_checkpoint == name


def test_checkpoint_before_init_still_succeeds(tmp_path):
    """No project_state.json yet -- checkpoint() must not raise trying to
    update_state a state that does not exist."""
    name = checkpoint(str(tmp_path), "before init")
    checkpoints_dir = os.path.join(str(tmp_path), ".ai", "checkpoints")
    with open(os.path.join(checkpoints_dir, name), encoding = "utf-8") as f:
        data = json.load(f)
    assert data["state"] is None
