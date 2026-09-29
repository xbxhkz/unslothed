# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""task_queue.json: add, transitions, derived readiness.

"ready" is never a stored value -- test_ready_tasks_is_derived_not_stored below
is the one that would catch a future change that starts persisting it."""

from __future__ import annotations

import pytest

from core.continuity import add_task, load_tasks, ready_tasks, set_task_status
from core.continuity.schemas import ContinuityError, Task


def _task(id, **over):
    base = dict(id = id, title = f"task {id}", status = "pending", depends_on = [],
                acceptance_criteria = ["it works"], owner = None)
    base.update(over)
    return Task(**base)


def test_load_tasks_on_an_uninitialized_project_returns_an_empty_queue(tmp_path):
    q = load_tasks(str(tmp_path))
    assert q.tasks == []


def test_add_task_then_load_round_trips(tmp_path):
    add_task(str(tmp_path), _task("t1"))
    q = load_tasks(str(tmp_path))
    assert [t.id for t in q.tasks] == ["t1"]


def test_add_task_rejects_a_duplicate_id(tmp_path):
    add_task(str(tmp_path), _task("t1"))
    with pytest.raises(ContinuityError):
        add_task(str(tmp_path), _task("t1"))


def test_add_task_rejects_a_depends_on_naming_no_real_task(tmp_path):
    """Review Focus: a hand-edited task graph will have typos. A dependency on
    an id that exists nowhere must be caught at add time, not silently accepted
    and then mis-evaluated by ready_tasks."""
    with pytest.raises(ContinuityError):
        add_task(str(tmp_path), _task("t1", depends_on = ["does-not-exist"]))


def test_ready_tasks_is_derived_not_stored(tmp_path):
    add_task(str(tmp_path), _task("t1", status = "pending"))
    add_task(str(tmp_path), _task("t2", status = "pending", depends_on = ["t1"]))
    assert [t.id for t in ready_tasks(str(tmp_path))] == ["t1"]
    set_task_status(str(tmp_path), "t1", "ready")
    set_task_status(str(tmp_path), "t1", "in_progress")
    set_task_status(str(tmp_path), "t1", "complete")
    # t2's status field is still "pending" on disk; readiness is computed, not read.
    loaded = load_tasks(str(tmp_path))
    assert next(t for t in loaded.tasks if t.id == "t2").status == "pending"
    assert [t.id for t in ready_tasks(str(tmp_path))] == ["t2"]


def test_illegal_transition_raises(tmp_path):
    add_task(str(tmp_path), _task("t1", status = "pending"))
    with pytest.raises(ContinuityError):
        # pending -> complete directly is not a legal edge (spec section 5)
        set_task_status(str(tmp_path), "t1", "complete")


def test_complete_requires_non_empty_acceptance_criteria(tmp_path):
    add_task(str(tmp_path), _task("t1", status = "pending", acceptance_criteria = []))
    set_task_status(str(tmp_path), "t1", "ready")
    set_task_status(str(tmp_path), "t1", "in_progress")
    with pytest.raises(ContinuityError):
        set_task_status(str(tmp_path), "t1", "complete")


def test_set_task_status_on_a_missing_id_raises(tmp_path):
    with pytest.raises(ContinuityError):
        set_task_status(str(tmp_path), "nope", "in_progress")
