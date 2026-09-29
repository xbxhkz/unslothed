# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""validate() finds problems; repair() fixes only what is safe to fix
automatically and NEVER marks a task complete -- that is the one thing spec
section 28 forbids outright."""

from __future__ import annotations

import json
import os

from core.continuity import add_task, repair, set_task_status, storage, validate, write_state
from core.continuity.schemas import ProjectState, Task


def _task(id, **over):
    base = dict(id = id, title = f"task {id}", status = "pending", depends_on = [],
                acceptance_criteria = ["ok"], owner = None)
    base.update(over)
    return Task(**base)


def _state(**over):
    base = dict(
        schema_version = 1, project = "demo", status = "active",
        current_phase = "phase-1", current_task = None, completion_percent = 0,
        last_checkpoint = None,
    )
    base.update(over)
    return ProjectState(**base)


def test_validate_clean_project_reports_nothing(tmp_path):
    add_task(str(tmp_path), _task("t1"))
    assert validate(str(tmp_path)) == []


def test_validate_reports_a_dependency_cycle(tmp_path):
    """Review Focus: A depends on B, B depends on A. Must be reported, not
    hang, and not silently leave both permanently un-ready with no explanation."""
    add_task(str(tmp_path), _task("t1"))
    add_task(str(tmp_path), _task("t2", depends_on = ["t1"]))
    # Hand-edit the queue to create the cycle -- add_task's own validation
    # (Task 3) would refuse to create one through the public API, so the test
    # writes the file directly to simulate a hand-edited or corrupted graph.
    path = os.path.join(str(tmp_path), ".ai", "task_queue.json")
    with open(path, encoding = "utf-8") as f:
        data = json.load(f)
    for t in data["tasks"]:
        if t["id"] == "t1":
            t["depends_on"] = ["t2"]
    with open(path, "w", encoding = "utf-8") as f:
        json.dump(data, f)

    problems = validate(str(tmp_path))
    assert any("cycle" in p.lower() for p in problems), problems


def test_validate_reports_corrupt_json(tmp_path):
    storage.ai_dir(str(tmp_path))
    path = os.path.join(str(tmp_path), ".ai", "task_queue.json")
    with open(path, "w", encoding = "utf-8") as f:
        f.write("{not json")
    problems = validate(str(tmp_path))
    assert any("task_queue.json" in p for p in problems), problems


def test_validate_reports_wrong_shaped_json(tmp_path):
    """Valid JSON, wrong shape -- a bare scalar where an object is expected.
    Different failure mode from a syntax error: read_json returns it happily,
    and validate must still survive turning it into a reported problem."""
    storage.ai_dir(str(tmp_path))
    path = os.path.join(str(tmp_path), ".ai", "project_state.json")
    with open(path, "w", encoding = "utf-8") as f:
        f.write("42")
    problems = validate(str(tmp_path))  # must not raise
    assert any("project_state.json" in p for p in problems), problems


def test_validate_reports_a_duplicate_id(tmp_path):
    # write task_queue.json directly with two tasks sharing an id -- add_task's
    # own guard (Task 3) would refuse to create this through the public API
    storage.ai_dir(str(tmp_path))
    path = os.path.join(str(tmp_path), ".ai", "task_queue.json")
    with open(path, "w", encoding = "utf-8") as f:
        json.dump({"schema_version": 1, "tasks": [
            {"id": "t1", "title": "a", "status": "pending", "depends_on": [], "acceptance_criteria": ["x"]},
            {"id": "t1", "title": "b", "status": "pending", "depends_on": [], "acceptance_criteria": ["y"]},
        ]}, f)
    problems = validate(str(tmp_path))
    assert any("duplicate" in p.lower() and "t1" in p for p in problems), problems


def test_validate_reports_an_unknown_dependency(tmp_path):
    add_task(str(tmp_path), _task("t1", depends_on = []))
    # hand-edit in a second task depending on an id that doesn't exist, since
    # add_task's own guard (Task 3) would refuse this through the public API
    path = os.path.join(str(tmp_path), ".ai", "task_queue.json")
    with open(path, encoding = "utf-8") as f:
        data = json.load(f)
    data["tasks"].append({"id": "t2", "title": "b", "status": "pending",
                          "depends_on": ["ghost"], "acceptance_criteria": ["x"]})
    with open(path, "w", encoding = "utf-8") as f:
        json.dump(data, f)
    problems = validate(str(tmp_path))
    assert any("t2" in p and "ghost" in p for p in problems), problems


def test_validate_reports_a_current_task_naming_nothing(tmp_path):
    write_state(str(tmp_path), _state(current_task = "does-not-exist"))
    problems = validate(str(tmp_path))
    assert any("does-not-exist" in p for p in problems), problems


def test_repair_never_completes_a_task(tmp_path):
    add_task(str(tmp_path), _task("t1"))
    set_task_status(str(tmp_path), "t1", "in_progress")
    repair(str(tmp_path))
    from core.continuity import load_tasks
    reloaded = load_tasks(str(tmp_path))
    assert next(t for t in reloaded.tasks if t.id == "t1").status == "in_progress"


def test_repair_regenerates_the_rendered_files(tmp_path):
    add_task(str(tmp_path), _task("t1"))
    report = repair(str(tmp_path))
    assert any("project_state.md" in r or "context_summary" in r for r in report)
