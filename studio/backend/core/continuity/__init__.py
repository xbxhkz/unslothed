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
from core.continuity.schemas import ContinuityError, ProjectState, Task, TaskQueue

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


_TASKS_FILENAME = "task_queue.json"

# Legal edges. A task is promoted from "pending" to "ready" by an explicit
# set_task_status call (e.g. once its prerequisites are satisfied outside this
# graph, or a human queues it up) -- that is a distinct fact from what
# ready_tasks() answers below, which is "which pending tasks' depends_on are
# all complete right now". ready_tasks never reads or writes a "ready" status;
# it is a pure query over depends_on + status computed fresh on every call, so
# there is no way for a stored value and the derived one to disagree.
_TRANSITIONS: dict[str, set[str]] = {
    "pending": {"ready"},
    "ready": {"in_progress"},
    "in_progress": {"complete", "failed", "blocked"},
    "blocked": {"ready"},
    "failed": {"abandoned", "pending"},
    "abandoned": set(),
    "complete": set(),
}
_VALID_STATUSES = frozenset(_TRANSITIONS)


def _tasks_path(project_dir: str) -> str:
    return os.path.join(storage.ai_dir(project_dir), _TASKS_FILENAME)


def load_tasks(project_dir: str) -> TaskQueue:
    raw = storage.read_json(_tasks_path(project_dir))
    if raw is None:
        return TaskQueue(schema_version = storage.CURRENT_SCHEMA_VERSION, tasks = [])
    return TaskQueue.from_dict(raw)


def _write_tasks(project_dir: str, queue: TaskQueue) -> None:
    storage.write_json_atomic(_tasks_path(project_dir), queue.to_dict())


def add_task(project_dir: str, task: Task) -> None:
    if task.status not in _VALID_STATUSES:
        raise ContinuityError(f"unknown status {task.status!r}")
    queue = load_tasks(project_dir)
    existing_ids = {t.id for t in queue.tasks}
    if task.id in existing_ids:
        raise ContinuityError(f"a task with id {task.id!r} already exists")
    missing_deps = [d for d in task.depends_on if d not in existing_ids]
    if missing_deps:
        raise ContinuityError(
            f"task {task.id!r} depends on unknown id(s) {missing_deps} -- add those tasks first"
        )
    queue.tasks.append(task)
    _write_tasks(project_dir, queue)


def set_task_status(project_dir: str, task_id: str, status: str) -> None:
    if status not in _VALID_STATUSES:
        raise ContinuityError(f"unknown status {status!r}")
    queue = load_tasks(project_dir)
    task = next((t for t in queue.tasks if t.id == task_id), None)
    if task is None:
        raise ContinuityError(f"no task with id {task_id!r}")
    allowed = _TRANSITIONS.get(task.status, set())
    if status not in allowed:
        raise ContinuityError(
            f"task {task_id!r}: {task.status!r} -> {status!r} is not a legal transition "
            f"(allowed from {task.status!r}: {sorted(allowed) or 'none'})"
        )
    if status == "complete" and not task.acceptance_criteria:
        raise ContinuityError(
            f"task {task_id!r} cannot be marked complete with no acceptance_criteria"
        )
    task.status = status
    _write_tasks(project_dir, queue)


def ready_tasks(project_dir: str) -> list[Task]:
    """Every pending task whose depends_on are all complete. This is the ONLY
    place readiness-by-dependency is decided -- it is a computed answer, never
    a field trusted from disk."""
    queue = load_tasks(project_dir)
    complete_ids = {t.id for t in queue.tasks if t.status == "complete"}
    return [
        t for t in queue.tasks
        if t.status == "pending" and all(d in complete_ids for d in t.depends_on)
    ]
