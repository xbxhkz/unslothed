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
import datetime
import json as _json
import os
import sys

from core.continuity import storage
from core.continuity.schemas import ContinuityError, ErrorEntry, ProjectState, Task, TaskQueue

# Allow tests to import this module using: from core.continuity import __init__
__init__ = sys.modules[__name__]

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

# Legal edges. "ready" is NEVER a key here and never appears as a write
# target -- it is a status a task can only be JUDGED to have (ready_tasks
# computes it from depends_on + status), never one it can be SET to. Deriving
# "ready" and then also allowing set_task_status(id, "ready") would be two
# sources of truth for the same fact, able to disagree. Because "ready" is
# absent from this dict, it is absent from _VALID_STATUSES too (built from
# this dict's keys below), so set_task_status(id, "ready") is refused with
# "unknown status" before the transition table is even consulted -- there is
# no special-case check needed to keep the promise this comment makes.
#
# Concretely: a pending task goes straight to in_progress once a caller has
# decided (typically by first calling ready_tasks()) that it is unblocked --
# there is no intermediate stored state to pass through. blocked returns to
# pending, not to a "ready" it was never blocked away from reaching that way.
_TRANSITIONS: dict[str, set[str]] = {
    "pending": {"in_progress"},
    "in_progress": {"complete", "failed", "blocked"},
    "blocked": {"pending"},
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


_ERRORS_FILENAME = "errors.jsonl"
# The real fallback target. A test never writes here directly -- it monkeypatches
# _append_fallback_note itself, so this path is only ever touched by a human
# running the real CLI or app tool.
_REMEMBER_NOW_PATH = os.path.expanduser(r"~\odysseus\.remember\now.md")


def _errors_path(project_dir: str) -> str:
    return os.path.join(storage.ai_dir(project_dir), "logs", _ERRORS_FILENAME)


def _append_error_line(project_dir: str, entry: ErrorEntry) -> None:
    path = _errors_path(project_dir)
    os.makedirs(os.path.dirname(path), exist_ok = True)
    with open(path, "a", encoding = "utf-8") as f:
        f.write(_json.dumps(entry.to_dict()) + "\n")


def _append_fallback_note(text: str) -> None:
    """Last resort: record_error's own write failed. Losing the record of a
    failure is bad; crashing the caller over it is worse, so this is wrapped in
    its own bare except and never propagates."""
    try:
        os.makedirs(os.path.dirname(_REMEMBER_NOW_PATH), exist_ok = True)
        with open(_REMEMBER_NOW_PATH, "a", encoding = "utf-8") as f:
            f.write(text)
    except BaseException:
        pass


def record_error(project_dir: str, *, what_tried: str, why_failed: str, symptom: str) -> None:
    """Never raises. A failed primary write falls back to .remember/now.md
    rather than silently losing the record of a failure."""
    entry = ErrorEntry(
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat(),
        project = os.path.basename(os.path.normpath(project_dir)),
        what_tried = what_tried, why_failed = why_failed, symptom = symptom,
    )
    try:
        _append_error_line(project_dir, entry)
    except BaseException:
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M")
        _append_fallback_note(
            f"\n## {now} | continuity-fallback\n{what_tried} -- FAILED: {why_failed}\n"
        )


def prior_failures(project_dir: str, symptom_query: str) -> list[ErrorEntry]:
    path = _errors_path(project_dir)
    if not os.path.isfile(path):
        return []
    query = symptom_query.strip().lower()
    if not query:
        return []
    hits = []
    with open(path, encoding = "utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = ErrorEntry.from_dict(_json.loads(line))
            if query in entry.symptom.lower():
                hits.append(entry)
    return hits


import time
import uuid as _uuid

_DEFAULT_CHECKPOINT_RETAIN = 20


def checkpoint(project_dir: str, note: str, *, retain: int = _DEFAULT_CHECKPOINT_RETAIN) -> str:
    """Snapshot project_state.json plus a note. Returns the written filename.
    Pruned to the RETAIN most recently written checkpoints -- ordered by a
    sequence number baked into the filename, not by os.listdir's order, which
    is not creation order on every filesystem."""
    state = load_state(project_dir)
    checkpoints_dir = os.path.join(storage.ai_dir(project_dir), "checkpoints")
    existing = sorted(
        (p for p in os.listdir(checkpoints_dir) if p.endswith(".json")),
        key = lambda p: int(p.split("-", 1)[0]),
    )
    next_seq = (int(existing[-1].split("-", 1)[0]) + 1) if existing else 0
    filename = f"{next_seq:08d}-{_uuid.uuid4().hex[:8]}.json"
    payload = {
        "seq": next_seq,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": note,
        "state": state.to_dict() if state is not None else None,
    }
    storage.write_json_atomic(os.path.join(checkpoints_dir, filename), payload)

    all_now = sorted(
        (p for p in os.listdir(checkpoints_dir) if p.endswith(".json")),
        key = lambda p: int(p.split("-", 1)[0]),
    )
    if len(all_now) > retain:
        for stale in all_now[: len(all_now) - retain]:
            try:
                os.unlink(os.path.join(checkpoints_dir, stale))
            except OSError:
                pass
    return filename
