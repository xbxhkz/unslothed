# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The continuity engine's public API.

Every function here operates on a project_dir the caller names explicitly.
Nothing in this module resolves a path on its own -- the app-side tool (added
in a later task) is what confines project_dir to a sandbox; this module trusts
whatever path it is given, the same way storage.py does.

The core RAISES ContinuityError rather than guessing at recovery. Never-raises
is each caller's own decision (the CLI catches and prints; the app tool catches
and returns a string) -- this module swallows exceptions in exactly two
places: record_error with its fallback note (documented at record_error), and
execute(), the app tool's own never-raises boundary.
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


def _errors_path(project_dir: str) -> str:
    return os.path.join(storage.ai_dir(project_dir), "logs", _ERRORS_FILENAME)


def _append_error_line(project_dir: str, entry: ErrorEntry) -> None:
    path = _errors_path(project_dir)
    os.makedirs(os.path.dirname(path), exist_ok = True)
    with open(path, "a", encoding = "utf-8") as f:
        f.write(_json.dumps(entry.to_dict()) + "\n")


def _append_fallback_note(path: str, text: str) -> bool:
    """Last resort: the primary write already failed. Losing the record of a
    failure is bad; crashing the caller over it is worse, so this stays
    never-raises -- but it reports whether the note landed rather than
    pretending it always does."""
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok = True)
        with open(path, "a", encoding = "utf-8") as f:
            f.write(text)
        return True
    except BaseException:
        return False


def record_error(
    project_dir: str, *, what_tried: str, why_failed: str, symptom: str,
    fallback_path: str | None = None,
) -> bool:
    """Never raises. Returns True if the entry landed somewhere findable (the
    primary errors.jsonl, or -- only when the caller explicitly opts in by
    passing fallback_path -- that file), False if it was lost entirely.

    fallback_path is caller-supplied, not a hardcoded default: this function
    has two real callers with different trust levels. The CLI runs as the
    user on their own machine and may reasonably pass ~/odysseus/.remember/now.md
    (still their own file, in their own home directory). The app-side tool is
    driven by a model with no approval gate on THIS call, so it must pass
    nothing -- writing model-controlled text outside the conversation's own
    sandbox on a write failure would be an exfiltration channel, not a safety
    net. When fallback_path is None and the primary write fails, the entry is
    genuinely lost, and this function says so honestly rather than claiming
    success.
    """
    entry = ErrorEntry(
        ts = datetime.datetime.now(datetime.timezone.utc).isoformat(),
        project = os.path.basename(os.path.normpath(project_dir)),
        what_tried = what_tried, why_failed = why_failed, symptom = symptom,
    )
    try:
        _append_error_line(project_dir, entry)
        return True
    except BaseException:
        if fallback_path is None:
            return False
        now = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M")
        return _append_fallback_note(
            fallback_path,
            f"\n## {now} | continuity-fallback\n{what_tried} -- FAILED: {why_failed}\n",
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


def _find_cycle(tasks: list[Task]) -> list[str] | None:
    """DFS cycle detection over depends_on edges. Returns the cycle's task ids,
    or None. Bounded by len(tasks) recursion depth, which this project's task
    graphs are nowhere near large enough to make a concern."""
    by_id = {t.id: t for t in tasks}
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {t.id: WHITE for t in tasks}
    stack: list[str] = []

    def visit(tid: str) -> list[str] | None:
        color[tid] = GRAY
        stack.append(tid)
        for dep in by_id.get(tid, Task(id = tid, title = "", status = "")).depends_on:
            if dep not in by_id:
                continue  # unknown ids are validate's own separate problem
            if color.get(dep) == GRAY:
                cycle_start = stack.index(dep)
                return stack[cycle_start:] + [dep]
            if color.get(dep) == WHITE:
                found = visit(dep)
                if found:
                    return found
        stack.pop()
        color[tid] = BLACK
        return None

    for t in tasks:
        if color[t.id] == WHITE:
            found = visit(t.id)
            if found:
                return found
    return None


def validate(project_dir: str) -> list[str]:
    """Read-only. Never raises -- a validator that can crash on the exact data
    it exists to check is not trustworthy. Problems are returned as plain
    strings; there is no downstream consumer yet that needs structure richer
    than 'read this to a human'."""
    problems: list[str] = []

    try:
        state = load_state(project_dir)
    except (ContinuityError, TypeError, AttributeError, KeyError) as exc:
        # Broader than ContinuityError on purpose: read_json returns whatever
        # valid JSON it finds (e.g. a bare 42), and from_dict's _require then
        # raises TypeError/AttributeError/KeyError trying to treat a non-dict
        # as a dict, not ContinuityError. validate's job is to survive that
        # too -- ContinuityError alone is correct for every other caller of
        # load_state/load_tasks (they should get a hard failure on malformed
        # data), but validate is specifically the "bulletproof against
        # garbage input" function, so its own net is wider than theirs.
        problems.append(f"project_state.json: {exc}")
        state = None

    try:
        tasks = load_tasks(project_dir).tasks
    except (ContinuityError, TypeError, AttributeError, KeyError) as exc:
        problems.append(f"task_queue.json: {exc}")
        tasks = []

    ids = [t.id for t in tasks]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        problems.append(f"duplicate task id(s): {sorted(dupes)}")

    known = set(ids)
    for t in tasks:
        for dep in t.depends_on:
            if dep not in known:
                problems.append(f"task {t.id!r} depends on unknown id {dep!r}")

    cycle = _find_cycle(tasks)
    if cycle:
        problems.append(f"dependency cycle: {' -> '.join(cycle)}")

    if state is not None and state.current_task and state.current_task not in known:
        problems.append(
            f"project_state.json names current_task {state.current_task!r}, "
            "which is not in task_queue.json"
        )

    return problems


def repair(project_dir: str) -> list[str]:
    """Regenerates ONLY derived data: the rendered .md files. Never touches a
    task's status, never adds an acceptance criterion, never marks anything
    complete -- see spec section 28. Readiness itself needs no repair, since it
    was never stored (Task 3)."""
    from core.continuity import render

    actions: list[str] = []
    summary = render.render_context_summary(project_dir)
    _write_text(os.path.join(storage.ai_dir(project_dir), "context_summary.md"), summary)
    actions.append("regenerated context_summary.md")

    state_md = render.render_project_state_md(project_dir)
    _write_text(os.path.join(storage.ai_dir(project_dir), "project_state.md"), state_md)
    actions.append("regenerated project_state.md")

    return actions


def _write_text(path: str, text: str) -> None:
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding = "utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def execute(name: str, arguments, *, session_id: str | None = None, **kwargs) -> str:
    """Handler for continuity_task. Never raises -- a continuity read/write
    failure must degrade the turn, not break it, the same never-raises
    boundary every other built-in tool in this fork keeps at its execute()."""
    try:
        return _execute(arguments, session_id = session_id)
    except BaseException as exc:  # noqa: BLE001 - a continuity failure must not break the turn
        return f"Error: continuity_task failed: {exc}"


def _execute(arguments, *, session_id: str | None) -> str:
    from core.continuity.sandbox import confine_project_dir

    args = arguments if isinstance(arguments, dict) else {}
    action = str(args.get("action") or "").strip()
    project_dir = confine_project_dir(session_id)

    if action == "status":
        from core.continuity.render import render_context_summary
        summary = render_context_summary(project_dir)
        if load_state(project_dir) is None:
            # continuity_task exposes no write_state/init action, so a
            # session that only ever calls add_task never gets a
            # project_state.json -- render_context_summary's early return
            # (Task 7) would then hide every task it has recorded. Fall back
            # to a plain task list so status still reflects what was tracked.
            tasks = load_tasks(project_dir).tasks
            if tasks:
                lines = [summary, "", "## Tasks"]
                lines += [f"- [{t.status}] {t.id}: {t.title}" for t in tasks]
                summary = "\n".join(lines)
        return summary

    if action == "add_task":
        task_id = str(args.get("id") or "").strip()
        if not task_id:
            return "Error: add_task needs an id."
        add_task(project_dir, Task(
            id = task_id, title = str(args.get("title") or task_id), status = "pending",
            depends_on = list(args.get("depends_on") or []),
            acceptance_criteria = list(args.get("acceptance_criteria") or []),
        ))
        return f"Added task {task_id!r}."

    if action == "set_status":
        task_id = str(args.get("id") or "").strip()
        status = str(args.get("status") or "").strip()
        if not task_id or not status:
            return "Error: set_status needs an id and a status."
        set_task_status(project_dir, task_id, status)
        return f"Task {task_id!r} is now {status!r}."

    if action == "record_error":
        # No fallback_path, deliberately -- see record_error's docstring.
        recorded = record_error(
            project_dir,
            what_tried = str(args.get("what_tried") or ""),
            why_failed = str(args.get("why_failed") or ""),
            symptom = str(args.get("symptom") or ""),
        )
        return "Recorded." if recorded else (
            "Error: could not record this -- the write failed and there is nowhere else "
            "to put it. The failure itself was not lost silently; it just could not be "
            "written down."
        )

    if action == "check_prior_failures":
        hits = prior_failures(project_dir, str(args.get("symptom") or ""))
        if not hits:
            return "No prior recorded failures match that."
        return "\n".join(f"- {h.what_tried} -- {h.why_failed}" for h in hits)

    if action == "checkpoint":
        name = checkpoint(project_dir, str(args.get("note") or ""))
        return f"Checkpointed as {name}."

    return f"Error: unknown continuity_task action {action!r}."
