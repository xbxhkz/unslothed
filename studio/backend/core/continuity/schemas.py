# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The continuity engine's data shapes.

Every dataclass here round-trips through to_dict()/from_dict() rather than a
generic asdict(): from_dict is where a missing or wrong-typed field becomes a
ContinuityError instead of a confusing KeyError three calls later, at whichever
place first happens to read the missing field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


class ContinuityError(Exception):
    """Raised by every core.continuity function on a bad read, a bad write, an
    illegal status transition, or a render that cannot fit its cap. The core
    never guesses at recovery in place of raising -- that is each caller's own
    decision to make, not this module's."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _require(d: dict, key: str, expected_type: type):
    if key not in d:
        raise ContinuityError(f"missing required field {key!r}")
    value = d[key]
    if not isinstance(value, expected_type):
        raise ContinuityError(
            f"field {key!r} must be {expected_type.__name__}, got {type(value).__name__}"
        )
    return value


@dataclass
class ProjectState:
    schema_version: int
    project: str
    status: str
    current_phase: str
    current_task: Optional[str]
    completion_percent: int
    last_checkpoint: Optional[str]
    completed: list[str] = field(default_factory = list)
    in_progress: list[str] = field(default_factory = list)
    remaining: list[str] = field(default_factory = list)
    next_action: str = ""
    modified_files: list[str] = field(default_factory = list)
    tests: dict = field(default_factory = dict)
    blocking_issues: list[str] = field(default_factory = list)
    important_decisions: list[str] = field(default_factory = list)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version, "project": self.project,
            "status": self.status, "current_phase": self.current_phase,
            "current_task": self.current_task,
            "completion_percent": self.completion_percent,
            "last_checkpoint": self.last_checkpoint, "completed": self.completed,
            "in_progress": self.in_progress, "remaining": self.remaining,
            "next_action": self.next_action, "modified_files": self.modified_files,
            "tests": self.tests, "blocking_issues": self.blocking_issues,
            "important_decisions": self.important_decisions,
        }

    @staticmethod
    def from_dict(d: dict) -> "ProjectState":
        return ProjectState(
            schema_version = _require(d, "schema_version", int),
            project = _require(d, "project", str),
            status = _require(d, "status", str),
            current_phase = _require(d, "current_phase", str),
            current_task = d.get("current_task"),
            completion_percent = _require(d, "completion_percent", int),
            last_checkpoint = d.get("last_checkpoint"),
            completed = list(d.get("completed") or []),
            in_progress = list(d.get("in_progress") or []),
            remaining = list(d.get("remaining") or []),
            next_action = d.get("next_action") or "",
            modified_files = list(d.get("modified_files") or []),
            tests = dict(d.get("tests") or {}),
            blocking_issues = list(d.get("blocking_issues") or []),
            important_decisions = list(d.get("important_decisions") or []),
        )


@dataclass
class Task:
    id: str
    title: str
    status: str
    depends_on: list[str] = field(default_factory = list)
    acceptance_criteria: list[str] = field(default_factory = list)
    owner: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id, "title": self.title, "status": self.status,
            "depends_on": self.depends_on,
            "acceptance_criteria": self.acceptance_criteria, "owner": self.owner,
        }

    @staticmethod
    def from_dict(d: dict) -> "Task":
        return Task(
            id = _require(d, "id", str), title = _require(d, "title", str),
            status = _require(d, "status", str),
            depends_on = list(d.get("depends_on") or []),
            acceptance_criteria = list(d.get("acceptance_criteria") or []),
            owner = d.get("owner"),
        )


@dataclass
class TaskQueue:
    schema_version: int
    tasks: list[Task] = field(default_factory = list)

    def to_dict(self) -> dict:
        return {"schema_version": self.schema_version,
                "tasks": [t.to_dict() for t in self.tasks]}

    @staticmethod
    def from_dict(d: dict) -> "TaskQueue":
        return TaskQueue(
            schema_version = _require(d, "schema_version", int),
            tasks = [Task.from_dict(t) for t in (d.get("tasks") or [])],
        )


@dataclass
class ErrorEntry:
    ts: str
    project: str
    what_tried: str
    why_failed: str
    symptom: str
    do_not_repeat: bool = True

    def to_dict(self) -> dict:
        return {
            "ts": self.ts, "project": self.project, "what_tried": self.what_tried,
            "why_failed": self.why_failed, "symptom": self.symptom,
            "do_not_repeat": self.do_not_repeat,
        }

    @staticmethod
    def from_dict(d: dict) -> "ErrorEntry":
        return ErrorEntry(
            ts = _require(d, "ts", str), project = _require(d, "project", str),
            what_tried = _require(d, "what_tried", str),
            why_failed = _require(d, "why_failed", str),
            symptom = _require(d, "symptom", str),
            do_not_repeat = bool(d.get("do_not_repeat", True)),
        )
