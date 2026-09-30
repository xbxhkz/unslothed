# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Renders project_state.json + task_queue.json into the two files a human (or
a fresh context) reads first.

Both outputs are GENERATED, never hand-authored: MEMORY.md already does
index-first progressive disclosure for durable cross-session facts, so these
render from the structured state rather than becoming a second thing someone
has to remember to update by hand.

Token counting is approximate: len(text) // 4, the common rule-of-thumb ratio
for English text. This is not a real tokenizer and is not meant to be exact --
it exists only to keep render_context_summary inside its budget, and the cap
is generous enough (1500) that the approximation's error does not matter.
"""

from __future__ import annotations

from core.continuity.schemas import ContinuityError, TaskQueue

_TARGET_TOKENS = 800
_HARD_CAP_TOKENS = 1500


def _approx_tokens(text: str) -> int:
    return len(text) // 4


def _completion_percent(queue: TaskQueue) -> int:
    """Derived from the live task queue, never read from ProjectState's own
    completion_percent field -- nothing in this package ever updates that
    field after init, so a stored copy can only go stale the moment the
    first task completes. Same "no second source of truth" principle
    already applied to "ready" status (Task 3)."""
    if not queue.tasks:
        return 0
    completed = sum(1 for t in queue.tasks if t.status == "complete")
    return round(100 * completed / len(queue.tasks))


def render_context_summary(project_dir: str) -> str:
    """~300-800 tokens target, 1500 hard cap. Raises ContinuityError rather
    than silently returning an oversized string if it cannot fit even after
    truncating the lists -- a context-reset summary that can silently blow its
    own budget has failed at the one thing it exists to guarantee."""
    from core.continuity import load_state, load_tasks, ready_tasks

    state = load_state(project_dir)
    if state is None:
        return "No project state recorded yet (.ai/project_state.json does not exist)."

    ready = ready_tasks(project_dir)
    queue = load_tasks(project_dir)
    in_progress = [t for t in queue.tasks if t.status == "in_progress"]

    def _fmt_list(items: list[str], limit: int) -> str:
        shown = items[:limit]
        text = "\n".join(f"- {i}" for i in shown)
        if len(items) > limit:
            text += f"\n- ... and {len(items) - limit} more"
        return text or "(none)"

    limit = 10
    while True:
        lines = [
            f"# {state.project} -- {state.status}",
            f"Phase: {state.current_phase}",
            f"Current task: {state.current_task or '(none)'}",
            f"Completion: {_completion_percent(queue)}%",
            "",
            "## Next action",
            state.next_action or "(not recorded)",
            "",
            "## In progress",
            _fmt_list([t.id for t in in_progress], limit),
            "",
            "## Ready to start",
            _fmt_list([t.id for t in ready], limit),
            "",
            "## Blocking issues",
            _fmt_list(state.blocking_issues, limit),
        ]
        summary = "\n".join(lines)
        if _approx_tokens(summary) <= _HARD_CAP_TOKENS:
            return summary
        if limit <= 1:
            raise ContinuityError(
                f"context summary for {state.project!r} exceeds the {_HARD_CAP_TOKENS}-token "
                "cap even at minimum truncation -- the underlying state is too large to summarize "
                "safely; trim blocking_issues or the task queue directly"
            )
        limit = max(1, limit // 2)


def render_project_state_md(project_dir: str) -> str:
    """A fuller, human-readable render -- no token cap, this is the level-2
    disclosure in the spec's progressive-disclosure model, read deliberately
    rather than loaded automatically."""
    from core.continuity import load_state, load_tasks

    state = load_state(project_dir)
    queue = load_tasks(project_dir)
    lines = ["# Project State", ""]
    if state is None:
        lines.append("(no project_state.json recorded yet)")
    else:
        lines += [
            f"**Project:** {state.project}",
            f"**Status:** {state.status}",
            f"**Phase:** {state.current_phase}",
            f"**Completion:** {_completion_percent(queue)}%",
            f"**Next action:** {state.next_action or '(not recorded)'}",
            "",
            "## Modified files",
            *([f"- {p}" for p in state.modified_files] or ["(none recorded)"]),
            "",
            "## Blocking issues",
            *([f"- {b}" for b in state.blocking_issues] or ["(none)"]),
        ]
    lines += ["", "## Tasks"]
    for t in queue.tasks:
        deps = f" (depends on {', '.join(t.depends_on)})" if t.depends_on else ""
        lines.append(f"- [{t.status}] {t.id}: {t.title}{deps}")
    if not queue.tasks:
        lines.append("(no tasks recorded)")
    return "\n".join(lines)
