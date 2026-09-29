# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""context_summary.md must fit its cap -- that is the whole point of a
context-reset summary: a caller must be able to trust it never blows the
budget it exists to protect. Token count is approximated as len(text) // 4
(documented in render.py's docstring); no tokenizer dependency."""

from __future__ import annotations

import pytest

from core.continuity import add_task, checkpoint, update_state, write_state
from core.continuity.render import _approx_tokens, render_context_summary, render_project_state_md
from core.continuity.schemas import ContinuityError, ProjectState, Task


def _state(**over):
    base = dict(schema_version = 1, project = "demo", status = "active",
                current_phase = "p1", current_task = None, completion_percent = 10,
                last_checkpoint = None, next_action = "do the next thing")
    base.update(over)
    return ProjectState(**base)


def test_render_context_summary_on_empty_project_is_short(tmp_path):
    write_state(str(tmp_path), _state())
    summary = render_context_summary(str(tmp_path))
    assert _approx_tokens(summary) <= 1500
    assert "demo" in summary


def test_render_context_summary_includes_next_action_and_blockers(tmp_path):
    write_state(str(tmp_path), _state(
        next_action = "fix the thing", blocking_issues = ["waiting on X"],
    ))
    summary = render_context_summary(str(tmp_path))
    assert "fix the thing" in summary
    assert "waiting on X" in summary


def test_render_context_summary_caps_even_with_a_huge_blocking_issues_list(tmp_path):
    huge = [f"blocker number {i} with a fair bit of explanatory text attached" for i in range(500)]
    write_state(str(tmp_path), _state(blocking_issues = huge))
    summary = render_context_summary(str(tmp_path))
    assert _approx_tokens(summary) <= 1500


def test_render_project_state_md_lists_tasks(tmp_path):
    write_state(str(tmp_path), _state())
    add_task(str(tmp_path), Task(id = "t1", title = "the first task", status = "pending",
                                  acceptance_criteria = ["ok"]))
    md = render_project_state_md(str(tmp_path))
    assert "the first task" in md
