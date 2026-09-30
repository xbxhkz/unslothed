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
    # Long enough that limit=10 alone exceeds the 1500-token (~6000 char) cap --
    # the previous version of this test used short items that never forced a
    # shrink iteration at all, making the cap check's own removal undetectable.
    huge = [f"blocker number {i}: " + ("x" * 700) for i in range(500)]
    write_state(str(tmp_path), _state(blocking_issues = huge))
    summary = render_context_summary(str(tmp_path))
    assert _approx_tokens(summary) <= 1500


def test_render_context_summary_raises_when_nothing_can_make_it_fit(tmp_path):
    """The other end of the shrink loop: even minimum truncation (limit=1)
    can't bring this under the cap, so it must raise rather than silently
    return an oversized string. Previously untested -- the existing huge-list
    test's data was too small to ever force a shrink at all, let alone reach
    the point where shrinking is exhausted."""
    write_state(str(tmp_path), _state(blocking_issues = ["x" * 8000]))
    with pytest.raises(ContinuityError):
        render_context_summary(str(tmp_path))


def test_render_project_state_md_lists_tasks(tmp_path):
    write_state(str(tmp_path), _state())
    add_task(str(tmp_path), Task(id = "t1", title = "the first task", status = "pending",
                                  acceptance_criteria = ["ok"]))
    md = render_project_state_md(str(tmp_path))
    assert "the first task" in md


def test_completion_percent_is_derived_from_the_live_queue_not_the_stale_stored_field(tmp_path):
    """ProjectState.completion_percent is never updated after init by anything
    in this package -- a render that trusted it would show 0% forever, no
    matter how many tasks actually completed. completion_percent = 10 here is
    a decoy: if the render ever regresses to reading it directly, this test
    catches it via the wrong percentage rather than a coincidentally-matching
    stale value."""
    from core.continuity import set_task_status

    write_state(str(tmp_path), _state(completion_percent = 10))
    add_task(str(tmp_path), Task(id = "t1", title = "one", status = "pending",
                                  acceptance_criteria = ["ok"]))
    add_task(str(tmp_path), Task(id = "t2", title = "two", status = "pending",
                                  acceptance_criteria = ["ok"]))
    set_task_status(str(tmp_path), "t1", "in_progress")
    set_task_status(str(tmp_path), "t1", "complete")

    summary = render_context_summary(str(tmp_path))
    assert "Completion: 50%" in summary
    assert "Completion: 10%" not in summary

    md = render_project_state_md(str(tmp_path))
    assert "**Completion:** 50%" in md
