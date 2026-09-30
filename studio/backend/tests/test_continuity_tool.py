# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The continuity_task tool handler: never raises, dispatches to the six
actions, and is confined to the session's own sandbox workdir. No test here
loads a model, starts a backend, or calls a real tool -- execute() is driven
directly, the way test_delegation_tool.py drives delegation.execute directly.
"""

from __future__ import annotations

import pytest

from core.continuity import execute
from core.continuity.schemas_tool import CONTINUITY_TOOL_NAMES, CONTINUITY_TOOLS


@pytest.fixture(autouse = True)
def _sandbox(tmp_path, monkeypatch):
    import core.inference.tools as tools_module
    monkeypatch.setattr(tools_module, "_get_workdir", lambda session_id: str(tmp_path))
    yield


def test_the_schema_is_shaped_like_the_other_tools():
    function = CONTINUITY_TOOLS[0]["function"]
    assert function["name"] == "continuity_task"
    assert CONTINUITY_TOOL_NAMES == frozenset({"continuity_task"})
    # project_dir must never be a parameter a model can supply -- confinement
    # is structural (Task 8), not a value the tool call carries.
    assert "project_dir" not in function["parameters"]["properties"]


def test_status_on_an_uninitialized_sandbox_says_so_rather_than_erroring():
    out = execute("continuity_task", {"action": "status"}, session_id = "s1")
    assert isinstance(out, str)
    assert "No project state" in out or "not recorded" in out


def test_add_task_then_status_round_trips():
    out = execute("continuity_task",
                   {"action": "add_task", "id": "t1", "title": "do the thing",
                    "acceptance_criteria": ["it works"]},
                   session_id = "s1")
    assert "t1" in out
    status_out = execute("continuity_task", {"action": "status"}, session_id = "s1")
    assert "t1" in status_out


def test_an_unknown_action_returns_an_error_string_not_a_raise():
    out = execute("continuity_task", {"action": "not_a_real_action"}, session_id = "s1")
    assert isinstance(out, str)
    assert "Error" in out or "error" in out


def test_a_core_error_is_caught_and_returned_as_a_string_not_raised():
    """The never-raises boundary. set_task_status on a missing id raises
    ContinuityError inside the core (Task 3); the tool handler must catch it."""
    out = execute("continuity_task",
                   {"action": "set_status", "id": "does-not-exist", "status": "in_progress"},
                   session_id = "s1")
    assert isinstance(out, str)
    assert "Error" in out or "error" in out


def test_record_error_then_check_prior_failures_round_trips():
    out = execute("continuity_task", {"action": "record_error", "what_tried": "loose regex",
                                      "why_failed": "matched both", "symptom": "inert control"},
                  session_id = "s1")
    assert out == "Recorded."
    hits = execute("continuity_task", {"action": "check_prior_failures", "symptom": "inert"},
                   session_id = "s1")
    assert "loose regex" in hits


def test_record_error_says_so_when_the_write_failed_rather_than_claiming_success(monkeypatch):
    """A model told "Recorded." has no reason to retry or to tell the user; when
    the write failed it must hear that. And the tool must not opt in to a
    fallback_path -- that would write model-controlled text outside this
    conversation's sandbox with no approval gate on the call."""
    from core.continuity import __init__ as continuity_module

    def _boom(*a, **k):
        raise OSError("disk full")

    fallback_calls = []
    monkeypatch.setattr(continuity_module, "_append_error_line", _boom)
    monkeypatch.setattr(continuity_module, "_append_fallback_note",
                        lambda path, text: fallback_calls.append(path) or True)

    out = execute("continuity_task", {"action": "record_error", "what_tried": "x",
                                      "why_failed": "y", "symptom": "z"}, session_id = "s1")

    assert out != "Recorded."
    # The honest message's own wording, not merely "Error...": execute()'s generic
    # never-raises wrapper also starts with "Error", and a crash inside
    # record_error must not pass for an honest report.
    assert out.startswith("Error: could not record this"), out
    assert fallback_calls == [], "the app tool must never opt in to a fallback path"


def test_two_sessions_do_not_see_each_others_tasks(tmp_path, monkeypatch):
    import core.inference.tools as tools_module
    workdirs = {"s1": str(tmp_path / "s1"), "s2": str(tmp_path / "s2")}
    monkeypatch.setattr(tools_module, "_get_workdir", lambda session_id: workdirs[session_id])
    execute("continuity_task", {"action": "add_task", "id": "t1", "title": "x",
                                 "acceptance_criteria": ["y"]}, session_id = "s1")
    out = execute("continuity_task", {"action": "status"}, session_id = "s2")
    assert "t1" not in out


def test_two_threads_sharing_one_session_do_not_see_each_others_tasks(tmp_path, monkeypatch):
    """The real shared-workspace case Important-3 was found from: _get_workdir
    can return the SAME root for every session_id (a project session, or the
    global-workspace setting), unlike the keyed-dict fixture above which
    already differs per session_id. Without thread_id namespacing, two chats
    sharing that one root would collide on the same project_state.json."""
    import core.inference.tools as tools_module
    monkeypatch.setattr(tools_module, "_get_workdir", lambda session_id: str(tmp_path))
    execute("continuity_task", {"action": "add_task", "id": "t1", "title": "x",
                                 "acceptance_criteria": ["y"]},
            session_id = "shared", thread_id = "thread-a")
    out = execute("continuity_task", {"action": "status"}, session_id = "shared",
                  thread_id = "thread-b")
    assert "t1" not in out
