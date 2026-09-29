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


def test_two_sessions_do_not_see_each_others_tasks(tmp_path, monkeypatch):
    import core.inference.tools as tools_module
    workdirs = {"s1": str(tmp_path / "s1"), "s2": str(tmp_path / "s2")}
    monkeypatch.setattr(tools_module, "_get_workdir", lambda session_id: workdirs[session_id])
    execute("continuity_task", {"action": "add_task", "id": "t1", "title": "x",
                                 "acceptance_criteria": ["y"]}, session_id = "s1")
    out = execute("continuity_task", {"action": "status"}, session_id = "s2")
    assert "t1" not in out
