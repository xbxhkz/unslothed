# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Registration, asserted by behaviour, not by "is it in ALL_TOOLS" -- that
alone has passed while a tool was unreachable in Studio three times in this
project's history. Reachability is driven through the REAL
_select_request_tools, exactly as test_delegation_seam.py does.
"""

from __future__ import annotations

import asyncio
import subprocess
import types
from pathlib import Path

import pytest

from core.inference import tool_audit
from core.inference import tools
from storage import tool_audit_db

_STUDIO_ALLOWLIST = ["web_search", "python", "terminal", "edit_file"]


@pytest.fixture(autouse = True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    tool_audit_db.reset_for_tests()
    tool_audit.reset_degraded_for_tests()
    yield


def _payload(**over):
    base = dict(enabled_tools = list(_STUDIO_ALLOWLIST), rag_scope = None,
                thread_id = None, bypass_permissions = False)
    base.update(over)
    return types.SimpleNamespace(**base)


def _select(payload, **kwargs):
    import routes.inference as routes_mod

    kwargs.setdefault("tools_on", True)
    kwargs.setdefault("mcp_allowed", False)
    return [t["function"]["name"]
            for t in asyncio.run(routes_mod._select_request_tools(payload, **kwargs))]


def test_continuity_task_is_registered():
    assert "continuity_task" in {t["function"]["name"] for t in tools.ALL_TOOLS}


def test_continuity_task_reaches_the_model_through_the_real_route():
    """continuity_task has no UI pill, exactly like check_tool_readiness and
    find_capability -- without the _addable re-add it would be registered,
    dispatched, safe-listed, fully tested, and unreachable in Studio, which
    has happened three times in this project already."""
    assert "continuity_task" in _select(_payload())


def test_execute_tool_dispatches_to_the_continuity_handler(tmp_path, monkeypatch):
    import core.inference.tools as tools_module
    monkeypatch.setattr(tools_module, "_get_workdir", lambda session_id: str(tmp_path))
    out = tools.execute_tool("continuity_task", {"action": "status"})
    assert "No project state" in out or "not recorded" in out


def test_continuity_task_is_always_safe():
    """Unlike ask_model, this reads and writes only local sandboxed files --
    no network, no code execution -- so it belongs on the safe list rather
    than needing a confirmation prompt."""
    assert tools.is_high_risk_tool_call("continuity_task", {"action": "status"}) is False


def test_continuity_task_is_present_in_the_anthropic_unprompted_set():
    inference = pytest.importorskip("routes.inference", reason = "inference stack not installed")
    assert "continuity_task" in inference._ANTHROPIC_UNPROMPTED_SAFE_TOOLS


def test_the_seams_stay_additive():
    repo = Path(tools.__file__).parents[4]
    base = subprocess.run(["git", "merge-base", "origin/main", "HEAD"], cwd = repo,
                          capture_output = True, text = True, check = True).stdout.strip()
    for path, ceiling in (
        ("studio/backend/core/inference/tools.py", 140),
        ("studio/backend/routes/inference.py", 80),
        ("studio/backend/main.py", 10),
    ):
        stat = subprocess.run(["git", "diff", "--numstat", base, "--", path], cwd = repo,
                              capture_output = True, text = True, check = True).stdout.split()
        assert stat, f"no diff recorded for {path}"
        insertions, deletions = int(stat[0]), int(stat[1])
        assert deletions == 0, f"{path} lost {deletions} line(s); the seam must be additive"
        assert insertions <= ceiling, f"{path} grew to {insertions}"
