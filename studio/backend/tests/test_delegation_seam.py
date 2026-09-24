# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Registration, asserted by behaviour.

Being in ALL_TOOLS proves nothing: a previous tool in this project was
registered, dispatched, safe-listed and fully tested -- and unreachable in every
Studio chat, because a request-level filter stripped it. Reachability is
therefore driven through the REAL _select_request_tools.

Nothing is imported from another test module: studio/__init__.py and
studio/backend/__init__.py make a bare `tests` package resolve elsewhere under
pytest, and a collection error interrupts the whole run.
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


def test_ask_model_is_registered():
    assert "ask_model" in {t["function"]["name"] for t in tools.ALL_TOOLS}


def test_ask_model_reaches_the_model_through_the_real_route():
    assert "ask_model" in _select(_payload())


def test_an_empty_selection_stays_empty():
    assert _select(_payload(enabled_tools = [])) == []


def test_execute_tool_dispatches_to_the_delegation_handler():
    """Exactly delegation's own missing-role refusal, so the test fails if the
    dispatch reaches anything but delegation.execute. Empty arguments return
    before the loader is touched, and this deliberately goes through the audited
    execute_tool shadow, exercising that path too."""
    out = tools.execute_tool("ask_model", {})
    assert out == "Error: ask_model needs a role.", out


def test_ask_model_is_NOT_always_safe():
    """It loads models, runs a sub-agent and can edit files."""
    assert tools.is_high_risk_tool_call("ask_model", {"role": "coding", "task": "t"}) is True


def test_a_read_only_tool_is_still_not_high_risk():
    """Control: the assertion above is not passing because everything is high risk."""
    assert tools.is_high_risk_tool_call("check_tool_readiness", {}) is False


def test_ask_model_is_absent_from_the_anthropic_unprompted_set():
    inference = pytest.importorskip("routes.inference", reason = "inference stack not installed")
    assert "ask_model" not in inference._ANTHROPIC_UNPROMPTED_SAFE_TOOLS


def test_the_roles_router_is_mounted():
    """Driven through app.openapi(), not app.routes.

    The plan asserted `any(r.path == "/api/model-roles" for r in app.routes)`.
    That cannot hold on FastAPI 0.141: include_router is lazy, so app.routes
    holds _IncludedRouter objects with no `.path` at all and the routers are
    flattened per request. The pre-existing /api/tool-audit mount fails that
    assertion too, which is how the plan's version was caught -- it would have
    stayed red however correctly main.py was edited. The OpenAPI paths are the
    public, version-stable view of what is actually served, and they still go
    red if the include_router line is removed."""
    from main import app

    assert "/api/model-roles" in app.openapi().get("paths", {})


def test_the_seams_stay_additive():
    repo = Path(tools.__file__).parents[4]
    base = subprocess.run(["git", "merge-base", "origin/main", "HEAD"], cwd = repo,
                          capture_output = True, text = True, check = True).stdout.strip()
    for path, ceiling in (
        ("studio/backend/core/inference/tools.py", 100),
        ("studio/backend/routes/inference.py", 70),
        ("studio/backend/main.py", 10),
    ):
        stat = subprocess.run(["git", "diff", "--numstat", base, "--", path], cwd = repo,
                              capture_output = True, text = True, check = True).stdout.split()
        assert stat, f"no diff recorded for {path}"
        insertions, deletions = int(stat[0]), int(stat[1])
        assert deletions == 0, f"{path} lost {deletions} line(s); the seam must be additive"
        assert insertions <= ceiling, f"{path} grew to {insertions}"
