# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""find_capability's five registration points, each asserted by BEHAVIOUR.

Reachability drives the REAL _select_request_tools with a Studio-shaped payload.
Being in ALL_TOOLS proves nothing: check_tool_readiness was in ALL_TOOLS and
unreachable in every Studio chat, and a test replicating the route's filter was
inert under mutation.
"""

from __future__ import annotations

import asyncio
import types
from pathlib import Path

import pytest

from core.inference import tool_audit
from core.inference import tool_readiness as tr
from core.inference import tools
from storage import tool_audit_db
from tests.test_tool_readiness_seam import _anthropic_gate_genexp

_ROUTE_FILE = Path(tools.__file__).parents[2] / "routes" / "inference.py"
_STUDIO_ALLOWLIST = ["web_search", "python", "terminal", "edit_file"]


@pytest.fixture(autouse = True)
def _isolated(tmp_path, monkeypatch):
    """execute_tool is the audit shadow; without this, calling it writes audit rows
    into the user's real studio.db -- silently, because the recorder never raises."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    tool_audit_db.reset_for_tests()
    tool_audit.reset_degraded_for_tests()
    tr.reset_for_tests()
    yield
    tr.reset_for_tests()


def _payload(**over):
    base = dict(
        enabled_tools = list(_STUDIO_ALLOWLIST),
        rag_scope = None,
        thread_id = None,
        bypass_permissions = False,
    )
    base.update(over)
    return types.SimpleNamespace(**base)


def _select(payload, **kwargs):
    import routes.inference as routes_mod

    kwargs.setdefault("tools_on", True)
    kwargs.setdefault("mcp_allowed", False)
    return [t["function"]["name"] for t in asyncio.run(routes_mod._select_request_tools(payload, **kwargs))]


# --- 1. reachability (the _addable re-add) ----------------------------------


def test_find_capability_reaches_the_model_through_the_real_route():
    assert "find_capability" in _select(_payload())


def test_the_narrowest_studio_allowlist_still_carries_it():
    assert "find_capability" in _select(_payload(enabled_tools = ["web_search"]))


def test_an_empty_selection_stays_empty_through_the_real_route():
    """The `tools_on and tools` gate is load-bearing; this tool must not break it."""
    assert _select(_payload(enabled_tools = [])) == []


# --- 2 and 3. ALL_TOOLS + dispatch ------------------------------------------


def test_execute_tool_dispatches_to_the_capability_map():
    out = tools.execute_tool("find_capability", {})
    assert out.startswith("Capability map"), out[:200]


def test_execute_tool_answers_one_capability():
    out = tools.execute_tool("find_capability", {"capability": "ocr"})
    assert "CAPABILITY: ocr" in out


# --- 4. _ALWAYS_SAFE_TOOLS ---------------------------------------------------


def test_find_capability_is_not_treated_as_high_risk():
    assert tools.is_high_risk_tool_call("find_capability", {}) is False


def test_an_unknown_tool_is_still_high_risk():
    """Control: the assertion above is not passing because everything is safe."""
    assert tools.is_high_risk_tool_call("zz_not_a_real_tool", {}) is True


# --- 5. the Anthropic Messages channel --------------------------------------


@pytest.fixture
def _inference_routes():
    return pytest.importorskip("routes.inference", reason = "inference stack not installed")


def test_find_capability_is_unprompted_on_the_anthropic_channel(_inference_routes):
    inf = _inference_routes
    run = types.FunctionType(_anthropic_gate_genexp(inf), vars(inf))
    assert any(run(iter([{"function": {"name": "find_capability"}}]))) is False


def test_terminal_still_trips_the_anthropic_gate(_inference_routes):
    """Control: the genexp is not answering False for everything."""
    inf = _inference_routes
    run = types.FunctionType(_anthropic_gate_genexp(inf), vars(inf))
    assert any(run(iter([{"function": {"name": "terminal"}}]))) is True


def test_the_upstream_anthropic_literal_was_not_edited(_inference_routes):
    source = _ROUTE_FILE.read_text(encoding = "utf-8")
    literal = source.split("_ANTHROPIC_UNPROMPTED_SAFE_TOOLS = frozenset(", 1)[1].split(")", 1)[0]
    assert "find_capability" not in literal, "upstream's literal was edited"


# --- readiness integration ---------------------------------------------------


def test_the_readiness_report_calls_find_capability_ready_not_unknown():
    """Without this, adding the tool makes check_tool_readiness's full report say
    'unknown -- nobody checked' about a discovery tool that plainly works."""
    out = tr.execute("check_tool_readiness", {"tool": "find_capability"})
    assert out.split()[1] == tr.READY, out
