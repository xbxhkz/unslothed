# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The check_tool_readiness tool, its dispatch, and the _ALWAYS_SAFE_TOOLS rebind.

The rebind test is behavioural on purpose. It depends on _ALWAYS_SAFE_TOOLS being
read as a module global at CALL time; if that were wrong the rebind would
silently do nothing, every readiness query would prompt the user for approval,
and no other test would notice. Asserting the rebind's presence rather than its
effect is exactly the inert control this project has produced nine times.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from core.inference import tool_audit
from core.inference import tools
from storage import tool_audit_db


@pytest.fixture(autouse = True)
def _isolated(tmp_path, monkeypatch):
    """REQUIRED, and not optional here.

    execute_tool is the audit shadow, so calling it writes an audit row. Without
    this redirect those rows land in the real ~/.unsloth/studio/studio.db -- and
    because the recorder is guarded to never raise, it would do so silently.
    The existing test_tool_audit_seam.py needs no fixture only because its checks
    are static; these actually invoke the tool.
    """
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    tool_audit_db.reset_for_tests()
    tool_audit.reset_degraded_for_tests()
    yield


def test_the_tool_is_registered_in_ALL_TOOLS():
    """INERT ON ITS OWN. Kept because it localises a failure, but note that it
    passed for the entire time the feature was unreachable in the product: being
    in ALL_TOOLS says nothing about surviving the route's enabled_tools filter.
    test_the_schema_survives_a_realistic_studio_payload below is the real one."""
    names = {t["function"]["name"] for t in tools.ALL_TOOLS}
    assert "check_tool_readiness" in names


# --- the route seam ---------------------------------------------------------
# Approach: REPLICATE the route's filter-then-re-add against the real ALL_TOOLS,
# rather than calling _select_request_tools.
#
# Why: the route helper is async, takes a pydantic ChatCompletionRequest, and
# before it returns it calls _thread_has_conversation_archive (a studio.db read)
# and awaits get_enabled_mcp_tools (an async DB read that network-probes every
# enabled MCP server). Driving all of that would test the DB and the MCP stack,
# not the filter.
#
# A replica can drift from the original, which would make it another inert
# control. So the one line that matters is NOT duplicated: _route_addable() reads
# the real `_addable = ...` expression out of routes/inference.py and evaluates
# it. Delete READINESS_TOOL_NAMES from that line and the payload test below fails
# directly, because it is using that line. (An earlier revision hardcoded the
# union here and a second test pinned it; the mutation proved the payload test
# itself still passed, which is the ninth inert control's exact shape.)

# tools.py is studio/backend/core/inference/tools.py, so parents[2] is backend/.
_ROUTE = Path(tools.__file__).parents[2] / "routes" / "inference.py"

# What Studio's pill-driven literal in chat-adapter.ts actually sends: upstream
# tool names only. This is the payload that made the feature dead.
_STUDIO_ENABLED_TOOLS = {"terminal", "python", "web_search"}


def _route_addable() -> frozenset:
    """The route's OWN `_addable` expression, read from source and evaluated.

    Not a copy of it. This is what keeps the replica honest: whatever names the
    real line unions together are the names these tests exercise."""
    from core.inference import tools as _t

    source = _ROUTE.read_text(encoding = "utf-8")
    lines = [ln for ln in source.splitlines() if ln.strip().startswith("_addable =")]
    assert len(lines) == 1, f"expected one _addable line in the route, found {len(lines)}"
    expression = lines[0].split("=", 1)[1].strip()
    # Resolved by name, not eval()'d: the route builds this from `A | B | C`,
    # where each part is a name imported out of core.inference.tools. Splitting
    # on "|" and looking each part up keeps the test to exactly the shape the
    # route is allowed to have, and an expression of any other shape fails loudly
    # here rather than being executed.
    parts = [p.strip() for p in expression.split("|")]
    assert all(p.isidentifier() for p in parts), f"unexpected _addable shape: {expression}"
    resolved = frozenset()
    for part in parts:
        names = getattr(_t, part, None)
        assert names is not None, f"_addable names {part}, which core.inference.tools lacks"
        resolved = resolved | frozenset(names)
    return resolved


def _select_like_the_route(enabled_tools: set[str] | None, *, tools_on: bool = True):
    """The filter-then-re-add from _select_request_tools, verbatim in shape."""
    from core.inference.tools import ALL_TOOLS

    if not tools_on:
        selected = []
    elif enabled_tools is not None:
        selected = [t for t in ALL_TOOLS if t["function"]["name"] in enabled_tools]
    else:
        selected = list(ALL_TOOLS)
    if tools_on and selected:
        _addable = _route_addable()
        _already = {t["function"]["name"] for t in selected}
        selected = selected + [
            t for t in ALL_TOOLS
            if t["function"]["name"] in _addable and t["function"]["name"] not in _already
        ]
    return {t["function"]["name"] for t in selected}


def test_the_schema_survives_a_realistic_studio_payload():
    """THE load-bearing route test. With every Studio pill ON the model received
    16 schemas and check_tool_readiness was not one of them: the filter dropped
    it (no pill names it) and the re-add block never listed it. The tool could
    not be called at all, in the product, for the whole branch."""
    assert "check_tool_readiness" in _select_like_the_route(_STUDIO_ENABLED_TOOLS)


def test_the_readiness_tool_is_not_injected_into_an_empty_selection():
    """The `and tools` gate is load-bearing: an empty selection must stay empty,
    or the loop-skip guard downstream never fires."""
    assert _select_like_the_route(set()) == set()
    assert _select_like_the_route(None, tools_on = False) == set()


def test_the_schema_is_not_offered_twice_when_no_allowlist_is_sent():
    """An omitted allowlist already means 'all tools'; the re-add de-duplicates."""
    from core.inference.tools import ALL_TOOLS

    selected = _select_like_the_route(None)
    assert len(selected) == len({t["function"]["name"] for t in ALL_TOOLS})


def test_the_route_really_adds_the_readiness_names():
    """Pins the replica above to the real source. If someone drops
    READINESS_TOOL_NAMES from _addable, this fails alongside it -- so the replica
    cannot silently drift into agreeing with itself."""
    source = _ROUTE.read_text(encoding = "utf-8")
    addable = [ln for ln in source.splitlines() if ln.strip().startswith("_addable =")]
    assert len(addable) == 1, f"expected one _addable line, found {len(addable)}"
    assert "READINESS_TOOL_NAMES" in addable[0], addable[0]


def test_execute_tool_dispatches_to_the_readiness_handler():
    out = tools.execute_tool("check_tool_readiness", {})
    assert isinstance(out, str) and out.strip(), "should return a non-empty report"
    assert "terminal" in out, "the report should list known tools"


def test_asking_about_one_tool_reports_only_that_tool():
    out = tools.execute_tool("check_tool_readiness", {"tool": "terminal"})
    assert "terminal" in out
    assert "webcam_look" not in out


def test_an_unknown_tool_name_is_reported_not_an_error():
    out = tools.execute_tool("check_tool_readiness", {"tool": "no_such_tool"})
    assert "unknown" in out.lower()


def test_readiness_is_not_treated_as_high_risk():
    """THE load-bearing test. Without the additive _ALWAYS_SAFE_TOOLS rebind,
    unknown tools fail closed and every readiness query prompts the user."""
    assert tools.is_high_risk_tool_call("check_tool_readiness", {}) is False


def test_a_tool_outside_the_safe_set_IS_still_high_risk():
    """Control: proves the assertion above is not passing because everything is
    considered safe."""
    assert tools.is_high_risk_tool_call("zz_not_a_real_tool", {}) is True


# --- the Anthropic Messages channel -----------------------------------------


def _anthropic_gate_genexp(inf):
    """The code object of the `_gated_tool_selected_pre` generator expression,
    pulled out of routes/inference.py's own compiled bytecode.

    Running THIS is the point. The rebind of _ANTHROPIC_UNPROMPTED_SAFE_TOOLS is
    only additive-safe if every reader looks the name up as a module global at
    call time; a reader that captured it early would leave the rebind inert, and
    asserting `"check_tool_readiness" in inf._ANTHROPIC_UNPROMPTED_SAFE_TOOLS`
    would still pass while the gate rejected the request. So the real reader is
    executed instead of inspected.
    """
    import types

    def walk(code):
        for const in code.co_consts:
            if not isinstance(const, types.CodeType):
                continue
            if const.co_name == "<genexpr>" and any(
                ins == "_ANTHROPIC_UNPROMPTED_SAFE_TOOLS" for ins in const.co_names
            ):
                return const
            found = walk(const)
            if found is not None:
                return found
        return None

    genexp = walk(inf.anthropic_messages.__code__)
    assert genexp is not None, "could not locate the gate genexp; the route changed shape"
    return genexp


@pytest.fixture
def _inference_routes():
    return pytest.importorskip(
        "routes.inference", reason = "inference stack not installed"
    )


def test_readiness_is_unprompted_on_the_anthropic_channel(_inference_routes):
    """THE load-bearing rebind test, and it is behavioural on purpose.

    _gated_tool_selected_pre decides whether the Messages channel must reject the
    request for want of a confirmation prompt it cannot present. Without the
    additive rebind, a client that explicitly names check_tool_readiness gets a
    400 under the default ("auto") permission mode."""
    import types

    inf = _inference_routes
    genexp = _anthropic_gate_genexp(inf)
    run = types.FunctionType(genexp, vars(inf))

    def gated(*names):
        return any(run(iter([{"function": {"name": n}} for n in names])))

    assert gated("check_tool_readiness") is False, (
        "the readiness tool trips the Anthropic confirmation gate; the additive "
        "rebind of _ANTHROPIC_UNPROMPTED_SAFE_TOOLS is inert or missing"
    )
    assert gated("web_search", "check_tool_readiness") is False


def test_a_genuinely_gated_tool_still_trips_the_anthropic_gate(_inference_routes):
    """Control: proves the assertion above is not passing because the genexp
    replica always answers False."""
    import types

    inf = _inference_routes
    run = types.FunctionType(_anthropic_gate_genexp(inf), vars(inf))
    assert any(run(iter([{"function": {"name": "terminal"}}]))) is True


def test_the_anthropic_rebind_is_additive_not_an_edited_literal(_inference_routes):
    """The frozenset literal is UPSTREAM code. Editing it to add a name would be
    a DELETION against upstream and would conflict the moment upstream touches
    that line, so the name must arrive by rebind."""
    source = _ROUTE.read_text(encoding = "utf-8")
    literal = source.split("_ANTHROPIC_UNPROMPTED_SAFE_TOOLS = frozenset(", 1)[1].split(")", 1)[0]
    assert "check_tool_readiness" not in literal, "upstream's literal was edited"
    assert "check_tool_readiness" in _inference_routes._ANTHROPIC_UNPROMPTED_SAFE_TOOLS


def test_the_route_seam_stays_additive():
    """routes/inference.py carries the same insertions-only contract as tools.py.
    Insertions may grow; deletions may not."""
    repo = Path(tools.__file__).parents[4]
    base = subprocess.run(
        ["git", "merge-base", "origin/main", "HEAD"],
        cwd = repo, capture_output = True, text = True, check = True,
    ).stdout.strip()
    stat = subprocess.run(
        ["git", "diff", "--numstat", base, "--", "studio/backend/routes/inference.py"],
        cwd = repo, capture_output = True, text = True, check = True,
    ).stdout.split()
    assert stat, "no diff recorded for routes/inference.py"
    assert int(stat[1]) == 0, (
        f"routes/inference.py lost {stat[1]} line(s); the seam must be additive"
    )


def test_the_seam_stays_additive():
    repo = Path(tools.__file__).parents[4]
    base = subprocess.run(
        ["git", "merge-base", "origin/main", "HEAD"],
        cwd = repo, capture_output = True, text = True, check = True,
    ).stdout.strip()
    stat = subprocess.run(
        ["git", "diff", "--numstat", base, "--", "studio/backend/core/inference/tools.py"],
        cwd = repo, capture_output = True, text = True, check = True,
    ).stdout.split()
    assert stat, "no diff recorded for tools.py"
    insertions, deletions = int(stat[0]), int(stat[1])
    assert deletions == 0, f"tools.py lost {deletions} line(s); the seam must be additive"
    assert insertions <= 90, f"seam grew to {insertions}; budget is ~88"
