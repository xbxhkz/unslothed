# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Registration and reachability tests for the five code-intelligence tools.

Registering in ALL_TOOLS and wiring execute_tool is necessary but not
sufficient: Studio's frontend sends an explicit `enabled_tools` allowlist
naming only upstream tools, and `_select_request_tools` filters ALL_TOOLS
down to it. Sub-project 1 shipped five vision tools that were silently
unreachable from every Studio chat for exactly this reason, while all its
registration tests stayed green -- because those tests called execute_tool
directly, the one path that bypasses the filter. TestReachability below
drives the real `_select_request_tools` instead.
"""
import asyncio
import os
import sys
import types

import pytest

from core.inference.assist_code import ASSIST_CODE_TOOL_NAMES

_STUDIO_ALLOWLIST = ["web_search", "python", "terminal", "edit_file"]
_FAKE_SERVER = os.path.join(os.path.dirname(__file__), "lsp_fake_server.py")


def _payload(**over):
    base = dict(enabled_tools = list(_STUDIO_ALLOWLIST), rag_scope = None,
                thread_id = None, bypass_permissions = False)
    base.update(over)
    return types.SimpleNamespace(**base)


def _select(payload, **kwargs):
    import routes.inference as routes_mod
    kwargs.setdefault("tools_on", True)
    kwargs.setdefault("mcp_allowed", False)
    return asyncio.run(routes_mod._select_request_tools(payload, **kwargs))


def _names(tools):
    return [t["function"]["name"] for t in tools]


class TestRegistry:
    def test_every_code_tool_is_in_all_tools(self):
        from core.inference.tools import ALL_TOOLS
        names = {t["function"]["name"] for t in ALL_TOOLS}
        assert ASSIST_CODE_TOOL_NAMES <= names

    def test_no_code_tool_collides_with_an_upstream_name(self):
        from core.inference.tools import ALL_TOOLS
        names = [t["function"]["name"] for t in ALL_TOOLS]
        for name in ASSIST_CODE_TOOL_NAMES:
            assert names.count(name) == 1

    def test_every_schema_documents_the_first_use_install(self):
        from core.inference.assist_code import ASSIST_CODE_TOOLS
        for tool in ASSIST_CODE_TOOLS:
            assert "install" in tool["function"]["description"].lower()


class TestReachability:
    def test_code_tools_survive_a_studio_shaped_allowlist(self):
        """Registration is necessary but not sufficient: Studio's frontend
        sends an enabled_tools allowlist naming only upstream tools."""
        names = _names(_select(_payload()))
        missing = ASSIST_CODE_TOOL_NAMES - set(names)
        assert not missing, f"code tools filtered out of a Studio chat request: {sorted(missing)}"

    def test_the_vision_tools_still_survive_too(self):
        """The re-add is shared; extending it must not displace sub-project 1."""
        from core.inference.tools import ASSIST_VISION_TOOL_NAMES
        names = set(_names(_select(_payload())))
        assert ASSIST_VISION_TOOL_NAMES <= names

    def test_an_empty_selection_stays_empty(self):
        """Upstream relies on an empty selection producing an empty catalogue
        so the tool loop is skipped. Breaking this broke four upstream tests
        in test_run_tools_locally_discriminator.py."""
        assert _names(_select(_payload(enabled_tools = []))) == []

    def test_a_selection_of_only_unimplemented_tools_stays_empty(self):
        assert _names(_select(_payload(enabled_tools = ["code_execution"]))) == []

    def test_no_tool_is_offered_twice_when_the_allowlist_is_omitted(self):
        names = _names(_select(_payload(enabled_tools = None)))
        for name in ASSIST_CODE_TOOL_NAMES:
            assert names.count(name) == 1

    def test_tools_off_admits_no_code_tool(self):
        names = set(_names(_select(_payload(), tools_on = False)))
        assert not (ASSIST_CODE_TOOL_NAMES & names)


class TestDispatch:
    def test_execute_tool_reaches_the_code_dispatcher(self, monkeypatch):
        from core.inference import tools
        from core.inference import assist_code
        monkeypatch.setattr(assist_code, "execute", lambda *a, **k: "SENTINEL")
        assert tools.execute_tool("code_diagnostics", {"path": "a.ts"}, session_id = "s") == "SENTINEL"

    def test_every_code_tool_returns_a_string_for_junk_arguments(self):
        from core.inference import tools
        for name in sorted(ASSIST_CODE_TOOL_NAMES):
            for args in ({}, {"path": None}, {"path": 5}, {"path": "../../etc/passwd"}):
                out = tools.execute_tool(name, args, session_id = "s")
                assert isinstance(out, str) and out


class TestErrorPropagation:
    """Dispatcher-level check for the task-8 corrections: a genuine
    jsonrpc.LspError from the language server must render as a
    distinguishable failure through the FULL call chain --
    tools.execute_tool -> assist_code.execute -> the pool.lease()'d
    session -> diagnostics.collect() -- never as an empty "no problems
    found" result. An empty result is a legitimate answer for a clean file;
    collapsing a server-side rejection into that same text would tell the
    model a false "all clear."

    Uses the real fake-server subprocess (lsp_fake_server.py), not a mock,
    so pool.lease()'s actual acquire/lease/release path is genuinely
    exercised end to end -- not just diagnostics.collect() in isolation,
    which Task 6 already covers on its own.
    """

    @pytest.fixture
    def rejecting_workspace(self, tmp_path, monkeypatch):
        from core.inference import tools
        from core.inference.assist_code import pool, servers
        monkeypatch.setattr(tools, "_get_workdir", lambda _sid = None: str(tmp_path))
        monkeypatch.setattr(
            servers, "server_command",
            lambda language, installer = None: [sys.executable, _FAKE_SERVER, "pull_error"],
        )
        (tmp_path / "a.ts").write_text("const a = 1\n")
        yield tmp_path
        pool.shutdown_all()

    def test_a_rejected_diagnostics_request_is_not_reported_as_clean(self, rejecting_workspace):
        from core.inference import tools
        out = tools.execute_tool("code_diagnostics", {"path": "a.ts"}, session_id = "propagation")
        assert "no problems" not in out.lower(), out
        assert "rejected" in out.lower(), out


class TestServerMessageIsCapped:
    """A server's rejection message must not carry a stack trace into the
    model's context.

    tsserver answers a request it cannot serve with a one-line diagnosis
    followed by a full JavaScript stack trace -- a dozen-plus lines of
    node_modules paths. Interpolated whole (which is what ``execute`` did),
    that is what a single failed ``code_symbols`` call spent on noise.

    Driven through the real ``execute`` dispatcher rather than by calling
    ``_short`` directly, so it exercises the actual interpolation site: with
    the cap reverted to ``e.server_message``, this test fails.
    """

    def test_a_stack_trace_is_reduced_to_its_first_line(self, monkeypatch):
        from core.inference import assist_code
        from core.inference.assist_code import jsonrpc

        trace = "\n".join(
            ["No Project.", "Error: No Project."]
            + [f"    at Object.thing (C:\\node_modules\\typescript\\lib\\typescript.js:{n}:11)"
               for n in range(186170, 186190)]
        )

        def boom(_arguments, _session_id, _budget):
            raise jsonrpc.LspError("workspace/symbol", {"code": 1, "message": trace})

        monkeypatch.setitem(assist_code._HANDLERS, "code_symbols", boom)
        out = assist_code.execute("code_symbols", {"query": "x", "path": "a.ts"})

        assert "rejected this request" in out, out
        assert "No Project." in out, out
        assert "typescript.js" not in out, out
        assert "    at " not in out, out
        assert len(out.splitlines()) == 1, out

    def test_a_single_long_line_is_truncated_visibly(self):
        from core.inference import assist_code
        assert assist_code._short("x" * 500).endswith("...")
        assert len(assist_code._short("x" * 500)) <= assist_code._MAX_SERVER_MESSAGE + 3

    def test_a_short_message_is_passed_through_unchanged(self):
        from core.inference import assist_code
        assert assist_code._short("Invalid params") == "Invalid params"
