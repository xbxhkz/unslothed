# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The vision tools must survive Studio's own request-level tool filter.

Registering a tool in ``ALL_TOOLS`` and wiring ``execute_tool`` is necessary
but NOT sufficient: every chat request that originates in Studio's UI carries
an explicit ``enabled_tools`` allowlist, built from a hard-coded, pill-driven
literal in ``chat-adapter.ts`` that names only the upstream tools. That filter

    tools = [t for t in ALL_TOOLS if t["function"]["name"] in payload.enabled_tools]

removed all five vision schemas from every Studio chat request, so the model
never saw them and could never call them -- while the registration tests, which
go straight to ``ALL_TOOLS`` and ``execute_tool``, stayed green.

These tests therefore drive the REAL ``_select_request_tools`` with a
Studio-shaped payload. A test that calls ``execute_tool`` directly cannot catch
this class of bug and is not what lives here.

Upstream hit the identical bug with ``search_conversation`` (see the comment at
``_select_request_tools``'s archive block) and fixed it by re-adding after the
filter; that is the precedent this follows.
"""

import asyncio
import types

import pytest

from core.inference.assist_vision import ASSIST_VISION_TOOL_NAMES

# Exactly what chat-adapter.ts sends: Search on, Code on, nothing else. It names
# no vision tool because no pill exists for one.
_STUDIO_ALLOWLIST = ["web_search", "python", "terminal", "edit_file"]


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
    return asyncio.run(routes_mod._select_request_tools(payload, **kwargs))


def _names(tools):
    return [t["function"]["name"] for t in tools]


class TestStudioAllowlistDoesNotStripVisionTools:
    def test_every_vision_schema_survives_a_studio_shaped_allowlist(self):
        """The whole point: the allowlist names NO vision tool, and all five
        must still reach the model."""
        names = _names(_select(_payload()))
        missing = ASSIST_VISION_TOOL_NAMES - set(names)
        assert not missing, (
            f"vision tools filtered out of a Studio chat request: {sorted(missing)}. "
            "They are registered in ALL_TOOLS but never reach the model."
        )

    def test_the_minimal_studio_allowlist_still_carries_them(self):
        """Search alone -- the narrowest allowlist Studio ever sends with
        tools on."""
        names = _names(_select(_payload(enabled_tools = ["web_search"])))
        assert ASSIST_VISION_TOOL_NAMES <= set(names)

    def test_an_empty_selection_stays_empty(self):
        """The opposite of what this test originally asserted, and the reason is
        worth keeping.

        `enabled_tools = []` is reachable: a chat with MCP on and every built-in
        pill off sends it. The first version of this test claimed the vision
        tools should ride along anyway. They must not. Upstream relies on an
        empty selection producing an EMPTY catalogue so the guard further down
        can skip the tool loop -- that is what stops the safetensors loop's
        "empty means allow all" semantic from reaching built-ins the caller
        never opted into. Re-adding unconditionally broke four upstream tests in
        test_run_tools_locally_discriminator.py.

        Answering "every built-in tool is off" by injecting five of them is also
        just wrong on its own terms, and it keeps `enabled_tools` satisfiable for
        third-party /v1/chat/completions clients, for whom the array is a
        contract.
        """
        names = _names(_select(_payload(enabled_tools = [])))
        assert names == [], f"an empty selection must stay empty, got {names}"

    def test_a_selection_of_only_unimplemented_tools_stays_empty(self):
        """The upstream case the discriminator tests actually exercise:
        `code_execution` is provider-hosted and has no local implementation, so
        the filter admits nothing and the catalogue must remain empty rather
        than being back-filled with vision tools."""
        names = _names(_select(_payload(enabled_tools = ["code_execution"])))
        assert names == [], f"unimplemented-only selection must stay empty, got {names}"

    def test_the_upstream_tools_the_allowlist_named_are_untouched(self):
        """The re-add must ADD, never replace what the filter admitted."""
        names = _names(_select(_payload()))
        assert set(_STUDIO_ALLOWLIST) <= set(names)

    def test_no_vision_tool_is_offered_twice(self):
        """An omitted allowlist means "all tools", so the re-add must not
        duplicate schemas already present -- a duplicated function name is a
        hard error on several providers."""
        names = _names(_select(_payload(enabled_tools = None)))
        for name in ASSIST_VISION_TOOL_NAMES:
            assert names.count(name) == 1, f"{name} offered {names.count(name)} times"


class TestToolsOffStaysOff:
    def test_tools_off_admits_no_vision_tool(self):
        """`tools_on = False` is the MCP-only / tools-disabled request. The
        re-add must not smuggle built-ins back into it."""
        names = _names(_select(_payload(), tools_on = False))
        assert not (ASSIST_VISION_TOOL_NAMES & set(names))

    def test_a_compaction_reset_still_admits_search_conversation_alone(self, monkeypatch):
        """Upstream's own invariant, which the vision re-add sits next to: after
        a reset the model is shown search_conversation and NOTHING else, because
        a large catalogue at a compaction boundary provokes guessed tool names.
        Breaking that would be the obvious way to get the vision re-add wrong."""
        import routes.inference as routes_mod

        monkeypatch.setattr(routes_mod, "_thread_has_conversation_archive", lambda _tid: True)
        monkeypatch.setattr(routes_mod, "_checkpoint_needs_search", lambda: True)
        names = _names(
            _select(_payload(thread_id = "t"), tools_on = False, checkpoint_fitted = True)
        )
        assert names == ["search_conversation"]
