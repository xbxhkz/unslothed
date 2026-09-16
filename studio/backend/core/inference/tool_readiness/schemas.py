# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""OpenAI-style schema for the readiness tool."""

CHECK_TOOL_READINESS_TOOL = {
    "type": "function",
    "function": {
        "name": "check_tool_readiness",
        # No "connections": enumerating MCP tools means an async DB read and a
        # network probe per server, which this tool promises not to do, so an MCP
        # server's liveness is not something it can report. An MCP tool name can
        # still be PASSED here -- it answers 'unknown', which is the honest
        # answer -- so the description invites that rather than implying a check
        # that does not exist.
        "description": (
            "Report whether tools will actually work before you call them: whether their "
            "models, language servers and packages are present. States are 'ready', "
            "'missing' (with what is absent and how to obtain it) and 'unknown' (no check "
            "exists -- calling it is the only way to find out). Read-only and cheap. Use it "
            "when a capability might be unavailable, or after a tool fails unexpectedly. "
            "It does not test network reachability or MCP server connections."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tool": {
                    "type": "string",
                    "description": (
                        "One tool name -- any name, including an MCP tool, which answers "
                        "'unknown'. Omit to report every built-in tool."
                    ),
                },
            },
            "required": [],
        },
    },
}

READINESS_TOOLS = [CHECK_TOOL_READINESS_TOOL]
READINESS_TOOL_NAMES = frozenset({"check_tool_readiness"})
