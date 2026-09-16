# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""OpenAI-style schema for the readiness tool."""

CHECK_TOOL_READINESS_TOOL = {
    "type": "function",
    "function": {
        "name": "check_tool_readiness",
        "description": (
            "Report whether tools will actually work before you call them: whether their "
            "models, language servers or connections are present. States are 'ready', "
            "'missing' (with what is absent and how to obtain it) and 'unknown' (no check "
            "exists -- calling it is the only way to find out). Read-only and cheap. Use it "
            "when a capability might be unavailable, or after a tool fails unexpectedly."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tool": {
                    "type": "string",
                    "description": "One tool name. Omit to report every tool with a known check.",
                },
            },
            "required": [],
        },
    },
}

READINESS_TOOLS = [CHECK_TOOL_READINESS_TOOL]
READINESS_TOOL_NAMES = frozenset({"check_tool_readiness"})
