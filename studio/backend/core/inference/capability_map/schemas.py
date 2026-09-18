# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""OpenAI-style schema for the capability map tool."""

FIND_CAPABILITY_TOOL = {
    "type": "function",
    "function": {
        "name": "find_capability",
        "description": (
            "Find which tools, installed software or model abilities can do something "
            "(read text in images, edit video, analyse PDFs), which is best, and whether it "
            "works right now. Omit 'capability' to see the whole map, including what is "
            "missing. Read-only and cheap: it checks what is installed and loaded, not "
            "network reachability, and it does not cover MCP tools."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "capability": {
                    "type": "string",
                    "description": (
                        "A capability name or alias, e.g. 'ocr', 'video editing', "
                        "'transcribe'. An unrecognised one returns the list of capabilities. "
                        "Omit for the whole map."
                    ),
                },
            },
            "required": [],
        },
    },
}

CAPABILITY_TOOLS = [FIND_CAPABILITY_TOOL]
CAPABILITY_TOOL_NAMES = frozenset({"find_capability"})
