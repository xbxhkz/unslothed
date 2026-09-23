# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""OpenAI-style schema for the delegation tool."""

ASK_MODEL_TOOL = {
    "type": "function",
    "function": {
        "name": "ask_model",
        "description": (
            "Hand a self-contained task to another model configured for a role (for example "
            "'coding' or 'vision'), then continue once it is done. The other model is loaded, "
            "works with its own tools, writes its result to a file, and the model you were "
            "using is loaded again. Expensive: it swaps models twice, so use it when the other "
            "model is genuinely better suited, not for small steps. Returns a short summary and "
            "the paths of the files it wrote."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "role": {
                    "type": "string",
                    "description": "The configured role to hand the task to, e.g. 'coding'.",
                },
                "task": {
                    "type": "string",
                    "description": (
                        "What to do, written so it stands alone -- the other model cannot see "
                        "this conversation."
                    ),
                },
                "context": {
                    "type": "string",
                    "description": "What you already established that it needs (optional).",
                },
            },
            "required": ["role", "task"],
        },
    },
}

DELEGATION_TOOLS = [ASK_MODEL_TOOL]
DELEGATION_TOOL_NAMES = frozenset({"ask_model"})
