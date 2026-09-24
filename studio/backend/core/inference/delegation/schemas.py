# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""OpenAI-style schema for the delegation tool.

The description is read twice, and both readers matter: the model decides whether
to call this from it, and the user reads it on the approval card. It therefore has
to be literally true. It said "It works with the same tools this conversation has",
which is the opposite of what ships -- the delegate is given ALL_TOOLS minus
ask_model, whichever pills the user switched off (see delegation/__init__.py's
module docstring, which had it right all along). A card that understates what is
being approved is worse than one that says nothing.
"""

ASK_MODEL_TOOL = {
    "type": "function",
    "function": {
        "name": "ask_model",
        "description": (
            "Hand a self-contained task to another model configured for a role (for example "
            "'coding' or 'vision'), then continue once it is done. The other model is loaded, "
            "works with its own tools, writes its result to a file, and the model you were "
            "using is loaded again. It works with this server's full tool set, not the tools "
            "switched on in this conversation, and its individual tool calls are not confirmed "
            "separately -- approving this call approves "
            "the work it does. Expensive: it swaps models twice, so use it when the other model "
            "is genuinely better suited, not for small steps. Returns a short summary and the "
            "paths of the files it wrote."
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
