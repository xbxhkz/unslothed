# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""OpenAI-style schema for continuity_task.

Named schemas_tool.py, not schemas.py, because that name is already the
dataclasses module in this package (ProjectState, Task, ...) -- delegation
keeps its tool schema and its internal dataclasses in separate files too, just
under different names (delegation/schemas.py is the tool schema there, and
delegation/subagent.py holds its own internal SubagentResult dataclass). This
package's split is the other way around for a reason worth stating: schemas.py
was written first (Task 1) and named for the data shapes, so the tool schema
gets the more specific name rather than forcing a rename across every earlier
task's imports.
"""

CONTINUITY_TASK_TOOL = {
    "type": "function",
    "function": {
        "name": "continuity_task",
        "description": (
            "Track your own progress on a multi-step task so it survives a context reset: "
            "record what you tried and whether it failed, add tasks with dependencies, mark "
            "them in progress or done, and check what state you're in. Scoped to this "
            "conversation when possible; may be shared with other chats in the same project "
            "or workspace if this server's settings put multiple chats in one folder. There is "
            "no separate project to name. Prefer this over re-deriving status by re-reading "
            "the conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "status", "add_task", "set_status",
                        "record_error", "check_prior_failures", "checkpoint",
                    ],
                    "description": "Which operation to perform.",
                },
                "id": {"type": "string", "description": "Task id (add_task, set_status)."},
                "title": {"type": "string", "description": "Task title (add_task)."},
                "status": {
                    "type": "string",
                    "description": "New status (set_status): in_progress, complete, failed, blocked, abandoned, pending. ('ready' is never set explicitly -- it is computed; use status action to see which pending tasks are unblocked.)",
                },
                "depends_on": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Task ids this one depends on (add_task, optional).",
                },
                "acceptance_criteria": {
                    "type": "array", "items": {"type": "string"},
                    "description": "What 'done' means for this task (add_task). Required to later mark it complete.",
                },
                "what_tried": {"type": "string", "description": "What was attempted (record_error)."},
                "why_failed": {"type": "string", "description": "Why it failed (record_error)."},
                "symptom": {
                    "type": "string",
                    "description": "Short tag for later matching (record_error, check_prior_failures).",
                },
                "note": {"type": "string", "description": "A note for this checkpoint (checkpoint)."},
            },
            "required": ["action"],
        },
    },
}

CONTINUITY_TOOLS = [CONTINUITY_TASK_TOOL]
CONTINUITY_TOOL_NAMES = frozenset({"continuity_task"})
