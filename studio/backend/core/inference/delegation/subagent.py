# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A bounded agent loop for a delegate model.

The model call and the tool executor are injected, exactly as
safetensors_agentic.run_safetensors_tool_loop injects its own. That is what lets
this be tested without a backend, and what lets a GGUF delegate (structured tool
calls) and a safetensors delegate (markup) share one loop.

Every bound here is load-bearing. The delegate runs INSIDE one tool call of the
primary's turn, so an unbounded loop is an unbounded turn.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

DELEGATE_MAX_ITERATIONS = 12
DELEGATE_TIMEOUT_S = 600.0


@dataclass(frozen = True)
class SubagentResult:
    text: str
    rounds: int
    stopped_because: str  # "answered" | "iteration-cap" | "timeout" | "cancelled"


def _arguments(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except BaseException:  # noqa: BLE001 - a malformed call is the model's error, not a crash
        return {}
    return parsed if isinstance(parsed, dict) else {}


def run_subagent(
    *,
    messages: list,
    tools: list,
    call_model: Callable[[list, list], dict],
    execute_tool: Callable[..., str],
    on_round: Optional[Callable[[str], None]] = None,
    max_iterations: int = DELEGATE_MAX_ITERATIONS,
    timeout_s: float = DELEGATE_TIMEOUT_S,
    cancel_event = None,
    now: Callable[[], float] = time.monotonic,
    **tool_kwargs,
) -> SubagentResult:
    history = list(messages)
    started = now()
    rounds = 0
    text = ""
    while True:
        if cancel_event is not None and cancel_event.is_set():
            return SubagentResult(text, rounds, "cancelled")
        if rounds >= max_iterations:
            return SubagentResult(text, rounds, "iteration-cap")
        if now() - started >= timeout_s:
            return SubagentResult(text, rounds, "timeout")

        message = call_model(history, tools)
        rounds += 1
        text = (message or {}).get("content") or text
        calls = (message or {}).get("tool_calls") or []
        if on_round is not None:
            names = ", ".join(c.get("function", {}).get("name", "?") for c in calls)
            on_round(f"round {rounds}: {message.get('content') or ''}{(' -> ' + names) if names else ''}")
        if not calls:
            return SubagentResult(text, rounds, "answered")

        history.append({"role": "assistant", "content": message.get("content"), "tool_calls": calls})
        for call in calls:
            function = call.get("function") or {}
            name = function.get("name") or ""
            arguments = _arguments(function.get("arguments"))
            try:
                result = execute_tool(name, arguments, **tool_kwargs)
            except BaseException as exc:  # noqa: BLE001 - reported to the delegate, never propagated
                result = f"Error: {name} failed: {exc}"
            history.append({"role": "tool", "tool_call_id": call.get("id"), "content": str(result)})
            if on_round is not None:
                on_round(f"  {name} -> {str(result)[:400]}")
            if cancel_event is not None and cancel_event.is_set():
                return SubagentResult(text, rounds, "cancelled")
