# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The bounded sub-agent loop.

The delegate is a real agent with real tools, so every bound here is
load-bearing: without them a delegate can loop forever inside one tool call of
the primary's turn.
"""

from __future__ import annotations

import threading

import pytest

from core.inference.delegation import subagent


def _assistant(text = None, calls = ()):
    return {
        "content": text,
        "tool_calls": [
            {"id": f"c{i}", "function": {"name": name, "arguments": args}}
            for i, (name, args) in enumerate(calls)
        ],
    }


def _script(*turns):
    """call_model returns each turn in order, then repeats the last."""
    seen = []

    def call_model(messages, tools):
        seen.append(list(messages))
        return turns[min(len(seen) - 1, len(turns) - 1)]

    call_model.seen = seen
    return call_model


def test_a_plain_answer_ends_the_loop():
    result = subagent.run_subagent(
        messages = [{"role": "user", "content": "hi"}],
        tools = [],
        call_model = _script(_assistant("done")),
        execute_tool = lambda name, arguments, **kw: "",
    )
    assert result.text == "done"
    assert result.rounds == 1
    assert result.stopped_because == "answered"


def test_tool_results_are_fed_back_and_the_loop_continues():
    executed = []

    def execute_tool(name, arguments, **kwargs):
        executed.append((name, arguments))
        return "tool said hello"

    call_model = _script(_assistant(calls = [("terminal", '{"command": "ls"}')]), _assistant("finished"))
    result = subagent.run_subagent(
        messages = [{"role": "user", "content": "go"}],
        tools = [{"function": {"name": "terminal"}}],
        call_model = call_model,
        execute_tool = execute_tool,
    )
    assert executed == [("terminal", {"command": "ls"})], "arguments are parsed from JSON"
    assert result.text == "finished"
    assert result.rounds == 2
    assert any(m.get("role") == "tool" and "tool said hello" in str(m.get("content"))
               for m in call_model.seen[-1]), "the tool result must reach the next turn"


def test_the_iteration_cap_stops_a_runaway_delegate():
    call_model = _script(_assistant(calls = [("terminal", "{}")]))
    result = subagent.run_subagent(
        messages = [], tools = [], call_model = call_model,
        execute_tool = lambda name, arguments, **kw: "again",
        max_iterations = 3,
    )
    assert result.rounds == 3
    assert result.stopped_because == "iteration-cap"


def test_the_timeout_stops_a_slow_delegate():
    clock = {"t": 0.0}

    def now():
        clock["t"] += 100.0
        return clock["t"]

    result = subagent.run_subagent(
        messages = [], tools = [],
        call_model = _script(_assistant(calls = [("terminal", "{}")])),
        execute_tool = lambda name, arguments, **kw: "again",
        max_iterations = 50, timeout_s = 250.0, now = now,
    )
    assert result.stopped_because == "timeout"
    assert result.rounds < 50


def test_cancellation_stops_the_loop():
    cancel = threading.Event()

    def execute_tool(name, arguments, **kwargs):
        cancel.set()
        return "x"

    result = subagent.run_subagent(
        messages = [], tools = [],
        call_model = _script(_assistant(calls = [("terminal", "{}")])),
        execute_tool = execute_tool, cancel_event = cancel,
    )
    assert result.stopped_because == "cancelled"


def test_a_tool_that_raises_is_reported_to_the_delegate_not_propagated():
    def execute_tool(name, arguments, **kwargs):
        raise OverflowError("boom")

    result = subagent.run_subagent(
        messages = [], tools = [],
        call_model = _script(_assistant(calls = [("terminal", "{}")]), _assistant("recovered")),
        execute_tool = execute_tool,
    )
    assert result.text == "recovered"


def test_malformed_tool_arguments_do_not_crash_the_loop():
    seen = []
    result = subagent.run_subagent(
        messages = [], tools = [],
        call_model = _script(_assistant(calls = [("terminal", "{not json")]), _assistant("ok")),
        execute_tool = lambda name, arguments, **kw: seen.append(arguments) or "",
    )
    assert seen == [{}], "unparseable arguments become an empty dict"
    assert result.text == "ok"


def test_each_round_is_reported_for_the_transcript():
    rounds = []
    subagent.run_subagent(
        messages = [], tools = [],
        call_model = _script(_assistant(calls = [("terminal", "{}")]), _assistant("done")),
        execute_tool = lambda name, arguments, **kw: "result",
        on_round = rounds.append,
    )
    joined = "\n".join(rounds)
    assert "terminal" in joined and "done" in joined
