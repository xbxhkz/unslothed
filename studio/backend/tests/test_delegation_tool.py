# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ask_model: swap, run, swap back.

The load-bearing rule is step 3 of the sequence: what gets restored is the model
that was RESIDENT when the call started, not whatever the "primary" role names.
Conflating them would silently change which model the user is talking to.

Two more rules are tested here because they are the ones that cost the user
something rather than costing a message: a delegation is refused outright when
what is loaded now could not be loaded again (the loader is asked first), and a
delegate that stops being the resident model mid-run fails loudly instead of
answering as whatever took its place.

No test loads a model: the loader is injected.
"""

from __future__ import annotations

import threading

import pytest

from core.inference import delegation
from core.inference import model_roles
from core.inference.delegation import subagent, workspace
from core.inference.delegation.schemas import ASK_MODEL_TOOL
from core.inference.model_roles import storage

# Captured before any fixture replaces it, so the one test that exercises the
# real adapter -- and the residency check inside it -- can put it back.
_REAL_CALL_MODEL_FOR = delegation._call_model_for


class FakeLoader:
    def __init__(self, resident = "repo/Primary:Q4", fail_on = None, restorable = True):
        self.resident = resident
        self.fail_on = fail_on
        self.restorable = restorable
        self.calls = []

    def resident_model_id(self):
        return self.resident

    def resident_is_restorable(self):
        return self.restorable

    def load(self, model_id, overrides = None):
        self.calls.append(model_id)
        if self.fail_on is not None and model_id == self.fail_on:
            raise delegation._loader_error("could not load " + model_id)
        self.resident = model_id


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "_workdir_for", lambda session_id: str(tmp_path))
    monkeypatch.setattr(storage, "get_role_bindings",
                        lambda: {"coding": {"model": "repo/Coder:Q4"}, "primary": {"model": "repo/Other:Q4"}})
    monkeypatch.setattr(model_roles, "_resolve_local", lambda model: ("/p", None, model))
    fake = FakeLoader()
    monkeypatch.setattr(delegation, "_loader", fake)
    monkeypatch.setattr(delegation, "_call_model_for",
                        lambda binding: (lambda messages, tools: {"content": "delegate answer", "tool_calls": []}))
    delegation._reset_for_tests()
    return fake


def _run(**over):
    arguments = {"role": "coding", "task": "rewrite the parser"}
    arguments.update(over)
    return delegation.execute("ask_model", arguments, session_id = "s1")


def test_a_delegation_swaps_runs_and_restores(rig):
    out = _run()
    assert rig.calls == ["repo/Coder:Q4", "repo/Primary:Q4"], "swap out, then back"
    assert rig.resident == "repo/Primary:Q4"
    assert "delegate answer" in out


def test_the_restore_targets_what_was_resident_not_the_primary_role(rig):
    """THE load-bearing test. The 'primary' role names repo/Other:Q4; the model
    actually loaded was repo/Primary:Q4, and that is what must come back."""
    _run()
    assert rig.calls[-1] == "repo/Primary:Q4"
    assert "repo/Other:Q4" not in rig.calls


def test_the_result_names_the_work_file(rig):
    out = _run()
    assert "delegations/" in out and "work.md" in out


def test_an_unavailable_role_never_loads_anything(rig, monkeypatch):
    monkeypatch.setattr(model_roles, "_resolve_local", lambda model: None)
    out = _run()
    assert rig.calls == [], "no swap may be attempted when the role is not ready"
    assert "not downloaded" in out.lower() or "missing" in out.lower()


def test_an_unbound_role_is_refused(rig):
    out = _run(role = "nonexistent")
    assert rig.calls == []
    assert "nonexistent" in out


def test_a_resident_model_that_could_not_come_back_is_refused_before_anything_happens(rig):
    """The loader is asked FIRST whether what is loaded now could be loaded again
    -- a Transformers model, or a GGUF whose id does not resolve back. Losing the
    delegation costs one message; losing the user's model costs their session."""
    rig.restorable = False
    out = _run()
    assert rig.calls == [], "nothing may be swapped when the swap could not be undone"
    assert "repo/Primary:Q4" in out, "the user must be told which model blocked it"


def test_the_model_is_restored_even_when_the_delegate_raises(rig, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("sub-agent exploded")

    monkeypatch.setattr(subagent, "run_subagent", boom)
    out = _run()
    assert rig.calls[-1] == "repo/Primary:Q4", "restore must happen in a finally"
    assert "exploded" in out or "failed" in out.lower()


def test_a_failed_restore_is_reported_loudly(rig, monkeypatch):
    rig.fail_on = "repo/Primary:Q4"
    out = _run()
    lowered = out.lower()
    assert "could not" in lowered or "failed" in lowered
    assert "repo/Primary:Q4" in out, "the user must be told which model did not come back"


def test_a_swap_underneath_the_delegate_fails_the_delegation(rig, monkeypatch):
    """load() releases the auto-switch lock, the swap gate and the keep-warm gate
    before it returns, so an ordinary API request can swap the model away between
    the load and the delegate's first token. The answer would then come from the
    wrong model, silently, and be written to work.md as the delegate's work."""
    monkeypatch.setattr(delegation, "_call_model_for", _REAL_CALL_MODEL_FOR)
    monkeypatch.setattr(delegation, "_model_call",
                        lambda: (lambda messages, tools: {"content": "wrong model's answer", "tool_calls": []}))

    loaded = rig.load

    def load_then_lose_it(model_id, overrides = None):
        loaded(model_id)
        if model_id == "repo/Coder:Q4":
            rig.resident = "repo/Interloper:Q4"  # another request swapped it away

    monkeypatch.setattr(rig, "load", load_then_lose_it)

    out = _run()
    assert "repo/Coder:Q4" in out and "repo/Interloper:Q4" in out, "name both models"
    assert "wrong model's answer" not in out
    assert rig.calls[-1] == "repo/Primary:Q4", "the restore still runs"


def test_the_delegate_keeps_working_while_it_is_still_resident(rig, monkeypatch):
    """The other half of the check above: the real adapter must not refuse a
    delegate that IS loaded, or every delegation would fail."""
    monkeypatch.setattr(delegation, "_call_model_for", _REAL_CALL_MODEL_FOR)
    monkeypatch.setattr(delegation, "_model_call",
                        lambda: (lambda messages, tools: {"content": "delegate answer", "tool_calls": []}))
    out = _run()
    assert "delegate answer" in out
    assert rig.calls == ["repo/Coder:Q4", "repo/Primary:Q4"]


def test_a_nested_delegation_is_refused(rig, monkeypatch):
    """Depth guard: the delegate must not delegate."""
    inner = {}

    def call_model(messages, tools):
        inner["out"] = delegation.execute("ask_model", {"role": "coding", "task": "again"}, session_id = "s1")
        return {"content": "done", "tool_calls": []}

    monkeypatch.setattr(delegation, "_call_model_for", lambda binding: call_model)
    _run()
    assert "already" in inner["out"].lower() or "nested" in inner["out"].lower()


def test_the_delegate_is_not_given_the_delegation_tool(rig, monkeypatch):
    """Loop prevention, layer one. ALL_TOOLS is faked because registering
    ask_model with the dispatcher is Task 8's job, so today the filter would be
    checked against a list that cannot contain what it removes."""
    import core.inference.tools as tools_module

    monkeypatch.setattr(tools_module, "ALL_TOOLS", [
        {"type": "function", "function": {"name": "read_file"}},
        ASK_MODEL_TOOL,
    ])
    names = [t["function"]["name"] for t in delegation._delegate_tools()]
    assert names == ["read_file"]


def test_the_window_cap_refuses_a_third_delegation(rig):
    _run()
    _run()
    out = _run()
    assert "limit" in out.lower() or "too many" in out.lower()
    assert rig.calls.count("repo/Coder:Q4") == 2, "the third must not swap"


def test_cancellation_is_passed_to_the_sub_agent(rig, monkeypatch):
    seen = {}

    def spy(**kwargs):
        seen.update(kwargs)
        return subagent.SubagentResult("x", 1, "answered")

    monkeypatch.setattr(subagent, "run_subagent", spy)
    cancel = threading.Event()
    delegation.execute("ask_model", {"role": "coding", "task": "t"}, session_id = "s1", cancel_event = cancel)
    assert seen.get("cancel_event") is cancel


def test_execute_never_raises_on_bad_arguments(rig):
    for arguments in (None, [], {"role": None}, {"role": "coding"}, {"task": "x"}, {"role": 5, "task": 6}):
        out = delegation.execute("ask_model", arguments, session_id = "s1")
        assert isinstance(out, str) and out


def test_the_schema_is_shaped_like_the_other_tools():
    from core.inference.delegation.schemas import (
        ASK_MODEL_TOOL, DELEGATION_TOOLS, DELEGATION_TOOL_NAMES,
    )

    function = ASK_MODEL_TOOL["function"]
    assert function["name"] == "ask_model"
    assert set(function["parameters"]["required"]) == {"role", "task"}
    assert DELEGATION_TOOLS == [ASK_MODEL_TOOL]
    assert DELEGATION_TOOL_NAMES == frozenset({"ask_model"})
