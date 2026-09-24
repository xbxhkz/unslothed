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
import types

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
    def __init__(self, resident = "repo/Primary:Q4", fail_on = None, restorable = True,
                 pinned = False):
        self.resident = resident
        self.fail_on = fail_on
        self.restorable = restorable
        # _loaded_by_user_action: True when the user loaded this model from the
        # picker by hand. The idle unloader spares a pinned model under "unload
        # API-loaded models only".
        self.pinned = pinned
        self.calls = []
        # (model_id, pinned) as each load leaves it, so a test can see the pin go
        # down on the delegate's load and come back up on the restore.
        self.pin_trail = []
        self.resident_reads = 0

    def resident_model_id(self):
        self.resident_reads += 1
        return self.resident

    def serves(self, model_id):
        """The real loader re-resolves and matches every alias of the resident
        model; an id-for-id compare is enough for the tests that do not care."""
        return self.resident == model_id

    def resident_is_restorable(self):
        return self.restorable

    def resident_is_user_pinned(self):
        return self.pinned

    def restore_user_pin(self, pinned):
        """Only ever sets it, exactly as the real helper does."""
        if pinned:
            self.pinned = True

    def load(self, model_id, overrides = None):
        self.calls.append(model_id)
        if self.fail_on is not None and model_id == self.fail_on:
            raise delegation._loader_error("could not load " + model_id)
        self.resident = model_id
        # What routes/inference.py:6222 does after every swap auto-switch performs:
        # the model now loaded came from an API request, so it is not user-pinned.
        self.pinned = False
        self.pin_trail.append((model_id, self.pinned))


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


@pytest.fixture
def recorded_tools(tmp_path, monkeypatch):
    """The REAL tools.execute_tool, with every tool but ``ask_model`` recorded.

    No test may run a real tool, so ``_execute_tool_unaudited`` -- the body the
    audit shadow calls, looked up as a module global on every call -- is replaced
    with a recorder that lets exactly one name through: ``ask_model``, the
    dispatch under test. Everything else is recorded and answered with a string.

    The audit shadow runs for real, so the audit DB is pointed at tmp_path.
    """
    import core.inference.tools as tools_module
    from core.inference import tool_audit
    from storage import tool_audit_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    tool_audit_db.reset_for_tests()
    tool_audit.reset_degraded_for_tests()

    dispatch = tools_module._execute_tool_unaudited
    seen = types.SimpleNamespace(calls = [])

    def recorder(name, arguments, **kwargs):
        if name == "ask_model":
            return dispatch(name, arguments, **kwargs)
        seen.calls.append((name, arguments, kwargs))
        return f"ran {name}"

    monkeypatch.setattr(tools_module, "_execute_tool_unaudited", recorder)
    return seen


def _run(**over):
    arguments = {"role": "coding", "task": "rewrite the parser"}
    arguments.update(over)
    return delegation.execute("ask_model", arguments, session_id = "s1")


def _one_tool_call(name, arguments):
    """A model that asks for one tool on its first turn and answers on its second."""
    turns = []

    def call_model(messages, tools):
        turns.append(1)
        if len(turns) == 1:
            return {
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": name, "arguments": arguments}}],
            }
        return {"content": "delegate answer", "tool_calls": []}

    return call_model


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
    """Depth guard: the delegate must not delegate.

    Asserted on the depth guard's OWN wording. The first version accepted
    "already" or "nested", and deleting the depth guard left every test passing:
    the nested call simply fell through to the process-wide gate, whose refusal
    ("another conversation is already delegating") also contains "already". The
    control the spec's testing table names was therefore never achievable. Only
    the depth guard says "cannot delegate again".
    """
    inner = {}

    def call_model(messages, tools):
        inner["out"] = delegation.execute("ask_model", {"role": "coding", "task": "again"}, session_id = "s1")
        return {"content": "done", "tool_calls": []}

    monkeypatch.setattr(delegation, "_call_model_for", lambda binding: call_model)
    _run()
    assert "cannot delegate again" in inner["out"].lower(), inner["out"]
    assert delegation.DELEGATION_BUSY_MESSAGE not in inner["out"], \
        "the process-wide gate's refusal is not evidence that the depth guard fired"


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


def test_the_delegate_is_not_given_the_delegation_tool_for_real():
    """The same rule against the REAL registry. Task 8 registered ask_model, so
    the filter now has something to remove -- until then the test above ran
    against a fake and could not fail. The faked one stays: it pins the filter's
    logic independently of what the registry happens to hold."""
    from core.inference.tools import ALL_TOOLS

    names = {t["function"]["name"] for t in delegation._delegate_tools()}
    assert "ask_model" in {t["function"]["name"] for t in ALL_TOOLS}, "not registered"
    assert "ask_model" not in names
    assert "terminal" in names, "the delegate still gets the ordinary tools"


def test_the_delegate_gets_the_registry_minus_exactly_one():
    """'Everything except ask_model' as a fact rather than a comment: it fails
    loudly if a future tool is accidentally filtered out too."""
    from core.inference.tools import ALL_TOOLS

    assert len(delegation._delegate_tools()) == len(ALL_TOOLS) - 1


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


# ── what the delegate is given, and what approving it approves ───────────


def test_the_tool_description_says_the_delegates_calls_are_not_confirmed(rig):
    """Both halves of the disclosure, because both are what the approval card says.

    The delegate's calls are not individually approved -- the primary's loop prompts
    once, for the delegation. And the delegate gets the server's whole tool set, not
    the pills this conversation left switched on: the description used to claim the
    opposite ("the same tools this conversation has") while the module docstring had
    it right, so the card understated what was being approved."""
    description = ASK_MODEL_TOOL["function"]["description"]
    assert "not confirmed separately" in description
    assert "full tool set" in description
    assert "not the tools switched on in this conversation" in description
    assert "same tools this conversation has" not in description, \
        "the sentence that said the opposite of what ships"


def test_the_delegate_is_told_its_files_are_the_whole_record(rig):
    """The caller cannot see the delegate's reasoning or its tool calls, only the
    files. Saying so is what makes the transcript a record rather than a side
    effect."""
    prompt = delegation._delegate_system_prompt("vision", {"work": "d/work.md"})
    assert "d/work.md" in prompt
    assert "on behalf of another model" in prompt


def test_the_execution_context_reaches_the_delegates_tools(rig, monkeypatch, recorded_tools):
    """Driven through the REAL tools.execute_tool, which is the only production
    caller and is exactly where this broke.

    The first version of this test called delegation.execute(...) directly with
    the full kwarg set. No caller supplies that: the dispatch in tools.py
    forwarded session_id and cancel_event and nothing else, so _TOOL_PASSTHROUGH
    collected {} on every real call and every line that reads it was dead. A
    delegate briefed to consult the chat's attached manual called
    search_knowledge_base, got rag_scope = None and was told "No documents are
    attached to this chat" -- in a chat with documents attached. The test passed
    the whole time, because it was the only caller that ever passed the kwargs.
    """
    import core.inference.tools as tools_module

    monkeypatch.setattr(delegation, "_call_model_for",
                        lambda binding: _one_tool_call("read_file", {"path": "a.txt"}))
    cancel = threading.Event()
    out = tools_module.execute_tool(
        "ask_model", {"role": "coding", "task": "t"},
        session_id = "s1", cancel_event = cancel, thread_id = "t1",
        rag_scope = {"mode": "dense"}, timeout = 42, disable_sandbox = False,
        website_policy = {"allow": []}, output_callback = lambda chunk: None,
        conversation_branch = [{"role": "user", "content": "earlier"}],
        conversation_budget_tokens = 1234, conversation_token_counter = len,
    )
    assert "delegate answer" in out, out
    assert recorded_tools.calls, "the delegate's tool call never reached execute_tool"
    _name, _arguments, kwargs = recorded_tools.calls[0]
    assert kwargs["session_id"] == "s1"
    assert kwargs["thread_id"] == "t1"
    assert kwargs["rag_scope"] == {"mode": "dense"}
    assert kwargs["timeout"] == 42
    assert kwargs["disable_sandbox"] is False
    assert kwargs["website_policy"] == {"allow": []}
    assert kwargs["cancel_event"] is cancel, "Stop must reach a long-running tool call"
    # Pinned individually, not as a group: these three were left unasserted in the
    # first pass, which put them back in exactly the unguarded shape this whole test
    # exists to prevent -- deleting all three from the dispatch left the suite green.
    # They decide which branch turns search_conversation reads and how it budgets them.
    assert kwargs["conversation_branch"] == [{"role": "user", "content": "earlier"}]
    assert kwargs["conversation_budget_tokens"] == 1234
    assert kwargs["conversation_token_counter"] is len
    assert "output_callback" not in kwargs, "the delegate's stdout is not the primary's"


# ── one delegation at a time, process-wide ──────────────────────────────


def test_a_second_conversation_is_refused_while_a_delegation_runs(rig, monkeypatch):
    """The depth guard is a threading.local, so it only stopped a delegate
    delegating on its own thread. Two chats on two worker threads interleave, each
    reads the other's delegate as "what was resident", both restores succeed, the
    user's model is gone -- and neither result carries a warning, because from
    inside either delegation nothing failed."""
    started = threading.Event()
    finish = threading.Event()
    second = {}

    def slow_call(messages, tools):
        started.set()
        finish.wait(5)
        return {"content": "delegate answer", "tool_calls": []}

    monkeypatch.setattr(delegation, "_call_model_for", lambda binding: slow_call)

    def other_conversation():
        started.wait(5)
        reads_before = rig.resident_reads
        second["out"] = delegation.execute(
            "ask_model", {"role": "coding", "task": "t"}, session_id = "s2"
        )
        second["read_resident"] = rig.resident_reads > reads_before
        finish.set()

    thread = threading.Thread(target = other_conversation)
    thread.start()
    first = _run()
    thread.join(10)

    assert not thread.is_alive(), "the second conversation must be refused, not parked"
    assert "already delegating" in second["out"].lower()
    assert second["read_resident"] is False, "refused before the resident model is read"
    assert "delegate answer" in first, "the first delegation is unaffected"


def test_the_gate_is_released_after_a_delegation_that_failed(rig, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("sub-agent exploded")

    monkeypatch.setattr(subagent, "run_subagent", boom)
    _run()
    assert delegation._delegation_gate.acquire(blocking = False), "a failure must not wedge it"
    delegation._delegation_gate.release()


def test_reset_for_tests_cannot_leave_the_gate_wedged(rig):
    """A test that dies mid-delegation must not take the rest of the session
    with it."""
    assert delegation._delegation_gate.acquire(blocking = False)
    delegation._reset_for_tests()
    assert delegation._delegation_gate.acquire(blocking = False), "reset must release it"
    delegation._delegation_gate.release()


def test_reset_does_not_free_a_gate_a_running_delegation_holds(rig, monkeypatch):
    """The other side of the same helper. threading.Lock has no owner, so release()
    from a thread that never acquired it succeeds -- a reset firing while another
    conversation is mid-swap would hand its slot away and let a second delegation
    start on top of it. locked() cannot tell the two cases apart (it is True either
    way), so the holder is recorded and compared."""
    started = threading.Event()
    finish = threading.Event()

    def slow_call(messages, tools):
        started.set()
        finish.wait(5)
        return {"content": "delegate answer", "tool_calls": []}

    monkeypatch.setattr(delegation, "_call_model_for", lambda binding: slow_call)

    thread = threading.Thread(target = _run)
    thread.start()
    try:
        assert started.wait(5), "the delegation never started"
        delegation._reset_for_tests()
        assert not delegation._delegation_gate.acquire(blocking = False), \
            "reset must not hand a running delegation's slot to another conversation"
    finally:
        finish.set()
        thread.join(10)


# ── what actually moved ─────────────────────────────────────────────────


def test_a_delegate_that_never_loaded_is_not_reported_as_a_lost_model(rig):
    """load() raises before it touches the backend when auto-switch is off or the
    delegate is not downloaded -- the likeliest first-run failure there is. Nothing
    was unloaded, so an unconditional restore fails for the same reason and tells
    the user their model is gone when it never moved."""
    rig.fail_on = "repo/Coder:Q4"
    out = _run()
    assert rig.calls == ["repo/Coder:Q4"], "the restore must not even be attempted"
    assert "WARNING" not in out, "nothing was swapped, so nothing was lost"
    assert "could not load repo/Coder:Q4" in out


def test_a_delegation_does_not_un_pin_the_users_own_model(rig):
    """The one defect whose damage lands AFTER the delegation reports success.

    The restore goes through auto-switch, which clears _loaded_by_user_action on
    every swap it performs -- right for an API-driven swap, wrong for putting a
    hand-loaded model back. With "unload API-loaded models only" on and an idle TTL
    set, the idle loop had been sparing the user's model; after one delegation it
    quietly stops, unloads it minutes later, and the next message pays a cold load
    with nothing connecting it to the delegation that caused it."""
    rig.pinned = True
    _run()
    assert rig.pin_trail == [("repo/Coder:Q4", False), ("repo/Primary:Q4", False)], \
        "the premise: both loads clear the pin, the delegate's included"
    assert rig.pinned is True, "the user's model must come back pinned as they left it"


def test_a_model_that_was_not_pinned_is_not_pinned_by_the_restore(rig):
    """The other direction: restore_user_pin only ever SETS the flag, so a model
    that was API-loaded must not come back looking hand-loaded and start being
    spared by an idle unloader scoped to API loads."""
    rig.pinned = False
    _run()
    assert rig.pinned is False


def test_a_delegation_that_had_nothing_to_restore_says_what_it_left_loaded(rig):
    """"Nothing was loaded before" and "nothing needs putting back" are not the same
    thing. A chat on an external provider delegates, a multi-gigabyte GGUF is loaded
    for the delegate, and it stays resident for the life of the process -- on a 6 GB
    card that is the whole VRAM budget, spent with nothing in the result saying so."""
    rig.resident = None
    out = _run()
    assert rig.calls == ["repo/Coder:Q4"], "there was nothing to put back"
    assert "WARNING" not in out, "nothing was lost, so this is not a warning"
    assert "nothing was put back" in out and "stays loaded" in out, out
    assert "repo/Coder:Q4" in out.split("Note:")[-1], "name the model left loaded"


def test_nothing_is_reported_as_left_loaded_when_the_delegate_never_loaded(rig):
    """The other half of the note, and the one that can hand the user a false VRAM
    report. With nothing resident AND the delegate's load failing, the note's serves()
    guard is all that stops the result announcing "repo/Coder:Q4 is loaded now and
    stays loaded" in the same breath as saying it could not be loaded. Running out of
    VRAM is the ordinary way to reach exactly this pair."""
    rig.resident = None
    rig.fail_on = "repo/Coder:Q4"
    out = _run()
    assert "could not load repo/Coder:Q4" in out, out
    assert "stays loaded" not in out, "nothing is loaded; do not claim the delegate is"
    assert "WARNING" not in out, "there was nothing to put back, so nothing was lost"


def test_a_media_role_is_refused_even_when_its_binding_would_resolve(rig, monkeypatch):
    """image and video bindings "record the preference and feed defaults" in v1 --
    they name a media model, and delegation loads onto the CHAT backend. Nothing
    stopped ask_model targeting one, so a bound image model that happened to resolve
    would unload the user's chat model to make room for something that cannot answer
    a task at all."""
    monkeypatch.setattr(storage, "get_role_bindings", lambda: {
        "image": {"model": "repo/Flux:Q4"},
        "video": {"model": "repo/Wan:Q4"},
        "primary": {"model": "repo/Other:Q4"},
    })
    for role in ("image", "VIDEO "):
        out = _run(role = role)
        assert rig.calls == [], "the chat backend must not be swapped to a media model"
        assert "not a chat model" in out, out


def test_a_delegate_that_died_mid_swap_still_puts_the_model_back(rig, monkeypatch):
    """The other direction, and the expensive one. Three of load()'s raise sites fire
    AFTER auto-switch has already unloaded the resident model (loader.py:416/:423/
    :425), and the delegate's launch running out of VRAM is the ordinary way to reach
    them. A flag set on load()'s return cannot tell "nothing moved" from "moved, then
    failed" -- and getting it wrong the second way loses the user's model in silence:
    no restore attempted, and a result that reads exactly like nothing happened.
    Asking the loader whether what we want is already served answers both."""
    def load_after_tearing_the_resident_down(model_id, overrides = None):
        rig.calls.append(model_id)
        rig.resident = None  # auto-switch unloaded it, then the launch failed
        raise delegation._loader_error("out of VRAM loading " + model_id)

    monkeypatch.setattr(rig, "load", load_after_tearing_the_resident_down)

    out = _run()
    # Asserted first and deliberately: the missing WARNING is only the symptom. The
    # defect is that the restore is never attempted, and a test that checked the
    # string alone would pass on a restore that ran and then said nothing.
    assert rig.calls == ["repo/Coder:Q4", "repo/Primary:Q4"], "the restore must be attempted"
    assert "WARNING" in out and "repo/Primary:Q4" in out, \
        "a restore that failed too leaves the user on another model; say which one"


# ── one id comparison, the loader's ─────────────────────────────────────


def test_the_per_turn_guard_accepts_any_alias_the_loader_serves(rig, monkeypatch):
    """Delegation's own string compare matched only the id resident_model_id
    advertises, so a binding naming a bare repo, a revision, or a standalone file's
    index alias was a FALSE refusal on the delegate's first turn -- after
    availability said READY, after load() confirmed the load, and after both swaps
    had been paid for."""
    monkeypatch.setattr(delegation, "_call_model_for", _REAL_CALL_MODEL_FOR)
    monkeypatch.setattr(delegation, "_model_call",
                        lambda: (lambda messages, tools: {"content": "delegate answer", "tool_calls": []}))

    loaded = rig.load

    def load_and_report_another_alias(model_id, overrides = None):
        loaded(model_id)
        if model_id == "repo/Coder:Q4":
            # The same model under a different id the resolver indexes.
            rig.resident = r"C:\models\qwen3-coder.gguf"

    monkeypatch.setattr(rig, "load", load_and_report_another_alias)
    monkeypatch.setattr(rig, "serves", lambda model_id: True)

    out = _run()
    assert "delegate answer" in out, "the loader says it serves the delegate; do not refuse"


def test_the_per_turn_guard_refuses_when_the_loader_says_no(rig, monkeypatch):
    """And the converse: serves() decides, not a string that happens to match."""
    monkeypatch.setattr(delegation, "_call_model_for", _REAL_CALL_MODEL_FOR)
    monkeypatch.setattr(delegation, "_model_call",
                        lambda: (lambda messages, tools: {"content": "wrong model's answer", "tool_calls": []}))
    monkeypatch.setattr(rig, "serves", lambda model_id: False)
    out = _run()
    assert "wrong model's answer" not in out
    assert "repo/Coder:Q4" in out, "name the model that was supposed to answer"
    assert rig.calls[-1] == "repo/Primary:Q4", "the restore still runs"


# ── what a failed attempt costs ─────────────────────────────────────────


def test_a_setup_that_failed_does_not_burn_a_delegation(rig, monkeypatch):
    """The window was spent at the moment it was checked, so two OSErrors out of
    workspace.create left the session refused for ten minutes having loaded
    nothing."""
    real_create = workspace.create
    remaining = {"failures": 2}

    def create(session_id, role):
        if remaining["failures"] > 0:
            remaining["failures"] -= 1
            raise OSError("read-only workdir")
        return real_create(session_id, role)

    monkeypatch.setattr(workspace, "create", create)
    assert "read-only" in _run()
    assert "read-only" in _run()
    out = _run()
    assert "delegate answer" in out, "a third attempt must still be allowed to run"
    assert rig.calls == ["repo/Coder:Q4", "repo/Primary:Q4"]


def test_a_load_that_never_reached_the_backend_does_not_burn_a_delegation(rig):
    """The same principle one line further down. The window was spent before the
    load, but load() raises before it touches the backend when auto-switch is off or
    the delegate is not downloaded -- no swap, nothing to pay for. Two of those and
    the user who then fixes the setting is told "the delegation limit was reached"
    for ten minutes."""
    rig.fail_on = "repo/Coder:Q4"
    assert "could not load repo/Coder:Q4" in _run()
    assert "could not load repo/Coder:Q4" in _run()
    rig.fail_on = None
    out = _run()
    assert "delegate answer" in out, "a load that cost no swap must cost no delegation"


def test_the_window_ledger_does_not_grow_one_entry_per_conversation(rig):
    """_recent kept a key for every session that ever delegated, for the life of
    the process. A session whose stamps have all expired loses its key."""
    delegation._window_record("s-gone", 0.0)
    assert "s-gone" in delegation._recent
    assert delegation._window_has_room("s-gone", delegation.DELEGATION_WINDOW_S + 1.0)
    assert "s-gone" not in delegation._recent, "an expired session must not be kept"


def test_write_brief_returns_its_text_even_when_the_write_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "_workdir_for", lambda session_id: str(tmp_path))
    files = workspace.create("s1", "coding")
    monkeypatch.setattr(workspace, "_write_text",
                        lambda path, text, append = False: (_ for _ in ()).throw(OSError("read-only")))
    body = workspace.write_brief(files, role = "coding", task = "rewrite the parser")
    assert "rewrite the parser" in body


def test_a_brief_that_could_not_be_written_still_reaches_the_delegate(rig, monkeypatch):
    """write_brief is never-raises by design. Re-reading brief.md made a silent
    write failure fail the delegation after both swaps, with the error text blaming
    the delegate model for an OSError on a file."""
    monkeypatch.setattr(workspace, "_write_text",
                        lambda path, text, append = False: (_ for _ in ()).throw(OSError("read-only")))
    seen = {}

    def spy(**kwargs):
        seen.update(kwargs)
        return subagent.SubagentResult("delegate answer", 1, "answered")

    monkeypatch.setattr(subagent, "run_subagent", spy)
    out = _run()
    assert "delegate answer" in out
    assert "rewrite the parser" in seen["messages"][1]["content"], "the text, not a re-read"


class _Unprintable:
    def __str__(self):
        raise RuntimeError("this value has no string form")


class _HostileMapping(dict):
    def get(self, *args, **kwargs):
        raise RuntimeError("this mapping refuses to be read")


def test_execute_never_raises_on_bad_arguments(rig):
    hostile = _HostileMapping(role = "coding", task = "t")
    for arguments in (
        None, [], {"role": None}, {"role": "coding"}, {"task": "x"}, {"role": 5, "task": 6},
        {"role": _Unprintable(), "task": "t"},      # a value whose str() raises
        {"role": "coding", "task": _Unprintable()},
        hostile,                                     # a dict subclass whose .get raises
    ):
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
