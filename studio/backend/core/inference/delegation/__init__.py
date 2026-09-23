# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hand a task to another role's model, then come back.

THE RULE THAT MATTERS: what gets restored is the model that was RESIDENT when
the call started -- never whatever the "primary" role happens to name. If the
user loaded a model by hand, that is what comes back. A delegation must be
invisible to the conversation except for its result.

Two questions are asked before anything moves, in this order:

* Could what is loaded now be loaded again? The loader answers that without
  raising (``resident_is_restorable``). A No is a plain refusal: losing a
  delegation costs one message, losing the user's model costs their session.
* Is the delegate still resident? Asked again before EVERY turn, not once after
  the load, because ``load()`` releases every lock it took before it returns.

Two budgets guard the primary's turn, because the delegate runs inside one of
its tool calls: the sub-agent's own caps, and a limit on how often delegation
may happen at all. A third guard is process-wide: one delegation at a time
anywhere in the server. Two interleaving delegations each read the other's
delegate as "what was resident", both restores succeed, the user's model is
gone, and neither result carries a warning -- from inside either one, nothing
failed.

KNOWN LIMITATION, not fixed here: the delegate is given ALL_TOOLS minus
ask_model, not the tools the user left switched on. ``execute_tool``'s signature
(tools.py:10039) carries no enabled-tool list and no permission mode; the only
code holding them is the tool loops, which sit above this layer in files this
fork does not edit. So a tool switched off in the UI is still reachable by a
delegate. The dangerous half of that exposure is covered by the per-call risk
gate in :func:`_delegate_execute_tool`, which refuses outright anything the
approval classifier would have prompted on; the rest is recorded here.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from core.inference.delegation import loader as _loader_module
from core.inference.delegation import subagent, workspace
from core.inference.delegation.schemas import DELEGATION_TOOL_NAMES  # noqa: F401 - re-export
from core.inference.tool_readiness import READY

MAX_DELEGATIONS_PER_WINDOW = 2
# The spec says "per assistant turn". A tool cannot observe turn boundaries --
# execute_tool receives a session id, not a turn id -- so this is the closest
# observable approximation, and the refusal says so.
DELEGATION_WINDOW_S = 600.0

DELEGATE_DENIED_MESSAGE = (
    "Error: {name} needs the user's approval and a delegated model cannot ask for it -- "
    "the approval prompt belongs to the conversation you were called from, which is "
    "blocked waiting on you. Say in your result what you needed and why; the model that "
    "called you can run it there, where the user can see the prompt."
)

DELEGATION_BUSY_MESSAGE = (
    "Error: another conversation is already delegating. Only one delegation runs at a "
    "time, because each one swaps the loaded model out and back. Try again shortly."
)

# What is forwarded from the primary's tool call into the delegate's. Collected by
# name rather than splatted: execute_tool has a closed signature (tools.py:10039),
# so one unexpected keyword would raise on every tool call the delegate makes.
# Deliberately absent: output_callback -- the delegate's stdout would paint onto the
# primary's ask_model card as if the primary had produced it. cancel_event is absent
# too, but only because run_subagent takes it as a named parameter of its own and so
# never forwards it; _delegate_execute_tool binds it instead.
_TOOL_PASSTHROUGH = (
    "timeout", "thread_id", "rag_scope", "disable_sandbox", "website_policy",
    "conversation_branch", "conversation_budget_tokens", "conversation_token_counter",
)

_loader = _loader_module
_active = threading.local()
_recent: dict[str, list[float]] = {}
_recent_guard = threading.Lock()
# Process-wide and non-blocking: a delegation can run for ten minutes, and a second
# conversation must be told no now rather than parked behind it. The thread-local
# depth guard above is checked first, so a nested call never contends this.
_delegation_gate = threading.Lock()


def _loader_error(message: str):
    return _loader_module.LoaderError(message)


def _reset_for_tests() -> None:
    with _recent_guard:
        _recent.clear()
    _active.delegation_id = None
    try:
        _delegation_gate.release()
    except RuntimeError:
        pass  # already released, which is the normal case


def active_delegation_id() -> Optional[str]:
    return getattr(_active, "delegation_id", None)


def _model_call():
    """The raw one-turn call against whatever backend is loaded.

    Separated from :func:`_call_model_for` so a test can inject a model without
    also losing the residency check that wraps it -- that check is the thing most
    worth testing here, and a fake that replaced the whole adapter would take it
    out along with the network."""
    from core.inference.delegation.model_call import call_loaded_model

    return call_loaded_model


def _call_model_for(binding):
    """A callable(messages, tools) -> assistant message for this role's model.

    Every turn re-checks that the delegate is still what is loaded. ``load()``
    releases the auto-switch lock, the process-wide swap gate and the keep-warm
    lifecycle gate before it returns, so an ordinary API request can swap the
    model away between the load and the delegate's first token. Without this the
    answer would come from whatever took its place, silently, and be written to
    work.md as the delegate's work. Raising instead fails the delegation visibly,
    and the restore in :func:`_execute`'s ``finally`` still runs. The check is a
    resident-id read -- two attribute reads off the backend -- so it costs
    nothing per round.
    """
    raw = _model_call()

    def call_model(messages, tools):
        _require_resident(binding.model)
        return raw(messages, tools)

    return call_model


def _require_resident(expected: str) -> None:
    """Raise unless ``expected`` is still the loaded model. See _call_model_for.

    The question is asked of the loader, not answered here. ``serves()`` is the
    same comparison auto-switch makes for itself and that ``load()`` confirms the
    swap with, so this guard and that check can never disagree. Delegation's own
    string compare did: it matched only the id ``resident_model_id`` advertises,
    so a binding naming a bare repo, a revision or a standalone file's index alias
    was a false refusal on the delegate's FIRST turn -- after availability had said
    READY and load() had confirmed the load, and after both swaps had been paid.
    """
    try:
        served = _loader.serves(expected)
    except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
        raise _loader_error(f"could not confirm {expected} is still loaded: {exc}") from exc
    if not served:
        raise _loader_error(
            f"{expected} is no longer the loaded model -- {_resident_name()} is -- so "
            "anything it answered now would not be the delegate's work"
        )


def _resident_name() -> str:
    """The loaded model's id, for a message. Never raises: this only decorates a
    refusal that has already been decided, and a failure to name the model must
    not replace the real reason with its own."""
    try:
        return _loader.resident_model_id() or "nothing"
    except BaseException:  # noqa: BLE001 - a label must not mask the refusal
        return "an unreadable model"


def _resident_problem() -> str:
    """Why the loaded model could not be put back, named. Never raises.

    ``resident_model_id`` is the loader's own explanation of the two unrestorable
    states and names the model in both, so it is read for the message -- after
    ``resident_is_restorable`` has already said No, which is the only reason it is
    safe to let it raise here."""
    try:
        resident = _loader.resident_model_id()
    except BaseException as exc:  # noqa: BLE001 - the message IS the answer here
        return str(exc) or type(exc).__name__
    if resident:
        return f"the loaded model {resident} could not be loaded back afterwards"
    return "the loader could not promise to restore what is loaded"


def _window_prune(session_id: str, now: float) -> list[float]:
    """The session's stamps with the expired ones dropped. Caller holds the guard.

    A session left with none loses its key: otherwise _recent grows one entry per
    conversation and never shrinks for the life of the process.
    """
    stamps = [t for t in _recent.get(session_id, ()) if now - t < DELEGATION_WINDOW_S]
    if stamps:
        _recent[session_id] = stamps
    else:
        _recent.pop(session_id, None)
    return stamps


def _window_has_room(session_id: str, now: float) -> bool:
    """Whether another delegation may start. Records nothing -- see _window_record."""
    with _recent_guard:
        return len(_window_prune(session_id, now)) < MAX_DELEGATIONS_PER_WINDOW


def _window_record(session_id: str, now: float) -> None:
    """Spend one of the window's delegations.

    Split from the check so a refusal, or a setup that failed before anything was
    swapped, costs nothing: two OSErrors out of workspace.create used to leave the
    session refused for ten minutes having loaded nothing at all. Called
    immediately before the first load, the first line that actually spends a swap;
    the process-wide gate means nothing can slip between the check and here.
    """
    with _recent_guard:
        stamps = _window_prune(session_id, now)
        stamps.append(now)
        _recent[session_id] = stamps


def execute(name: str, arguments, **kwargs) -> str:
    """Handler for ask_model. Never raises."""
    try:
        return _execute(arguments, **kwargs)
    except BaseException as exc:  # noqa: BLE001 - a delegation must not break the turn
        return f"Error: delegation failed: {exc}"


def _execute(arguments, *, session_id = None, cancel_event = None, **kwargs) -> str:
    """The depth guard and the process-wide gate, around the delegation itself.

    The thread-local depth guard is checked first: it is the one that can tell a
    delegate "you cannot delegate again", and checking it first means a nested call
    never contends the gate. The gate is non-blocking, so a second conversation is
    refused immediately rather than parked behind a ten-minute delegation.
    """
    if active_delegation_id() is not None:
        return (
            "Error: a delegation is already running; the model you delegated to "
            "cannot delegate again. Do this part yourself."
        )

    if not _delegation_gate.acquire(blocking = False):
        return DELEGATION_BUSY_MESSAGE
    try:
        return _delegate(arguments, session_id = session_id, cancel_event = cancel_event, **kwargs)
    finally:
        # Wraps everything, including the restore: a delegation that died still has
        # to leave the next conversation able to run one.
        _delegation_gate.release()


def _delegate(arguments, *, session_id = None, cancel_event = None, **kwargs) -> str:
    from core.inference import model_roles

    args = arguments if isinstance(arguments, dict) else {}
    role = str(args.get("role") or "").strip()
    task = str(args.get("task") or "").strip()
    if not role:
        return "Error: ask_model needs a role."
    if not task:
        return "Error: ask_model needs a task, written so it stands alone."

    state = model_roles.availability(role)
    if state.state != READY:
        return f"Error: the role {role!r} is not usable: {state.detail}"
    binding = model_roles.resolve(role)
    if binding is None:
        return f"Error: no model is bound to the role {role!r}."

    # Asked BEFORE a file is written or anything is swapped: a delegation that
    # could not put the user's model back is not worth running at all.
    if not _loader.resident_is_restorable():
        return (
            "Error: delegation was refused because it could not put your model back afterwards: "
            f"{_resident_problem()}."
        )

    # str(session_id) means every call that carries no session id shares the one
    # bucket 'None'. Left that way deliberately: a tool call without a session has
    # nothing else to key on, and lumping them together errs towards refusing.
    session_key = str(session_id)
    # Checked here, spent at the load below. A refusal costs no directory, and a
    # setup that fails before the first swap costs no delegation.
    if not _window_has_room(session_key, time.monotonic()):
        return (
            f"Error: the delegation limit was reached -- already delegated "
            f"{MAX_DELEGATIONS_PER_WINDOW} times in the last {int(DELEGATION_WINDOW_S / 60)} "
            "minutes (each swaps models twice). Continue with the current model."
        )

    # Only a backstop now: resident_is_restorable() above already refused the two
    # states this raises in.
    resident = _loader.resident_model_id()
    files = workspace.create(session_id, role)
    # The brief's text comes back from the writer, not from re-reading the file.
    # write_brief is never-raises by design, so a delegate prompt built by opening
    # brief.md would fail the whole delegation -- after both swaps -- on a write
    # that was allowed to fail, and blame the delegate model for it.
    brief_text = workspace.write_brief(
        files, role = role, task = task, context = str(args.get("context") or "")
    )
    paths = workspace.relative_paths(files)

    passthrough = {key: kwargs[key] for key in _TOOL_PASSTHROUGH if key in kwargs}

    _active.delegation_id = files.delegation_id
    swapped = False
    try:
        _window_record(session_key, time.monotonic())
        # No overrides are passed: load() ignores them and applies the model's own
        # saved launch settings through auto-switch, exactly as an API request
        # naming that model would. A second conversion path would disagree with it.
        _loader.load(binding.model)
        swapped = True
        result = subagent.run_subagent(
            messages = [
                {"role": "system", "content": f"You are the {role} model. Write your result to {paths['work']}."},
                {"role": "user", "content": brief_text},
            ],
            tools = _delegate_tools(),
            call_model = _call_model_for(binding),
            execute_tool = _delegate_execute_tool(
                bypass = bool(passthrough.get("disable_sandbox")),
                cancel_event = cancel_event,
            ),
            on_round = lambda line: workspace.append_transcript(files, line),
            cancel_event = cancel_event,
            session_id = session_id,
            **passthrough,
        )
        workspace.write_work(files, result.text or "(the delegate produced no text)")
        summary = (result.text or "").strip()
        note = "" if result.stopped_because == "answered" else f" (stopped: {result.stopped_because})"
        body = (
            f"{binding.model} answered as {role}{note}.\n\n"
            f"{summary[:2000]}\n\n"
            f"Files: {paths['work']} (result), {paths['transcript']} (what it did), {paths['brief']}."
        )
    except BaseException as exc:  # noqa: BLE001 - reported below, after the restore
        body = f"Error: the {role} model failed: {exc}"
    finally:
        _active.delegation_id = None
        # Only restore what was actually swapped out. load() raises before it
        # touches the backend when auto-switch is off or the delegate is not
        # downloaded -- the likeliest first-run failure there is -- and an
        # unconditional restore would then fail for the same reason and tell the
        # user "the model now loaded is not the one you were using" about a model
        # that never moved. Never quietly lying must not become loudly lying.
        restore_note = _restore(resident) if swapped else ""
    return body if not restore_note else f"{body}\n\n{restore_note}"


def _restore(resident: Optional[str]) -> str:
    """Put back what was loaded. A failure here is reported, never swallowed:
    otherwise the next turn runs on a different model than the user believes."""
    if not resident:
        return ""
    try:
        _loader.load(resident)
        return ""
    except BaseException as exc:  # noqa: BLE001
        return (
            f"WARNING: could not reload {resident} after the delegation ({exc}). "
            f"The model now loaded is not the one you were using."
        )


def _delegate_tools() -> list:
    """Everything the primary can use, minus delegation itself."""
    from core.inference.tools import ALL_TOOLS

    return [t for t in ALL_TOOLS if t.get("function", {}).get("name") not in DELEGATION_TOOL_NAMES]


def _delegate_execute_tool(*, bypass: bool, cancel_event = None):
    """The audited executor, with a gate standing in for the approval prompt.

    The delegate's tool calls happen inside one of the primary's, so they never
    reach a tool loop -- and every approval prompt in this codebase lives in a tool
    loop, above execute_tool (studio_tool_loop.py:1165, llama_cpp.py:23959,
    safetensors_agentic.py:1252). execute_tool's signature carries no permission
    mode and no enabled-tool list, so the delegate cannot inherit the handshake. It
    is held to less instead: anything the classifier would have prompted on is
    refused outright.

    ``bypass`` is the loops' ``bypass_permissions``, which reaches here as
    ``disable_sandbox`` -- the same flag they gate the prompt on
    (studio_tool_loop.py:1142/:1224, llama_cpp.py:23930/:24016). When it is set the
    user switched the gate off for the primary, so the delegate runs ungated too
    and matches it.

    ``cancel_event`` is bound here rather than forwarded: run_subagent takes it as
    a named parameter of its own, so it never lands in the ``**tool_kwargs`` it
    passes on. Without it a delegate sitting in a long terminal call ignores Stop
    until that call returns, not merely until the next round.
    """
    from core.inference.tools import execute_tool

    def run(name, arguments, **kwargs) -> str:
        kwargs.setdefault("cancel_event", cancel_event)
        return execute_tool(name, arguments, **kwargs)

    if bypass:
        return run

    def gated(name, arguments, **kwargs) -> str:
        from core.inference.tools import is_high_risk_tool_call

        try:
            risky = is_high_risk_tool_call(
                name, arguments if isinstance(arguments, dict) else {}
            )
        except BaseException:  # noqa: BLE001 - an unclassifiable call is a refused call
            risky = True
        if risky:
            return DELEGATE_DENIED_MESSAGE.format(name = name)
        return run(name, arguments, **kwargs)

    return gated
