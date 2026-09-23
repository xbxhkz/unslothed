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
may happen at all.
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

_loader = _loader_module
_active = threading.local()
_recent: dict[str, list[float]] = {}
_recent_guard = threading.Lock()


def _loader_error(message: str):
    return _loader_module.LoaderError(message)


def _reset_for_tests() -> None:
    with _recent_guard:
        _recent.clear()
    _active.delegation_id = None


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


def _same_model(resident: Optional[str], expected: str) -> bool:
    """Whether ``resident`` is what a request for ``expected`` would be served by.

    Case-folded, the same way both halves of a swap identity are folded in the
    loader. A binding that names no quant ("repo/Model") is satisfied by any quant
    of that repo, because that is what auto-switch does with a bare id; a binding
    that names one ("repo/Model:Q4_K_M") is held to it.
    """
    if not resident or not expected:
        return False
    left = str(resident).strip().lower()
    right = str(expected).strip().lower()
    if left == right:
        return True
    return ":" not in right and left.split(":", 1)[0] == right


def _require_resident(expected: str) -> None:
    """Raise unless ``expected`` is still the loaded model. See _call_model_for."""
    try:
        resident = _loader.resident_model_id()
    except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
        raise _loader_error(f"could not confirm {expected} is still loaded: {exc}") from exc
    if not _same_model(resident, expected):
        raise _loader_error(
            f"{expected} is no longer the loaded model -- {resident or 'nothing'} is -- so "
            "anything it answered now would not be the delegate's work"
        )


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


def _window_ok(session_id: str, now: float) -> bool:
    with _recent_guard:
        stamps = [t for t in _recent.get(session_id, []) if now - t < DELEGATION_WINDOW_S]
        if len(stamps) >= MAX_DELEGATIONS_PER_WINDOW:
            _recent[session_id] = stamps
            return False
        stamps.append(now)
        _recent[session_id] = stamps
        return True


def execute(name: str, arguments, **kwargs) -> str:
    """Handler for ask_model. Never raises."""
    try:
        return _execute(arguments, **kwargs)
    except BaseException as exc:  # noqa: BLE001 - a delegation must not break the turn
        return f"Error: delegation failed: {exc}"


def _execute(arguments, *, session_id = None, cancel_event = None, **kwargs) -> str:
    from core.inference import model_roles

    if active_delegation_id() is not None:
        return (
            "Error: a delegation is already running; the model you delegated to "
            "cannot delegate again. Do this part yourself."
        )

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

    if not _window_ok(str(session_id), time.monotonic()):
        return (
            f"Error: the delegation limit was reached -- already delegated "
            f"{MAX_DELEGATIONS_PER_WINDOW} times in the last {int(DELEGATION_WINDOW_S / 60)} "
            "minutes (each swaps models twice). Continue with the current model."
        )

    # Only a backstop now: resident_is_restorable() above already refused the two
    # states this raises in.
    resident = _loader.resident_model_id()
    files = workspace.create(session_id, role)
    workspace.write_brief(files, role = role, task = task, context = str(args.get("context") or ""))
    paths = workspace.relative_paths(files)

    _active.delegation_id = files.delegation_id
    try:
        # No overrides are passed: load() ignores them and applies the model's own
        # saved launch settings through auto-switch, exactly as an API request
        # naming that model would. A second conversion path would disagree with it.
        _loader.load(binding.model)
        result = subagent.run_subagent(
            messages = [
                {"role": "system", "content": f"You are the {role} model. Write your result to {paths['work']}."},
                {"role": "user", "content": open(files.brief, encoding = "utf-8").read()},
            ],
            tools = _delegate_tools(),
            call_model = _call_model_for(binding),
            execute_tool = _delegate_execute_tool(),
            on_round = lambda line: workspace.append_transcript(files, line),
            cancel_event = cancel_event,
            session_id = session_id,
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
        restore_note = _restore(resident)
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


def _delegate_execute_tool():
    from core.inference.tools import execute_tool

    return execute_tool
