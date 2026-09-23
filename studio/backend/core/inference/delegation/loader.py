# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The one place that swaps the chat model.

It drives the route's own OpenAI auto-switch, ``_maybe_auto_switch_model``, and
nothing below it. That function is the in-process swap protocol, not a step in
it: it resolves ``repo:VARIANT`` to a concrete local path with the local model
resolver, applies the model's saved launch settings through
``model_override_load_kwargs``, takes the auto-switch lock, the process-wide swap
gate and the keep-warm lifecycle gate, calls the loader with
``current_request_counted = True`` so the drain does not wait on the very request
that asked for the swap, retries without a stale ``gpu_ids`` pin, and records the
advertised id afterwards. An adapter that called the loader underneath it got one
of those nine steps, hung on the drain forever, and spent ~53s probing
``repo:VARIANT`` as a remote repo -- after unloading the user's model.

Two rules keep this module usable from a tool call:

* Nothing here imports ``routes.inference`` or ``main``. Both are read through
  ``sys.modules``: core code must not depend on the route layer, and a tool
  worker must not drag the route module's imports in behind it.
* Every entry point is synchronous. The caller runs in a tool worker thread with
  no running loop, so the one coroutine involved is run on a loop of its own.
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Optional

# Recorded as the subject of the loads a delegation causes, so the API monitor's
# lifecycle rows say what asked for them. An internal label, not a user id.
DELEGATION_SUBJECT = "delegation"

# Auto-switch returns without loading anything when its toggle is off (see the
# early return in _maybe_auto_switch_model), so delegation cannot swap without it.
# The toggle is named here exactly as the UI names it -- "Switch model by request"
# in Settings > API, section "Model auto-switch (OpenAI API)" -- because whoever
# reads this message has to go and find it.
AUTO_SWITCH_OFF_MESSAGE = (
    "delegation swaps models through Model auto-switch, which is off. Turn on "
    "'Switch model by request' in Unsloth Studio under Settings > API, then try again."
)


class LoaderError(RuntimeError):
    """A swap failed, or did not happen. Raised so the caller can report it
    rather than carry on against a model that is not the one it asked for."""


def _route():
    """The inference route module, if this process has one, else None."""
    return sys.modules.get("routes.inference")


def _llama_backend():
    """The llama.cpp backend the route owns, via the route's own getter."""
    getter = getattr(_route(), "get_llama_cpp_backend", None)
    return getter() if getter is not None else None


def _gguf_identity():
    """``(model_identifier, hf_variant, advertised)`` for the resident GGUF, else None.

    This is ``llama_keepwarm._loaded_identity``, the function the idle-unload
    stash captures with -- it stores the quant "so the reload restores the exact
    freed variant", which is this module's problem exactly. Called rather than
    copied, so the two cannot drift.
    """
    backend = _llama_backend()
    if backend is None:
        return None
    from core.inference.llama_keepwarm import _loaded_identity

    return _loaded_identity(backend)


def _format_gguf_id(identity) -> str:
    """The resident GGUF's id as ``load()`` takes it back: the advertised repo id
    when an auto-switch load set one, else the identifier, plus the loaded quant."""
    _identifier, variant, advertised = identity
    return f"{advertised}:{variant}" if variant else advertised


def _resolvable_id(identity) -> Optional[str]:
    """The resident GGUF as an id that resolves BACK through the resolver ``load()``
    swaps on, or None when no form of it does.

    ``"<id>:<variant>"`` first, then the bare ``"<id>"``. The bare fallback is not
    cosmetic: a standalone ``.gguf`` gets its ``hf_variant`` parsed out of its own
    filename (``llama_cpp.py``, "For local GGUF files, extract variant from
    filename if absent"), while the resolver's entry for a single file lists no
    variants at all, so the quant-qualified form misses and only the bare one
    resolves. Reporting the prettier form there would hand the caller an id that
    cannot be reloaded, which is a model the user loses rather than a delegation
    they are told no about.
    """
    _identifier, variant, advertised = identity
    candidates = (f"{advertised}:{variant}", advertised) if variant else (advertised,)
    for candidate in candidates:
        if candidate and _resolve_local(candidate) is not None:
            return candidate
    return None


def _resolve_local(model_id: str):
    """``(load_path, gguf_variant, loader_id)`` for a downloaded local GGUF, else None.

    The resolver auto-switch itself resolves with, so "is this here" is answered
    the same way on both sides of the call. May scan: this runs in a tool worker
    thread, never on the event loop, and the scan is cached for 5s.
    """
    from core.inference.local_model_resolver import resolve_local_gguf

    return resolve_local_gguf(model_id)


def _transformers_resident() -> Optional[str]:
    """The Transformers/safetensors model the orchestrator holds, else None.

    ``peek_inference_backend`` "Never constructs one", and the orchestrator is
    read through ``sys.modules`` so a process that never imported it reads as
    "nothing loaded" instead of paying for the import (and its torch warm).
    """
    peek = getattr(sys.modules.get("core.inference.orchestrator"), "peek_inference_backend", None)
    if peek is None:
        return None
    return getattr(peek(), "active_model_name", None) or None


def resident_model_id() -> Optional[str]:
    """What is loaded right now, as an id ``load()`` can put back.

    Normally ``"<id>:<hf_variant>"`` -- the advertised repo id when an auto-switch
    load recorded one, else the model identifier. Dropping the quant would let the
    restore auto-pick a different one. The form returned is always one that
    resolves back through the resolver the swap uses (see :func:`_resolvable_id`),
    so for a standalone ``.gguf`` this is the bare id: the id that works beats the
    id that reads better.

    None when nothing is loaded.

    Raises LoaderError in the two states that are neither "nothing loaded" nor
    restorable -- a resident Transformers model, and a GGUF no form of whose id
    resolves. Both are backstops: :func:`resident_is_restorable` answers the same
    question without raising, and a caller is expected to ask it first. They raise
    rather than answer None because None reads as "nothing to put back", and a
    caller acting on that restores nothing and says nothing, which is how a user's
    model gets silently replaced by the delegate's.
    """
    try:
        identity = _gguf_identity()
        active = None if identity is not None else _transformers_resident()
        restorable = _resolvable_id(identity) if identity is not None else None
    except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
        raise LoaderError(f"could not read the resident model: {_reason(exc)}") from exc
    if identity is not None:
        if restorable is None:
            raise LoaderError(
                f"the resident model '{_format_gguf_id(identity)}' is not in this server's "
                "local model index under any id that would load it back, so it could not be "
                "restored after a delegation. Check resident_is_restorable() before swapping."
            )
        return restorable
    if active:
        raise LoaderError(
            f"'{active}' is loaded through Transformers, and delegation can only reload a "
            "GGUF, so it could not be put back afterwards. Check resident_is_restorable() "
            "before swapping."
        )
    return None


def resident_is_restorable() -> bool:
    """Whether a swap now could put back what is loaded now.

    False while a Transformers/safetensors model is resident: loading a GGUF
    unloads it (``_load_model_impl`` tears the Unsloth model down before the GGUF
    launch) and auto-switch can only ever load a downloaded local GGUF, so it
    could not be restored.

    False, too, for a resident GGUF whose id does not resolve back through the
    resolver the swap runs on -- a standalone file outside every scan root, say.
    "A GGUF is loaded" is not the same question as "this can be loaded again", and
    only the second one is worth anything to a caller about to unload it. Asking
    costs a resolver lookup (cached, and this is never called from the event loop);
    not asking costs the user their model.

    True for a resident GGUF that does resolve, and when nothing is loaded.

    Never raises: a backend this cannot read is one it cannot promise to restore,
    so an unreadable answer is a refusal.
    """
    try:
        if _transformers_resident() is not None:
            return False
        identity = _gguf_identity()
        if identity is None:
            return True
        return _resolvable_id(identity) is not None
    except BaseException:  # noqa: BLE001 - fail closed: cannot tell means cannot promise
        return False


def _request_stand_in() -> types.SimpleNamespace:
    """What auto-switch gets where a route passes its FastAPI ``Request``.

    A tool call has no Request, but the swap path reads exactly four things off
    one, and every read is a ``getattr`` with a default or sits inside a guard:
    ``.scope`` (checked with ``isinstance(..., dict)``), ``.url.path`` (only used
    when it is a ``str``), ``.headers`` and ``.state`` (``None`` is handled), and
    ``.app.state.llama_parallel_slots``. That last one is why this carries the
    real app: without it ``_resolve_parallel_slots`` falls back to one slot, so
    every delegation -- including the restore of the user's own model -- would
    quietly reload it with fewer slots than the server was started with.
    """
    return types.SimpleNamespace(app = getattr(sys.modules.get("main"), "app", None))


def _reason(exc: BaseException) -> str:
    """What to show for a failure: an HTTPException carries its message in
    ``detail``, and an exception with no message shows its type rather than
    nothing at all."""
    detail = getattr(exc, "detail", None)
    text = str(detail) if detail else str(exc)
    return text or type(exc).__name__


def _resident_label() -> str:
    """The resident model, for an error message. Never raises: it is only ever
    the second half of a sentence about a failure that already happened."""
    try:
        identity = _gguf_identity()
        if identity is not None:
            return _format_gguf_id(identity)
        return _transformers_resident() or "nothing"
    except BaseException:  # noqa: BLE001 - a label must not replace the real error
        return "unreadable"


def _auto_switch_serves(model_id: str) -> bool:
    """Whether what is resident now is what auto-switch calls ``model_id``.

    Auto-switch can dedupe or fall through -- an unknown name is not an error
    there, it just serves the loaded model -- so a call that returned proves
    nothing on its own.

    The test is auto-switch's own ``_already_serving``: a closure inside
    ``_maybe_auto_switch_model``, so it cannot be called from here, and kept
    step for step. It compares the resolver's ``(load_path, variant,
    override_id)`` triple against the backend's identifier and advertised id,
    case-insensitively -- the same folding ``_switch_key`` applies to both halves
    of a swap identity -- and holds an explicit quant to the loaded one while a
    bare id is satisfied by any quant of that repo.
    """
    from core.inference.openai_auto_download import looks_like_quant, split_model_ref

    resolved = _resolve_local(model_id)  # the same resolution the swap ran on
    if resolved is None:
        return False
    target_id, variant, override_id = resolved
    _base, requested_variant = split_model_ref(model_id)
    bare = not looks_like_quant(requested_variant)

    backend = _llama_backend()
    if backend is None or not getattr(backend, "is_loaded", False):
        return False
    identifier = getattr(backend, "model_identifier", None)
    if not identifier:
        return False
    loaded_keys = {identifier.lower()}
    advertised = getattr(backend, "_openai_advertised_id", None)
    if advertised:
        loaded_keys.add(advertised.lower())
    if loaded_keys.isdisjoint({target_id.lower(), override_id.lower()}):
        return False
    if bare:
        return True
    if variant:
        return (getattr(backend, "hf_variant", None) or "").lower() == variant.lower()
    return True


def serves(model_id: str) -> bool:
    """Whether a request for ``model_id`` would be answered by what is loaded now.

    The same comparison auto-switch makes for itself, so delegation's per-turn
    guard and the loader's own post-load check can never disagree. It re-resolves
    ``model_id`` and matches the resident backend against both the resolved target
    and the override id, so every alias the resolver indexes counts -- a bare repo
    id, the concrete load path, a standalone file's path-free name. A caller
    comparing :func:`resident_model_id`'s output as a string matches only the one
    form that function happens to advertise, and refuses the rest.

    Raises LoaderError if residency cannot be read at all.
    """
    try:
        return _auto_switch_serves(model_id)
    except LoaderError:
        raise
    except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
        raise LoaderError(f"could not confirm {model_id} is loaded: {_reason(exc)}") from exc


def _run_coroutine(coro):
    """Run a coroutine from this (synchronous) tool thread.

    A fresh loop is correct here: every lock the swap path takes across the
    process -- the swap gate and the keep-warm lifecycle gate -- is a
    ``threading.Lock`` polled off ``asyncio.sleep`` precisely so it serialises
    across loops, and the one per-loop ``asyncio.Lock`` is created per loop on
    demand.
    """
    return asyncio.run(coro)


def load(model_id: str, overrides: Optional[dict] = None) -> None:
    """Make ``model_id`` the loaded chat model. Raises LoaderError on any failure.

    ``overrides`` is accepted for signature compatibility and is ignored. The
    model loads with its own saved per-model launch settings, exactly as it would
    if an API request had named it: auto-switch applies those through
    ``model_override_load_kwargs``, which maps the UI's keys, strips the llama
    flags it manages itself, and scopes the GGUF-only ones. A second conversion
    path here would silently disagree with that one.
    """
    try:
        from utils.openai_auto_switch_settings import get_openai_auto_switch_enabled

        enabled = get_openai_auto_switch_enabled()
    except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
        raise LoaderError(f"could not read the model auto-switch setting: {_reason(exc)}") from exc
    if not enabled:
        raise LoaderError(AUTO_SWITCH_OFF_MESSAGE)

    switch = getattr(_route(), "_maybe_auto_switch_model", None)
    if switch is None:
        raise LoaderError(
            "the inference route is not loaded in this process, so nothing can swap the model"
        )

    # Refuse a model that is not here BEFORE handing over. Auto-switch's miss path
    # offers to fetch it (``_maybe_auto_download_model``), and that fetch schedules
    # its watcher with ``asyncio.create_task`` on the running loop -- which here is
    # the one ``asyncio.run`` closes moments later, when the refusal unwinds this
    # call. The watcher would be cancelled at its first await, leaving a "download"
    # row running forever in the API monitor and releasing the single-flight slot
    # while the download is still going, which that module warns admits a second
    # multi-gigabyte fetch of the same repo. A delegation must not start a download
    # mid-turn regardless; this also spares it a pointless Hub round trip.
    try:
        available = _resolve_local(model_id)
    except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
        raise LoaderError(f"could not look up {model_id}: {_reason(exc)}") from exc
    if available is None:
        raise LoaderError(
            f"{model_id} is not downloaded on this server, so delegation cannot load it. "
            "Download it in Unsloth Studio first."
        )

    try:
        _run_coroutine(switch(model_id, _request_stand_in(), DELEGATION_SUBJECT))
    except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
        raise LoaderError(f"could not load {model_id}: {_reason(exc)}") from exc

    # Through serves(), not _auto_switch_serves() directly: the post-load check and
    # delegation's per-turn guard are then genuinely one path, and cannot drift.
    try:
        served = serves(model_id)
    except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
        raise LoaderError(f"could not confirm {model_id} is loaded: {_reason(exc)}") from exc
    if not served:
        raise LoaderError(
            f"auto-switch did not load {model_id}: the resident model is {_resident_label()}. "
            "It loads a downloaded local GGUF and nothing else."
        )
