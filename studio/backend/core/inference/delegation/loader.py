# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The one place that swaps the chat model.

It calls the ROUTE's own _load_model_impl rather than assembling a load from the
arbiter and the backend. That function is ~450 lines of GPU arbitration, load
intent, config resolution and draft-model handling; a second implementation
would drift from it, which is the failure this fork has already paid for once in
its readiness probes.

Two things it does NOT have, and does not need:
  * a FastAPI Request -- _load_model_impl uses it only for
    _request_used_api_key (whose own comment says it is "Total by construction"
    and "must never fail a load") and _resolve_parallel_slots (a getattr chain
    defaulting to one slot, which is right for a single delegate).
  * a user subject -- an internal label is passed instead.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

DELEGATION_SUBJECT = "delegation"


class LoaderError(RuntimeError):
    """A swap failed. Raised so the caller can report it rather than guess."""


def resident_model_id() -> Optional[str]:
    """What is loaded right now, read from the backend, or None.

    Read through sys.modules: core code must not import the route module."""
    module = sys.modules.get("routes.inference")
    backend = getattr(module, "_llama_cpp_backend", None) if module is not None else None
    if backend is None or not getattr(backend, "is_loaded", False):
        return None
    return getattr(backend, "model_identifier", None) or None


def _run_coroutine(coro):
    """Run a coroutine from this (synchronous) tool thread."""
    return asyncio.run(coro)


def load(model_id: str, overrides: Optional[dict] = None) -> None:
    """Load ``model_id``. Raises LoaderError on any failure."""
    module = sys.modules.get("routes.inference")
    if module is None:
        raise LoaderError("the inference route is not loaded in this process")
    impl = getattr(module, "_load_model_impl", None)
    if impl is None:
        raise LoaderError("the inference route does not expose a loader")
    try:
        from models.inference import LoadRequest

        # LoadRequest has no `extra` policy set, so today pydantic silently
        # ignores unknown kwargs -- a role override like {"n_ctx": 16384} would
        # do nothing (the real field is max_seq_length) without this filter
        # ever telling anyone. Filtering to real fields makes that explicit and
        # keeps this adapter working if upstream ever switches to rejecting
        # extra fields outright.
        allowed = set(LoadRequest.model_fields)
        filtered_overrides = {k: v for k, v in (overrides or {}).items() if k in allowed}

        # model_path, not model_name: the field is "Model identifier or local path".
        request = LoadRequest(model_path = model_id, **filtered_overrides)
        _run_coroutine(impl(request, None, DELEGATION_SUBJECT))
    except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
        raise LoaderError(f"could not load {model_id}: {exc}") from exc
