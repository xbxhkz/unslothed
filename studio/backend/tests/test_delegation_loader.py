# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The swap adapter.

No test here loads a model. What IS tested is the wiring: that the adapter reads
the resident model from the backend rather than guessing, that it reuses the
route's own loader instead of re-implementing it, and that a load failure
surfaces as LoaderError rather than an arbitrary exception type.

Re-implementing the loader is the failure this project keeps paying for -- the
route's _load_model_impl is ~450 lines of arbiter, load intent, config and
draft-model handling.
"""

from __future__ import annotations

import inspect
import sys
import types

import pytest

from core.inference.delegation import loader


def _route_module(monkeypatch, backend = None, loader_calls = None):
    module = types.ModuleType("routes.inference")
    module._llama_cpp_backend = backend

    async def _load_model_impl(request, fastapi_request, current_subject, **kwargs):
        if loader_calls is not None:
            loader_calls.append((request, fastapi_request, current_subject))

    module._load_model_impl = _load_model_impl
    monkeypatch.setitem(sys.modules, "routes.inference", module)
    return module


def test_the_resident_model_is_read_from_the_backend(monkeypatch):
    backend = types.SimpleNamespace(is_loaded = True, model_identifier = "repo/Model:Q4")
    _route_module(monkeypatch, backend = backend)
    assert loader.resident_model_id() == "repo/Model:Q4"


def test_nothing_loaded_reads_as_none(monkeypatch):
    backend = types.SimpleNamespace(is_loaded = False, model_identifier = "stale")
    _route_module(monkeypatch, backend = backend)
    assert loader.resident_model_id() is None


def test_no_route_module_reads_as_none(monkeypatch):
    monkeypatch.delitem(sys.modules, "routes.inference", raising = False)
    assert loader.resident_model_id() is None


def test_load_delegates_to_the_routes_own_loader(monkeypatch):
    calls = []
    _route_module(monkeypatch, loader_calls = calls)
    monkeypatch.setattr(loader, "_run_coroutine", lambda coro: __import__("asyncio").run(coro))
    loader.load("repo/Model:Q4", {})
    assert len(calls) == 1, "the route's loader must be what runs"
    request, fastapi_request, subject = calls[0]
    assert fastapi_request is None, "no Request is available inside a tool call"
    assert isinstance(subject, str) and subject
    assert request.model_path == "repo/Model:Q4", "LoadRequest's field is model_path"


def test_a_load_failure_becomes_LoaderError(monkeypatch):
    module = _route_module(monkeypatch)

    async def _boom(request, fastapi_request, current_subject, **kwargs):
        raise RuntimeError("no VRAM")

    module._load_model_impl = _boom
    monkeypatch.setattr(loader, "_run_coroutine", lambda coro: __import__("asyncio").run(coro))
    with pytest.raises(loader.LoaderError) as excinfo:
        loader.load("repo/Model:Q4", {})
    assert "no VRAM" in str(excinfo.value)


def test_the_adapter_does_not_reimplement_loading():
    """Control against drift: the adapter must call the route's loader, not
    assemble its own from the arbiter and backend."""
    source = inspect.getsource(loader)
    assert "_load_model_impl" in source
    assert "acquire_for" not in source, "the route's loader already arbitrates; do not do it twice"


def test_unknown_override_keys_are_dropped_known_ones_pass_through(monkeypatch):
    """LoadRequest has no `extra` policy set, so pydantic silently ignores unknown
    kwargs today -- a role override like {"n_ctx": 16384} would do nothing, since
    the real field is max_seq_length. The adapter filters overrides to
    LoadRequest's real fields so that mistake is explicit rather than silent, and
    so the adapter keeps working if upstream ever switches to rejecting extras."""
    calls = []
    _route_module(monkeypatch, loader_calls = calls)
    monkeypatch.setattr(loader, "_run_coroutine", lambda coro: __import__("asyncio").run(coro))
    loader.load("repo/Model:Q4", {"n_ctx": 16384, "max_seq_length": 8192})
    assert len(calls) == 1
    request, _fastapi_request, _subject = calls[0]
    assert not hasattr(request, "n_ctx"), "unknown override key must not reach LoadRequest"
    assert request.max_seq_length == 8192, "a real LoadRequest field must pass through"
