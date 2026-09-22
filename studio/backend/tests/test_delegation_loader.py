# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The swap adapter.

No test here loads a model, starts llama-server, or imports the real route
module. What IS tested is the wiring: that the adapter drives the route's whole
auto-switch protocol instead of the loader underneath it, that it refuses when
auto-switch is off, that it proves the swap actually happened, and that the
resident id it reports is one it can take back.

Every fake has an explicit signature -- no ``**kwargs`` -- and the auto-switch
fake's is checked against the real function's. The first version of this adapter
called ``_load_model_impl`` without ``current_request_counted = True``, so every
real swap hung on the drain forever; the fake's ``**kwargs`` swallowed exactly
that argument, and no test could see it.
"""

from __future__ import annotations

import ast
import inspect
import sys
import types
from pathlib import Path

import pytest

import core.inference.local_model_resolver as local_model_resolver
from core.inference.delegation import loader

# A stand-in for the local GGUF index, in the shape resolve_local_gguf returns:
# (load_path, gguf_variant, loader_id). One repo cached with two quants, and a
# second repo, so a swap has somewhere to go.
_REPO = "unsloth/Qwen3-4B-GGUF"
_PATH = r"C:\hf-cache\models--unsloth--Qwen3-4B-GGUF\snapshots\aaaa"
_OTHER_REPO = "unsloth/Qwen3-30B-GGUF"
_OTHER_PATH = r"C:\hf-cache\models--unsloth--Qwen3-30B-GGUF\snapshots\bbbb"

_INDEX = {
    "unsloth/qwen3-4b-gguf": (_PATH, "Q4_K_M", _REPO),
    "unsloth/qwen3-4b-gguf:q4_k_m": (_PATH, "Q4_K_M", _REPO),
    "unsloth/qwen3-4b-gguf:q8_0": (_PATH, "Q8_0", _REPO),
    "unsloth/qwen3-30b-gguf": (_OTHER_PATH, "Q4_K_M", _OTHER_REPO),
    "unsloth/qwen3-30b-gguf:q4_k_m": (_OTHER_PATH, "Q4_K_M", _OTHER_REPO),
    _PATH.lower(): (_PATH, "Q4_K_M", _REPO),
}


def _resolve_local_gguf(requested, *, allow_scan = True):
    """The real resolver's signature, answering from the table above."""
    if not isinstance(requested, str):
        return None
    return _INDEX.get(requested.strip().lower())


def _loaded(model_identifier, hf_variant = None, advertised = None, is_loaded = True):
    """A llama.cpp backend as the route's one looks to a reader: the attributes
    llama_keepwarm._loaded_identity and auto-switch's _already_serving read."""
    return types.SimpleNamespace(
        is_loaded = is_loaded,
        model_identifier = model_identifier,
        hf_variant = hf_variant,
        _openai_advertised_id = advertised,
    )


def _loaded_by_auto_switch(model_id):
    """The backend state a real auto-switch load leaves behind for *model_id*:
    the concrete load path as the identifier, the repo id advertised, and the
    resolved quant. None when the id resolves to nothing local."""
    resolved = _resolve_local_gguf(model_id)
    if resolved is None:
        return None
    load_path, variant, override_id = resolved
    return _loaded(load_path, hf_variant = variant, advertised = override_id)


def _install_route(monkeypatch, backend, *, swaps = True, switch_error = None):
    """A fake ``routes.inference`` carrying both entry points, so a test can tell
    which one the adapter used."""
    module = types.ModuleType("routes.inference")
    state = types.SimpleNamespace(backend = backend, switch_calls = [], load_impl_calls = [])

    def get_llama_cpp_backend():
        return state.backend

    async def _maybe_auto_switch_model(
        requested_model,
        fastapi_request,
        current_subject,
        *,
        require_vision = False,
        require_image = True,
        modality_label = "image or audio",
    ):
        state.switch_calls.append(
            {
                "requested_model": requested_model,
                "fastapi_request": fastapi_request,
                "current_subject": current_subject,
                "require_vision": require_vision,
                "require_image": require_image,
                "modality_label": modality_label,
            }
        )
        if switch_error is not None:
            raise switch_error
        if swaps:
            swapped = _loaded_by_auto_switch(requested_model)
            if swapped is not None:
                state.backend = swapped

    async def _load_model_impl(
        request,
        fastapi_request,
        current_subject,
        *,
        current_request_counted = False,
        on_reload_confirmed = None,
        load_cancel_event = None,
    ):
        # This fake LOADS the model, like the real one. So an adapter that
        # regressed to calling it would satisfy every other assertion in this
        # file, and the "never called" one is the only thing standing between
        # that regression and a swap that hangs on the drain forever.
        state.load_impl_calls.append(
            {
                "model_path": getattr(request, "model_path", None),
                "current_request_counted": current_request_counted,
            }
        )
        swapped = _loaded_by_auto_switch(getattr(request, "model_path", None))
        if swapped is not None:
            state.backend = swapped

    module.get_llama_cpp_backend = get_llama_cpp_backend
    module._maybe_auto_switch_model = _maybe_auto_switch_model
    module._load_model_impl = _load_model_impl
    monkeypatch.setitem(sys.modules, "routes.inference", module)
    return state


def _install_auto_switch_setting(monkeypatch, enabled):
    module = types.ModuleType("utils.openai_auto_switch_settings")

    def get_openai_auto_switch_enabled():
        return enabled

    module.get_openai_auto_switch_enabled = get_openai_auto_switch_enabled
    monkeypatch.setitem(sys.modules, "utils.openai_auto_switch_settings", module)


def _install_orchestrator(monkeypatch, active_model_name = None, *, constructed = True):
    """A fake ``core.inference.orchestrator``. ``constructed = False`` is a
    process where no orchestrator was ever built, which peek reports as None."""
    module = types.ModuleType("core.inference.orchestrator")

    def peek_inference_backend():
        if not constructed:
            return None
        return types.SimpleNamespace(active_model_name = active_model_name)

    module.peek_inference_backend = peek_inference_backend
    monkeypatch.setitem(sys.modules, "core.inference.orchestrator", module)


def _install_main(monkeypatch, app):
    module = types.ModuleType("main")
    module.app = app
    monkeypatch.setitem(sys.modules, "main", module)
    return app


@pytest.fixture(autouse = True)
def _hermetic(monkeypatch):
    """No test reaches a real setting, a real orchestrator, a real app or the
    filesystem index unless it installs one itself."""
    monkeypatch.setattr(local_model_resolver, "resolve_local_gguf", _resolve_local_gguf)
    _install_auto_switch_setting(monkeypatch, True)
    _install_orchestrator(monkeypatch, None)
    _install_main(monkeypatch, types.SimpleNamespace(state = types.SimpleNamespace()))


# ── load(): what it drives ──────────────────────────────────────────────


def test_load_drives_the_routes_auto_switch(monkeypatch):
    state = _install_route(monkeypatch, _loaded_by_auto_switch(_OTHER_REPO))
    loader.load(f"{_REPO}:Q4_K_M")
    assert len(state.switch_calls) == 1, "the route's auto-switch must be what runs"
    call = state.switch_calls[0]
    assert call["requested_model"] == f"{_REPO}:Q4_K_M"
    assert call["current_subject"] == loader.DELEGATION_SUBJECT
    assert isinstance(call["current_subject"], str) and call["current_subject"]


def test_load_never_calls_the_loader_underneath_auto_switch(monkeypatch):
    """The whole point of the redesign. _load_model_impl is one step of a
    nine-step protocol; calling it directly skips the resolution of
    "repo:VARIANT", every serialising lock, the saved launch settings and
    current_request_counted = True, which is what made the drain wait on the
    delegating request itself and hang forever."""
    state = _install_route(monkeypatch, _loaded_by_auto_switch(_OTHER_REPO))
    loader.load(f"{_REPO}:Q4_K_M")
    assert state.load_impl_calls == [], "the adapter must not call _load_model_impl itself"
    assert len(state.switch_calls) == 1


def test_the_stand_in_carries_the_servers_own_app(monkeypatch):
    """Without the real app, _resolve_parallel_slots falls back to one slot, so a
    delegation would reload the user's own model with fewer slots than the server
    was started with."""
    app = _install_main(monkeypatch, types.SimpleNamespace(state = types.SimpleNamespace(
        llama_parallel_slots = 4
    )))
    state = _install_route(monkeypatch, _loaded_by_auto_switch(_OTHER_REPO))
    loader.load(_REPO)
    stand_in = state.switch_calls[0]["fastapi_request"]
    assert stand_in.app is app
    assert stand_in.app.state.llama_parallel_slots == 4


def test_the_stand_in_has_no_app_when_main_is_not_imported(monkeypatch):
    monkeypatch.delitem(sys.modules, "main", raising = False)
    state = _install_route(monkeypatch, _loaded_by_auto_switch(_OTHER_REPO))
    loader.load(_REPO)
    assert state.switch_calls[0]["fastapi_request"].app is None


def test_auto_switch_off_is_refused_by_name(monkeypatch):
    """Auto-switch returns without loading anything when its toggle is off, so a
    swap attempted with it off would 'succeed' having done nothing."""
    _install_auto_switch_setting(monkeypatch, False)
    state = _install_route(monkeypatch, _loaded_by_auto_switch(_OTHER_REPO))
    with pytest.raises(loader.LoaderError) as excinfo:
        loader.load(_REPO)
    assert "Switch model by request" in str(excinfo.value), "name the toggle the user must turn on"
    assert state.switch_calls == [], "nothing may be attempted while the setting is off"


def test_a_swap_that_did_not_happen_is_a_LoaderError(monkeypatch):
    """Auto-switch dedupes and falls through by design: an unknown name is not an
    error there, it just keeps serving the loaded model."""
    state = _install_route(monkeypatch, _loaded_by_auto_switch(_OTHER_REPO), swaps = False)
    with pytest.raises(loader.LoaderError) as excinfo:
        loader.load(f"{_REPO}:Q4_K_M")
    message = str(excinfo.value)
    assert _REPO in message and _OTHER_REPO in message, "name the target and what is resident"
    assert len(state.switch_calls) == 1


def test_a_model_that_resolves_to_nothing_local_is_a_LoaderError(monkeypatch):
    _install_route(monkeypatch, _loaded_by_auto_switch(_REPO))
    with pytest.raises(loader.LoaderError) as excinfo:
        loader.load("openai/gpt-4")
    assert "openai/gpt-4" in str(excinfo.value)


def test_an_exception_from_auto_switch_becomes_a_LoaderError(monkeypatch):
    _install_route(
        monkeypatch,
        _loaded_by_auto_switch(_OTHER_REPO),
        switch_error = RuntimeError("no VRAM"),
    )
    with pytest.raises(loader.LoaderError) as excinfo:
        loader.load(_REPO)
    assert "no VRAM" in str(excinfo.value)


def test_no_route_module_is_a_LoaderError(monkeypatch):
    monkeypatch.delitem(sys.modules, "routes.inference", raising = False)
    with pytest.raises(loader.LoaderError):
        loader.load(_REPO)


def test_overrides_are_accepted_and_change_nothing(monkeypatch):
    """Role overrides are not applied here: auto-switch loads the model with its
    own saved launch settings, through model_override_load_kwargs."""
    state = _install_route(monkeypatch, _loaded_by_auto_switch(_OTHER_REPO))
    loader.load(_REPO)
    state.backend = _loaded_by_auto_switch(_OTHER_REPO)
    loader.load(_REPO, {"n_ctx": 16384, "gpu_ids": [1], "max_seq_length": 8192})
    plain, with_overrides = dict(state.switch_calls[0]), dict(state.switch_calls[1])
    assert plain.pop("fastapi_request").app is with_overrides.pop("fastapi_request").app
    assert plain == with_overrides


# ── the identity check, mirrored from auto-switch's _already_serving ────


def test_an_explicit_quant_must_be_the_quant_that_loaded(monkeypatch):
    """A bare repo id is satisfied by any quant, an explicit ":QUANT" is not --
    otherwise a delegation asking for Q8_0 would run on a resident Q4_K_M and
    nothing would say so."""
    state = _install_route(monkeypatch, _loaded_by_auto_switch(_OTHER_REPO), swaps = False)
    state.backend = _loaded(_PATH, hf_variant = "Q4_K_M", advertised = _REPO)
    with pytest.raises(loader.LoaderError):
        loader.load(f"{_REPO}:Q8_0")


def test_a_bare_id_is_served_by_any_quant_of_that_repo(monkeypatch):
    state = _install_route(monkeypatch, _loaded_by_auto_switch(_OTHER_REPO), swaps = False)
    state.backend = _loaded(_PATH, hf_variant = "Q8_0", advertised = _REPO)
    loader.load(_REPO)  # must not raise


def test_a_model_loaded_by_path_counts_as_serving_its_repo_id(monkeypatch):
    """A manual load records the on-disk path as the identifier and advertises
    nothing, so the check has to match the resolver's load path too."""
    state = _install_route(monkeypatch, _loaded_by_auto_switch(_OTHER_REPO), swaps = False)
    state.backend = _loaded(_PATH, hf_variant = "Q4_K_M", advertised = None)
    loader.load(f"{_REPO}:Q4_K_M")  # must not raise


# ── resident_model_id() / resident_is_restorable() ──────────────────────


def test_the_resident_id_carries_the_quant(monkeypatch):
    """Dropping the quant lets the restore auto-pick a different one -- the idle
    unload stash keeps the pair for the same reason."""
    _install_route(monkeypatch, _loaded(_REPO, hf_variant = "Q4_K_M", advertised = None))
    assert loader.resident_model_id() == f"{_REPO}:Q4_K_M"


def test_the_resident_id_prefers_the_advertised_repo_id(monkeypatch):
    """An auto-switch load records the concrete snapshot path as its identifier
    and the repo id as the advertised one; the path is not what a reload takes."""
    _install_route(monkeypatch, _loaded(_PATH, hf_variant = "Q8_0", advertised = _REPO))
    assert loader.resident_model_id() == f"{_REPO}:Q8_0"


def test_the_resident_id_is_bare_when_no_quant_is_known(monkeypatch):
    _install_route(monkeypatch, _loaded(_REPO, hf_variant = None, advertised = None))
    assert loader.resident_model_id() == _REPO


def test_nothing_loaded_reads_as_none(monkeypatch):
    _install_route(monkeypatch, _loaded("stale/model", hf_variant = "Q4_K_M", is_loaded = False))
    assert loader.resident_model_id() is None


def test_no_route_module_reads_as_none(monkeypatch):
    monkeypatch.delitem(sys.modules, "routes.inference", raising = False)
    assert loader.resident_model_id() is None


def test_the_resident_id_round_trips_through_load(monkeypatch):
    """What Task 6 does: remember the resident model, swap to the delegate, put
    the first one back."""
    state = _install_route(monkeypatch, _loaded(_REPO, hf_variant = "Q4_K_M", advertised = None))
    resident = loader.resident_model_id()
    loader.load(_OTHER_REPO)
    assert state.backend.model_identifier == _OTHER_PATH
    loader.load(resident)  # the restore; raises if it did not take
    assert state.switch_calls[-1]["requested_model"] == resident
    assert state.backend.model_identifier == _PATH


def test_a_transformers_model_is_not_restorable(monkeypatch):
    """Loading a GGUF unloads it and auto-switch cannot load it back, so Task 6
    has to refuse BEFORE swapping rather than fail to restore afterwards."""
    _install_route(monkeypatch, _loaded(None, is_loaded = False))
    _install_orchestrator(monkeypatch, "unsloth/Llama-3.2-1B-Instruct")
    assert loader.resident_is_restorable() is False


def test_a_transformers_model_does_not_read_as_nothing_loaded(monkeypatch):
    """Answering None here is how a user's model gets silently replaced by the
    delegate's: the caller restores nothing and reports nothing."""
    _install_route(monkeypatch, _loaded(None, is_loaded = False))
    _install_orchestrator(monkeypatch, "unsloth/Llama-3.2-1B-Instruct")
    with pytest.raises(loader.LoaderError) as excinfo:
        loader.resident_model_id()
    assert "unsloth/Llama-3.2-1B-Instruct" in str(excinfo.value)


def test_a_resident_gguf_is_restorable(monkeypatch):
    _install_route(monkeypatch, _loaded(_PATH, hf_variant = "Q4_K_M", advertised = _REPO))
    assert loader.resident_is_restorable() is True


def test_nothing_loaded_is_restorable(monkeypatch):
    _install_route(monkeypatch, _loaded(None, is_loaded = False))
    _install_orchestrator(monkeypatch, None, constructed = False)
    assert loader.resident_is_restorable() is True


def test_an_unreadable_backend_is_not_restorable(monkeypatch):
    module = types.ModuleType("core.inference.orchestrator")

    def peek_inference_backend():
        raise RuntimeError("orchestrator is mid-teardown")

    module.peek_inference_backend = peek_inference_backend
    monkeypatch.setitem(sys.modules, "core.inference.orchestrator", module)
    assert loader.resident_is_restorable() is False


# ── the fakes themselves ────────────────────────────────────────────────


def test_the_auto_switch_fake_matches_the_real_signature(monkeypatch):
    """Read from routes/inference.py's source, not by importing it: importing the
    route module pulls in the whole inference stack. A fake that has drifted from
    the real signature is what hid the missing current_request_counted, so this
    file is only worth as much as this check."""
    source = (Path(__file__).resolve().parents[1] / "routes" / "inference.py").read_text(
        encoding = "utf-8"
    )
    real = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_maybe_auto_switch_model"
    )
    _install_route(monkeypatch, _loaded(None, is_loaded = False))
    fake = inspect.signature(sys.modules["routes.inference"]._maybe_auto_switch_model)
    assert [a.arg for a in real.args.args] == [
        name
        for name, p in fake.parameters.items()
        if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    assert [a.arg for a in real.args.kwonlyargs] == [
        name for name, p in fake.parameters.items() if p.kind is inspect.Parameter.KEYWORD_ONLY
    ]
    assert real.args.vararg is None and real.args.kwarg is None
    assert not any(
        p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for p in fake.parameters.values()
    ), "a **kwargs fake swallows the argument a test was written to catch"
