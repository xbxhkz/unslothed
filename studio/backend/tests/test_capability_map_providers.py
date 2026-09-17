# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Model-ability and cached-model checks.

Both read state that is ALREADY in memory or on disk. The backend getters in this
codebase construct what they are asked about, so a check that called one would
start the subsystem it was describing.

WHY THE TRIPWIRES RECORD INSTEAD OF RAISING: check_requirement swallows every
exception into an "unknown" answer. A tripwire getter that raised would be caught
by that guard, the test would see "unknown", and nothing would fail -- an inert
control. So each fake getter appends to a list, and the tests assert the list
stays empty.
"""

from __future__ import annotations

import inspect
import sys
import types

import pytest

from core.inference import tool_readiness as tr
from core.inference.capability_map import Requirement, providers
from core.inference.tool_readiness import MISSING, READY, UNKNOWN

_ROUTE = "routes.inference"
_ORCH = "core.inference.orchestrator"


@pytest.fixture(autouse = True)
def _no_backends_in_memory(monkeypatch):
    """Every test starts with NEITHER backend module loaded and states exactly
    what is in memory. monkeypatch restores the real modules afterwards."""
    tr.reset_for_tests()
    monkeypatch.delitem(sys.modules, _ROUTE, raising = False)
    monkeypatch.delitem(sys.modules, _ORCH, raising = False)
    yield
    tr.reset_for_tests()


def _vision():
    return providers.check_requirement(Requirement("model", "vision"))


def _route_module(monkeypatch, backend, calls):
    module = types.ModuleType(_ROUTE)
    module._llama_cpp_backend = backend

    def get_llama_cpp_backend():
        calls.append("get_llama_cpp_backend")
        return backend

    module.get_llama_cpp_backend = get_llama_cpp_backend
    monkeypatch.setitem(sys.modules, _ROUTE, module)


def _orchestrator_module(monkeypatch, orchestrator, calls):
    module = types.ModuleType(_ORCH)
    module.peek_inference_backend = lambda: orchestrator

    def get_inference_backend():
        calls.append("get_inference_backend")
        return orchestrator

    module.get_inference_backend = get_inference_backend
    monkeypatch.setitem(sys.modules, _ORCH, module)


def _llama(*, loaded, vision, name = "Qwen2.5-VL-7B"):
    return types.SimpleNamespace(is_loaded = loaded, is_vision = vision, model_identifier = name)


def _orch(active, *, vision, display = None):
    models = {active: {"is_vision": vision, "display_name": display or active}} if active else {}
    return types.SimpleNamespace(active_model_name = active, models = models)


# --- vision ---------------------------------------------------------------


def test_vision_is_unknown_when_no_backend_module_is_in_memory():
    result = _vision()
    assert result.state == UNKNOWN
    assert "backend" in result.detail


def test_vision_is_ready_and_names_a_loaded_gguf_vision_model(monkeypatch):
    _route_module(monkeypatch, _llama(loaded = True, vision = True), [])
    result = _vision()
    assert result.state == READY
    assert "Qwen2.5-VL-7B" in result.detail


def test_vision_is_missing_and_names_a_loaded_model_without_vision(monkeypatch):
    _route_module(monkeypatch, _llama(loaded = True, vision = False, name = "Llama-3.1-8B"), [])
    result = _vision()
    assert result.state == MISSING
    assert "Llama-3.1-8B" in result.detail


def test_vision_is_unknown_when_the_backend_exists_but_nothing_is_loaded(monkeypatch):
    """An external or API model may be serving, with abilities this process
    cannot see -- so this is 'unknown', not 'missing'."""
    _route_module(monkeypatch, _llama(loaded = False, vision = False), [])
    assert _vision().state == UNKNOWN


def test_vision_is_ready_from_a_transformers_model(monkeypatch):
    _orchestrator_module(monkeypatch, _orch("qwen-vl", vision = True, display = "Qwen VL"), [])
    result = _vision()
    assert result.state == READY
    assert "Qwen VL" in result.detail


def test_either_backend_having_vision_is_enough(monkeypatch):
    _route_module(monkeypatch, _llama(loaded = True, vision = False, name = "text-model"), [])
    _orchestrator_module(monkeypatch, _orch("qwen-vl", vision = True), [])
    assert _vision().state == READY


def test_the_checks_never_call_a_creating_getter(monkeypatch):
    """THE side-effect guard. Exercised on a ready path AND an unknown path."""
    calls = []
    _route_module(monkeypatch, _llama(loaded = True, vision = True), calls)
    _orchestrator_module(monkeypatch, _orch(None, vision = False), calls)
    assert _vision().state == READY
    _route_module(monkeypatch, _llama(loaded = False, vision = False), calls)
    assert _vision().state == UNKNOWN
    assert calls == [], f"a check constructed a backend: {calls}"


def test_an_ability_other_than_vision_is_unknown(monkeypatch):
    _route_module(monkeypatch, _llama(loaded = True, vision = True), [])
    assert providers.check_requirement(Requirement("model", "audio")).state == UNKNOWN


# --- cached models ----------------------------------------------------------


def test_whisper_is_missing_until_a_model_is_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    result = providers.check_requirement(Requirement("cached_model", "whisper"))
    assert result.state == MISSING
    assert "first use" in result.detail


def test_whisper_is_ready_once_a_pt_file_is_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    (tmp_path / "whisper").mkdir()
    (tmp_path / "whisper" / "tiny.en.pt").write_bytes(b"x")
    result = providers.check_requirement(Requirement("cached_model", "whisper"))
    assert result.state == READY
    assert "tiny.en.pt" in result.detail


def test_a_non_model_file_in_the_whisper_cache_does_not_count(tmp_path, monkeypatch):
    """Control: without this, a check that accepted any file would pass above."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    (tmp_path / "whisper").mkdir()
    (tmp_path / "whisper" / "notes.txt").write_text("x")
    assert providers.check_requirement(Requirement("cached_model", "whisper")).state == MISSING


def test_the_whisper_cache_dir_honours_XDG_CACHE_HOME(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert providers._whisper_cache_dir() == str(tmp_path / "whisper")


def test_the_pinned_whisper_cache_dir_matches_whisper_itself():
    """Drift guard. The path is pinned because importing whisper loads torch; if
    whisper ever builds download_root differently, this fails instead of the
    check silently looking in the wrong place."""
    whisper = pytest.importorskip("whisper")
    source = inspect.getsource(whisper.load_model)
    assert 'default = os.path.join(os.path.expanduser("~"), ".cache")' in source
    assert 'download_root = os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")' in source


def test_an_unrecognised_cached_model_is_unknown():
    assert providers.check_requirement(Requirement("cached_model", "llama")).state == UNKNOWN
