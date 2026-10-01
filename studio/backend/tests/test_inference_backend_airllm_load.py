# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""InferenceBackend.load_model()'s AirLLM branch, with airllm mocked entirely via sys.modules
injection (same pattern as tests/test_mlx_inference_backend.py's _install_fake_mlx). No real
model, GPU, or subprocess."""

import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


class _FakeNotEnoughSpaceException(Exception):
    pass


class _FakeAirLLMModel:
    def __init__(self):
        self.tokenizer = SimpleNamespace(eos_token_id = 0, pad_token_id = None)

    def generate(self, *args, **kwargs):
        raise AssertionError("not called in load_model tests")


def _install_fake_airllm(monkeypatch, calls, *, raise_exc = None):
    airllm_pkg = types.ModuleType("airllm")
    airllm_pkg.NotEnoughSpaceException = _FakeNotEnoughSpaceException

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append((path, kwargs))
            if raise_exc is not None:
                raise raise_exc
            return _FakeAirLLMModel()

    airllm_pkg.AutoModel = _FakeAutoModel
    monkeypatch.setitem(sys.modules, "airllm", airllm_pkg)
    return airllm_pkg


def _make_backend():
    # InferenceBackend.__init__ calls get_default_models() -> hw.get_device(), which blocks
    # on a real torch/GPU probe (~2.9s on a cold host, confirmed by
    # tests/test_inference_backend_singleton.py's own docstring) -- exactly the real-GPU
    # dependency this plan's test discipline forbids. Bypass __init__ via __new__ and set
    # only the attributes load_model()'s pre-dispatch checks and the new AirLLM branch
    # actually read (confirmed by reading InferenceBackend.__init__ and the top of
    # load_model() directly): self.models, self.active_model_name, self.loading_models.
    from core.inference.inference import InferenceBackend

    backend = InferenceBackend.__new__(InferenceBackend)
    backend.models = {}
    backend.active_model_name = None
    backend.loading_models = set()
    backend.loaded_local_models = []
    return backend


def _airllm_config(identifier = "fake/oversized-model"):
    return SimpleNamespace(
        identifier = identifier,
        path = identifier,
        is_airllm = True,
        is_vision = False,
        is_lora = False,
        is_audio = False,
        is_gguf = False,
        # load_model()'s pre-existing, unconditional dict-init block (predates this task, runs
        # for every config regardless of is_airllm) reads these two fields too. A real
        # ModelConfig always has them (both default -- see utils/models/model_config.py:3628-29),
        # so the double must too.
        audio_type = None,
        has_audio_input = False,
    )


def test_airllm_branch_not_reached_without_explicit_flag(monkeypatch):
    # With is_airllm=False, load_model() must fall through to the existing
    # FastLanguageModel-based path untouched -- but that real path would otherwise try a
    # real network/GPU model load, which this test must not risk hitting (could hang or be
    # slow/non-deterministic depending on the environment). get_device_map()/
    # get_visible_gpu_count() run unconditionally before ANY dispatch branch (lines 366-368)
    # and are mocked for the same real-hardware-independence reason as the tests below.
    # FastLanguageModel.from_pretrained is the real text-path's actual model call
    # (core/inference/inference.py:658, imported at module level from `unsloth` -- confirmed
    # by reading the file); it only runs once execution falls all the way through the
    # is_airllm/is_audio/is_vision chain to the final `else:` branch, so raising a sentinel
    # from it proves the AirLLM branch (and nothing before it) was skipped, without
    # attempting any real model load.
    calls = []
    _install_fake_airllm(monkeypatch, calls)
    backend = _make_backend()
    config = _airllm_config()
    config.is_airllm = False

    # load_model()'s own outer try/except re-wraps ANY exception as a plain Exception via
    # format_error_message() (confirmed by reading the tail of load_model()) -- the sentinel
    # must be asserted as the base Exception type, not RuntimeError, or this test would
    # itself fail against the real (correct) behavior.
    sentinel = RuntimeError("reached the non-AirLLM path")
    with patch("core.inference.inference.fit_airllm_context_length") as mock_fit, \
         patch("core.inference.inference.FastLanguageModel") as mock_flm, \
         patch("core.inference.inference.get_device_map", return_value = "sequential"), \
         patch("core.inference.inference.get_visible_gpu_count", return_value = 1):
        mock_flm.from_pretrained.side_effect = sentinel
        with pytest.raises(Exception, match = "reached the non-AirLLM path"):
            backend.load_model(config, max_seq_length = 2048)

    mock_fit.assert_not_called()
    assert calls == []


def test_airllm_branch_computes_max_seq_len_never_library_default(monkeypatch):
    # get_device_map()/get_visible_gpu_count() run unconditionally before the is_airllm
    # dispatch (core/inference/inference.py:366-368, both imported at module level -- confirmed
    # by reading the file) and would otherwise probe real hardware via get_device(); mocked here
    # for determinism and so this test passes identically with or without a real GPU present.
    calls = []
    _install_fake_airllm(monkeypatch, calls)
    backend = _make_backend()
    config = _airllm_config()

    with patch("core.inference.inference.fit_airllm_context_length", return_value = 131072) as mock_fit, \
         patch("core.inference.inference.resolve_selected_cuda_ordinal", return_value = 0), \
         patch("core.inference.inference.resolve_diffusion_device_target") as mock_target, \
         patch("core.inference.inference.apply_diffusion_device_ordinal"), \
         patch("core.inference.inference.get_device_map", return_value = "sequential"), \
         patch("core.inference.inference.get_visible_gpu_count", return_value = 1):
        mock_target.return_value = SimpleNamespace(torch_device = "cuda:0")
        assert backend.load_model(config, max_seq_length = 2048, hf_token = "tok") is True

    mock_fit.assert_called_once()
    assert len(calls) == 1
    _, kwargs = calls[0]
    assert kwargs["max_seq_len"] == 131072
    assert kwargs["max_seq_len"] != 512
    assert kwargs["device"] == "cuda:0"


def test_airllm_branch_stores_model_and_tokenizer(monkeypatch):
    calls = []
    _install_fake_airllm(monkeypatch, calls)
    backend = _make_backend()
    config = _airllm_config()

    with patch("core.inference.inference.fit_airllm_context_length", return_value = 65536), \
         patch("core.inference.inference.resolve_selected_cuda_ordinal", return_value = None), \
         patch("core.inference.inference.resolve_diffusion_device_target") as mock_target, \
         patch("core.inference.inference.apply_diffusion_device_ordinal"), \
         patch("core.inference.inference.get_device_map", return_value = "sequential"), \
         patch("core.inference.inference.get_visible_gpu_count", return_value = 1):
        mock_target.return_value = SimpleNamespace(torch_device = "cuda")
        assert backend.load_model(config, max_seq_length = 2048) is True

    assert config.identifier in backend.models
    assert isinstance(backend.models[config.identifier]["model"], _FakeAirLLMModel)
    assert backend.models[config.identifier]["tokenizer"] is not None


def test_airllm_sizing_refusal_raises_with_marker_text(monkeypatch):
    from core.inference.airllm_sizing import AirLLMSizingError

    calls = []
    _install_fake_airllm(monkeypatch, calls)
    backend = _make_backend()
    config = _airllm_config()

    with patch(
        "core.inference.inference.fit_airllm_context_length",
        side_effect = AirLLMSizingError("AirLLM sizing: cannot fit even one layer"),
    ), patch("core.inference.inference.resolve_selected_cuda_ordinal", return_value = None), \
       patch("core.inference.inference.resolve_diffusion_device_target") as mock_target, \
       patch("core.inference.inference.apply_diffusion_device_ordinal"), \
       patch("core.inference.inference.get_device_map", return_value = "sequential"), \
       patch("core.inference.inference.get_visible_gpu_count", return_value = 1):
        mock_target.return_value = SimpleNamespace(torch_device = "cuda")
        with pytest.raises(Exception) as exc_info:
            backend.load_model(config, max_seq_length = 2048)

    assert "AirLLM load refused:" in str(exc_info.value)
    assert calls == []  # from_pretrained must never be called once sizing refuses


def test_airllm_not_enough_space_exception_wrapped_with_marker(monkeypatch):
    calls = []
    _install_fake_airllm(monkeypatch, calls, raise_exc = _FakeNotEnoughSpaceException("Not enough space. Free space under X: need 50GB, have 10GB"))
    backend = _make_backend()
    config = _airllm_config()

    # No extra patch for airllm.NotEnoughSpaceException needed: load_model()'s AirLLM branch
    # does a LOCAL `import airllm` inside the function body (Step 3 below -- deliberately not
    # a module-level import), so it resolves from sys.modules at call time and picks up the
    # SAME fake module _install_fake_airllm already injected there. Patching
    # "core.inference.inference.airllm...." would fail anyway: that name is never a module
    # level attribute of core.inference.inference, only a local variable inside load_model().
    with patch("core.inference.inference.fit_airllm_context_length", return_value = 65536), \
         patch("core.inference.inference.resolve_selected_cuda_ordinal", return_value = None), \
         patch("core.inference.inference.resolve_diffusion_device_target") as mock_target, \
         patch("core.inference.inference.apply_diffusion_device_ordinal"), \
         patch("core.inference.inference.get_device_map", return_value = "sequential"), \
         patch("core.inference.inference.get_visible_gpu_count", return_value = 1):
        mock_target.return_value = SimpleNamespace(torch_device = "cuda")
        with pytest.raises(Exception) as exc_info:
            backend.load_model(config, max_seq_length = 2048)

    assert "AirLLM load refused:" in str(exc_info.value)


def test_airllm_defensive_import_error_wrapped_with_marker(monkeypatch):
    calls = []
    _install_fake_airllm(monkeypatch, calls, raise_exc = ImportError("bitsandbytes not found"))
    backend = _make_backend()
    config = _airllm_config()

    with patch("core.inference.inference.fit_airllm_context_length", return_value = 65536), \
         patch("core.inference.inference.resolve_selected_cuda_ordinal", return_value = None), \
         patch("core.inference.inference.resolve_diffusion_device_target") as mock_target, \
         patch("core.inference.inference.apply_diffusion_device_ordinal"), \
         patch("core.inference.inference.get_device_map", return_value = "sequential"), \
         patch("core.inference.inference.get_visible_gpu_count", return_value = 1):
        mock_target.return_value = SimpleNamespace(torch_device = "cuda")
        with pytest.raises(Exception) as exc_info:
            backend.load_model(config, max_seq_length = 2048)

    assert "AirLLM load refused:" in str(exc_info.value)


def test_route_classifies_airllm_refusal_as_400():
    from routes.inference import _is_airllm_load_refusal

    assert _is_airllm_load_refusal("AirLLM load refused: Not enough space. Free space under X") is True
    assert _is_airllm_load_refusal("Some unrelated CUDA error") is False
