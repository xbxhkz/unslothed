# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""is_airllm must survive the orchestrator -> worker subprocess boundary, and an AirLLM load must
always run the remote-code consent gate (airllm hardcodes trust_remote_code=True internally).
No GPU, network, model files, or real subprocess."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import worker

# Local-path-shaped id: keeps ModelConfig.from_identifier() off every networked branch (same
# convention as tests/test_model_config_airllm_flag.py).
_FAKE_LOCAL_ID = "C:\\fake\\local\\airllm-test-model"


def _patched_from_identifier_probes():
    return (
        patch("utils.models.model_config.is_vision_model", return_value = False),
        patch("utils.models.model_config.detect_audio_type", return_value = None),
        patch("utils.models.model_config.is_model_cached", return_value = True),
    )


# ── C1: worker side ───────────────────────────────────────────────────


@pytest.mark.parametrize("flag", [True, False])
def test_build_model_config_threads_is_airllm(flag):
    p1, p2, p3 = _patched_from_identifier_probes()
    with p1, p2, p3:
        mc = worker._build_model_config({"model_name": _FAKE_LOCAL_ID, "is_airllm": flag})
    assert mc.is_airllm is flag


def test_build_model_config_is_airllm_defaults_false_when_key_absent():
    p1, p2, p3 = _patched_from_identifier_probes()
    with p1, p2, p3:
        mc = worker._build_model_config({"model_name": _FAKE_LOCAL_ID})
    assert mc.is_airllm is False


# ── C1: orchestrator side ─────────────────────────────────────────────


def _spawned_sub_config(config):
    """Drive InferenceOrchestrator.load_model() up to the spawn and return the dict it sends
    to the worker (same isolation as test_gpu_selection.py's orchestrator tests)."""

    class DummyThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return None

    with patch("core.inference.orchestrator.threading.Thread", DummyThread):
        from core.inference.orchestrator import InferenceOrchestrator
        orchestrator = InferenceOrchestrator()

    with (
        patch(
            "core.inference.orchestrator.prepare_gpu_selection",
            return_value = ([0], {"selection_mode": "auto"}),
        ),
        patch.object(orchestrator, "_ensure_subprocess_alive", return_value = False),
        patch.object(orchestrator, "_spawn_subprocess") as mock_spawn,
        patch.object(
            orchestrator,
            "_wait_response",
            return_value = {"success": True, "model_info": {}},
        ),
        patch("utils.transformers_version.needs_transformers_5", return_value = False),
    ):
        assert orchestrator.load_model(config = config) is True

    return mock_spawn.call_args.args[0]


def test_orchestrator_sub_config_forwards_is_airllm_true():
    config = SimpleNamespace(identifier = "org/big-model", gguf_variant = None, is_airllm = True)
    assert _spawned_sub_config(config)["is_airllm"] is True


def test_orchestrator_sub_config_is_airllm_false_when_unset():
    config = SimpleNamespace(identifier = "org/model", gguf_variant = None)
    assert _spawned_sub_config(config)["is_airllm"] is False


# ── C2: consent gate is forced for AirLLM ─────────────────────────────


class _Q:
    def __init__(self):
        self.sent = []

    def put(self, item, *a, **k):
        self.sent.append(item)


def _drive_handle_load(monkeypatch, *, is_airllm, request_trust):
    mc = SimpleNamespace(
        identifier = "org/big-model",
        display_name = "big-model",
        is_vision = False,
        is_lora = False,
        base_model = None,
        is_audio = False,
        audio_type = None,
        has_audio_input = False,
        is_airllm = is_airllm,
    )
    gate_calls = []
    load_calls = []

    def _fake_gates(targets, **kwargs):
        gate_calls.append(kwargs)
        return True

    backend = SimpleNamespace(
        device = "mlx",  # skips the SSM-kernel install branch
        active_model_name = mc.identifier,
        models = {mc.identifier: {}},
        load_model = lambda **kw: load_calls.append(kw) or True,
    )
    monkeypatch.setattr(worker, "_build_model_config", lambda cfg: mc)
    monkeypatch.setattr(worker, "_run_security_gates", _fake_gates)
    monkeypatch.setattr(worker, "_resolve_lora_4bit", lambda mc_, v: False)
    monkeypatch.setattr(worker, "_needs_nemotron_trust", lambda *a, **k: False)

    fake_xet = type(sys)("utils.hf_xet_fallback")
    fake_xet.start_watchdog = lambda **k: SimpleNamespace(set = lambda: None)
    monkeypatch.setitem(sys.modules, "utils.hf_xet_fallback", fake_xet)

    q = _Q()
    worker._handle_load(
        backend,
        {
            "model_name": mc.identifier,
            "trust_remote_code": request_trust,
            "is_airllm": is_airllm,
        },
        q,
    )
    loaded = [m for m in q.sent if m.get("type") == "loaded"]
    assert loaded and loaded[0].get("success"), q.sent
    return gate_calls, load_calls


def test_airllm_load_forces_remote_code_consent_gate(monkeypatch):
    gate_calls, load_calls = _drive_handle_load(
        monkeypatch, is_airllm = True, request_trust = False,
    )
    assert len(gate_calls) == 1
    assert gate_calls[0]["trust_remote_code"] is True
    assert load_calls[0]["trust_remote_code"] is True


def test_non_airllm_load_keeps_request_trust_remote_code(monkeypatch):
    gate_calls, load_calls = _drive_handle_load(
        monkeypatch, is_airllm = False, request_trust = False,
    )
    assert gate_calls[0]["trust_remote_code"] is False
    assert load_calls[0]["trust_remote_code"] is False
