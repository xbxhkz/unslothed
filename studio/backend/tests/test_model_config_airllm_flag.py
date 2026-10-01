# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""is_airllm must default False and thread through LoadRequest -> ModelConfig.from_identifier()
exactly like the existing is_lora flag, never auto-detected. No GPU, network, or model files."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


def test_load_request_is_airllm_defaults_false():
    from models.inference import LoadRequest

    request = LoadRequest(model_path = "some/repo")
    assert request.is_airllm is False


def test_load_request_is_airllm_explicit_true():
    from models.inference import LoadRequest

    request = LoadRequest(model_path = "some/repo", is_airllm = True)
    assert request.is_airllm is True


def test_model_config_dataclass_is_airllm_defaults_false():
    from utils.models.model_config import ModelConfig

    config = ModelConfig(
        identifier = "some/repo",
        display_name = "some/repo",
        path = "some/repo",
        is_local = False,
        is_cached = False,
        is_vision = False,
        is_lora = False,
    )
    assert config.is_airllm is False


def test_from_identifier_threads_is_airllm_true():
    from utils.models.model_config import ModelConfig

    # A Windows-absolute-path-shaped identifier (contains ":" and "\\") is recognized as
    # local by is_local_path() regardless of whether it exists on disk (utils/paths/
    # path_utils.py:135) -- this steers from_identifier() away from every networked branch
    # (remote GGUF detection, remote LoRA auto-detection via hf_model_info) without a single
    # patch for them: a nonexistent local path safely short-circuits detect_gguf_model()
    # (suffix/glob check, no crash on a missing dir) and _looks_like_lora_adapter()
    # (Path.is_dir() is False, short-circuits immediately). Only the three probes in the
    # function's final shared tail (confirmed by reading from_identifier() directly) need
    # stubbing: is_vision_model and detect_audio_type (both DEFINED in this module, not
    # imported -- confirmed by reading the file), and is_model_cached (imported from
    # utils.paths into this module's namespace).
    fake_local_id = "C:\\fake\\local\\airllm-test-model"
    with patch("utils.models.model_config.is_vision_model", return_value = False), \
         patch("utils.models.model_config.detect_audio_type", return_value = None), \
         patch("utils.models.model_config.is_model_cached", return_value = True):
        config = ModelConfig.from_identifier(
            model_id = fake_local_id,
            is_airllm = True,
        )
    assert config is not None
    assert config.is_airllm is True


def test_from_identifier_is_airllm_defaults_false():
    from utils.models.model_config import ModelConfig

    fake_local_id = "C:\\fake\\local\\airllm-test-model"
    with patch("utils.models.model_config.is_vision_model", return_value = False), \
         patch("utils.models.model_config.detect_audio_type", return_value = None), \
         patch("utils.models.model_config.is_model_cached", return_value = True):
        config = ModelConfig.from_identifier(model_id = fake_local_id)
    assert config is not None
    assert config.is_airllm is False
