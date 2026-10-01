# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pure-unit tests for AirLLM's context-length sizing formula. No real model, GPU, or
network -- config metadata and free-VRAM numbers are faked throughout."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.airllm_sizing import AirLLMSizingError, fit_airllm_context_length


def _fake_config(
    hidden_size = 4096,
    intermediate_size = 11008,
    num_hidden_layers = 32,
    num_attention_heads = 32,
    num_key_value_heads = 32,
    vocab_size = 32000,
    torch_dtype = "bfloat16",
    tie_word_embeddings = False,
    max_position_embeddings = None,
):
    # max_position_embeddings defaults to None (absent) so the tests below that probe the raw
    # VRAM formula are not masked by the trained-position clamp; the clamp has its own tests.
    return SimpleNamespace(
        hidden_size = hidden_size,
        intermediate_size = intermediate_size,
        num_hidden_layers = num_hidden_layers,
        num_attention_heads = num_attention_heads,
        num_key_value_heads = num_key_value_heads,
        vocab_size = vocab_size,
        torch_dtype = torch_dtype,
        tie_word_embeddings = tie_word_embeddings,
        max_position_embeddings = max_position_embeddings,
        text_config = None,
    )


def _fake_memory(free_mib):
    total_mib = (free_mib + 2000) if free_mib is not None else None
    return SimpleNamespace(free_mib = free_mib, total_mib = total_mib)


def test_small_model_ample_vram_yields_large_max_seq_len():
    # A 7B-class model (Llama-2-7B-shaped config) against a nearly-empty 16GB card should get
    # a context far larger than what resident loading of an equivalently-sized model would
    # ever allow -- that's the entire point of this load path (spec section 4.3).
    # The fake config deliberately declares no max_position_embeddings (the real Llama-2-7B
    # limit, 4096, would clamp this to 4096): this test measures the VRAM-budget formula
    # itself; the position clamp is covered by test_result_clamped_to_max_position_embeddings.
    with patch("core.inference.airllm_sizing.AutoConfig") as mock_auto_config, \
         patch("core.inference.airllm_sizing.resolve_diffusion_device_target") as mock_target, \
         patch("core.inference.airllm_sizing.snapshot_device_memory") as mock_snapshot:
        mock_auto_config.from_pretrained.return_value = _fake_config()
        mock_target.return_value = SimpleNamespace(torch_device = "cuda:0")
        mock_snapshot.return_value = _fake_memory(free_mib = 15000)

        result = fit_airllm_context_length("fake/llama-7b-shaped")

    assert result > 8192  # far above what 15GB could resident-fit a 7B model's KV cache to
                            # under normal (non-streaming) loading


def test_max_seq_len_never_falls_back_to_airllm_library_default():
    with patch("core.inference.airllm_sizing.AutoConfig") as mock_auto_config, \
         patch("core.inference.airllm_sizing.resolve_diffusion_device_target") as mock_target, \
         patch("core.inference.airllm_sizing.snapshot_device_memory") as mock_snapshot:
        mock_auto_config.from_pretrained.return_value = _fake_config()
        mock_target.return_value = SimpleNamespace(torch_device = "cuda:0")
        mock_snapshot.return_value = _fake_memory(free_mib = 15000)

        result = fit_airllm_context_length("fake/llama-7b-shaped")

    assert result != 512  # AirLLM's own AirLLMBaseModel.__init__ default
    assert isinstance(result, int)


def test_model_too_wide_to_fit_even_one_layer_refuses_cleanly():
    # An absurdly wide model (hidden_size pushed far past normal ranges) whose single
    # streamed unit alone exceeds the entire fake free-VRAM budget.
    with patch("core.inference.airllm_sizing.AutoConfig") as mock_auto_config, \
         patch("core.inference.airllm_sizing.resolve_diffusion_device_target") as mock_target, \
         patch("core.inference.airllm_sizing.snapshot_device_memory") as mock_snapshot:
        mock_auto_config.from_pretrained.return_value = _fake_config(
            hidden_size = 65536, intermediate_size = 180000, vocab_size = 256000,
        )
        mock_target.return_value = SimpleNamespace(torch_device = "cuda:0")
        mock_snapshot.return_value = _fake_memory(free_mib = 4000)  # small card

        with pytest.raises(AirLLMSizingError):
            fit_airllm_context_length("fake/absurdly-wide-model")


def test_unreadable_free_vram_refuses_rather_than_guessing():
    with patch("core.inference.airllm_sizing.AutoConfig") as mock_auto_config, \
         patch("core.inference.airllm_sizing.resolve_diffusion_device_target") as mock_target, \
         patch("core.inference.airllm_sizing.snapshot_device_memory") as mock_snapshot:
        mock_auto_config.from_pretrained.return_value = _fake_config()
        mock_target.return_value = SimpleNamespace(torch_device = "cuda:0")
        mock_snapshot.return_value = _fake_memory(free_mib = None)

        with pytest.raises(AirLLMSizingError):
            fit_airllm_context_length("fake/llama-7b-shaped")


def test_large_vocab_embed_head_can_dominate_over_decoder_layer():
    # A model with a huge vocab (e.g. 256k tokens) relative to a modest hidden size can have
    # embed_tokens/lm_head larger than any single decoder layer -- the formula must compare
    # both and use the larger, not assume a decoder layer always dominates.
    with patch("core.inference.airllm_sizing.AutoConfig") as mock_auto_config, \
         patch("core.inference.airllm_sizing.resolve_diffusion_device_target") as mock_target, \
         patch("core.inference.airllm_sizing.snapshot_device_memory") as mock_snapshot:
        mock_auto_config.from_pretrained.return_value = _fake_config(
            hidden_size = 2048, intermediate_size = 5632, vocab_size = 256000,
        )
        mock_target.return_value = SimpleNamespace(torch_device = "cuda:0")
        mock_snapshot.return_value = _fake_memory(free_mib = 15000)

        # Must not raise, and must still produce a sane positive context.
        result = fit_airllm_context_length("fake/huge-vocab-model")

    assert result > 0


def test_grouped_query_attention_narrows_kv_cache_vs_full_attention():
    # num_key_value_heads < num_attention_heads (GQA) should yield a LARGER max_seq_len than
    # the same model under full multi-head attention, since the KV cache is narrower per head.
    with patch("core.inference.airllm_sizing.AutoConfig") as mock_auto_config, \
         patch("core.inference.airllm_sizing.resolve_diffusion_device_target") as mock_target, \
         patch("core.inference.airllm_sizing.snapshot_device_memory") as mock_snapshot:
        mock_target.return_value = SimpleNamespace(torch_device = "cuda:0")
        mock_snapshot.return_value = _fake_memory(free_mib = 15000)

        mock_auto_config.from_pretrained.return_value = _fake_config(num_key_value_heads = 32)
        mha_result = fit_airllm_context_length("fake/mha-model")

        mock_auto_config.from_pretrained.return_value = _fake_config(num_key_value_heads = 8)
        gqa_result = fit_airllm_context_length("fake/gqa-model")

    assert gqa_result > mha_result


def _fit_with(fake_config, free_mib = 15000):
    with patch("core.inference.airllm_sizing.AutoConfig") as mock_auto_config, \
         patch("core.inference.airllm_sizing.resolve_diffusion_device_target") as mock_target, \
         patch("core.inference.airllm_sizing.snapshot_device_memory") as mock_snapshot:
        mock_auto_config.from_pretrained.return_value = fake_config
        mock_target.return_value = SimpleNamespace(torch_device = "cuda:0")
        mock_snapshot.return_value = _fake_memory(free_mib = free_mib)
        return fit_airllm_context_length("fake/model")


def test_tied_embeddings_reserve_embed_alongside_decoder_layer():
    # AirLLM keeps a tied embedding GPU-resident for the whole run (airllm_base.py's
    # _install_streaming_hooks), so it coexists with every streamed decoder layer: the floor
    # is decoder_layer + embed, leaving less room for KV cache than the untied max(...) case.
    untied = _fit_with(_fake_config(tie_word_embeddings = False))
    tied = _fit_with(_fake_config(tie_word_embeddings = True))
    assert 0 < tied < untied


def test_tied_embeddings_can_refuse_where_untied_fits():
    # Budget chosen between the two floors: Llama-7B-shaped bf16 decoder layer ~386 MiB,
    # embed ~250 MiB. 2048 overhead + 386 fits; 2048 + 386 + 250 does not.
    free_mib = 2048 + 500
    assert _fit_with(_fake_config(tie_word_embeddings = False), free_mib = free_mib) > 0
    with pytest.raises(AirLLMSizingError):
        _fit_with(_fake_config(tie_word_embeddings = True), free_mib = free_mib)


def test_result_clamped_to_max_position_embeddings():
    # Ample VRAM would allow far more than the model's trained 4096 positions; AirLLM never
    # enforces max_seq_len itself, so the sizing result must be capped.
    unclamped = _fit_with(_fake_config())
    assert unclamped > 4096
    clamped = _fit_with(_fake_config(max_position_embeddings = 4096))
    assert clamped == 4096


def test_position_limit_above_vram_budget_does_not_raise_result():
    # The clamp only ever lowers the result: a generous position limit leaves the
    # VRAM-derived value untouched.
    unclamped = _fit_with(_fake_config())
    assert _fit_with(_fake_config(max_position_embeddings = unclamped * 4)) == unclamped


def test_position_limit_read_from_text_config_of_multimodal_config():
    text_config = _fake_config(max_position_embeddings = 2048)
    outer = SimpleNamespace(text_config = text_config, vocab_size = 32000)
    assert _fit_with(outer) == 2048
