# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Context-length sizing for AirLLM's per-layer disk-to-GPU streaming load path.

NOT a reuse of core/inference/llama_cpp.py's GGUF ctx-auto-fit logic: that formula assumes
the model's own weights compete with the KV cache for the same VRAM budget, which is true for
normal resident loading but false here -- AirLLM keeps only its single largest streamed unit
(a decoder layer, or embed_tokens/lm_head for a large-vocab model) resident at a time, so the
budget is dominated by the KV cache accumulated across the WHOLE generation, not by total
model size. See the design spec's section 4.3 for the derivation.

Exception: with ``tie_word_embeddings`` set, AirLLM's ``_install_streaming_hooks`` loads the
embedding onto the GPU ONCE and keeps it resident (lm_head is re-tied to it), streaming only
the decoder layers and final norm -- so the embedding coexists with every streamed layer and
the KV cache, and the resident floor is ``decoder_layer + embed``, not their max.

The result is also clamped to the model's ``max_position_embeddings`` when the config declares
one: AirLLM only stores ``max_seq_len`` and never enforces it during generation, so this
function is the only thing keeping the served context inside the model's trained range.
"""

from typing import Any, Optional

from transformers import AutoConfig

from core.inference.diffusion_device import resolve_diffusion_device_target
from core.inference.diffusion_memory import snapshot_device_memory

# Same CUDA-context-plus-activation-buffer margin diffusion_memory.py's
# DEFAULT_BASE_OVERHEAD_MIB already uses for a single resident pipeline -- kept as a local
# constant rather than importing a diffusion-owned name for this unrelated (chat) owner.
_FIXED_OVERHEAD_MIB = 2048

_DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
}


class AirLLMSizingError(Exception):
    """Raised when this host's free VRAM cannot support even one token of KV cache for the
    model's single largest streamed unit. The caller must refuse the load before starting
    AirLLM's slow layer-split/download step -- never pass a zero/negative max_seq_len through."""


def _resolve_text_config(config: Any) -> Any:
    """Unwrap a multimodal config's text_config, same fallback AirLLM's own AirLLMBaseModel
    .__init__ uses for dtype resolution (confirmed by reading airllm_base.py)."""
    text_config = getattr(config, "text_config", None)
    return text_config if text_config is not None else config


def _dtype_byte_width(config: Any, compression: Optional[str]) -> int:
    if compression in ("4bit", "8bit"):
        # Conservative: real nf4 packing is ~0.5 bytes/param, but using 1 byte here only makes
        # the resulting max_seq_len an UNDERESTIMATE (safe direction) rather than risking an
        # OOM from an overestimate.
        return 1
    torch_dtype = getattr(config, "torch_dtype", None) or getattr(config, "dtype", None)
    if isinstance(torch_dtype, str):
        return _DTYPE_BYTES.get(torch_dtype, 2)
    return 2  # bf16/fp16 default, matches AirLLMBaseModel.__init__'s own fallback


def fit_airllm_context_length(
    model_id_or_path: str,
    *,
    hf_token: Optional[str] = None,
    trust_remote_code: bool = False,
    compression: Optional[str] = None,
    ordinal: Optional[int] = None,
) -> int:
    """The largest max_seq_len this host's current free VRAM can support for an AirLLM load
    of ``model_id_or_path``. Raises AirLLMSizingError if even one token of KV cache cannot
    fit alongside the model's single largest streamed unit."""
    token_kwargs = {"token": hf_token} if hf_token else {}
    config = AutoConfig.from_pretrained(
        model_id_or_path, trust_remote_code = trust_remote_code, **token_kwargs,
    )
    text_config = _resolve_text_config(config)

    hidden_size = int(text_config.hidden_size)
    intermediate_size = int(getattr(text_config, "intermediate_size", 4 * hidden_size))
    num_hidden_layers = int(text_config.num_hidden_layers)
    num_attention_heads = int(text_config.num_attention_heads)
    num_key_value_heads = int(
        getattr(text_config, "num_key_value_heads", None) or num_attention_heads
    )
    vocab_size = int(getattr(text_config, "vocab_size", None) or getattr(config, "vocab_size", 0))
    head_dim = int(getattr(text_config, "head_dim", None) or (hidden_size // num_attention_heads))

    dtype_bytes = _dtype_byte_width(text_config, compression)

    kv_hidden = head_dim * num_key_value_heads
    attn_params = 2 * hidden_size * hidden_size + 2 * hidden_size * kv_hidden
    mlp_params = 3 * hidden_size * intermediate_size  # gate/up/down (SwiGLU-style MLP)
    decoder_layer_mib = (attn_params + mlp_params) * dtype_bytes / (1024 * 1024)
    embed_mib = vocab_size * hidden_size * dtype_bytes / (1024 * 1024)

    # Untied: embed_tokens and lm_head are part of AirLLM's own streamed
    # embed->layers->norm->lm_head rotation (confirmed in airllm_base.py), not separately
    # GPU-resident -- compare both and take the larger as the one unit that must coexist with
    # the KV cache. Tied: AirLLM pins the embedding on the GPU for the whole run and streams
    # only decoder layers + norm, so both are resident at once. Read from the OUTER config,
    # exactly as AirLLMBaseModel._install_streaming_hooks does (self.config.tie_word_embeddings).
    if bool(getattr(config, "tie_word_embeddings", False)):
        largest_streamed_unit_mib = decoder_layer_mib + embed_mib
    else:
        largest_streamed_unit_mib = max(decoder_layer_mib, embed_mib)

    target = resolve_diffusion_device_target(ordinal = ordinal)
    memory = snapshot_device_memory(target)
    free_vram_mib = memory.free_mib
    if free_vram_mib is None:
        raise AirLLMSizingError(
            f"AirLLM sizing: could not read free VRAM on this host; refusing to guess a "
            f"context length for '{model_id_or_path}'."
        )

    budget_mib = free_vram_mib - _FIXED_OVERHEAD_MIB - largest_streamed_unit_mib
    if budget_mib <= 0:
        raise AirLLMSizingError(
            f"AirLLM sizing: '{model_id_or_path}' needs {largest_streamed_unit_mib:.0f} MiB "
            f"for its largest streamed unit plus {_FIXED_OVERHEAD_MIB} MiB fixed overhead, "
            f"but only {free_vram_mib} MiB VRAM is free. This model cannot run via AirLLM on "
            f"this host."
        )

    kv_bytes_per_token = 2 * num_hidden_layers * num_key_value_heads * head_dim * dtype_bytes
    max_seq_len = int((budget_mib * 1024 * 1024) // kv_bytes_per_token)
    if max_seq_len < 1:
        raise AirLLMSizingError(
            f"AirLLM sizing: '{model_id_or_path}' cannot fit even one token of KV cache in "
            f"the {budget_mib:.0f} MiB remaining after its largest streamed unit and fixed "
            f"overhead. This model cannot run via AirLLM on this host."
        )

    # AirLLM never enforces max_seq_len itself, so cap at the trained position limit when the
    # config declares one (no clamp, and no invented fallback, when it doesn't).
    max_position_embeddings = getattr(text_config, "max_position_embeddings", None) or getattr(
        config, "max_position_embeddings", None
    )
    try:
        max_position_embeddings = int(max_position_embeddings or 0)
    except (TypeError, ValueError):
        max_position_embeddings = 0
    if max_position_embeddings > 0:
        max_seq_len = min(max_seq_len, max_position_embeddings)
    return max_seq_len
