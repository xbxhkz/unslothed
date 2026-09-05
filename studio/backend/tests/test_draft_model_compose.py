# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Composition of llama_extra_args for a pinned drafter.

The flag vocabulary is imported from upstream rather than retyped, so the
canaries here are load-bearing: if upstream renames or moves those sets, these
tests fail loudly instead of the composer silently missing a spelling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from routes.draft_model import compose_draft_args


class TestUpstreamCanaries:
    """These pin the upstream names the composer depends on. A rename upstream
    must break a test here, not leak into production as an unrecognised flag."""

    def test_flag_sets_are_importable_and_populated(self):
        from core.inference.llama_cpp import _HF_DRAFT_FLAGS, _LOCAL_DRAFT_FLAGS
        assert "-md" in _LOCAL_DRAFT_FLAGS
        assert "--model-draft" in _LOCAL_DRAFT_FLAGS
        assert "--spec-draft-model" in _LOCAL_DRAFT_FLAGS
        assert "-hfd" in _HF_DRAFT_FLAGS
        assert "--spec-draft-hf" in _HF_DRAFT_FLAGS
        assert not (_HF_DRAFT_FLAGS & _LOCAL_DRAFT_FLAGS)

    def test_flag_name_helper_is_importable(self):
        from core.inference.llama_server_args import _flag_name
        assert _flag_name("--model-draft=/x.gguf") == "--model-draft"
        assert _flag_name("-md") == "-md"


class TestComposition:
    def test_pins_a_local_drafter_on_empty_args(self):
        assert compose_draft_args([], ("local", "/models/d.gguf")) == [
            "--model-draft", "/models/d.gguf"
        ]

    def test_pins_an_hf_drafter_on_empty_args(self):
        assert compose_draft_args([], ("hf", "unsloth/Qwen3-0.6B-GGUF")) == [
            "--spec-draft-hf", "unsloth/Qwen3-0.6B-GGUF"
        ]

    def test_unrelated_args_survive_untouched_and_in_order(self):
        existing = ["--threads", "8", "--flash-attn", "-c", "4096"]
        out = compose_draft_args(existing, ("local", "/d.gguf"))
        assert out[:5] == existing, "hand-written args must not be reordered or dropped"
        assert out[5:] == ["--model-draft", "/d.gguf"]

    def test_an_existing_drafter_flag_is_replaced_not_duplicated(self):
        existing = ["--threads", "8", "-md", "/old.gguf", "--flash-attn"]
        out = compose_draft_args(existing, ("local", "/new.gguf"))
        assert out == ["--threads", "8", "--flash-attn", "--model-draft", "/new.gguf"]

    def test_the_equals_spelling_is_recognised_and_removed(self):
        existing = ["--model-draft=/old.gguf", "--threads", "8"]
        out = compose_draft_args(existing, ("local", "/new.gguf"))
        assert "--model-draft=/old.gguf" not in out
        assert out == ["--threads", "8", "--model-draft", "/new.gguf"]

    def test_every_spelling_is_removed_including_the_rare_ones(self):
        existing = ["-hfrd", "a/b", "--hf-repo-draft", "c/d", "--spec-draft-model", "/e.gguf"]
        out = compose_draft_args(existing, ("hf", "x/y"))
        assert out == ["--spec-draft-hf", "x/y"]

    def test_clearing_removes_the_drafter_and_restores_auto_discovery(self):
        existing = ["--threads", "8", "-md", "/old.gguf"]
        assert compose_draft_args(existing, None) == ["--threads", "8"]

    # --- negative controls -------------------------------------------------
    # Each must FAIL against a naive substring implementation. Verified in Step 2.

    def test_control_a_value_that_merely_contains_md_is_not_stripped(self):
        """`--cache-type-k` with value `md_something` must survive. A composer
        that removes any token containing "-md" eats it."""
        existing = ["--alias", "my-md-model", "--threads", "8"]
        out = compose_draft_args(existing, ("local", "/d.gguf"))
        assert "--alias" in out and "my-md-model" in out

    def test_control_a_drafter_flags_value_is_removed_with_it(self):
        """Removing `-md` but leaving `/old.gguf` behind would hand llama-server
        a stray positional argument."""
        out = compose_draft_args(["-md", "/old.gguf"], None)
        assert out == []
