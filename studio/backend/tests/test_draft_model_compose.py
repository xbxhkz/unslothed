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

from routes.draft_model import compose_draft_args, current_draft_pin


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


class TestReadingThePinBack:
    """The read-back half. A picker that cannot see what is already pinned
    renders "Automatic" for a pinned model, which is how the shared-args risk
    the spec flagged actually bites: the user re-picks over a choice they
    could not see, or assumes the feature did nothing.

    These share `_draft_flag_spans` with the composer above, so a spelling the
    stripper removes is always a spelling the reader recognises. That is the
    property being pinned here -- an independent reader is free to drift, and
    the first one written did, recognising two spellings out of seven.
    """

    @pytest.mark.parametrize("flag", ["-md", "--model-draft", "--spec-draft-model"])
    def test_every_local_spelling_is_read_back(self, flag):
        assert current_draft_pin([flag, "/d.gguf"]) == ("local", "/d.gguf")

    @pytest.mark.parametrize(
        "flag", ["-hfd", "-hfrd", "--hf-repo-draft", "--spec-draft-hf"]
    )
    def test_every_hf_spelling_is_read_back(self, flag):
        assert current_draft_pin([flag, "a/b"]) == ("hf", "a/b")

    def test_the_equals_spelling_is_read_back(self):
        assert current_draft_pin(["--model-draft=/d.gguf"]) == ("local", "/d.gguf")

    def test_the_underscore_spelling_is_read_back(self):
        """`_flag_name` normalises `_` to `-`, so this spelling is stripped by
        the composer. A reader that missed it would show "Automatic" for a pin
        the composer would nonetheless replace."""
        assert current_draft_pin(["--model_draft", "/d.gguf"]) == ("local", "/d.gguf")

    def test_no_drafter_flag_reads_as_no_pin(self):
        assert current_draft_pin(["--threads", "8", "--flash-attn"]) is None
        assert current_draft_pin([]) is None
        assert current_draft_pin(None) is None

    def test_the_last_flag_wins_as_llama_server_would(self):
        args = ["-md", "/first.gguf", "--spec-draft-hf", "org/second"]
        assert current_draft_pin(args) == ("hf", "org/second")

    def test_a_pin_is_read_out_of_a_crowd_of_unrelated_args(self):
        args = ["--threads", "8", "--alias", "my-md-model", "-md", "/d.gguf", "-c", "4096"]
        assert current_draft_pin(args) == ("local", "/d.gguf")

    def test_a_valueless_trailing_flag_pins_nothing(self):
        """A malformed trailing `-md` owns no value; reporting it as a pin on
        "" would seed the picker with an empty selection that reads as a real
        choice."""
        assert current_draft_pin(["--threads", "8", "-md"]) is None

    # --- negative control --------------------------------------------------
    def test_control_a_value_that_merely_contains_md_is_not_read_as_a_pin(self):
        """The reader must match parsed flag names, not substrings -- the same
        control the composer carries, because they are now the same scanner."""
        assert current_draft_pin(["--alias", "my-md-model"]) is None

    def test_control_reader_and_stripper_agree_on_every_spelling(self):
        """Binds the two halves together. For each spelling: whatever the
        reader reports as pinned is exactly what the stripper removes, so a
        vocabulary change cannot land on one side only."""
        for flag in [
            "-md", "--model-draft", "--spec-draft-model",
            "-hfd", "-hfrd", "--hf-repo-draft", "--spec-draft-hf",
            "--model_draft",
        ]:
            args = ["--threads", "8", flag, "value", "--flash-attn"]
            assert current_draft_pin(args) is not None, flag
            assert compose_draft_args(args, None) == ["--threads", "8", "--flash-attn"], flag
