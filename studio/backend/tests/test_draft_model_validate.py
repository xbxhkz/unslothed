# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Validation of a pinned drafter.

Auto-discovery is safe by construction: it only selects colocated,
name-matched sidecars. Pinning removes that protection, so these are the
checks that replace it. Each has a negative control proving it is the check
that fires, not a neighbour.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from routes.draft_model import (
    VERDICT_MISSING,
    VERDICT_NO_TARGET,
    VERDICT_OK,
    VERDICT_OUTSIDE,
    validate_choice,
)


def _fake_gguf(path: Path, size: int = 2048) -> Path:
    """A file that is not a real GGUF. Enough for existence/size/confinement,
    which run before any header parse."""
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"\0" * size)
    return path


class TestExistence:
    def test_a_missing_local_drafter_is_rejected_with_a_stated_reason(self, tmp_path):
        v = validate_choice(str(tmp_path / "target.gguf"), ("local", str(tmp_path / "nope.gguf")))
        assert not v.ok
        assert v.reason == VERDICT_MISSING
        assert "nope.gguf" in v.detail, "the reason must name the file, not just fail"

    def test_an_existing_local_drafter_passes_existence(self, tmp_path):
        target = _fake_gguf(tmp_path / "target.gguf")
        draft = _fake_gguf(tmp_path / "draft.gguf", size = 4096)
        v = validate_choice(str(target), ("local", str(draft)))
        assert v.reason != VERDICT_MISSING
        assert v.size_bytes == 4096, "size must be reported so the UI can warn before a 409"


class TestNoTarget:
    def test_a_local_choice_with_no_target_is_rejected_not_silently_unconfined(self, tmp_path):
        """With no target there is nothing to confine against, so an unconfined
        pass would accept any file on disk."""
        draft = _fake_gguf(tmp_path / "anywhere" / "d.gguf")
        v = validate_choice(None, ("local", str(draft)))
        assert not v.ok
        assert v.reason == VERDICT_NO_TARGET

    def test_a_remote_choice_with_no_target_is_still_fine(self, tmp_path):
        v = validate_choice(None, ("hf", "unsloth/Qwen3-0.6B-GGUF"))
        assert v.ok


class TestConfinement:
    def test_a_drafter_outside_the_target_tree_is_rejected(self, tmp_path):
        target = _fake_gguf(tmp_path / "models" / "target.gguf")
        outside = _fake_gguf(tmp_path / "elsewhere" / "secret.gguf")
        v = validate_choice(str(target), ("local", str(outside)))
        assert not v.ok
        assert v.reason == VERDICT_OUTSIDE

    def test_a_sibling_of_the_target_is_allowed(self, tmp_path):
        target = _fake_gguf(tmp_path / "models" / "target.gguf")
        sibling = _fake_gguf(tmp_path / "models" / "draft.gguf")
        v = validate_choice(str(target), ("local", str(sibling)))
        assert v.reason != VERDICT_OUTSIDE

    # --- negative control --------------------------------------------------
    def test_control_a_symlink_escape_is_rejected(self, tmp_path):
        """A path INSIDE the tree that resolves outside it. A confinement check
        using string prefixes on the unresolved path admits this."""
        target = _fake_gguf(tmp_path / "models" / "target.gguf")
        outside = _fake_gguf(tmp_path / "elsewhere" / "secret.gguf")
        link = tmp_path / "models" / "innocent.gguf"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable (Windows without developer mode)")
        v = validate_choice(str(target), ("local", str(link)))
        assert not v.ok, "a symlink out of the tree must not be admitted"
        assert v.reason == VERDICT_OUTSIDE

    def test_control_a_dotdot_traversal_escape_is_rejected(self, tmp_path):
        """Portable twin of the symlink control. `relative_to` on an UNRESOLVED
        path succeeds here -- the component list ['...','models','..','elsewhere']
        has ['...','models'] as a lexical prefix -- so without `.resolve()` this
        escaping path reads as confined. Verified against CPython's pathlib.
        """
        target = _fake_gguf(tmp_path / "models" / "target.gguf")
        outside = _fake_gguf(tmp_path / "elsewhere" / "secret.gguf")
        sneaky = tmp_path / "models" / ".." / "elsewhere" / "secret.gguf"
        assert Path(sneaky).exists(), "fixture must exist, or MISSING would mask OUTSIDE"
        v = validate_choice(str(target), ("local", str(sneaky)))
        assert not v.ok
        assert v.reason == VERDICT_OUTSIDE


class TestRemote:
    def test_an_hf_repo_is_not_subjected_to_local_path_checks(self, tmp_path):
        target = _fake_gguf(tmp_path / "target.gguf")
        v = validate_choice(str(target), ("hf", "unsloth/Qwen3-0.6B-GGUF"))
        assert v.reason not in (VERDICT_MISSING, VERDICT_OUTSIDE)


import struct

from routes.draft_model import (
    VERDICT_VOCAB_MISMATCH,
    VERDICT_VOCAB_UNKNOWN,
    read_gguf_vocab_size,
)

_GGUF_MAGIC = 0x46554747
_TYPE_ARRAY = 9
_TYPE_STRING = 8


def _write_gguf_with_vocab(path: Path, n_tokens: int) -> Path:
    """A minimal but REAL GGUF header whose tokenizer.ggml.tokens array has
    ``n_tokens`` entries. Built rather than mocked: the parser under test reads
    bytes, so a mock would test nothing about the parsing."""
    path.parent.mkdir(parents = True, exist_ok = True)
    key = b"tokenizer.ggml.tokens"
    body = struct.pack("<IIQQ", _GGUF_MAGIC, 3, 0, 1)
    body += struct.pack("<Q", len(key)) + key
    body += struct.pack("<I", _TYPE_ARRAY)
    body += struct.pack("<I", _TYPE_STRING)
    body += struct.pack("<Q", n_tokens)
    for i in range(n_tokens):
        tok = f"t{i}".encode()
        body += struct.pack("<Q", len(tok)) + tok
    path.write_bytes(body)
    return path


class TestVocabulary:
    def test_vocab_size_is_read_from_the_token_array_length(self, tmp_path):
        g = _write_gguf_with_vocab(tmp_path / "m.gguf", 7)
        assert read_gguf_vocab_size(str(g)) == 7

    def test_a_non_gguf_file_reads_as_unknown_not_zero(self, tmp_path):
        p = tmp_path / "not.gguf"
        p.write_bytes(b"\0" * 512)
        assert read_gguf_vocab_size(str(p)) is None

    def test_matching_vocabularies_pass(self, tmp_path):
        t = _write_gguf_with_vocab(tmp_path / "m" / "target.gguf", 32)
        d = _write_gguf_with_vocab(tmp_path / "m" / "draft.gguf", 32)
        v = validate_choice(str(t), ("local", str(d)))
        assert v.ok
        assert v.vocab_target == 32 and v.vocab_draft == 32

    def test_mismatched_vocabularies_are_rejected(self, tmp_path):
        t = _write_gguf_with_vocab(tmp_path / "m" / "target.gguf", 32)
        d = _write_gguf_with_vocab(tmp_path / "m" / "draft.gguf", 64)
        v = validate_choice(str(t), ("local", str(d)))
        assert not v.ok
        assert v.reason == VERDICT_VOCAB_MISMATCH
        assert "32" in v.detail and "64" in v.detail

    def test_an_unreadable_vocabulary_is_reported_not_silently_passed(self, tmp_path):
        """Fail visibly. A silent pass here is the exact shape of the
        sqlite-vec and update_flow.py defects this project has already paid
        for: a feature that quietly does not work."""
        t = _write_gguf_with_vocab(tmp_path / "m" / "target.gguf", 32)
        d = _fake_gguf(tmp_path / "m" / "draft.gguf")   # not a GGUF
        v = validate_choice(str(t), ("local", str(d)))
        assert v.reason == VERDICT_VOCAB_UNKNOWN
        assert not v.ok

    # --- negative control --------------------------------------------------
    def test_control_the_vocab_gate_fires_on_its_own(self, tmp_path):
        """The mismatched pair must pass existence, confinement and size, so a
        VOCAB_MISMATCH verdict cannot be a neighbouring check misfiring."""
        t = _write_gguf_with_vocab(tmp_path / "m" / "target.gguf", 32)
        d = _write_gguf_with_vocab(tmp_path / "m" / "draft.gguf", 64)
        assert d.is_file()                              # existence would pass
        assert d.parent == Path(str(t)).parent          # confinement would pass
        v = validate_choice(str(t), ("local", str(d)))
        assert v.reason == VERDICT_VOCAB_MISMATCH


class TestVocabularyParserRobustness:
    """The path is user-named, so a hostile or corrupted file is in scope, not
    just a well-formed non-GGUF. Each of these must degrade to ``None``
    (unknown), never raise -- an uncaught exception here is a 500 the user
    never asked for, and defeats the "returns None, not 0" contract this
    module exists to uphold."""

    def test_a_desynced_array_header_returns_unknown(self, tmp_path):
        """A truncated array header (4-byte type marker, missing the u64 count)
        desyncs the walk. The parser must return None rather than misread the
        remaining bytes.

        NOTE: this does NOT exercise recursion depth -- the stream desyncs before
        _skip_value can descend. test_a_properly_encoded_deeply_nested_array_does_
        not_blow_the_stack is what covers that, with a correctly encoded payload.
        """
        p = tmp_path / "nested.gguf"
        key = b"x"
        body = struct.pack("<IIQQ", _GGUF_MAGIC, 3, 0, 1)
        body += struct.pack("<Q", len(key)) + key
        depth = 1200
        for _ in range(depth):
            body += struct.pack("<I", 9)          # TYPE_ARRAY
        body += struct.pack("<I", 8) + struct.pack("<Q", 0)
        p.write_bytes(body)
        assert read_gguf_vocab_size(str(p)) is None

    def test_a_properly_encoded_deeply_nested_array_does_not_blow_the_stack(self, tmp_path):
        """Same intent as the test above, but with a byte layout that actually
        reaches the recursive branch under the pre-fix parser.

        Verified by running against the pre-fix code: the test above, as
        specified, writes only a 4-byte TYPE_ARRAY marker per nesting level,
        omitting each level's 8-byte element count. `_skip_value`'s array
        branch reads a 12-byte header per level (4-byte atype + 8-byte alen),
        so that payload desyncs immediately -- two consecutive 4-byte markers
        get reinterpreted as one 8-byte alen, and the very next recursive call
        hits a short read and returns False before any real depth is reached.
        Against the pre-fix parser it returned None without recursing, so it
        does not exercise the stack-depth bug on its own.

        This version writes a proper 12-byte (atype=ARRAY, alen=1) header per
        level, which pre-fix actually recursed 1200 Python-call frames deep and
        raised RecursionError (confirmed before applying the non-recursive
        rewrite). Kept alongside the test above rather than replacing it: that
        one still documents and guards the desync/truncation-on-malformed-input
        path, which is a real, separate case worth keeping covered.
        """
        p = tmp_path / "nested_well_formed.gguf"
        key = b"x"
        body = struct.pack("<IIQQ", _GGUF_MAGIC, 3, 0, 1)
        body += struct.pack("<Q", len(key)) + key
        depth = 1200
        body += struct.pack("<I", _TYPE_ARRAY)               # outer KV vtype
        for _ in range(depth - 1):
            body += struct.pack("<IQ", _TYPE_ARRAY, 1)       # atype=ARRAY, alen=1
        body += struct.pack("<IQ", _TYPE_STRING, 0)          # innermost leaf
        p.write_bytes(body)
        assert read_gguf_vocab_size(str(p)) is None

    def test_an_absurd_key_length_does_not_allocate(self, tmp_path):
        """A 40-byte file declaring a ~16 EB key length. Must return None promptly."""
        p = tmp_path / "huge.gguf"
        body = struct.pack("<IIQQ", _GGUF_MAGIC, 3, 0, 1)
        body += struct.pack("<Q", (1 << 64) - 1)
        p.write_bytes(body)
        assert read_gguf_vocab_size(str(p)) is None
