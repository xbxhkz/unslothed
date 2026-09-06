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
    VERDICT_INVALID_REPO_ID,
    VERDICT_MISSING,
    VERDICT_NO_TARGET,
    VERDICT_OK,
    VERDICT_OUTSIDE,
    VERDICT_UNVERIFIED,
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


class TestSplitShardConfinement:
    """The spec names this in both its Validation table and its Testing
    section: confinement must cover "split-shard symlinks", not just the
    launch path. llama-server opens the sibling shards of a split GGUF
    implicitly, so a check that stops at the file actually named on the
    command line confines nothing for a split set."""

    # Real GGUF headers with MATCHING vocabularies throughout, so every other
    # check in validate_choice passes. Without that, a shard-loop regression
    # would still be masked by a vocab_unknown rejection and these controls
    # would prove nothing about confinement -- the inert-control failure this
    # project has hit eight times.
    def _shard(self, path: Path) -> Path:
        return _write_gguf_with_vocab(path, 32)

    def test_a_complete_split_set_beside_the_target_is_allowed(self, tmp_path):
        target = _write_gguf_with_vocab(tmp_path / "models" / "target.gguf", 32)
        first = self._shard(tmp_path / "models" / "d-00001-of-00002.gguf")
        self._shard(tmp_path / "models" / "d-00002-of-00002.gguf")
        v = validate_choice(str(target), ("local", str(first)))
        assert v.ok, (
            "a legitimately colocated split set must not be rejected by the "
            "shard check the escaping case needs"
        )

    # --- negative controls -------------------------------------------------
    def test_control_a_symlinked_second_shard_outside_the_tree_is_rejected(self, tmp_path):
        """The spec's own wording. Shard 1 is a real file inside the permitted
        directory with a matching vocabulary, so EVERY other check passes and
        only the shard loop can reject. SKIPS without symlink support -- see
        the monkeypatched twin below, which runs everywhere."""
        target = _write_gguf_with_vocab(tmp_path / "models" / "target.gguf", 32)
        first = self._shard(tmp_path / "models" / "d-00001-of-00002.gguf")
        outside = self._shard(tmp_path / "elsewhere" / "secret.gguf")
        second = tmp_path / "models" / "d-00002-of-00002.gguf"
        try:
            second.symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable (Windows without developer mode)")
        assert not second.resolve().is_relative_to((tmp_path / "models").resolve()), (
            "fixture must actually escape, or this control proves nothing"
        )
        v = validate_choice(str(target), ("local", str(first)))
        assert not v.ok, "a second shard pointing out of the tree must not be admitted"
        assert v.reason == VERDICT_OUTSIDE
        assert "shard" in v.detail, "the reason must say it was a shard, not the pick itself"

    def test_control_an_escaping_shard_is_rejected_on_every_platform(
        self, tmp_path, monkeypatch
    ):
        """Portable twin of the symlink control, and the one that actually runs
        on this host.

        Why it monkeypatches the enumerator rather than building the escape on
        disk: ``colocated_split_shards`` lists ``path.parent.iterdir()``, whose
        entries are single name components under one real directory. Without
        per-file symlinks every entry therefore resolves inside that directory,
        so shard 1 and shard 2 cannot disagree about confinement -- the
        divergence the spec describes is not constructible on a symlink-less
        filesystem. What IS worth binding on every platform is the loop that
        consumes the enumeration: that it resolves each shard and rejects one
        that escapes. So the enumeration (upstream, already tested) is replaced
        and the shard loop under test is fed a `..` traversal, exactly the
        escape shape the launch-path control uses.
        """
        import routes.draft_model as mod

        target = _write_gguf_with_vocab(tmp_path / "models" / "target.gguf", 32)
        first = self._shard(tmp_path / "models" / "d-00001-of-00002.gguf")
        outside = self._shard(tmp_path / "elsewhere" / "secret.gguf")
        escaping = tmp_path / "models" / ".." / "elsewhere" / "secret.gguf"
        assert Path(escaping).exists(), "fixture must exist, or MISSING would mask OUTSIDE"
        assert outside.exists()

        def fake_shards(path):
            return [Path(first), escaping], True

        monkeypatch.setattr(mod, "colocated_split_shards", fake_shards)
        # Everything else about this pick is valid -- shard 1 exists, is
        # confined, and its vocabulary matches the target's -- so the verdict
        # below is the shard loop's alone. Without it: ok=True.
        v = validate_choice(str(target), ("local", str(first)))
        assert not v.ok, (
            "the shard loop must resolve each shard; on the UNRESOLVED path "
            "relative_to() succeeds and this escape reads as confined"
        )
        assert v.reason == VERDICT_OUTSIDE
        assert "shard" in v.detail

    def test_an_enumeration_failure_does_not_crash_the_verdict(self, tmp_path, monkeypatch):
        """A permissions error listing the directory must degrade to checking
        the one path we do know, not become a 500."""
        import routes.draft_model as mod

        target = _write_gguf_with_vocab(tmp_path / "models" / "target.gguf", 32)
        draft = _write_gguf_with_vocab(tmp_path / "models" / "d.gguf", 32)

        def boom(path):
            raise OSError("permission denied")

        monkeypatch.setattr(mod, "colocated_split_shards", boom)
        v = validate_choice(str(target), ("local", str(draft)))
        assert v.ok, "an unlistable directory must not turn a valid pick into a rejection"


class TestRemote:
    def test_an_hf_repo_is_not_subjected_to_local_path_checks(self, tmp_path):
        target = _fake_gguf(tmp_path / "target.gguf")
        v = validate_choice(str(target), ("hf", "unsloth/Qwen3-0.6B-GGUF"))
        assert v.reason not in (VERDICT_MISSING, VERDICT_OUTSIDE)

    def test_a_well_shaped_repo_is_accepted_but_marked_unverified(self, tmp_path):
        """The spec's rule for a remote drafter is an explicit could-not-verify
        state, "never a silent pass". An affirmative ok with reason "ok" IS
        that silent pass: nothing about the repository was checked, and the
        caller has no way to tell that from a fully validated local pick."""
        v = validate_choice(str(_fake_gguf(tmp_path / "t.gguf")), ("hf", "unsloth/Qwen3-0.6B-GGUF"))
        assert v.ok, "a plausible repo id is still usable; the load path resolves it"
        assert v.reason == VERDICT_UNVERIFIED
        assert v.reason != VERDICT_OK, (
            "reporting VERDICT_OK would tell the UI this was checked when it was not"
        )
        assert "unverified" in v.detail.lower()

    @pytest.mark.parametrize(
        "bad",
        [
            "/etc/passwd",          # a path, not a repo id
            "a/b/c",                # three segments
            "unsloth/../secrets",   # traversal
            " unsloth/Qwen3",       # leading space
            "unsloth/Qwen3.git",    # git suffix
            "",                     # empty
        ],
    )
    def test_a_malformed_repo_id_is_rejected_not_pinned(self, bad):
        """A shape failure is a rejection. There is no useful thing downstream
        can do with it, and admitting it writes a path or a typo into
        --spec-draft-hf where it fails much later and much less clearly."""
        v = validate_choice(None, ("hf", bad))
        assert not v.ok
        assert v.reason == VERDICT_INVALID_REPO_ID

    # --- negative control --------------------------------------------------
    def test_control_the_shape_check_fires_on_its_own(self):
        """The rejected id differs from the accepted one ONLY in shape: same
        namespace, same style of name, no filesystem or network involvement in
        either. So an INVALID_REPO_ID verdict cannot be a neighbouring check
        misfiring."""
        good = validate_choice(None, ("hf", "unsloth/Qwen3-0.6B-GGUF"))
        bad = validate_choice(None, ("hf", "unsloth/Qwen3--0.6B-GGUF"))
        assert good.ok and good.reason == VERDICT_UNVERIFIED
        assert not bad.ok and bad.reason == VERDICT_INVALID_REPO_ID


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

    def test_an_overflowing_key_length_is_caught_by_the_backstop(self, tmp_path):
        """A 32-byte file declaring a ~16 EB key length. Must return None.

        RENAMED, because the old name ("does not allocate") claimed to bind
        `_MAX_KEY_LEN` and no longer does: the commit that added that bound
        also added OverflowError to the except tuple, and 2**64-1 is large
        enough that `f.read()` raises OverflowError before any bound is
        consulted. Delete `_MAX_KEY_LEN` and this still passes -- via the
        backstop. So this is a backstop test, and it is named like one; what
        binds the bound is the control below.
        """
        p = tmp_path / "huge.gguf"
        body = struct.pack("<IIQQ", _GGUF_MAGIC, 3, 0, 1)
        body += struct.pack("<Q", (1 << 64) - 1)
        p.write_bytes(body)
        assert read_gguf_vocab_size(str(p)) is None

    # --- negative controls for the bounds themselves -----------------------
    def test_control_a_key_length_past_the_bound_is_never_read(self, tmp_path, monkeypatch):
        """Binds `_MAX_KEY_LEN`, which the backstop cannot stand in for.

        The declared length is one byte past the bound: small enough that
        `f.read()` would SUCCEED (a short read on a small file, no exception),
        so nothing in the except tuple can fire. The bound is therefore the
        only thing that can stop the read from being attempted, and the
        observable difference is that the read is not attempted at all --
        which is the property the bound exists for, since the whole point is
        not allocating an attacker-chosen buffer.

        Proven to fail with the bound removed: the spy then records the
        1048577-byte read.
        """
        import routes.draft_model as mod

        p = tmp_path / "past_bound.gguf"
        body = struct.pack("<IIQQ", _GGUF_MAGIC, 3, 0, 1)
        body += struct.pack("<Q", mod._MAX_KEY_LEN + 1)
        p.write_bytes(body)

        reads: list[int] = []
        real_open = open

        class _ReadSpy:
            def __init__(self, fh):
                self._fh = fh

            def read(self, n = -1):
                reads.append(n)
                return self._fh.read(n)

            def seek(self, *a, **k):
                return self._fh.seek(*a, **k)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return self._fh.__exit__(*exc)

        # A module-level `open` shadows the builtin for this module only.
        monkeypatch.setattr(
            mod, "open", lambda *a, **k: _ReadSpy(real_open(*a, **k)), raising = False
        )
        assert read_gguf_vocab_size(str(p)) is None
        assert reads, "the spy must actually have been used"
        assert max(reads) <= mod._MAX_KEY_LEN, (
            f"a read of {max(reads)} bytes was attempted for an attacker-declared "
            "key length; the bound must refuse it before the read"
        )

    def test_control_an_absurd_vocab_array_length_reads_as_unknown(self, tmp_path):
        """Binds `_MAX_LEN` on the RETURNED value (the fifth, previously
        unbounded, length in this parser).

        Sharper than its four siblings, which only skip past a bad length:
        this one is returned. Unbounded, this file reads as a vocabulary of
        2**63, which is not None -- so two such files compare EQUAL and
        validate_choice admits the pair, the exact silent pass the
        "None, never 0" contract exists to prevent.

        Proven to fail with the bound removed: it then returns 9223372036854775808.
        """
        p = tmp_path / "absurd_vocab.gguf"
        key = b"tokenizer.ggml.tokens"
        body = struct.pack("<IIQQ", _GGUF_MAGIC, 3, 0, 1)
        body += struct.pack("<Q", len(key)) + key
        body += struct.pack("<I", _TYPE_ARRAY)
        body += struct.pack("<I", _TYPE_STRING)
        body += struct.pack("<Q", 1 << 63)
        p.write_bytes(body)
        assert read_gguf_vocab_size(str(p)) is None

    def test_control_two_absurd_headers_are_not_admitted_as_a_matching_pair(self, tmp_path):
        """What the bound above actually protects: end to end, through
        validate_choice. Two crafted siblings declaring the same absurd
        vocabulary must be rejected as unknown, not accepted as a match."""
        t = tmp_path / "m" / "target.gguf"
        d = tmp_path / "m" / "draft.gguf"
        t.parent.mkdir(parents = True, exist_ok = True)
        key = b"tokenizer.ggml.tokens"
        for p in (t, d):
            body = struct.pack("<IIQQ", _GGUF_MAGIC, 3, 0, 1)
            body += struct.pack("<Q", len(key)) + key
            body += struct.pack("<I", _TYPE_ARRAY)
            body += struct.pack("<I", _TYPE_STRING)
            body += struct.pack("<Q", 1 << 63)
            p.write_bytes(body)
        v = validate_choice(str(t), ("local", str(d)))
        assert not v.ok
        assert v.reason == VERDICT_VOCAB_UNKNOWN
