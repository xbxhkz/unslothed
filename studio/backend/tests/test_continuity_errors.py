# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""logs/errors.jsonl: append-only, queried by a substring match on symptom.

record_error is never-raises internally: if the primary write to errors.jsonl
fails, it falls back to appending into .remember/now.md rather than losing the
record of a failure silently. That fallback is tested here against a redirected
fallback path (monkeypatched), not the real ~/odysseus/.remember, since a test
must never touch the user's real notes.
"""

from __future__ import annotations

from core.continuity import prior_failures, record_error
from core.continuity import __init__ as continuity_module


def test_record_then_query_by_matching_symptom(tmp_path):
    record_error(str(tmp_path), what_tried = "loose regex", why_failed = "matched both brands",
                 symptom = "inert negative control")
    hits = prior_failures(str(tmp_path), "inert")
    assert len(hits) == 1
    assert hits[0].what_tried == "loose regex"


def test_query_matching_nothing_returns_empty_not_everything(tmp_path):
    record_error(str(tmp_path), what_tried = "a", why_failed = "b", symptom = "xyz")
    assert prior_failures(str(tmp_path), "this matches nothing recorded") == []


def test_multiple_entries_all_queryable(tmp_path):
    record_error(str(tmp_path), what_tried = "a", why_failed = "b", symptom = "git checkout on uncommitted work")
    record_error(str(tmp_path), what_tried = "c", why_failed = "d", symptom = "inert control")
    assert len(prior_failures(str(tmp_path), "git checkout")) == 1
    assert len(prior_failures(str(tmp_path), "inert")) == 1


def test_record_error_falls_back_when_the_primary_write_fails(tmp_path, monkeypatch):
    """A write failure must not lose the record, and must not raise into the
    caller either -- record_error is never-raises internally."""
    fallback_calls = []
    monkeypatch.setattr(
        continuity_module, "_append_fallback_note",
        lambda text: fallback_calls.append(text),
    )

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(continuity_module.storage, "write_json_atomic", _boom)
    # errors.jsonl append does not go through write_json_atomic (it is an
    # append-only file, not a replace-whole-file one) -- patch the actual append
    # primitive instead. See the implementation step below for its name.
    monkeypatch.setattr(continuity_module, "_append_error_line", _boom)

    record_error(str(tmp_path), what_tried = "x", why_failed = "y", symptom = "z")
    assert len(fallback_calls) == 1
    assert "x" in fallback_calls[0] and "y" in fallback_calls[0]
