# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""logs/errors.jsonl: append-only, queried by a substring match on symptom.

record_error is never-raises internally, and reports honestly whether the entry
landed anywhere: True for errors.jsonl, or for a fallback file ONLY when the
caller explicitly passes fallback_path; False when it was lost. The core
hardcodes no fallback target -- the app tool passes none (a model-driven write
must not escape the sandbox), the CLI passes the human's own .remember/now.md.
Every fallback here is a tmp_path file, never the real ~/odysseus/.remember,
since a test must never touch the user's real notes.
"""

from __future__ import annotations

from core.continuity import prior_failures, record_error
from core.continuity import __init__ as continuity_module


def _fail_the_primary_write(monkeypatch):
    # errors.jsonl is append-only, not replace-whole-file, so it does not go
    # through write_json_atomic -- patch the actual append primitive.
    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(continuity_module, "_append_error_line", _boom)


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


def test_a_torn_line_does_not_make_later_entries_unreachable(tmp_path):
    """errors.jsonl is append-only text, not atomically written like the JSON
    files -- a partial write from a killed process (or a hand edit) is a real
    possibility, and one bad line must not sink every entry after it."""
    record_error(str(tmp_path), what_tried = "a", why_failed = "b", symptom = "good entry one")
    errors_path = continuity_module._errors_path(str(tmp_path))
    with open(errors_path, "a", encoding = "utf-8") as f:
        f.write('{"ts": "2026-01-01", "project": "x", "what_tried": "torn\n')
    record_error(str(tmp_path), what_tried = "c", why_failed = "d", symptom = "good entry two")
    hits = prior_failures(str(tmp_path), "good entry")
    assert {h.symptom for h in hits} == {"good entry one", "good entry two"}


def test_record_error_returns_true_when_the_primary_write_lands(tmp_path):
    assert record_error(str(tmp_path), what_tried = "a", why_failed = "b", symptom = "c") is True


def test_record_error_with_no_fallback_path_reports_the_loss_and_writes_nowhere(tmp_path, monkeypatch):
    """The app tool's shape: no fallback_path. When the primary write fails the
    entry is genuinely lost, and record_error must say so -- not claim success,
    and not quietly write model-controlled text to a file outside the sandbox."""
    _fail_the_primary_write(monkeypatch)
    fallback_calls = []
    monkeypatch.setattr(continuity_module, "_append_fallback_note",
                        lambda path, text: fallback_calls.append((path, text)) or True)
    before = sorted(tmp_path.rglob("*"))

    recorded = record_error(str(tmp_path), what_tried = "x", why_failed = "y", symptom = "z")

    assert recorded is False
    assert fallback_calls == [], f"wrote a fallback note nobody asked for: {fallback_calls}"
    assert sorted(tmp_path.rglob("*")) == before
    assert not hasattr(continuity_module, "_REMEMBER_NOW_PATH"), \
        "the core must not hardcode a fallback target; only the CLI knows that path"


def test_record_error_with_a_fallback_path_lands_there_when_the_primary_write_fails(tmp_path, monkeypatch):
    """The CLI's shape: a human on their own machine opts in to a fallback file."""
    _fail_the_primary_write(monkeypatch)
    fallback = tmp_path / "remember" / "now.md"

    recorded = record_error(str(tmp_path), what_tried = "loose regex", why_failed = "matched both",
                            symptom = "inert control", fallback_path = str(fallback))

    assert recorded is True
    text = fallback.read_text(encoding = "utf-8")
    assert "loose regex" in text and "matched both" in text


def test_record_error_reports_false_when_the_fallback_fails_too(tmp_path, monkeypatch):
    """Both writes fail: still never raises, and says so rather than claiming success.
    The fallback fails for real here (its directory would have to be a plain file),
    so _append_fallback_note's own failure branch is exercised, not stubbed."""
    _fail_the_primary_write(monkeypatch)
    blocker = tmp_path / "blocker"
    blocker.write_text("a plain file where a directory would have to be", encoding = "utf-8")

    recorded = record_error(str(tmp_path), what_tried = "x", why_failed = "y", symptom = "z",
                            fallback_path = str(blocker / "now.md"))

    assert recorded is False
