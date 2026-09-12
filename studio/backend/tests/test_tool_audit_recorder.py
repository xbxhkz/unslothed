# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The audit recorder wrapping execute_tool.

The invariant that matters most is negative: auditing must never break a tool
call. A test asserting that is the difference between an audit feature and an
outage.
"""

from __future__ import annotations

import pytest

from core.inference import tool_audit
from storage import tool_audit_db


@pytest.fixture(autouse = True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    tool_audit_db.reset_for_tests()
    tool_audit.reset_degraded_for_tests()
    yield


def _fake_tool(name, arguments, **kwargs):
    return "tool output"


def test_successful_call_is_recorded_and_result_passes_through():
    out = tool_audit.around(_fake_tool, "terminal", {"command": "ls"}, session_id = "s1")
    assert out == "tool output"
    entries = tool_audit_db.query_entries()
    assert len(entries) == 1
    assert entries[0]["tool_name"] == "terminal"
    assert entries[0]["outcome"] == "ok"
    assert entries[0]["session_id"] == "s1"


def test_raising_tool_is_recorded_as_error_and_the_exception_propagates():
    def boom(name, arguments, **kwargs):
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError, match = "kaboom"):
        tool_audit.around(boom, "python", {"code": "1/0"})
    entry = tool_audit_db.query_entries()[0]
    assert entry["outcome"] == "error"
    assert "kaboom" in entry["error_text"]


def test_a_raising_RECORDER_does_not_break_the_tool_call(monkeypatch):
    """THE critical invariant. If auditing fails, the tool still works."""
    def explode(**kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(tool_audit_db, "record_start", explode)
    out = tool_audit.around(_fake_tool, "terminal", {"command": "ls"})
    assert out == "tool output", "a failed audit write must not cost the caller its result"
    assert tool_audit.degraded_count() >= 1, "and the failure must be counted, not silent"


def test_arguments_are_redacted_before_storage():
    tool_audit.around(_fake_tool, "terminal", {"api_key": "hunter2"})
    entry = tool_audit_db.query_entries()[0]
    assert "hunter2" not in entry["arguments_json"]
    assert entry["redacted"] is True


def test_large_results_are_capped_but_length_is_true():
    big = "x" * 50_000

    def big_tool(name, arguments, **kwargs):
        return big

    tool_audit.around(big_tool, "python", {})
    entry = tool_audit_db.query_entries()[0]
    assert entry["result_bytes"] == 50_000, "the TRUE length must survive truncation"
    assert len(entry["result_head"]) <= tool_audit.RESULT_CAP_BYTES
    assert len(entry["result_tail"]) <= tool_audit.RESULT_CAP_BYTES


def test_disable_sandbox_is_recorded():
    tool_audit.around(_fake_tool, "terminal", {"command": "ls"}, disable_sandbox = True)
    assert tool_audit_db.query_entries()[0]["disable_sandbox"] is True


def test_a_raising___str___does_not_break_the_tool_call():
    """A successful tool result whose __str__ raises must still be returned.

    The coercion to text for storage happens after the tool has already
    succeeded; a bad __str__ on the result must degrade the audit, not cost
    the caller its result.
    """
    class Explodes:
        def __str__(self):
            raise OverflowError("nope")

    bad = Explodes()

    def bad_tool(name, arguments, **kwargs):
        return bad

    out = tool_audit.around(bad_tool, "python", {})
    assert out is bad, "a failed result-to-text coercion must not cost the caller its result"
    assert tool_audit.degraded_count() >= 1, "and the failure must be counted, not silent"


def test_a_raising___repr___does_not_replace_the_original_exception():
    """The mirror of the __str__ fix above, one line away: repr(exc) must run
    INSIDE the guarded region. If it ran in around()'s own frame, an exception
    whose __repr__ raises would replace the caller's real exception with the
    audit's formatting failure."""
    class BadRepr(RuntimeError):
        def __repr__(self):
            raise OverflowError("nope")

    def boom(name, arguments, **kwargs):
        raise BadRepr("boom")

    with pytest.raises(BadRepr):
        tool_audit.around(boom, "python", {})
    assert tool_audit.degraded_count() >= 1, "the repr() failure must be counted, not silent"


def test_a_secret_in_an_exception_message_is_redacted_in_error_text():
    """Exceptions routinely embed the very arguments that were just scrubbed
    from arguments_json; error_text must not be the unredacted back door."""
    def boom(name, arguments, **kwargs):
        raise RuntimeError("upstream rejected Authorization: Bearer sk-live-abcdef0123456789xyz")

    with pytest.raises(RuntimeError):
        tool_audit.around(boom, "terminal", {})
    entry = tool_audit_db.query_entries()[0]
    assert "sk-live-abcdef0123456789xyz" not in entry["error_text"]


def test_an_ordinary_error_message_is_NOT_mangled():
    """Control for the above: over-redacting error_text would be as bad as
    under-redacting it."""
    def boom(name, arguments, **kwargs):
        raise RuntimeError("connection refused on port 8080")

    with pytest.raises(RuntimeError):
        tool_audit.around(boom, "terminal", {})
    entry = tool_audit_db.query_entries()[0]
    assert "connection refused on port 8080" in entry["error_text"]


def test_error_text_is_kept_whole_even_when_redacted():
    """redact_text must not be given an implicit cap -- 'errors kept whole'
    means whole, not unredacted."""
    def boom(name, arguments, **kwargs):
        raise RuntimeError("frame\n" * 5000 + "Bearer sk-live-abcdef0123456789xyz")

    with pytest.raises(RuntimeError):
        tool_audit.around(boom, "terminal", {})
    entry = tool_audit_db.query_entries()[0]
    assert "sk-live-abcdef0123456789xyz" not in entry["error_text"]
    assert len(entry["error_text"]) > tool_audit.RESULT_CAP_BYTES * 2


def _mcp_tool_returning(result):
    def mcp_tool(name, arguments, **kwargs):
        return result
    return mcp_tool


def test_a_credential_referencing_mcp_call_has_its_result_withheld():
    """Redaction layer 1 (spec): upstream's own classifier fires on an MCP
    call whose arguments name a credential path, so the result must never
    reach the row -- the demonstrated failure was ~/.aws/credentials landing
    verbatim in result_head with redacted=0."""
    out = tool_audit.around(
        _mcp_tool_returning("aws_secret_access_key=hunter2"),
        "mcp__testserver__read_file",
        {"path": "~/.aws/credentials"},
    )
    assert out == "aws_secret_access_key=hunter2", "the tool call itself must still succeed"
    entry = tool_audit_db.query_entries()[0]
    assert entry["redacted"] is True
    assert "hunter2" not in (entry["result_head"] or "")
    assert entry["result_head"] == tool_audit.WITHHELD_MARKER
    assert entry["result_tail"] == ""
    assert entry["result_bytes"] == len(b"aws_secret_access_key=hunter2"), (
        "result_bytes must still be the TRUE length so a withheld row is not "
        "mistaken for a genuinely empty one"
    )


def test_a_non_mcp_call_with_the_same_shaped_arguments_is_unaffected():
    """Only MCP tools are in scope for layer 1: a first-party tool reading the
    same-looking path is a different trust surface and must not be withheld."""
    out = tool_audit.around(
        lambda name, arguments, **kw: "aws_secret_access_key=hunter2",
        "terminal",
        {"path": "~/.aws/credentials"},
    )
    assert out == "aws_secret_access_key=hunter2"
    entry = tool_audit_db.query_entries()[0]
    assert entry["result_head"] == "aws_secret_access_key=hunter2"
    assert entry["result_head"] != tool_audit.WITHHELD_MARKER


def test_an_ordinary_mcp_call_still_has_its_result_stored():
    """THE negative control for layer 1. Without it, withholding everything an
    MCP tool returns would also make this fix 'look correct'."""
    out = tool_audit.around(
        _mcp_tool_returning("42 results found"),
        "mcp__testserver__search",
        {"query": "hello world"},
    )
    assert out == "42 results found"
    entry = tool_audit_db.query_entries()[0]
    assert entry["result_head"] == "42 results found"
    assert entry["result_head"] != tool_audit.WITHHELD_MARKER
    assert entry["redacted"] is False
