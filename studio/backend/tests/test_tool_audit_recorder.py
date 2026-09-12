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
