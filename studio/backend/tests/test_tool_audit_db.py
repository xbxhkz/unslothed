# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Storage for the tool audit log.

Owns its own table and schema rather than extending studio_db's _ensure_schema,
so the fork adds a file instead of editing one.
"""

from __future__ import annotations

import time

import pytest

from storage import tool_audit_db


@pytest.fixture(autouse = True)
def _isolated_db(tmp_path, monkeypatch):
    """Point the studio data root at a temp dir so tests never touch real data."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    tool_audit_db.reset_for_tests()
    yield


def _start(**over):
    kw = dict(
        tool_name = "terminal",
        arguments_json = '{"command": "ls"}',
        paths_json = "[]",
        redacted = False,
        session_id = "sess-1",
        thread_id = "thread-1",
        disable_sandbox = False,
    )
    kw.update(over)
    return tool_audit_db.record_start(**kw)


def test_record_start_creates_a_running_row():
    row_id = _start()
    entry = tool_audit_db.get_entry(row_id)
    assert entry is not None
    assert entry["outcome"] == "running"
    assert entry["tool_name"] == "terminal"
    assert entry["duration_ms"] is None


def test_record_finish_completes_the_row():
    row_id = _start()
    tool_audit_db.record_finish(
        row_id,
        outcome = "ok",
        duration_ms = 12,
        result_head = "total 0",
        result_tail = "",
        result_bytes = 7,
        result_sha256 = "abc",
        error_text = None,
    )
    entry = tool_audit_db.get_entry(row_id)
    assert entry["outcome"] == "ok"
    assert entry["duration_ms"] == 12
    assert entry["result_bytes"] == 7


def test_error_text_is_stored_whole():
    """A truncated traceback is useless; this column is deliberately uncapped."""
    row_id = _start()
    long_tb = "Traceback\n" + ("frame\n" * 5000)
    tool_audit_db.record_finish(
        row_id,
        outcome = "error",
        duration_ms = 1,
        result_head = "",
        result_tail = "",
        result_bytes = 0,
        result_sha256 = "",
        error_text = long_tb,
    )
    assert tool_audit_db.get_entry(row_id)["error_text"] == long_tb


def test_query_filters_by_tool_and_session():
    _start(tool_name = "terminal", session_id = "a")
    _start(tool_name = "python", session_id = "a")
    _start(tool_name = "terminal", session_id = "b")
    assert len(tool_audit_db.query_entries(tool_name = "terminal")) == 2
    assert len(tool_audit_db.query_entries(session_id = "a")) == 2
    assert len(tool_audit_db.query_entries(tool_name = "terminal", session_id = "b")) == 1


def test_prune_by_row_cap_writes_its_own_row():
    """A forensic log that silently drops history is worse than no log: 'it never
    happened' and 'it scrolled off' must stay distinguishable."""
    for _ in range(10):
        _start()
    deleted = tool_audit_db.prune(max_rows = 4)
    assert deleted > 0
    remaining = tool_audit_db.query_entries(limit = 100)
    markers = [e for e in remaining if e["tool_name"] == "__prune__"]
    assert len(markers) == 1, "prune must record itself"
    assert str(deleted) in markers[0]["arguments_json"]


def test_prune_by_age():
    old_id = _start()
    with tool_audit_db._connect() as conn:
        conn.execute(
            "UPDATE tool_audit SET ts = ? WHERE id = ?",
            (time.time() - 400 * 86400, old_id),
        )
        conn.commit()
    fresh_id = _start()
    tool_audit_db.prune(max_age_days = 365)
    assert tool_audit_db.get_entry(old_id) is None
    assert tool_audit_db.get_entry(fresh_id) is not None
