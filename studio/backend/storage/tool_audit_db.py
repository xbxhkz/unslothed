# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persistent record of every tool invocation.

Owns its own table and its own schema check rather than extending studio_db's
_ensure_schema: the fork adds a file instead of editing one, which keeps the
upstream storage module at a zero-line diff.

The table is deliberately append-mostly. Rows are written twice -- once on entry
with outcome='running', once on exit -- so a crash mid-tool leaves visible
evidence rather than nothing at all.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from storage.studio_db import get_connection

_SCHEMA_READY = False

_DDL = """
CREATE TABLE IF NOT EXISTS tool_audit (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ts               REAL    NOT NULL,
    duration_ms      INTEGER,
    session_id       TEXT,
    thread_id        TEXT,
    tool_name        TEXT    NOT NULL,
    arguments_json   TEXT    NOT NULL,
    paths_json       TEXT,
    redacted         INTEGER NOT NULL DEFAULT 0,
    disable_sandbox  INTEGER NOT NULL DEFAULT 0,
    outcome          TEXT    NOT NULL,
    result_head      TEXT,
    result_tail      TEXT,
    result_bytes     INTEGER,
    result_sha256    TEXT,
    error_text       TEXT
);
CREATE INDEX IF NOT EXISTS idx_tool_audit_ts ON tool_audit (ts DESC);
CREATE INDEX IF NOT EXISTS idx_tool_audit_tool ON tool_audit (tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_audit_session ON tool_audit (session_id);
"""

# Written as a normal row so the log can always explain its own gaps.
PRUNE_MARKER_TOOL = "__prune__"


def reset_for_tests() -> None:
    """Forget that the schema was created; used by tests that swap data roots."""
    global _SCHEMA_READY
    _SCHEMA_READY = False


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    conn = get_connection()
    try:
        global _SCHEMA_READY
        if not _SCHEMA_READY:
            conn.executescript(_DDL)
            conn.commit()
            _SCHEMA_READY = True
        yield conn
    finally:
        conn.close()


def record_start(
    *,
    tool_name: str,
    arguments_json: str,
    paths_json: str,
    redacted: bool,
    session_id: Optional[str],
    thread_id: Optional[str],
    disable_sandbox: bool,
) -> int:
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO tool_audit
                (ts, session_id, thread_id, tool_name, arguments_json, paths_json,
                 redacted, disable_sandbox, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running')
            """,
            (
                time.time(),
                session_id,
                thread_id,
                tool_name,
                arguments_json,
                paths_json,
                1 if redacted else 0,
                1 if disable_sandbox else 0,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def record_finish(
    row_id: int,
    *,
    outcome: str,
    duration_ms: int,
    result_head: str,
    result_tail: str,
    result_bytes: int,
    result_sha256: str,
    error_text: Optional[str],
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            UPDATE tool_audit
               SET outcome = ?, duration_ms = ?, result_head = ?, result_tail = ?,
                   result_bytes = ?, result_sha256 = ?, error_text = ?
             WHERE id = ?
            """,
            (
                outcome,
                duration_ms,
                result_head,
                result_tail,
                result_bytes,
                result_sha256,
                error_text,
                row_id,
            ),
        )
        conn.commit()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["redacted"] = bool(d.get("redacted"))
    d["disable_sandbox"] = bool(d.get("disable_sandbox"))
    return d


def get_entry(row_id: int) -> Optional[dict[str, Any]]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM tool_audit WHERE id = ?", (row_id,)).fetchone()
        return _row_to_dict(row) if row else None


def query_entries(
    *,
    limit: int = 100,
    offset: int = 0,
    tool_name: Optional[str] = None,
    session_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if tool_name:
        clauses.append("tool_name = ?")
        params.append(tool_name)
    if session_id:
        clauses.append("session_id = ?")
        params.append(session_id)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    params.extend([max(1, min(limit, 1000)), max(0, offset)])
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT * FROM tool_audit{where} ORDER BY ts DESC, id DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def prune(*, max_age_days: int = 365, max_rows: int = 250_000) -> int:
    """Delete old rows, then excess rows. Records what it did as a row of its own."""
    cutoff = time.time() - max_age_days * 86400
    with _connect() as conn:
        deleted = conn.execute(
            "DELETE FROM tool_audit WHERE ts < ? AND tool_name != ?",
            (cutoff, PRUNE_MARKER_TOOL),
        ).rowcount
        over = conn.execute("SELECT COUNT(*) FROM tool_audit").fetchone()[0] - max_rows
        if over > 0:
            deleted += conn.execute(
                """
                DELETE FROM tool_audit WHERE id IN (
                    SELECT id FROM tool_audit
                     WHERE tool_name != ?
                     ORDER BY ts ASC, id ASC
                     LIMIT ?
                )
                """,
                (PRUNE_MARKER_TOOL, over),
            ).rowcount
        if deleted:
            conn.execute(
                """
                INSERT INTO tool_audit
                    (ts, tool_name, arguments_json, paths_json, redacted,
                     disable_sandbox, outcome, duration_ms)
                VALUES (?, ?, ?, '[]', 0, 0, 'ok', 0)
                """,
                (
                    time.time(),
                    PRUNE_MARKER_TOOL,
                    json.dumps(
                        {
                            "deleted": deleted,
                            "max_age_days": max_age_days,
                            "max_rows": max_rows,
                        }
                    ),
                ),
            )
        conn.commit()
        return deleted
