# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Delegate tool calls must be distinguishable from the primary's.

Attribution goes in its own column rather than in session_id: approvals are
keyed by session, so changing it would send the prompt somewhere the user is not
looking.

Nothing is imported from another test module: studio/__init__.py and
studio/backend/__init__.py make a bare `tests` package resolve elsewhere under
pytest, and a collection error interrupts the whole run. The delegation rig
below therefore COPIES the shape of test_delegation_tool.py's rather than
importing it.

The load-bearing test here is the last one. Every other test sets the
thread-local by hand, which proves the column works and proves nothing about the
wiring -- whether a delegation actually sets that thread-local while its
delegate's tools run, and on the thread that reads it. That one drives a real
delegation through the real audited execute_tool shadow instead.
"""

from __future__ import annotations

import pytest

from core.inference import delegation
from core.inference import model_roles
from core.inference import tool_audit
from core.inference.delegation import workspace
from core.inference.model_roles import storage
from storage import tool_audit_db

# The tool_audit table exactly as it stood at commit 46e8b89a, written out
# literally rather than derived from _DDL: the point of the migration test is to
# stand up a PRE-CHANGE installed database, and a stub built from three columns
# would make every later INSERT fail on a missing column -- silently, because
# around() never raises -- so the test would pass while proving nothing.
_SCHEMA_AT_46E8B89A = """
CREATE TABLE tool_audit (
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


@pytest.fixture(autouse = True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    tool_audit_db.reset_for_tests()
    tool_audit.reset_degraded_for_tests()
    delegation._reset_for_tests()
    yield
    delegation._reset_for_tests()


def _fake_tool(name, arguments, **kwargs):
    return "ok"


def test_a_primary_tool_call_has_no_delegation_id():
    tool_audit.around(_fake_tool, "terminal", {"command": "ls"}, session_id = "s1")
    row = tool_audit_db.query_entries()[0]
    assert row["delegation_id"] is None


def test_a_delegate_tool_call_carries_the_delegation_id():
    delegation._active.delegation_id = "deleg-123"
    try:
        tool_audit.around(_fake_tool, "terminal", {"command": "ls"}, session_id = "s1")
    finally:
        delegation._active.delegation_id = None
    row = tool_audit_db.query_entries()[0]
    assert row["delegation_id"] == "deleg-123"


def test_the_session_id_is_unchanged_by_a_delegation():
    """Approvals are keyed by session; attribution must not move it."""
    delegation._active.delegation_id = "deleg-123"
    try:
        tool_audit.around(_fake_tool, "terminal", {}, session_id = "s1")
    finally:
        delegation._active.delegation_id = None
    assert tool_audit_db.query_entries()[0]["session_id"] == "s1"


def test_a_broken_attribution_lookup_still_records_the_row():
    """Never-raises, at the one seam this task adds. Attribution is decoration on
    a row that has to be written either way -- a tool call must not lose its audit
    record because the delegation module could not be asked."""
    def explode():
        raise RuntimeError("delegation module is unimportable")

    original = delegation.active_delegation_id
    delegation.active_delegation_id = explode
    try:
        out = tool_audit.around(_fake_tool, "terminal", {}, session_id = "s1")
    finally:
        delegation.active_delegation_id = original
    assert out == "ok"
    rows = tool_audit_db.query_entries()
    assert len(rows) == 1, "the row is still written; only the attribution is lost"
    assert rows[0]["delegation_id"] is None


def test_an_existing_database_gains_the_column_and_can_still_be_written_to():
    """The table already exists in installed databases, so this needs a
    migration, not just a new CREATE TABLE.

    Three things are asserted together, because the first two alone would pass on
    a migration that added the column to a table nothing could be inserted into:
    the column is added, a row written AFTERWARDS lands and carries its
    delegation_id, and nothing was swallowed on the way.
    """
    tool_audit.around(_fake_tool, "terminal", {}, session_id = "s1")

    # Stand the pre-change schema back up, the way an installed database looks.
    with tool_audit_db._connect() as conn:
        conn.executescript("DROP TABLE tool_audit;")
        conn.executescript(_SCHEMA_AT_46E8B89A)
        conn.commit()
    tool_audit_db.reset_for_tests()

    delegation._active.delegation_id = "deleg-migrated"
    try:
        tool_audit.around(_fake_tool, "terminal", {"command": "ls"}, session_id = "s2")
    finally:
        delegation._active.delegation_id = None

    with tool_audit_db._connect() as conn:
        columns = {r[1] for r in conn.execute("PRAGMA table_info(tool_audit)").fetchall()}
    assert "delegation_id" in columns, "the migration never ran"

    rows = tool_audit_db.query_entries()
    assert len(rows) == 1, "a migrated database must still accept writes"
    assert rows[0]["session_id"] == "s2"
    assert rows[0]["delegation_id"] == "deleg-migrated"
    # The one that catches an INSERT that failed quietly: around() never raises,
    # so a broken write shows up here or nowhere.
    assert tool_audit.degraded_count() == 0, "something was swallowed on the way"


# -- the real thing: a delegation, end to end -----------------------------


class FakeLoader:
    """No test may load a model. Copied in shape from test_delegation_tool.py."""

    def __init__(self, resident = "repo/Primary:Q4"):
        self.resident = resident
        self.calls = []

    def resident_model_id(self):
        return self.resident

    def serves(self, model_id):
        return self.resident == model_id

    def resident_is_restorable(self):
        return True

    def load(self, model_id, overrides = None):
        self.calls.append(model_id)
        self.resident = model_id


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "_workdir_for", lambda session_id: str(tmp_path))
    monkeypatch.setattr(storage, "get_role_bindings",
                        lambda: {"coding": {"model": "repo/Coder:Q4"}})
    monkeypatch.setattr(model_roles, "_resolve_local", lambda model: ("/p", None, model))
    fake = FakeLoader()
    monkeypatch.setattr(delegation, "_loader", fake)
    delegation._reset_for_tests()
    return fake


def _one_tool_call(name, arguments):
    """A model that asks for one tool on its first turn and answers on its second."""
    turns = []

    def call_model(messages, tools):
        turns.append(1)
        if len(turns) == 1:
            return {
                "content": "",
                "tool_calls": [{"id": "c1", "function": {"name": name, "arguments": arguments}}],
            }
        return {"content": "delegate answer", "tool_calls": []}

    return call_model


def test_a_real_delegations_tool_call_is_attributed_to_it(rig, monkeypatch):
    """THE load-bearing test of this task.

    Every test above sets delegation._active.delegation_id by hand, which proves
    the column stores what it is given and nothing about whether a delegation
    ever gives it anything. This one runs a delegation for real -- the same
    _delegate that sets the thread-local, the same audited execute_tool shadow
    the delegate's tools go through -- and compares the row against the
    delegation's OWN id, taken from workspace.create rather than from the same
    thread-local the code under test wrote.

    No real tool runs: tools._execute_tool_unaudited is replaced with a recorder,
    which leaves the audited shadow around it running for real.
    """
    import core.inference.tools as tools_module

    seen = []

    def recorder(name, arguments, **kwargs):
        seen.append(name)
        return "ok"

    monkeypatch.setattr(tools_module, "_execute_tool_unaudited", recorder)

    created = {}
    real_create = workspace.create

    def spy_create(session_id, role):
        files = real_create(session_id, role)
        created["delegation_id"] = files.delegation_id
        return files

    monkeypatch.setattr(workspace, "create", spy_create)
    monkeypatch.setattr(delegation, "_call_model_for",
                        lambda binding: _one_tool_call("read_file", {"path": "a.txt"}))

    out = delegation.execute(
        "ask_model", {"role": "coding", "task": "rewrite the parser"}, session_id = "s1"
    )

    assert "delegate answer" in out, out
    assert seen == ["read_file"], "the delegate's tool must have gone through the real shadow"
    assert created.get("delegation_id"), "the delegation never created its workspace"

    rows = tool_audit_db.query_entries(tool_name = "read_file")
    assert len(rows) == 1
    assert rows[0]["delegation_id"] == created["delegation_id"], \
        "the delegate's row must name the delegation that made the call"

    # Same process, same thread, after the delegation's finally has run. The
    # thread-local must be cleared, or every later tool call in this conversation
    # is attributed to a delegation that already ended.
    tools_module.execute_tool("terminal", {"command": "ls"}, session_id = "s1")
    after = tool_audit_db.query_entries(tool_name = "terminal")
    assert len(after) == 1
    assert after[0]["delegation_id"] is None, \
        "a call outside a delegation must carry no attribution"
    assert tool_audit.degraded_count() == 0
