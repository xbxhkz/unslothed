# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read-only API over the tool audit log."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from routes.tool_audit import router
from storage import tool_audit_db


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    tool_audit_db.reset_for_tests()
    app = FastAPI()
    app.include_router(router, prefix = "/api/tool-audit")
    # Every endpoint depends on get_current_subject; these tests exercise
    # behaviour behind auth, not auth itself, so the dependency is overridden
    # rather than satisfied with a real credential. dependency_overrides is
    # the supported seam for this -- Depends() captures the function object at
    # import time, so monkeypatching the auth module attribute would not
    # affect routes already bound to it.
    app.dependency_overrides[get_current_subject] = lambda: "test-subject"
    return TestClient(app)


def _seed(tool_name = "terminal", session_id = "s1"):
    return tool_audit_db.record_start(
        tool_name = tool_name,
        arguments_json = '{"command": "ls"}',
        paths_json = "[]",
        redacted = False,
        session_id = session_id,
        thread_id = None,
        disable_sandbox = False,
    )


def test_entries_returns_rows_newest_first(client):
    _seed(tool_name = "terminal")
    _seed(tool_name = "python")
    r = client.get("/api/tool-audit/entries")
    assert r.status_code == 200
    body = r.json()
    assert [e["tool_name"] for e in body["entries"]] == ["python", "terminal"]


def test_entries_filters_by_tool(client):
    _seed(tool_name = "terminal")
    _seed(tool_name = "python")
    r = client.get("/api/tool-audit/entries", params = {"tool_name": "python"})
    assert [e["tool_name"] for e in r.json()["entries"]] == ["python"]


def test_single_entry_and_404(client):
    row_id = _seed()
    assert client.get(f"/api/tool-audit/entries/{row_id}").status_code == 200
    assert client.get("/api/tool-audit/entries/999999").status_code == 404


def test_status_reports_degradation(client):
    from core.inference import tool_audit

    tool_audit.reset_degraded_for_tests()
    body = client.get("/api/tool-audit/status").json()
    assert body["degraded"] is False
    assert body["failed_writes"] == 0


def test_entries_requires_authentication(tmp_path, monkeypatch):
    """These endpoints return a forensic record of every tool run on the
    machine, arguments included. Without a live auth dependency they would be
    a complete activity log readable by anyone who can reach the port, so
    this asserts the guard is actually wired -- not just present on the
    other, overridden client -- and stays red if a future edit drops it."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    tool_audit_db.reset_for_tests()
    app = FastAPI()
    app.include_router(router, prefix = "/api/tool-audit")
    unauthenticated_client = TestClient(app)

    r = unauthenticated_client.get("/api/tool-audit/entries")

    assert r.status_code in (401, 403)
