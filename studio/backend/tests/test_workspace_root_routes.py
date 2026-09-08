# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HTTP contract for the chosen workspace root."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


@pytest.fixture
def client(monkeypatch):
    """Auth is replaced via dependency_overrides, NOT monkeypatch: Depends()
    captures the function object at import time, so reassigning the module
    attribute would leave real auth in place and the tests would prove nothing."""
    from auth.authentication import get_current_subject
    from routes.settings import router
    import utils.workspace_root as mod

    store = {}
    monkeypatch.setattr(mod, "_read_setting", lambda key, fallback = None: store.get(key, fallback))
    monkeypatch.setattr(mod, "_write_setting", lambda mapping: store.update(mapping))

    app = FastAPI()
    app.include_router(router, prefix = "/api/settings")
    app.dependency_overrides[get_current_subject] = lambda: "test-subject"
    return TestClient(app)


class TestGlobalRootEndpoints:
    def test_unset_reads_as_null(self, client):
        r = client.get("/api/settings/workspace-root")
        assert r.status_code == 200
        assert r.json()["path"] is None

    def test_put_then_get_round_trips(self, client, tmp_path):
        r = client.put("/api/settings/workspace-root", json = {"path": str(tmp_path)})
        assert r.status_code == 200
        assert client.get("/api/settings/workspace-root").json()["path"] is not None

    def test_clearing_sets_null(self, client, tmp_path):
        client.put("/api/settings/workspace-root", json = {"path": str(tmp_path)})
        client.put("/api/settings/workspace-root", json = {"path": None})
        assert client.get("/api/settings/workspace-root").json()["path"] is None

    def test_a_dangerous_path_is_ACCEPTED_and_warned_about(self, client, tmp_path, monkeypatch):
        """Warn-only, at the HTTP boundary. A 4xx here would be a spec violation."""
        import utils.workspace_root as mod
        monkeypatch.setattr(mod, "_studio_root", lambda: str(tmp_path))
        r = client.put("/api/settings/workspace-root", json = {"path": str(tmp_path)})
        assert r.status_code == 200, "warn-only: a flagged path must still be accepted"
        assert any(w["code"] == "studio_data" for w in r.json()["warnings"])

    def test_preview_does_not_store(self, client, tmp_path):
        r = client.post("/api/settings/workspace-root/preview", json = {"path": str(tmp_path)})
        assert r.status_code == 200
        assert r.json()["exists"] is True
        assert client.get("/api/settings/workspace-root").json()["path"] is None, (
            "preview must not have side effects"
        )

    def test_preview_reports_a_missing_directory(self, client, tmp_path):
        r = client.post("/api/settings/workspace-root/preview",
                        json = {"path": str(tmp_path / "nope")})
        assert r.json()["exists"] is False


class TestProjectPatchModel:
    def test_chatprojectpatch_accepts_rootpath(self):
        from routes.chat_history import ChatProjectPatch
        assert "rootPath" in ChatProjectPatch.model_fields
