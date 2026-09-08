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
        """A cheap structural smoke check -- the two tests below are the real
        coverage, exercising the field through the model, model_dump, storage
        and the persisted row. This one just fails fast on an outright typo."""
        from routes.chat_history import ChatProjectPatch
        assert "rootPath" in ChatProjectPatch.model_fields


class TestProjectPatchRootPathWiring:
    """Behavioural coverage for ChatProjectPatch.rootPath: through the
    Pydantic model, model_dump(exclude_unset = True), update_chat_project, to
    the row PATCH /projects/{id} actually persists.

    Calls the route function directly (as test_project_workspace_location.py's
    test_creating_a_project_says_which_folder_failed does for save_project)
    rather than through a TestClient: current_subject is a plain keyword here,
    not a Depends() the ASGI layer resolves, so there is no auth wiring to
    stand up for a route this module does not otherwise touch.
    """

    def _make_project(self, tmp_path, monkeypatch):
        import time

        from routes.chat_history import ChatProject, save_project

        # Otherwise project creation writes into the real user's Documents
        # folder -- this is the only thing Studio writes there.
        monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects_home"))
        now = int(time.time() * 1000)
        project = save_project(
            ChatProject(
                id = "11111111-2222-3333-4444-555555555555",
                name = "Probe",
                instructions = "",
                archived = False,
                createdAt = now,
                updatedAt = now,
            ),
            current_subject = "test-subject",
        )
        return project.id

    def test_patching_rootpath_changes_the_persisted_value(self, tmp_path, monkeypatch):
        from routes.chat_history import ChatProjectPatch, patch_project
        from storage.studio_db import get_chat_project

        project_id = self._make_project(tmp_path, monkeypatch)
        chosen_root = tmp_path / "chosen-root"

        patch_project(
            project_id,
            ChatProjectPatch(rootPath = str(chosen_root)),
            current_subject = "test-subject",
        )

        stored = get_chat_project(project_id)
        assert Path(stored["rootPath"]) == chosen_root.resolve()

    def test_omitting_rootpath_leaves_the_stored_root_untouched(self, tmp_path, monkeypatch):
        """Patching an unrelated field must not clobber rootPath. This is what
        proves exclude_unset is doing its job: a naive model_dump() without it
        would serialize the omitted rootPath as None and wipe the stored root."""
        from routes.chat_history import ChatProjectPatch, patch_project
        from storage.studio_db import get_chat_project

        project_id = self._make_project(tmp_path, monkeypatch)
        root_before = get_chat_project(project_id)["rootPath"]

        patch_project(
            project_id,
            ChatProjectPatch(name = "Renamed"),
            current_subject = "test-subject",
        )

        after = get_chat_project(project_id)
        assert after["rootPath"] == root_before
        assert after["name"] == "Renamed"
