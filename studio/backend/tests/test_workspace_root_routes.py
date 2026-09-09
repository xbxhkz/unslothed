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


def _app(monkeypatch, *, via_api_key: bool):
    """Auth is replaced via dependency_overrides, NOT monkeypatch: Depends()
    captures the function object at import time, so reassigning the module
    attribute would leave real auth in place and the tests would prove nothing.

    ``authenticated_via_api_key`` is overridden as well as the subject: the real
    dependency reads an Authorization header these tests do not send, and
    HTTPBearer's auto_error would answer 403 before any route ran.
    """
    from auth.authentication import authenticated_via_api_key, get_current_subject
    from routes.settings import router
    import utils.workspace_root as mod

    store = {}
    monkeypatch.setattr(mod, "_read_setting", lambda key, fallback = None: store.get(key, fallback))
    monkeypatch.setattr(mod, "_write_setting", lambda mapping: store.update(mapping))

    app = FastAPI()
    app.include_router(router, prefix = "/api/settings")
    app.dependency_overrides[get_current_subject] = lambda: "test-subject"
    app.dependency_overrides[authenticated_via_api_key] = lambda: via_api_key
    return TestClient(app)


@pytest.fixture
def client(monkeypatch):
    """An interactive UI session, which is what the settings UI is."""
    return _app(monkeypatch, via_api_key = False)


@pytest.fixture
def api_key_client(monkeypatch):
    """A programmatic caller holding an sk-unsloth key, not a UI session."""
    return _app(monkeypatch, via_api_key = True)


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


class TestOnlyTheUiMaySetIt:
    """Warn-only governs WHICH paths may be chosen, not WHO may choose one.

    This setting decides where `terminal` runs with full read and write, so it
    is gated exactly like the llama.cpp executable path in the same file.
    """

    def test_an_api_key_may_not_set_the_workspace_root(self, api_key_client, tmp_path):
        r = api_key_client.put("/api/settings/workspace-root", json = {"path": str(tmp_path)})
        assert r.status_code == 403

    def test_an_api_key_that_is_refused_stores_nothing(self, api_key_client, tmp_path):
        api_key_client.put("/api/settings/workspace-root", json = {"path": str(tmp_path)})
        assert api_key_client.get("/api/settings/workspace-root").json()["path"] is None


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

    def test_an_unusable_root_is_refused_before_it_is_committed(self, tmp_path, monkeypatch):
        """The bad value must never reach the row.

        update_chat_project used to write and commit it, and only the ensure
        call after that discovered the folder could not be made -- leaving the
        project permanently pointed at a folder that works nowhere, and 500ing
        every later read of the project list.
        """
        from fastapi import HTTPException

        from routes.chat_history import ChatProjectPatch, patch_project
        from storage import studio_db
        from storage.studio_db import get_chat_project

        project_id = self._make_project(tmp_path, monkeypatch)
        before = get_chat_project(project_id)["rootPath"]

        blocked = tmp_path / "no-entry"
        real_ensure_dir = studio_db.ensure_dir

        def refuse(path):
            if "no-entry" in str(path):
                raise PermissionError(13, "Permission denied", str(path))
            return real_ensure_dir(path)

        monkeypatch.setattr(studio_db, "ensure_dir", refuse)

        with pytest.raises(HTTPException) as caught:
            patch_project(
                project_id,
                ChatProjectPatch(rootPath = str(blocked)),
                current_subject = "test-subject",
            )

        assert caught.value.status_code == 400
        assert str(blocked) in str(caught.value.detail)
        assert get_chat_project(project_id)["rootPath"] == before, (
            "a refused folder must not have been committed"
        )

    def test_a_stored_unusable_root_does_not_break_the_project_list(self, tmp_path, monkeypatch):
        """One bad row must not 500 GET /projects.

        That list is the only surface from which the user can re-point the
        project, so taking it out makes the bad value unrecoverable.
        """
        from routes import chat_history
        from storage import studio_db

        project_id = self._make_project(tmp_path, monkeypatch)
        blocked = tmp_path / "no-entry"
        conn = studio_db.get_connection()
        try:
            conn.execute(
                "UPDATE chat_projects SET root_path = ? WHERE id = ?", (str(blocked), project_id)
            )
            conn.commit()
        finally:
            conn.close()

        real_ensure_dir = studio_db.ensure_dir

        def refuse(path):
            if "no-entry" in str(path):
                raise PermissionError(13, "Permission denied", str(path))
            return real_ensure_dir(path)

        monkeypatch.setattr(studio_db, "ensure_dir", refuse)

        listed = chat_history.list_projects(current_subject = "test-subject")
        assert [p.id for p in listed.projects] == [project_id]
        assert chat_history.get_project(project_id, current_subject = "test-subject").id == project_id

    def test_a_path_that_is_not_a_path_at_all_is_reported_as_a_bad_folder(self):
        """Path() raises ValueError, not OSError, on an embedded NUL -- uncaught,
        it escapes as a bare 500 from wherever the project was being read."""
        from storage.studio_db import ProjectWorkspaceError, _ensure_project_workspace

        with pytest.raises(ProjectWorkspaceError):
            _ensure_project_workspace("C:\\bad\x00path")

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
