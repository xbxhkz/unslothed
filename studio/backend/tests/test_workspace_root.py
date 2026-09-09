# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Advisory classification of a user-chosen workspace root.

The policy is warn-only: nothing here may refuse a path. These tests pin both
halves of that -- that the dangerous cases DO warn, and that warning is all
they do.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from utils.workspace_root import (
    WARN_BROAD,
    WARN_INSTALL_DIR,
    WARN_STUDIO_DATA,
    WARN_SYSTEM_DIR,
    classify_root,
)


def _codes(path):
    return {w.code for w in classify_root(str(path))}


class TestDangerousRoots:
    def test_studio_data_dir_warns_about_auth_and_models(self, tmp_path, monkeypatch):
        import utils.workspace_root as mod
        monkeypatch.setattr(mod, "_studio_root", lambda: str(tmp_path / "studio"))
        warnings = classify_root(str(tmp_path / "studio"))
        assert WARN_STUDIO_DATA in {w.code for w in warnings}
        msg = " ".join(w.message for w in warnings).lower()
        assert "auth" in msg, "the warning must say WHAT is at stake, not just that there is risk"

    def test_a_subdirectory_of_the_studio_data_dir_also_warns(self, tmp_path, monkeypatch):
        import utils.workspace_root as mod
        monkeypatch.setattr(mod, "_studio_root", lambda: str(tmp_path / "studio"))
        assert WARN_STUDIO_DATA in _codes(tmp_path / "studio" / "cache")

    def test_the_install_directory_warns(self, tmp_path, monkeypatch):
        import utils.workspace_root as mod
        monkeypatch.setattr(mod, "_install_root", lambda: str(tmp_path / "app"))
        assert WARN_INSTALL_DIR in _codes(tmp_path / "app")

    def test_a_filesystem_root_warns_as_a_system_directory(self):
        root = os.path.abspath(os.sep)
        assert WARN_SYSTEM_DIR in _codes(root)


class TestBroadRoots:
    def test_the_home_directory_warns_as_broad(self, tmp_path, monkeypatch):
        monkeypatch.setenv("USERPROFILE", str(tmp_path / "me"))
        monkeypatch.setenv("HOME", str(tmp_path / "me"))
        assert WARN_BROAD in _codes(tmp_path / "me")


class TestOrdinaryRoots:
    def test_an_ordinary_project_folder_warns_about_nothing(self, tmp_path, monkeypatch):
        import utils.workspace_root as mod
        monkeypatch.setattr(mod, "_studio_root", lambda: str(tmp_path / "studio"))
        monkeypatch.setattr(mod, "_install_root", lambda: str(tmp_path / "app"))
        monkeypatch.setenv("USERPROFILE", str(tmp_path / "me"))
        monkeypatch.setenv("HOME", str(tmp_path / "me"))
        assert classify_root(str(tmp_path / "me" / "code" / "my-project")) == []


class TestWarnOnlyIsStructural:
    def test_classify_never_refuses(self, tmp_path, monkeypatch):
        """Every classification returns warnings. There is no rejection channel
        at all -- the policy is enforced by the return type, not by discipline."""
        import utils.workspace_root as mod
        monkeypatch.setattr(mod, "_studio_root", lambda: str(tmp_path / "studio"))
        for candidate in (os.path.abspath(os.sep), str(tmp_path / "studio"), str(tmp_path / "x")):
            result = classify_root(candidate)
            assert isinstance(result, list)
            assert all(hasattr(w, "code") and hasattr(w, "message") for w in result)

    # --- negative controls ------------------------------------------------
    # Each must FAIL if its own rule is deleted, and must NOT fire on a
    # neighbour's path. Verified in Step 2.

    def test_control_studio_rule_does_not_fire_on_a_sibling_named_similarly(
        self, tmp_path, monkeypatch
    ):
        """A string-prefix check would match `studio-notes` against `studio`."""
        import utils.workspace_root as mod
        monkeypatch.setattr(mod, "_studio_root", lambda: str(tmp_path / "studio"))
        assert WARN_STUDIO_DATA not in _codes(tmp_path / "studio-notes")

    def test_control_broad_does_not_fire_on_a_folder_inside_home(
        self, tmp_path, monkeypatch
    ):
        """Only home ITSELF is broad. If a nested folder warns, the rule is
        matching a prefix rather than the directory."""
        monkeypatch.setenv("USERPROFILE", str(tmp_path / "me"))
        monkeypatch.setenv("HOME", str(tmp_path / "me"))
        assert WARN_BROAD not in _codes(tmp_path / "me" / "code")


from utils.workspace_root import WORKSPACE_ROOT_KEY, get_global_root, set_global_root


class TestGlobalDefault:
    def test_unset_reads_as_none(self, monkeypatch):
        import utils.workspace_root as mod
        monkeypatch.setattr(mod, "_read_setting", lambda key, fallback = None: fallback)
        assert get_global_root() is None

    def test_a_stored_value_is_returned_expanded(self, monkeypatch, tmp_path):
        import utils.workspace_root as mod
        monkeypatch.setattr(mod, "_read_setting", lambda key, fallback = None: str(tmp_path))
        assert get_global_root() == os.path.realpath(str(tmp_path))

    def test_a_blank_stored_value_reads_as_none(self, monkeypatch):
        """An empty string must not become a workdir of "" -- that would
        resolve to the process's cwd, which is not a directory the user chose."""
        import utils.workspace_root as mod
        monkeypatch.setattr(mod, "_read_setting", lambda key, fallback = None: "   ")
        assert get_global_root() is None

    def test_a_missing_directory_reads_as_none(self, monkeypatch, tmp_path):
        """A root that no longer exists must not be returned: _get_workdir would
        then makedirs() it and silently recreate a folder the user deleted."""
        import utils.workspace_root as mod
        monkeypatch.setattr(mod, "_read_setting", lambda key, fallback = None: str(tmp_path / "gone"))
        assert get_global_root() is None

    def test_setting_writes_under_the_documented_key(self, monkeypatch, tmp_path):
        seen = {}
        import utils.workspace_root as mod
        monkeypatch.setattr(mod, "_write_setting", lambda mapping: seen.update(mapping))
        set_global_root(str(tmp_path))
        assert seen == {WORKSPACE_ROOT_KEY: os.path.realpath(str(tmp_path))}

    def test_clearing_writes_none(self, monkeypatch):
        seen = {}
        import utils.workspace_root as mod
        monkeypatch.setattr(mod, "_write_setting", lambda mapping: seen.update(mapping))
        set_global_root(None)
        assert seen == {WORKSPACE_ROOT_KEY: None}


class TestResolutionOrder:
    """The three-way order: project root, then global default, then sandbox."""

    def test_the_global_default_is_used_when_no_project_applies(self, monkeypatch, tmp_path):
        from core.inference import tools
        chosen = tmp_path / "chosen"
        chosen.mkdir()
        monkeypatch.setattr(tools, "_project_workdir_for", lambda sid: None)
        import utils.workspace_root as mod
        monkeypatch.setattr(mod, "_read_setting", lambda key, fallback = None: str(chosen))
        assert os.path.realpath(tools._get_workdir("sess-global")) == os.path.realpath(str(chosen))

    def test_a_project_root_beats_the_global_default(self, monkeypatch, tmp_path):
        from core.inference import tools
        proj = tmp_path / "proj"; proj.mkdir()
        glob = tmp_path / "glob"; glob.mkdir()
        monkeypatch.setattr(tools, "_project_workdir_for", lambda sid: str(proj))
        import utils.workspace_root as mod
        monkeypatch.setattr(mod, "_read_setting", lambda key, fallback = None: str(glob))
        assert os.path.realpath(tools._get_workdir("sess-both")) == os.path.realpath(str(proj))

    def test_a_session_less_call_does_not_use_the_global_root(self, monkeypatch, tmp_path):
        """An anonymous caller gets the sandbox, never the user's chosen folder.
        This is also what keeps _claim_sandbox from marking a directory Studio
        does not own -- which is why the guard below it needs no edit."""
        from core.inference import tools
        chosen = tmp_path / "chosen"; chosen.mkdir()
        monkeypatch.setattr(tools, "_project_workdir_for", lambda sid: None)
        import utils.workspace_root as mod
        monkeypatch.setattr(mod, "_read_setting", lambda key, fallback = None: str(chosen))
        result = tools._get_workdir(None)
        assert os.path.realpath(result) != os.path.realpath(str(chosen))
        # Not merely "somewhere else": in the sandbox, where it was before this
        # feature existed.
        assert tools._contained_in_root(result, tools.sandbox_root())

    # --- the control that protects every existing chat --------------------
    def test_control_with_nothing_set_the_sandbox_is_unchanged(self, monkeypatch):
        """The opt-in invariant. If this fails, the feature has changed
        behaviour for users who never asked for it."""
        from core.inference import tools
        import utils.workspace_root as mod
        monkeypatch.setattr(mod, "_read_setting", lambda key, fallback = None: None)
        monkeypatch.setattr(tools, "_project_workdir_for", lambda sid: None)
        first = tools._get_workdir("sess-untouched")
        tools._workdirs.pop("sess-untouched", None)
        second = tools._get_workdir("sess-untouched")
        assert first == second
        # Containment only. The `or "sandbox" in first.lower()` this used to
        # start with short-circuited on every run -- the sandbox path contains
        # that substring by construction -- so the assertion that matters never
        # executed.
        assert tools._contained_in_root(first, tools.sandbox_root())

    def test_the_two_resolvers_agree_for_a_global_root_session(self, monkeypatch, tmp_path):
        """_get_workdir is where tools WRITE; resolve_sandbox_workdir is where
        the UI READS (file download, folder listing, "open chat folder"). They
        answer the same question and must not disagree: while they did, tools
        wrote to the chosen folder and every file card in the transcript 404'd
        against an empty sandbox.

        The read is taken FIRST, uncached, because that is the order the UI
        hits it in -- listing a chat's folder before any tool has run.
        """
        from core.inference import tools
        chosen = tmp_path / "chosen"; chosen.mkdir()
        monkeypatch.setattr(tools, "_project_workdir_for", lambda sid: None)
        import utils.workspace_root as mod
        monkeypatch.setattr(mod, "_read_setting", lambda key, fallback = None: str(chosen))

        tools._workdirs.pop("sess-agree", None)
        served = os.path.realpath(tools.resolve_sandbox_workdir("sess-agree"))
        written = os.path.realpath(tools._get_workdir("sess-agree"))
        tools._workdirs.pop("sess-agree", None)

        assert served == written
        assert served == os.path.realpath(str(chosen))

    def test_a_session_less_read_does_not_use_the_global_root(self, monkeypatch, tmp_path):
        """The read side carries _get_workdir's session_id gate too: an
        anonymous caller is answered from the sandbox, not the user's folder."""
        from core.inference import tools
        chosen = tmp_path / "chosen"; chosen.mkdir()
        monkeypatch.setattr(tools, "_project_workdir_for", lambda sid: None)
        import utils.workspace_root as mod
        monkeypatch.setattr(mod, "_read_setting", lambda key, fallback = None: str(chosen))
        result = os.path.realpath(tools.resolve_sandbox_workdir(None))
        assert result != os.path.realpath(str(chosen))
        assert tools._contained_in_root(result, tools.sandbox_root())


class TestSetGlobalRootInvalidatesTheCache:
    """New-1: _get_workdir and resolve_sandbox_workdir consult ``_workdirs``
    in opposite orders relative to the global root -- the former checks the
    cache first, the latter checks the global root first. Setting the root
    mid-session used to leave a session's cached sandbox in place, so tools
    kept writing there while every read-only route already served the newly
    chosen folder. The fix is set_global_root invalidating the cache on a
    successful write.
    """

    def test_the_two_resolvers_agree_after_the_global_root_is_set_mid_session(
        self, monkeypatch, tmp_path
    ):
        from core.inference import tools
        import utils.workspace_root as mod

        chosen = tmp_path / "chosen"; chosen.mkdir()
        monkeypatch.setattr(tools, "_project_workdir_for", lambda sid: None)

        # Step 1: no global root set yet. _get_workdir claims and caches the
        # sandbox -- exactly a chat that ran a tool before Settings was
        # touched.
        monkeypatch.setattr(mod, "_read_setting", lambda key, fallback = None: None)
        tools._workdirs.pop("sess-mid", None)
        sandboxed = os.path.realpath(tools._get_workdir("sess-mid"))
        assert tools._contained_in_root(sandboxed, tools.sandbox_root())

        # Step 2: the user sets the global root through the real setter --
        # not by patching _read_setting alone, since the invalidation lives
        # in the setter itself.
        monkeypatch.setattr(mod, "_write_setting", lambda mapping: None)
        mod.set_global_root(str(chosen))
        # Reads now answer with the new value -- standing in for the setting
        # the mocked _write_setting above did not actually persist.
        monkeypatch.setattr(mod, "_read_setting", lambda key, fallback = None: str(chosen))

        # Step 3: both resolvers must agree, and both must point at the
        # chosen folder rather than the stale cached sandbox.
        served = os.path.realpath(tools.resolve_sandbox_workdir("sess-mid"))
        written = os.path.realpath(tools._get_workdir("sess-mid"))
        tools._workdirs.pop("sess-mid", None)

        assert served == written
        assert served == os.path.realpath(str(chosen))
        assert served != sandboxed


class TestProjectRoot:
    def test_a_supplied_root_is_honoured_at_creation(self, tmp_path, monkeypatch):
        from storage import studio_db
        chosen = tmp_path / "mycode"; chosen.mkdir()
        captured = {}
        monkeypatch.setattr(studio_db, "get_chat_project", lambda pid: captured.get(pid))
        monkeypatch.setattr(studio_db, "_ensure_project_workspace", lambda p: os.path.realpath(p))
        resolved = studio_db._resolve_project_root(
            {"id": "p1", "name": "P", "rootPath": str(chosen)}, existing = None
        )
        assert os.path.realpath(resolved) == os.path.realpath(str(chosen))

    def test_a_blank_supplied_root_falls_through_to_the_default(self, tmp_path, monkeypatch):
        """A whitespace-only rootPath is not a real choice -- it must not win
        over the generated default, the same way an unset one must not."""
        from storage import studio_db
        monkeypatch.setattr(studio_db, "_ensure_project_workspace", lambda p: os.path.realpath(p))
        monkeypatch.setattr(studio_db, "_default_project_root", lambda proj: str(tmp_path / "DEFAULT"))
        resolved = studio_db._resolve_project_root(
            {"id": "p1", "name": "P", "rootPath": "   "}, existing = None
        )
        assert os.path.realpath(resolved) == os.path.realpath(str(tmp_path / "DEFAULT"))

    def test_an_existing_root_is_kept_when_none_is_supplied(self, tmp_path, monkeypatch):
        from storage import studio_db
        monkeypatch.setattr(studio_db, "_ensure_project_workspace", lambda p: os.path.realpath(p))
        resolved = studio_db._resolve_project_root(
            {"id": "p1", "name": "P"}, existing = {"rootPath": str(tmp_path / "old")}
        )
        assert os.path.realpath(resolved) == os.path.realpath(str(tmp_path / "old"))

    def test_a_supplied_root_does_not_override_an_existing_one(self, tmp_path, monkeypatch):
        """The anti-relocation guarantee. An established project's files are
        already in its folder, so upsert must never move it -- re-pointing is
        PATCH's job. Without this test, swapping the branch order in
        _resolve_project_root would pass every other test in the suite.
        """
        from storage import studio_db
        old = tmp_path / "old"; old.mkdir()
        new = tmp_path / "new"; new.mkdir()
        monkeypatch.setattr(studio_db, "_ensure_project_workspace", lambda p: os.path.realpath(p))
        resolved = studio_db._resolve_project_root(
            {"id": "p1", "name": "P", "rootPath": str(new)},
            existing = {"rootPath": str(old)},
        )
        assert os.path.realpath(resolved) == os.path.realpath(str(old))
        assert os.path.realpath(resolved) != os.path.realpath(str(new))

    def test_rootpath_is_patchable(self, tmp_path):
        """Behavioural, not structural: writes a row through the real (per-test
        isolated) sqlite database, patches rootPath through update_chat_project,
        and reads it back -- so a rootPath entry wired to the wrong column, or
        never persisted, fails this even though the string "rootPath" would
        still appear in the source.
        """
        from storage import studio_db
        conn = studio_db.get_connection()
        try:
            conn.execute(
                """
                INSERT INTO chat_projects
                    (id, name, instructions, root_path, archived, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ("p1", "P", "", str(tmp_path / "old"), 0, 1_700_000_000_000, 1_700_000_000_000),
            )
            conn.commit()
        finally:
            conn.close()

        new_root = str(tmp_path / "new")
        updated = studio_db.update_chat_project("p1", {"rootPath": new_root})

        assert updated is not None
        assert updated["rootPath"] == new_root
        assert studio_db.get_chat_project("p1")["rootPath"] == new_root

    # --- negative control -------------------------------------------------
    def test_control_a_supplied_root_beats_the_default(self, tmp_path, monkeypatch):
        """The whole defect being fixed: the old code computed
        _default_project_root and ignored the caller. If the default wins here,
        nothing has changed."""
        from storage import studio_db
        chosen = tmp_path / "mycode"; chosen.mkdir()
        monkeypatch.setattr(studio_db, "_ensure_project_workspace", lambda p: os.path.realpath(p))
        monkeypatch.setattr(studio_db, "_default_project_root", lambda proj: str(tmp_path / "DEFAULT"))
        resolved = studio_db._resolve_project_root(
            {"id": "p1", "name": "P", "rootPath": str(chosen)}, existing = None
        )
        assert "DEFAULT" not in resolved


def _insert_project(project_id: str, name: str, root_path: str) -> None:
    """A project row written straight to the (per-test, isolated) database.

    Straight to SQL rather than through upsert_chat_project so the test states
    the stored root itself, including the auto-generated shape, and nothing
    resolves it on the way in.
    """
    from storage import studio_db
    conn = studio_db.get_connection()
    try:
        conn.execute(
            """
            INSERT INTO chat_projects
                (id, name, instructions, root_path, archived, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (project_id, name, "", root_path, 0, 1_700_000_000_000, 1_700_000_000_000),
        )
        conn.commit()
    finally:
        conn.close()


class TestWhereAProjectRootPutsTheAi:
    """Which directory a project's chats actually work in.

    The spec asserted a project chat "already resolves its tool working
    directory to that project's rootPath". It did not: it resolved to a
    freshly created, empty `<rootPath>/sandbox`, so a project pointed at a
    repository saw none of it. These exercise the real database and the real
    resolver -- every earlier resolution test patched `_project_workdir_for`
    out, which is exactly why nothing caught it.
    """

    def test_a_user_chosen_root_gives_the_ai_the_root_itself(self, tmp_path, monkeypatch):
        from core.inference import tools
        monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects_home"))
        chosen = tmp_path / "myrepo"; chosen.mkdir()
        (chosen / "README.md").write_text("the user's files", encoding = "utf-8")
        _insert_project("p-chosen", "My Repo", str(chosen))

        workdir = tools._get_project_workdir(tools.project_session_id("p-chosen"))

        assert workdir == os.path.realpath(str(chosen))
        assert os.path.isfile(os.path.join(workdir, "README.md")), (
            "the point of choosing a folder is that the AI sees what is in it"
        )
        assert not os.path.isdir(chosen / "sandbox"), (
            "and that Studio does not leave an unused folder of its own in there"
        )

    def test_a_generated_root_still_uses_its_sandbox_subfolder(self, tmp_path, monkeypatch):
        """The regression test that protects every project that already exists.

        Their files are in `<root>/sandbox`; moving them to the root would
        strand every one of them. If the carve-out is dropped -- if the
        user-chosen rule is applied to all roots -- this must fail.
        """
        from core.inference import tools
        from storage import studio_db
        projects_home = tmp_path / "projects_home"
        monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(projects_home))
        generated = studio_db._default_project_root({"id": "p-generated", "name": "Probe"})
        _insert_project("p-generated", "Probe", generated)

        workdir = tools._get_project_workdir(tools.project_session_id("p-generated"))

        assert workdir == os.path.realpath(os.path.join(generated, "sandbox"))
        assert os.path.isdir(workdir)

    def test_the_projects_folder_itself_is_not_read_as_a_user_choice(self, tmp_path, monkeypatch):
        """The boundary case of the containment check. A sibling named like the
        projects folder is outside it; the folder itself is not a choice."""
        from storage import studio_db
        projects_home = tmp_path / "projects_home"
        monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(projects_home))
        assert studio_db._is_user_chosen_root(str(projects_home)) is False
        assert studio_db._is_user_chosen_root(str(projects_home / "probe-1234")) is False
        assert studio_db._is_user_chosen_root(str(tmp_path / "projects_home-old")) is True

    def test_workspace_for_confines_to_the_chosen_directory(self, tmp_path, monkeypatch):
        """Spec Testing bullet 5, and the test that would have surfaced all of
        this: the language-server workspace root for a file in the user's
        repository must be the repository, not a sandbox subfolder of it that
        the file is not even inside."""
        from core.inference import tools
        from core.inference.assist_code import paths as assist_paths
        monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects_home"))
        chosen = tmp_path / "myrepo"; chosen.mkdir()
        (chosen / "src").mkdir()
        source = chosen / "src" / "main.py"
        source.write_text("print('hi')\n", encoding = "utf-8")
        _insert_project("p-ws", "My Repo", str(chosen))

        session = tools.project_session_id("p-ws")
        tools._workdirs.pop(session, None)
        try:
            resolved = assist_paths.workspace_for(str(source), session_id = session)
            confined, err = assist_paths.resolve_file(str(source), session_id = session)
        finally:
            tools._workdirs.pop(session, None)

        assert os.path.realpath(resolved) == os.path.realpath(str(chosen))
        assert err is None and confined is not None, (
            "a file in the chosen folder must be readable from it"
        )


class TestAClearedProjectRoot:
    """"Clear (use the global folder)" has to mean what it says.

    It sent the project to a generated Documents folder instead, while the
    button and use-chat-projects.ts both promised the global folder.
    """

    def test_a_cleared_root_falls_through_to_the_global_folder(self, tmp_path, monkeypatch):
        from core.inference import tools
        from storage import studio_db
        monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "projects_home"))
        chosen = tmp_path / "global"; chosen.mkdir()
        import utils.workspace_root as mod
        monkeypatch.setattr(mod, "_read_setting", lambda key, fallback = None: str(chosen))
        _insert_project("p-cleared", "Cleared", None)

        project = studio_db.ensure_chat_project_workspace("p-cleared")
        assert project is not None
        assert project["rootPath"] is None, (
            "a generated root written back here would pin the project to it, and "
            "a later change to the global folder would stop reaching this project"
        )

        session = tools.project_session_id("p-cleared")
        tools._workdirs.pop(session, None)
        try:
            assert os.path.realpath(tools._get_workdir(session)) == os.path.realpath(str(chosen))
        finally:
            tools._workdirs.pop(session, None)

    def test_control_with_no_global_folder_a_root_is_still_generated(self, tmp_path, monkeypatch):
        """The other half. With nothing chosen globally there is nothing to fall
        through to, so the generated Documents folder is still correct -- and a
        project row that predates the root_path column still gets one."""
        from storage import studio_db
        projects_home = tmp_path / "projects_home"
        monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(projects_home))
        import utils.workspace_root as mod
        monkeypatch.setattr(mod, "_read_setting", lambda key, fallback = None: None)
        _insert_project("p-nogl", "No Global", None)

        project = studio_db.ensure_chat_project_workspace("p-nogl")

        assert project is not None and project["rootPath"] is not None
        assert studio_db._is_user_chosen_root(project["rootPath"]) is False


class TestAStorageHiccupDoesNotMoveALiveChat:
    def test_a_read_failure_keeps_the_last_known_root(self, tmp_path, monkeypatch):
        """"Could not read the setting" is not "the user has not chosen one".
        Read as unset, a single SQLITE_BUSY moves the AI out of the user's
        folder and into an empty sandbox mid-conversation."""
        import utils.workspace_root as mod
        chosen = tmp_path / "chosen"; chosen.mkdir()
        monkeypatch.setattr(mod, "_read_setting", lambda key, fallback = None: str(chosen))
        assert get_global_root() == os.path.realpath(str(chosen))

        def boom(key, fallback = None):
            raise RuntimeError("database is locked")

        monkeypatch.setattr(mod, "_read_setting", boom)
        assert get_global_root() == os.path.realpath(str(chosen))

    def test_a_root_the_user_actually_cleared_still_falls_through(self, tmp_path, monkeypatch):
        """The cache must not outlive the value. Only a failed READ keeps it;
        a successful read of "nothing set" clears it on the spot."""
        import utils.workspace_root as mod
        chosen = tmp_path / "chosen"; chosen.mkdir()
        monkeypatch.setattr(mod, "_read_setting", lambda key, fallback = None: str(chosen))
        assert get_global_root() == os.path.realpath(str(chosen))
        monkeypatch.setattr(mod, "_read_setting", lambda key, fallback = None: None)
        assert get_global_root() is None

    def test_a_read_failure_does_not_return_a_since_deleted_root(self, tmp_path, monkeypatch):
        """New-2: the cached last-known root can itself have been deleted by
        the user since it was last confirmed. A read failure must not hand it
        back unchecked -- _get_workdir would makedirs() it right back into
        existence, which is the one outcome this function's own docstring
        says must not happen."""
        import utils.workspace_root as mod
        chosen = tmp_path / "chosen"; chosen.mkdir()
        monkeypatch.setattr(mod, "_read_setting", lambda key, fallback = None: str(chosen))
        assert get_global_root() == os.path.realpath(str(chosen))

        chosen.rmdir()

        def boom(key, fallback = None):
            raise RuntimeError("database is locked")

        monkeypatch.setattr(mod, "_read_setting", boom)
        assert get_global_root() is None
