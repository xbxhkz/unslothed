# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""continuity_task never takes a project_dir argument from the model -- it is
always the calling session's own sandbox workdir, resolved the same way
resolve_image_bytes confines an image path (core/inference/assist_vision/paths.py).
"""

from __future__ import annotations

from core.continuity.sandbox import confine_project_dir


def test_confine_project_dir_returns_a_real_directory(tmp_path, monkeypatch):
    import core.inference.tools as tools_module
    monkeypatch.setattr(tools_module, "_get_workdir", lambda session_id: str(tmp_path))
    result = confine_project_dir("some-session")
    assert result == str(tmp_path)


def test_confine_project_dir_works_with_no_session_id(tmp_path, monkeypatch):
    """_get_workdir is None-safe by Studio's own design (anonymous sandbox) --
    there must be no session_id-omitted escape hatch."""
    import core.inference.tools as tools_module
    monkeypatch.setattr(tools_module, "_get_workdir", lambda session_id: str(tmp_path))
    assert confine_project_dir(None) == str(tmp_path)


def test_two_thread_ids_resolve_to_different_paths_under_a_shared_workdir(tmp_path, monkeypatch):
    """The real shared-workspace case: _get_workdir returns ONE fixed root
    regardless of session_id (a project session, or the global-workspace
    setting). thread_id namespacing is the only thing that keeps two chats
    sharing that root from colliding on the same files."""
    import core.inference.tools as tools_module
    monkeypatch.setattr(tools_module, "_get_workdir", lambda session_id: str(tmp_path))
    a = confine_project_dir("shared-session", "thread-a")
    b = confine_project_dir("shared-session", "thread-b")
    assert a != b
    assert a.startswith(str(tmp_path))
    assert b.startswith(str(tmp_path))


def test_no_thread_id_falls_back_to_the_bare_session_workdir(tmp_path, monkeypatch):
    """Backward compatible for any caller with no thread context, e.g. a
    future non-conversational caller of confine_project_dir."""
    import core.inference.tools as tools_module
    monkeypatch.setattr(tools_module, "_get_workdir", lambda session_id: str(tmp_path))
    assert confine_project_dir("some-session", None) == str(tmp_path)


def test_a_path_hostile_thread_id_is_sanitized_to_one_safe_component(tmp_path, monkeypatch):
    import core.inference.tools as tools_module
    import os
    monkeypatch.setattr(tools_module, "_get_workdir", lambda session_id: str(tmp_path))
    result = confine_project_dir("s", "../../etc/a/b")
    resolved = os.path.abspath(result)
    workdir = os.path.abspath(str(tmp_path))
    assert resolved.startswith(workdir + os.sep)
    # exactly one path segment was added under continuity-threads/, not several
    relative = os.path.relpath(resolved, os.path.join(workdir, "continuity-threads"))
    assert os.sep not in relative
