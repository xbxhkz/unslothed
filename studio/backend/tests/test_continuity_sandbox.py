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
