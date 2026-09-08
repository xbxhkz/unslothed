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
