# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import os
import subprocess
import pytest
from core.inference.assist_code import paths


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    from core.inference import tools
    monkeypatch.setattr(tools, "_get_workdir", lambda _sid = None: str(tmp_path))
    return tmp_path


class TestResolveFile:
    def test_a_bare_filename_resolves_against_the_workdir(self, workdir):
        (workdir / "a.ts").write_text("export const x = 1\n")
        got, err = paths.resolve_file("a.ts", session_id = "s")
        assert err is None
        assert got == os.path.abspath(str(workdir / "a.ts"))

    def test_a_path_outside_the_workdir_is_refused(self, workdir, tmp_path_factory):
        outside = tmp_path_factory.mktemp("elsewhere") / "secret.ts"
        outside.write_text("x")
        got, err = paths.resolve_file(str(outside), session_id = "s")
        assert got is None
        assert "outside" in err.lower()

    def test_confinement_applies_with_no_session_id(self, workdir, tmp_path_factory):
        """No session_id must NOT be an escape hatch."""
        outside = tmp_path_factory.mktemp("elsewhere2") / "secret.ts"
        outside.write_text("x")
        got, err = paths.resolve_file(str(outside), session_id = None)
        assert got is None
        assert "outside" in err.lower()

    def test_a_missing_file_says_so_not_something_vaguer(self, workdir):
        got, err = paths.resolve_file("nope.ts", session_id = "s")
        assert got is None
        assert "not found" in err.lower()

    def test_an_empty_path_is_rejected(self, workdir):
        got, err = paths.resolve_file("   ", session_id = "s")
        assert got is None
        assert "required" in err.lower()


class TestResolveWorkspace:
    def test_an_existing_directory_resolves(self, workdir):
        (workdir / "proj").mkdir()
        got, err = paths.resolve_workspace("proj", session_id = "s")
        assert err is None
        assert got == os.path.abspath(str(workdir / "proj"))

    def test_a_file_is_not_a_workspace(self, workdir):
        (workdir / "a.ts").write_text("x")
        got, err = paths.resolve_workspace("a.ts", session_id = "s")
        assert got is None
        assert "directory" in err.lower()

    def test_a_workspace_outside_the_workdir_is_refused(self, workdir, tmp_path_factory):
        outside = tmp_path_factory.mktemp("elsewhere3")
        got, err = paths.resolve_workspace(str(outside), session_id = "s")
        assert got is None
        assert "outside" in err.lower()


class TestWorkspaceFor:
    @pytest.mark.parametrize("marker", ["package.json", "tsconfig.json", ".git"])
    def test_the_nearest_project_marker_wins(self, workdir, marker):
        proj = workdir / "proj"
        (proj / "src").mkdir(parents = True)
        if marker == ".git":
            (proj / marker).mkdir()
        else:
            (proj / marker).write_text("{}")
        f = proj / "src" / "a.ts"
        f.write_text("x")
        assert paths.workspace_for(str(f), session_id = "s") == os.path.abspath(str(proj))

    def test_no_marker_falls_back_to_the_workdir(self, workdir):
        (workdir / "loose").mkdir()
        f = workdir / "loose" / "a.ts"
        f.write_text("x")
        assert paths.workspace_for(str(f), session_id = "s") == os.path.abspath(str(workdir))

    def test_the_search_never_climbs_past_the_workdir(self, workdir, tmp_path_factory):
        """A marker ABOVE the sandbox must not be selected as the root."""
        parent_marker = workdir.parent / "package.json"
        parent_marker.write_text("{}")
        (workdir / "x").mkdir()
        f = workdir / "x" / "a.ts"
        f.write_text("x")
        assert paths.workspace_for(str(f), session_id = "s") == os.path.abspath(str(workdir))

    def test_string_prefix_collision_does_not_escape_confinement(self, workdir, tmp_path_factory):
        """A sibling directory whose name string-prefixes the workdir must not escape.

        This catches the bug where `str.startswith` was used without a separator boundary.
        E.g., workdir "...\\session1" would incorrectly match sibling "...\\session1-evil"
        via `"...\\session1-evil".startswith("...\\session1")`.
        """
        # Create a marker inside a sibling that string-prefixes the workdir
        sibling = workdir.parent / (os.path.basename(str(workdir)) + "-evil")
        (sibling / "src").mkdir(parents=True)
        (sibling / "package.json").write_text("{}")

        # File inside the evil sibling
        f = sibling / "src" / "a.ts"
        f.write_text("x")

        # Must return the workdir, not the sibling
        result = paths.workspace_for(str(f), session_id="s")
        assert result == os.path.abspath(str(workdir)), (
            f"Expected workdir {os.path.abspath(str(workdir))}, "
            f"but got {result} (sibling directory escaped confinement)"
        )

    def test_symlink_junction_escape_is_detected(self, workdir, tmp_path_factory):
        """A symlink/junction inside the sandbox pointing outside must not escape.

        This catches the bug where only abspath was used, not realpath. Windows junctions
        inside the sandbox pointing to external directories would be traversed transparently.
        """
        # Create an external directory with a marker
        external = tmp_path_factory.mktemp("external")
        (external / ".git").mkdir()

        # Create a symlink/junction inside the workdir pointing outside
        link = workdir / "sneaky_link"
        link_str = str(link)
        target_str = str(external)

        try:
            # Try creating a directory symlink (requires Windows 10+ or Linux)
            os.symlink(target_str, link_str, target_is_directory=True)
        except (OSError, NotImplementedError):
            # Fall back to mklink for older Windows versions
            try:
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", link_str, target_str],
                    check=True,
                    capture_output=True,
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                pytest.skip("Could not create directory symlink or junction on this system")

        # File inside the external directory via the junction
        f = link / "a.ts"
        f.write_text("x")

        # workspace_for should return the workdir, not climb into the external directory
        result = paths.workspace_for(str(f), session_id="s")
        result_realpath = os.path.realpath(result)
        workdir_realpath = os.path.realpath(str(workdir))

        assert result_realpath == workdir_realpath, (
            f"Expected realpath {workdir_realpath}, "
            f"but got {result_realpath} (symlink/junction escaped confinement)"
        )
