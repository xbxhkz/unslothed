# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The vision tools must be REACHABLE through Studio's own loop.

Being defined is not the same as being wired up. These tests go through the
real ``ALL_TOOLS`` list and the real ``execute_tool`` dispatcher, so a tool
that exists but was never registered fails here.
"""

import io

import pytest
from PIL import Image

from core.inference.tools import ALL_TOOLS, execute_tool

_NAMES = {
    "remove_background", "detect_shapes", "webcam_look",
    "edit_image_prompt", "face_swap",
}


def _png(path, size=(16, 16)):
    Image.new("RGB", size, (10, 20, 30)).save(path, format="PNG")
    return str(path)


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Make ``tmp_path`` the confined sandbox for session ``"t"``.

    Mirrors ``test_edit_file_tool.py``'s own ``workdir`` fixture: vision
    tools resolve every ``image_path`` through the same
    ``core.inference.tools._get_workdir``/``_is_outside_workdir`` pair
    ``edit_file`` uses, so a test image written under pytest's own
    ``tmp_path`` is otherwise flagged as outside the session's real sandbox
    (which lives under the per-test ``UNSLOTH_STUDIO_HOME``, not
    ``tmp_path``) before it ever reaches the "does this file exist" check.
    """
    from core.inference import tools

    monkeypatch.setattr(tools, "_get_workdir", lambda session_id=None: str(tmp_path))
    return tmp_path


class TestRegistration:
    def test_every_vision_tool_appears_in_all_tools(self):
        names = {t["function"]["name"] for t in ALL_TOOLS}
        assert _NAMES <= names, f"missing: {_NAMES - names}"

    def test_every_schema_is_openai_shaped(self):
        for tool in ALL_TOOLS:
            if tool["function"]["name"] in _NAMES:
                assert tool["type"] == "function"
                assert tool["function"]["description"].strip()
                assert tool["function"]["parameters"]["type"] == "object"

    def test_the_upstream_tool_names_are_not_disturbed(self):
        """The two-line edit must ADD tools, never replace Studio's own."""
        names = {t["function"]["name"] for t in ALL_TOOLS}
        assert {"web_search", "python", "terminal", "edit_file"} <= names


class TestDispatch:
    def test_dispatch_reaches_the_tool_and_returns_a_string(self, tmp_path, monkeypatch):
        import core.inference.assist_vision as av

        monkeypatch.setattr(
            av, "execute",
            lambda name, arguments, **kw: f"dispatched:{name}:{arguments.get('image_path')}",
        )
        out = execute_tool("remove_background", {"image_path": _png(tmp_path / "a.png")},
                           session_id="t")
        assert isinstance(out, str)
        assert out.startswith("dispatched:remove_background:")

    def test_a_missing_image_path_returns_error_text_not_an_exception(self, workdir):
        out = execute_tool("remove_background",
                           {"image_path": str(workdir / "nope.png")}, session_id="t")
        assert isinstance(out, str)
        assert "not found" in out.lower()

    def test_detect_shapes_finding_nothing_is_not_reported_as_an_error(self, workdir, monkeypatch):
        from core.inference.assist_vision import shape_detect

        monkeypatch.setattr(shape_detect, "detect", lambda *a, **k: [])
        out = execute_tool("detect_shapes", {"image_path": _png(workdir / "b.png")},
                           session_id="t")
        assert "error" not in out.lower()
        assert "no recognizable" in out.lower()

    def test_no_returned_string_contains_a_base64_data_uri(self, workdir, monkeypatch):
        """Data URIs get replayed into model context forever."""
        from core.inference.assist_vision import shape_detect

        monkeypatch.setattr(shape_detect, "detect", lambda *a, **k: [])
        out = execute_tool("detect_shapes", {"image_path": _png(workdir / "c.png")},
                           session_id="t")
        assert "data:image" not in out
        assert "base64" not in out
