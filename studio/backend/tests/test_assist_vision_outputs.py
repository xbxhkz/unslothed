# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where a vision tool's output lands, and whether anything can read it back.

The tools used to write results with ``tempfile.mkstemp``, i.e. into the system
temp directory -- which is OUTSIDE the session workdir. Every returned path was
therefore refused by ``resolve_image_bytes`` ("is outside this conversation's
working directory"), which made chaining structurally impossible:
``remove_background`` -> ``detect_shapes`` could never work, ``edit_file`` /
``python`` / ``terminal`` could not touch a result, and the user could not find
one. Nothing deleted them either, so every call leaked a multi-MB PNG.

The assertion that matters is the ROUND TRIP: take the path out of the string a
real tool returned and feed it straight back into ``resolve_image_bytes``. A
test that only checks the file exists on disk passes against the broken
behaviour and misses the entire point.
"""

import io
import os

import pytest
from PIL import Image

from core.inference.assist_vision.paths import resolve_image_bytes
from core.inference.tools import execute_tool

_SESSION = "vision-out-session"


def _png(path, size = (16, 16), color = (10, 20, 30)):
    Image.new("RGB", size, color).save(path, format = "PNG")
    return str(path)


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Make ``tmp_path`` the confined sandbox, as the real executor would."""
    from core.inference import tools

    d = tmp_path / "workdir"
    d.mkdir()
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(d))
    return d


def _path_from(result: str) -> str:
    """Pull the written path out of a tool's returned sentence.

    Deliberately parses the REAL returned string rather than reaching for an
    internal handle: if the model cannot recover a usable path from what the
    tool said, the tool is broken regardless of what it wrote to disk.
    """
    marker = "written to: "
    assert marker in result, f"no written path in tool output: {result!r}"
    tail = result.split(marker, 1)[1]
    # The face_swap line appends a parenthetical note after the path.
    return tail.split(" (")[0].strip()


class TestOutputsAreReadableBack:
    def test_remove_background_output_round_trips_through_resolve_image_bytes(
        self, workdir, monkeypatch
    ):
        """The whole point: a path a tool returned must resolve again."""
        from core.inference.assist_vision import bg_removal

        monkeypatch.setattr(
            bg_removal, "remove_background",
            lambda data, **kw: _rgba_png(),
        )
        out = execute_tool(
            "remove_background",
            {"image_path": _png(workdir / "in.png")},
            session_id = _SESSION,
        )
        path = _path_from(out)

        data, err = resolve_image_bytes(path, session_id = _SESSION)
        assert err is None, f"tool wrote somewhere nothing can read it back: {err}"
        assert Image.open(io.BytesIO(data)).format == "PNG"

    def test_a_bare_filename_from_the_output_also_resolves(self, workdir, monkeypatch):
        """A model naturally refers to a file it just made by bare filename;
        that must resolve against the workdir too."""
        from core.inference.assist_vision import bg_removal

        monkeypatch.setattr(bg_removal, "remove_background", lambda data, **kw: _rgba_png())
        out = execute_tool(
            "remove_background",
            {"image_path": _png(workdir / "in.png")},
            session_id = _SESSION,
        )
        data, err = resolve_image_bytes(
            os.path.basename(_path_from(out)), session_id = _SESSION
        )
        assert err is None
        assert data

    def test_the_output_of_one_tool_is_valid_input_to_the_next(self, workdir, monkeypatch):
        """remove_background -> detect_shapes chaining, which the temp-dir
        write made structurally impossible."""
        from core.inference.assist_vision import bg_removal, shape_detect

        monkeypatch.setattr(bg_removal, "remove_background", lambda data, **kw: _rgba_png())
        monkeypatch.setattr(shape_detect, "detect", lambda *a, **k: [])

        first = execute_tool(
            "remove_background",
            {"image_path": _png(workdir / "in.png")},
            session_id = _SESSION,
        )
        second = execute_tool(
            "detect_shapes",
            {"image_path": _path_from(first)},
            session_id = _SESSION,
        )
        assert "failed" not in second.lower(), second
        assert "outside" not in second.lower(), second
        assert "no recognizable" in second.lower()

    def test_the_written_file_is_inside_the_session_workdir(self, workdir, monkeypatch):
        from core.inference.assist_vision import bg_removal

        monkeypatch.setattr(bg_removal, "remove_background", lambda data, **kw: _rgba_png())
        out = execute_tool(
            "remove_background",
            {"image_path": _png(workdir / "in.png")},
            session_id = _SESSION,
        )
        path = os.path.abspath(_path_from(out))
        assert os.path.commonpath([path, str(workdir)]) == str(workdir), (
            f"{path} is not under the session workdir {workdir}"
        )

    def test_two_calls_do_not_collide(self, workdir, monkeypatch):
        """Colliding names would have the second call silently overwrite the
        first result the model was just told about."""
        from core.inference.assist_vision import bg_removal

        monkeypatch.setattr(bg_removal, "remove_background", lambda data, **kw: _rgba_png())
        src = _png(workdir / "in.png")
        a = _path_from(execute_tool("remove_background", {"image_path": src}, session_id = _SESSION))
        b = _path_from(execute_tool("remove_background", {"image_path": src}, session_id = _SESSION))
        assert a != b
        assert os.path.isfile(a) and os.path.isfile(b)

    def test_the_returned_path_is_a_path_not_a_data_uri(self, workdir, monkeypatch):
        from core.inference.assist_vision import bg_removal

        monkeypatch.setattr(bg_removal, "remove_background", lambda data, **kw: _rgba_png())
        out = execute_tool(
            "remove_background",
            {"image_path": _png(workdir / "in.png")},
            session_id = _SESSION,
        )
        assert "data:image" not in out
        assert "base64" not in out


def _rgba_png(size = (16, 16)):
    buf = io.BytesIO()
    Image.new("RGBA", size, (10, 20, 30, 128)).save(buf, format = "PNG")
    return buf.getvalue()
