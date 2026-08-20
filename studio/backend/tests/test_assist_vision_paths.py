# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Input resolution for the vision tools.

Every vision tool takes an ``image_path`` and must turn it into bytes without
becoming a file-exfiltration primitive. These tests execute the real resolver
against a real temp filesystem -- no mocks -- because the failure that matters
(reading a file outside the allowed area) is a filesystem behaviour, not a
code-shape property.
"""

import io

import pytest
from PIL import Image

from core.inference.assist_vision.paths import resolve_image_bytes
from core.inference import tools


def _write_png(path, size=(8, 8), color=(200, 50, 50)):
    Image.new("RGB", size, color).save(path, format="PNG")
    return path


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Point the session workdir at a tmp subdir, as the executor would.

    A subdir of tmp_path rather than tmp_path itself: tests that request both
    ``tmp_path`` and ``workdir`` need genuinely distinct directories, since
    pytest caches ``tmp_path`` per test and a bare ``return tmp_path`` here
    would make "outside the workdir" paths land inside it.
    """
    d = tmp_path / "workdir"
    d.mkdir()
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(d))
    return d


class TestResolveImageBytes:
    def test_a_real_file_resolves_to_its_bytes(self, tmp_path):
        p = _write_png(tmp_path / "a.png")
        data, err = resolve_image_bytes(str(p))
        assert err is None
        assert Image.open(io.BytesIO(data)).size == (8, 8)

    def test_a_missing_file_returns_error_text_not_an_exception(self, tmp_path):
        data, err = resolve_image_bytes(str(tmp_path / "nope.png"))
        assert data is None
        assert "not found" in err.lower()

    def test_a_directory_is_rejected(self, tmp_path):
        data, err = resolve_image_bytes(str(tmp_path))
        assert data is None
        assert "not a file" in err.lower()

    def test_an_oversized_file_is_rejected_without_reading_it_all(self, tmp_path):
        big = tmp_path / "big.png"
        with open(big, "wb") as f:
            f.seek(1024)
            f.write(b"x")
        data, err = resolve_image_bytes(str(big), max_bytes=512)
        assert data is None
        assert "too large" in err.lower()

    def test_a_non_image_file_is_rejected_with_readable_text(self, tmp_path):
        junk = tmp_path / "notes.txt"
        junk.write_text("this is not an image", encoding="utf-8")
        data, err = resolve_image_bytes(str(junk))
        assert data is None
        assert "image" in err.lower()

    def test_an_empty_path_is_rejected(self):
        data, err = resolve_image_bytes("")
        assert data is None
        assert err


class TestConfinement:
    def test_a_path_outside_the_session_workdir_is_refused(self, tmp_path, workdir):
        outside = tmp_path / "outside.png"
        _write_png(outside)
        data, err = resolve_image_bytes(str(outside), session_id="t")
        assert data is None
        assert err and ("outside" in err.lower() or "not allowed" in err.lower())

    def test_a_path_inside_the_session_workdir_is_allowed(self, workdir):
        inside = _write_png(workdir / "inside.png")
        data, err = resolve_image_bytes(str(inside), session_id="t")
        assert err is None
        assert data
