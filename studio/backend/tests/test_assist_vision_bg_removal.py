# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Background removal via the bundled U2Net ONNX model.

The model is injectable (``session=``) so these tests exercise the real
pre/post-processing -- resize, mask application, PNG encoding -- against a
stub that returns a known mask, without needing the 168 MB weight file.
The assertions are on real output pixels: a test that only checked "it
returned bytes" would pass against an implementation that ignored the mask
entirely.
"""

import io

import numpy as np
import pytest
from PIL import Image

from core.inference.assist_vision.bg_removal import remove_background


class _Input:
    name = "input"


class _HalfMaskSession:
    """Returns a mask that keeps the left half and drops the right half."""

    def get_inputs(self):
        return [_Input()]

    def run(self, output_names, input_feed):
        arr = next(iter(input_feed.values()))
        n, c, h, w = arr.shape
        mask = np.zeros((n, 1, h, w), dtype=np.float32)
        mask[:, :, :, : w // 2] = 1.0
        return [mask]


def _png(size=(64, 64), color=(200, 50, 50)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class TestRemoveBackground:
    def test_output_is_a_transparent_png_of_the_same_size(self):
        out = remove_background(_png(), session=_HalfMaskSession())
        img = Image.open(io.BytesIO(out))
        assert img.format == "PNG"
        assert img.mode == "RGBA"
        assert img.size == (64, 64)

    def test_the_mask_actually_drives_transparency(self):
        """Fails if the mask is computed but never applied."""
        out = remove_background(_png(), session=_HalfMaskSession())
        img = Image.open(io.BytesIO(out)).convert("RGBA")
        left_alpha = img.getpixel((4, 32))[3]
        right_alpha = img.getpixel((60, 32))[3]
        assert left_alpha > 200, "kept region should be opaque"
        assert right_alpha < 55, "dropped region should be transparent"

    def test_corrupt_bytes_raise_rather_than_returning_a_bad_image(self):
        """pytest.raises(Exception) passed on literally any error, including an
        AttributeError from a gutted implementation. PIL raises
        UnidentifiedImageError for undecodable bytes, so pin that."""
        from PIL import UnidentifiedImageError

        with pytest.raises(UnidentifiedImageError):
            remove_background(b"not an image", session=_HalfMaskSession())


class TestModelResolution:
    """No weights ship in git, so first use downloads -- openly, and never
    behind a user's back when they have said where the file is."""

    def test_a_broken_explicit_override_is_an_error_not_a_surprise_download(
        self, tmp_path, monkeypatch
    ):
        """UNSLOTH_U2NET_PATH pointing at nothing is a user mistake. Downloading
        176 MB somewhere they did not ask for would be the wrong repair."""
        from core.inference.assist_vision import bg_removal, models

        monkeypatch.setenv("UNSLOTH_U2NET_PATH", str(tmp_path / "nope.onnx"))
        monkeypatch.setattr(bg_removal, "_session", None)
        monkeypatch.setattr(
            models, "download_model",
            lambda *a, **k: pytest.fail("must not download over an explicit override"),
        )
        with pytest.raises(RuntimeError) as excinfo:
            bg_removal._get_session()
        msg = str(excinfo.value)
        assert "UNSLOTH_U2NET_PATH" in msg
        assert "nope.onnx" in msg

    def test_a_missing_model_downloads_to_the_shared_cache(self, tmp_path, monkeypatch):
        from core.inference.assist_vision import bg_removal

        monkeypatch.delenv("UNSLOTH_U2NET_PATH", raising=False)
        monkeypatch.setenv("UNSLOTH_VISION_MODEL_DIR", str(tmp_path))
        monkeypatch.setattr(bg_removal, "_session", None)

        called = {}

        def _fake_download(url, filename, **kw):
            called["url"] = url
            called["filename"] = filename
            called["env_var"] = kw.get("env_var")
            dest = tmp_path / filename
            dest.write_bytes(b"fake onnx")
            return str(dest)

        monkeypatch.setattr(bg_removal, "download_model", _fake_download)
        # Stop at session construction: the point is which path was resolved.
        monkeypatch.setattr(
            bg_removal, "_model_path", lambda: str(tmp_path / "u2net.onnx")
        )
        with pytest.raises(Exception):
            bg_removal._get_session()  # onnxruntime rejects the fake bytes

        assert called["filename"] == "u2net.onnx"
        assert called["url"].startswith("https://")
        assert called["env_var"] == "UNSLOTH_U2NET_PATH", (
            "the download must know which env var to name if it fails"
        )
