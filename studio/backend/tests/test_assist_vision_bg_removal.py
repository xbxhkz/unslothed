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


class _HalfMaskSession:
    """Returns a mask that keeps the left half and drops the right half."""

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
        with pytest.raises(Exception):
            remove_background(b"not an image", session=_HalfMaskSession())
