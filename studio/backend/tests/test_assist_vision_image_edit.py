# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Prompt-based image editing, driven by Studio's own diffusion backend.

Unlike the other vision tools this is not a port -- Assist drove its own
sd-server, and this drives Studio's existing img2img. The backend is
injectable so these tests pin the contract (the prompt and the source image
both reach it, and its result is what comes back) without loading a
diffusion model.
"""

import io

import pytest
from PIL import Image

from core.inference.assist_vision.image_edit import edit_image


def _png(size=(32, 32), color=(10, 120, 200)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class _Backend:
    def __init__(self):
        self.calls = []

    def __call__(self, *, image_bytes, prompt, strength):
        self.calls.append({"prompt": prompt, "strength": strength, "size": len(image_bytes)})
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), (240, 30, 30)).save(buf, format="PNG")
        return buf.getvalue()


class TestEditImage:
    def test_the_prompt_and_source_image_reach_the_backend(self):
        backend = _Backend()
        edit_image(_png(), "make the sky orange", backend=backend)
        assert backend.calls[0]["prompt"] == "make the sky orange"
        assert backend.calls[0]["size"] > 0

    def test_the_backend_result_is_returned_not_the_original(self):
        """Fails if the source image is passed through untouched."""
        out = edit_image(_png(), "anything", backend=_Backend())
        assert Image.open(io.BytesIO(out)).convert("RGB").getpixel((0, 0)) == (240, 30, 30)

    def test_an_empty_prompt_is_rejected(self):
        with pytest.raises(ValueError):
            edit_image(_png(), "   ", backend=_Backend())

    def test_strength_is_forwarded(self):
        backend = _Backend()
        edit_image(_png(), "x", backend=backend, strength=0.25)
        assert backend.calls[0]["strength"] == 0.25
