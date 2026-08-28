# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Face swapping, and the guardrails that make shipping it defensible.

InsightFace's models are non-commercial/research-only, so they are never
bundled and never fetched until the user has explicitly accepted the licence.
Outputs carry provenance metadata identifying them as AI-face-swapped.

These tests assert the GATE holds even when the models are injected -- a gate
that only guarded model construction would be bypassed by any caller that
supplies its own analyzer, which is exactly the bug this shape once had.
"""

import io

import numpy as np
import pytest
from PIL import Image

from core.inference.assist_vision import face_swap


def _png(size=(64, 64), color=(200, 100, 50)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class _Face:
    def __init__(self, kps=True):
        self.kps = np.zeros((5, 2), dtype=np.float32) if kps else None


class _Analyzer:
    def __init__(self, faces_by_call):
        self._faces = list(faces_by_call)

    def get(self, img):
        return self._faces.pop(0)


class _Swapper:
    def get(self, target_img, target_face, source_face, paste_back=True):
        return 255 - target_img  # distinguishable, so a dropped result is visible


@pytest.fixture
def accepted(monkeypatch):
    monkeypatch.setattr(face_swap, "licence_accepted", lambda: True)


class TestLicenceGate:
    def test_swapping_without_acceptance_is_refused_even_with_models_injected(self, monkeypatch):
        monkeypatch.setattr(face_swap, "licence_accepted", lambda: False)
        with pytest.raises(face_swap.LicenseNotAcceptedError):
            face_swap.swap_face(
                _png(), _png(),
                analyzer=_Analyzer([[_Face()], [_Face()]]), swapper=_Swapper(),
            )

    def test_recording_acceptance_makes_licence_accepted_true(self, tmp_path, monkeypatch):
        monkeypatch.setenv("UNSLOTH_VISION_MODEL_DIR", str(tmp_path))
        assert face_swap.licence_accepted() is False
        face_swap.record_licence_acceptance()
        assert face_swap.licence_accepted() is True


class TestSwap:
    def test_output_is_png_carrying_provenance_metadata(self, accepted):
        out = face_swap.swap_face(
            _png(), _png(color=(200, 100, 50)),
            analyzer=_Analyzer([[_Face()], [_Face()]]), swapper=_Swapper(),
        )
        img = Image.open(io.BytesIO(out))
        assert img.format == "PNG"
        assert img.text.get("assist:ai-edited") == "face-swap"

    def test_the_swapper_result_is_actually_used(self, accepted):
        """Fails if the target image is re-encoded untouched."""
        out = face_swap.swap_face(
            _png(), _png(color=(200, 100, 50)),
            analyzer=_Analyzer([[_Face()], [_Face()]]), swapper=_Swapper(),
        )
        px = Image.open(io.BytesIO(out)).convert("RGB").getpixel((0, 0))
        assert all(abs(a - b) <= 2 for a, b in zip(px, (55, 155, 205)))

    def test_no_face_in_source_is_an_error(self, accepted):
        with pytest.raises(face_swap.NoFaceDetectedError):
            face_swap.swap_face(_png(), _png(), analyzer=_Analyzer([[]]), swapper=_Swapper())

    def test_a_face_without_landmarks_is_an_error_not_a_crash(self, accepted):
        """Guards the real bug: kps=None crashed inside InsightFace as
        "'NoneType' object has no attribute 'shape'"."""
        with pytest.raises(face_swap.NoFaceDetectedError):
            face_swap.swap_face(
                _png(), _png(),
                analyzer=_Analyzer([[_Face(kps=False)], [_Face()]]), swapper=_Swapper(),
            )
