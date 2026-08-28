# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Object/shape detection (torchvision Mask R-CNN) and webcam capture.

Models and the camera are both injectable, so these tests run the real
filtering, per-label numbering and mask thresholding against stubs -- no
170 MB download, no physical camera. Numbering is asserted explicitly
because a later sub-project depends on being able to say "the 2nd person".
"""

import io

import numpy as np
import pytest
from PIL import Image

from core.inference.assist_vision import shape_detect, webcam, yolo


def _png(size=(4, 4), color=(10, 20, 30)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class _FakeMaskRCNN:
    def __init__(self, boxes, labels, scores, masks):
        self._out = {
            "boxes": np.array(boxes, dtype=np.float32),
            "labels": np.array(labels, dtype=np.int64),
            "scores": np.array(scores, dtype=np.float32),
            "masks": np.array(masks, dtype=np.float32),
        }

    def __call__(self, tensors):
        return [self._out]


_CATS = ["__background__", "person", "dog", "N/A"]


class TestShapeDetect:
    def test_a_detection_carries_label_box_and_thresholded_mask(self):
        model = _FakeMaskRCNN(
            boxes=[[1, 2, 3, 4]], labels=[1], scores=[0.92],
            masks=[[[[0.1, 0.9], [0.8, 0.2]]]],
        )
        dets = shape_detect.detect(_png(size=(2, 2)), model=model, categories=_CATS)
        assert len(dets) == 1
        assert dets[0]["label"] == "person"
        assert dets[0]["box"] == [1, 2, 3, 4]
        assert dets[0]["mask"].dtype == bool
        assert dets[0]["mask"].tolist() == [[False, True], [True, False]]

    def test_detections_are_numbered_per_label_not_globally(self):
        """person #1, person #2, dog #1 -- a global counter would give 1,2,3."""
        model = _FakeMaskRCNN(
            boxes=[[0, 0, 1, 1]] * 3, labels=[1, 1, 2], scores=[0.9, 0.9, 0.9],
            masks=[[[[0.9, 0.9], [0.9, 0.9]]]] * 3,
        )
        dets = shape_detect.detect(_png(size=(2, 2)), model=model, categories=_CATS)
        assert [(d["label"], d["index"]) for d in dets] == [
            ("person", 1), ("person", 2), ("dog", 1),
        ]

    def test_low_confidence_detections_are_dropped(self):
        model = _FakeMaskRCNN(
            boxes=[[0, 0, 1, 1]], labels=[1], scores=[0.2],
            masks=[[[[0.9, 0.9], [0.9, 0.9]]]],
        )
        assert shape_detect.detect(_png(size=(2, 2)), model=model, categories=_CATS, conf=0.4) == []

    def test_background_and_placeholder_labels_are_skipped(self):
        model = _FakeMaskRCNN(
            boxes=[[0, 0, 1, 1]] * 2, labels=[0, 3], scores=[0.99, 0.99],
            masks=[[[[0.9, 0.9], [0.9, 0.9]]]] * 2,
        )
        assert shape_detect.detect(_png(size=(2, 2)), model=model, categories=_CATS) == []

    def test_finding_nothing_is_an_empty_list_not_an_error(self):
        model = _FakeMaskRCNN(boxes=[], labels=[], scores=[], masks=np.empty((0, 1, 2, 2)))
        assert shape_detect.detect(_png(size=(2, 2)), model=model, categories=_CATS) == []

    def test_injection_bypasses_real_model_construction(self, monkeypatch):
        def _boom():
            raise AssertionError("must not build the real Mask R-CNN")

        monkeypatch.setattr(shape_detect, "_get_model", _boom)
        monkeypatch.setattr(shape_detect, "_get_categories", _boom)
        model = _FakeMaskRCNN(boxes=[], labels=[], scores=[], masks=np.empty((0, 1, 2, 2)))
        shape_detect.detect(_png(size=(2, 2)), model=model, categories=_CATS)


class TestWebcamAndYolo:
    def test_capture_returns_jpeg_bytes_from_an_injected_grabber(self):
        # `grabber(index) -> frame` is the real seam capture_frame_jpeg calls
        # (mirroring _default_grabber's own return contract, which already
        # does the open/isOpened/read/release dance and hands back a frame,
        # not a capture object) -- the double mimics that directly rather
        # than reimplementing a cv2.VideoCapture-shaped stand-in that the
        # production code never calls methods on.
        frame = np.zeros((16, 16, 3), dtype=np.uint8)

        data = webcam.capture_frame_jpeg(grabber=lambda index: frame)
        assert data[:2] == b"\xff\xd8", "should be a JPEG"

    def test_summarize_reports_nothing_found_without_raising(self):
        assert "no recognizable" in yolo.summarize([]).lower()

    def test_capture_encodes_without_cv2_when_a_grabber_is_injected(self, monkeypatch):
        """The grabber seam exists so this runs with no camera -- and so it must
        also run with no OpenCV. An unconditional `import cv2` on the encode
        step defeated that on any machine without it."""
        import builtins

        real_import = builtins.__import__

        def _no_cv2(name, *args, **kwargs):
            if name == "cv2":
                raise ImportError("No module named 'cv2'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_cv2)
        frame = np.zeros((8, 8, 3), dtype = np.uint8)
        data = webcam.capture_frame_jpeg(grabber = lambda index: frame)
        assert data[:2] == b"\xff\xd8", "should still be a JPEG via the PIL fallback"


class TestAnnotate:
    """Direct coverage for the drawing step on the success path of two of the
    five tools, which had none."""

    def _jpeg(self, size = (40, 30)):
        buf = io.BytesIO()
        Image.new("RGB", size, (30, 30, 30)).save(buf, format = "JPEG")
        return buf.getvalue()

    def _dets(self):
        return [{"label": "person", "confidence": 0.91, "box": [2, 2, 20, 20],
                 "position": "top-left"}]

    def test_it_returns_decodable_image_bytes_in_the_requested_format(self):
        out = yolo.annotate(self._jpeg(), self._dets(), ".png")
        assert out is not None
        assert Image.open(io.BytesIO(out)).format == "PNG"

    def test_the_default_format_is_jpeg(self):
        out = yolo.annotate(self._jpeg(), self._dets())
        assert Image.open(io.BytesIO(out)).format == "JPEG"

    def test_it_actually_draws_something(self):
        """Fails if the boxes are computed but never rendered."""
        plain = self._jpeg()
        out = yolo.annotate(plain, self._dets(), ".png")
        before = np.asarray(Image.open(io.BytesIO(plain)).convert("RGB"))
        after = np.asarray(Image.open(io.BytesIO(out)).convert("RGB"))
        assert after.shape == before.shape
        assert not np.array_equal(before, after), "no pixels changed"

    def test_the_annotated_size_matches_the_source(self):
        out = yolo.annotate(self._jpeg(size = (64, 48)), self._dets(), ".png")
        assert Image.open(io.BytesIO(out)).size == (64, 48)

    def test_undecodable_input_returns_none_not_the_original_bytes(self):
        """Returning the input made the caller write a JPEG/WEBP under a .png
        name -- a file that then failed to open for whatever read it next."""
        assert yolo.annotate(b"not an image at all", self._dets(), ".png") is None

    def test_zero_detections_still_produces_an_image(self):
        out = yolo.annotate(self._jpeg(), [], ".png")
        assert out is not None
        assert Image.open(io.BytesIO(out)).format == "PNG"
