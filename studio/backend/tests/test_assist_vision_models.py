# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The shared model-location / download / formatting helpers.

These were three unrelated conventions and two near-identical formatters spread
across the package. They are one module now, and this is where that module is
held to its contract -- including the parts that had zero coverage before:
``position`` (used by BOTH detectors), ``annotate`` (on the success path of two
of the five tools), and the pluralisation whose absence made ``detect_shapes``
say "2 person" where ``webcam_look`` said "2 people" about the same picture.

No weights ship in git, so first use downloads. That download must be
DISCLOSED -- announced before it starts, and explained by name, URL and manual
override if it fails. A silent multi-hundred-MB fetch triggered by a chat
message is the failure mode these tests exist to prevent.
"""

import io
import logging
import os

import numpy as np
import pytest
from PIL import Image

from core.inference.assist_vision import models


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_VISION_MODEL_DIR", str(tmp_path))
    return tmp_path


class TestModelRoot:
    def test_the_env_override_wins(self, model_dir):
        assert models.model_root() == str(model_dir)

    def test_the_default_is_the_shared_unsloth_cache(self, monkeypatch):
        monkeypatch.delenv("UNSLOTH_VISION_MODEL_DIR", raising = False)
        root = models.model_root()
        assert root.endswith(os.path.join(".unsloth", "assist_vision_models"))

    def test_every_model_resolves_under_one_root(self, model_dir):
        """The point of the extraction: one convention, not three."""
        for name in ("u2net.onnx", "yolov8n.pt"):
            assert os.path.dirname(models.model_path(name)) == str(model_dir)


class _RecordingLogger:
    """Records rendered log lines.

    A double rather than ``caplog``: this package logs through Studio's
    structlog logger, which writes to stdout and never reaches the stdlib
    handler caplog installs. Recording here also makes the ORDERING (announce,
    then fetch) directly observable, which is the property that matters.
    """

    def __init__(self):
        self.lines = []

    def info(self, msg, *args):
        self.lines.append(msg % args if args else msg)

    def text(self):
        return "\n".join(self.lines).lower()


@pytest.fixture
def spy_logger(monkeypatch):
    spy = _RecordingLogger()
    monkeypatch.setattr(models, "logger", spy)
    return spy


class TestDownloadDisclosure:
    def test_an_existing_file_is_not_refetched_and_says_nothing(self, model_dir, spy_logger):
        dest = model_dir / "already.bin"
        dest.write_bytes(b"cached")

        def _boom(*a, **k):
            raise AssertionError("must not download a file that is already cached")

        path = models.download_model(
            "https://example.invalid/already.bin", "already.bin",
            _opener = _boom,
        )
        assert path == str(dest)
        assert "downloading" not in spy_logger.text()

    def test_a_real_download_announces_itself_before_it_starts(self, model_dir, spy_logger):
        """The log line must name the file, its size and where it comes from,
        and must be emitted BEFORE the request opens -- announcing a 176 MB
        fetch only once it has finished is not disclosure."""
        log_at_open_time = []

        def _opener(url):
            log_at_open_time.append(spy_logger.text())
            return io.BytesIO(b"payload")

        models.download_model(
            "https://example.invalid/u2net.onnx", "u2net.onnx",
            size_bytes = 176 * 1024 * 1024, env_var = "UNSLOTH_U2NET_PATH",
            _opener = _opener,
        )

        assert log_at_open_time, "opener was never called"
        announced = log_at_open_time[0]
        assert "downloading" in announced, (
            "the download was announced only after it finished, or not at all"
        )
        assert "u2net.onnx" in announced
        assert "176 mb" in announced
        assert "example.invalid" in announced

    def test_the_downloaded_bytes_land_at_the_destination(self, model_dir):
        models.download_model(
            "https://example.invalid/m.bin", "m.bin",
            _opener = lambda url: io.BytesIO(b"weights"),
        )
        assert (model_dir / "m.bin").read_bytes() == b"weights"

    def test_a_failed_download_leaves_no_partial_file(self, model_dir):
        """A truncated file would later load as a corrupt model."""
        class _Halfway(io.RawIOBase):
            def readable(self):
                return True

            def readinto(self, b):
                raise OSError("connection reset")

        with pytest.raises(RuntimeError):
            models.download_model(
                "https://example.invalid/m.bin", "m.bin",
                _opener = lambda url: io.BufferedReader(_Halfway()),
            )
        assert not (model_dir / "m.bin").exists()
        assert not list(model_dir.glob(".m.bin.*")), "temp file left behind"

    def test_a_failure_names_the_file_the_url_and_the_manual_override(self, model_dir):
        def _opener(url):
            raise OSError("no route to host")

        with pytest.raises(RuntimeError) as excinfo:
            models.download_model(
                "https://example.invalid/u2net.onnx", "u2net.onnx",
                size_bytes = 1024, env_var = "UNSLOTH_U2NET_PATH",
                _opener = _opener,
            )
        msg = str(excinfo.value)
        assert "u2net.onnx" in msg
        assert "https://example.invalid/u2net.onnx" in msg
        assert "UNSLOTH_U2NET_PATH" in msg, (
            "the error must name the env var that lets a user supply the file"
        )


class TestPosition:
    """Direct coverage for the helper BOTH detectors phrase their output with,
    which previously had none."""

    @pytest.mark.parametrize(
        "cx, cy, expected",
        [
            (50, 50, "top-left"),
            (150, 50, "top"),
            (250, 50, "top-right"),
            (50, 150, "left"),
            (150, 150, "center"),
            (250, 150, "right"),
            (50, 250, "bottom-left"),
            (150, 250, "bottom"),
            (250, 250, "bottom-right"),
        ],
    )
    def test_the_nine_grid_cells(self, cx, cy, expected):
        assert models.position(cx, cy, 300, 300) == expected

    def test_the_centre_cell_is_named_center_not_middle_center(self):
        """Reads as prose in a sentence to the model, so it is special-cased."""
        assert models.position(150, 150, 300, 300) == "center"


class TestFormatDetections:
    def _raw(self, *items):
        return list(items)

    def test_low_confidence_is_dropped(self):
        raw = self._raw(("person", 0.2, 0, 0, 10, 10))
        assert models.format_detections(raw, 100, 100, conf = 0.4) == []

    def test_boxes_are_rounded_and_position_added(self):
        raw = self._raw(("dog", 0.9, 1.4, 2.6, 3.5, 4.5))
        det = models.format_detections(raw, 100, 100)[0]
        # round() is banker's rounding, so both .5 cases go to even: 3.5 -> 4
        # and 4.5 -> 4. Pinned as-is rather than "fixed" -- a box edge landing
        # a pixel either way is meaningless, and the existing detector tests
        # already encode this behaviour.
        assert det["box"] == [1, 3, 4, 4]
        assert det["position"] == "top-left"
        assert det["confidence"] == 0.9

    def test_detections_are_numbered_within_their_own_label(self):
        raw = self._raw(
            ("person", 0.9, 0, 0, 1, 1),
            ("person", 0.9, 0, 0, 1, 1),
            ("dog", 0.9, 0, 0, 1, 1),
        )
        dets = models.format_detections(raw, 10, 10)
        assert [(d["label"], d["index"]) for d in dets] == [
            ("person", 1), ("person", 2), ("dog", 1),
        ]

    def test_a_mask_is_carried_through_only_when_present(self):
        """This is what lets the box-only and mask-producing detectors share
        one formatter."""
        mask = np.ones((2, 2), dtype = bool)
        with_mask = models.format_detections(
            [("person", 0.9, 0, 0, 1, 1, mask)], 10, 10
        )[0]
        without = models.format_detections([("person", 0.9, 0, 0, 1, 1)], 10, 10)[0]
        assert with_mask["mask"] is mask
        assert "mask" not in without


class TestSummarize:
    def test_nothing_found_is_a_normal_sentence_not_an_error(self):
        out = models.summarize([])
        assert "no recognizable" in out.lower()
        assert "error" not in out.lower()

    def test_plural_labels_read_naturally(self):
        """The bug the duplicated formatter had: "2 person"."""
        dets = models.format_detections(
            [("person", 0.9, 0, 0, 1, 1), ("person", 0.8, 0, 0, 1, 1)], 10, 10
        )
        out = models.summarize(dets)
        assert "2 people" in out
        assert "2 person" not in out

    def test_a_single_detection_stays_singular(self):
        dets = models.format_detections([("person", 0.9, 0, 0, 1, 1)], 10, 10)
        assert "1 person" in models.summarize(dets)
