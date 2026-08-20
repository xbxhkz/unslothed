# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Webcam single-frame capture -> JPEG bytes. Real capture uses OpenCV
(cv2.VideoCapture); the grabber is injectable so tests never need a device.
The camera is opened, one frame read, and released immediately -- never held."""
import logging

logger = logging.getLogger(__name__)


def _default_grabber(index):
    import cv2
    cap = cv2.VideoCapture(index)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"could not open camera {index}")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"no frame from camera {index}")
        return frame  # BGR numpy array
    finally:
        cap.release()


def capture_frame_jpeg(*, grabber=None, index=None):
    """Grab one webcam frame and return JPEG bytes. Raises RuntimeError on
    failure (no camera / no frame). `grabber(index) -> frame` is injectable.

    `index` defaults to 0 (the first camera) when omitted -- this package has
    no persisted-settings lookup of its own, so unlike some of this module's
    sibling tools, camera selection is entirely the caller's responsibility
    (e.g. the agent tool threading through an explicit `index` argument)
    rather than being read from a stored preference here.
    """
    if index is None:
        index = 0
    grab = grabber or _default_grabber
    frame = grab(index)
    import cv2
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("failed to JPEG-encode webcam frame")
    return bytes(buf.tobytes())
