# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""YOLO object detection on a JPEG frame: detect -> annotate -> summarize.
The model is a lazy singleton (ultralytics). Formatting lives in the shared
`models` module, so tests need no ultralytics and both detectors phrase their
findings identically.

First use may DOWNLOAD the ~6 MB yolov8n weights. ultralytics fetches them
implicitly from a bare filename, which used to happen with nothing said about
it; `_get_model` now logs the fetch before letting it start."""
import logging
import os

from .models import format_detections, model_path, position as _position, summarize  # noqa: F401

logger = logging.getLogger(__name__)

_WEIGHTS_FILENAME = "yolov8n.pt"
_WEIGHTS_SIZE_BYTES = 6 * 1024 * 1024

_model = None


def weights_path():
    """Where yolov8n.pt should be loaded from.

    Prefers a copy already in the shared model cache; falls back to the bare
    filename, which is ultralytics' cue to resolve and download it. The bare
    string is deliberate, not an oversight -- but the caller announces it, so
    the download is no longer silent.
    """
    cached = model_path(_WEIGHTS_FILENAME)
    return cached if os.path.isfile(cached) else _WEIGHTS_FILENAME


def _get_model():
    global _model
    if _model is None:
        path = weights_path()
        if not os.path.isfile(path):
            # Disclosed before ultralytics is even imported, so the log line
            # precedes the network call rather than trailing it.
            logger.info(
                "assist_vision: ultralytics will download %s (~6 MB) on first "
                "use of webcam_look; set UNSLOTH_VISION_MODEL_DIR and place the "
                "file there to supply it yourself",
                _WEIGHTS_FILENAME,
            )
        from ultralytics import YOLO
        _model = YOLO(path)
    return _model


def detect(jpeg, *, model=None, conf=0.4):
    """Detect objects in JPEG bytes -> list[Detection]."""
    import cv2
    import numpy as np
    arr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return []
    h, w = arr.shape[:2]
    m = model or _get_model()
    raw = []
    for r in m(arr, verbose=False):
        names = r.names
        for b in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            raw.append((names[int(b.cls[0])], float(b.conf[0]), x1, y1, x2, y2))
    return format_detections(raw, w, h, conf)


def annotate(jpeg, dets, fmt=".jpg"):
    """Draw boxes + labels -> encoded image bytes in `fmt` (default JPEG,
    matching this module's original convention; pass fmt=".png" for callers
    that need PNG, e.g. shape_detect's save convention, which -- like every
    other image tool in this app -- persists PNGs).

    Returns None when the image cannot be decoded or re-encoded, rather than
    handing back the input unchanged: callers write the result under a .png
    name, so returning the original produced a JPEG/WEBP with a .png
    extension -- a file that then failed to open for whatever read it next.
    Callers report their findings without an annotated copy instead.
    """
    import cv2
    import numpy as np
    arr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return None
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        cv2.rectangle(arr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
        cv2.putText(arr, f"{d['label']} {int(d['confidence'] * 100)}%",
                    (int(x1), max(0, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 200, 0), 1)
    ok, buf = cv2.imencode(fmt, arr)
    return bytes(buf.tobytes()) if ok else None
