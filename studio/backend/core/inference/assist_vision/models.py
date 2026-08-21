# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One model-location convention, one download path, one detection formatter.

Before this module the package had THREE unrelated conventions for where a
model file lives -- ``UNSLOTH_VISION_MODEL_DIR`` -> ``~/.unsloth/
assist_vision_models`` (duplicated verbatim in ``shape_detect`` and
``face_swap``), ``UNSLOTH_U2NET_PATH`` -> a package-local ``weights/``
directory, and ``yolo.weights_path()`` with no env override at all -- and two
near-identical detection formatters. Both detectors now share ``position`` and
``format_detections`` here, so a phrasing or filtering change lands once.

No weights ship in git. A 168 MB binary in a fork of an actively developed
upstream is the wrong trade, so every model is fetched on first use instead --
and fetched OPENLY: ``download_model`` logs what it is about to pull and how
big it is BEFORE the request starts, and turns a failure into a message naming
the file, the URL, and the env var that lets a user drop the file in by hand.
An undisclosed multi-hundred-MB network fetch triggered by a chat message is
exactly the surprise this avoids.

``urllib``, not ``requests``: this is a plain unauthenticated GET of a public
file and the stdlib already does it, so nothing here needs to care which HTTP
client the surrounding install happens to carry.
"""
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from collections import OrderedDict

try:  # Studio's structured logger when present, plain logging in bare tests.
    from loggers import get_logger
    logger = get_logger(__name__)
except Exception:  # pragma: no cover - exercised only outside the app
    import logging
    logger = logging.getLogger(__name__)

_MODEL_DIR_ENV = "UNSLOTH_VISION_MODEL_DIR"


def model_root() -> str:
    """The one directory every vision model is cached in."""
    override = os.environ.get(_MODEL_DIR_ENV)
    if override:
        return override
    return os.path.join(os.path.expanduser("~"), ".unsloth", "assist_vision_models")


def model_path(filename: str) -> str:
    return os.path.join(model_root(), filename)


def _human(size_bytes):
    if not size_bytes:
        return "unknown size"
    mb = size_bytes / (1024 * 1024)
    return f"~{mb:.0f} MB"


def _declared_length(response):
    """``Content-Length`` as an int, or None when it isn't usable.

    None means "cannot verify" -- a server that omits the header, a chunked
    response, or a test double with no headers at all. The caller skips the
    size check in that case rather than failing a download it cannot judge.
    """
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        return int(headers.get("Content-Length"))
    except (TypeError, ValueError):
        return None


def download_model(url, filename, *, size_bytes = None, env_var = None, _opener = None):
    """Fetch ``url`` into ``model_root()/filename`` once, and say so out loud.

    Returns the local path. If the file is already there this is a no-op and
    logs nothing -- only a real download announces itself.

    Downloads to a temp file in the same directory and renames, so an install
    interrupted halfway cannot leave a truncated file that later loads as a
    corrupt model. The byte count is checked against ``Content-Length`` before
    the rename, because a dropped connection does NOT raise -- see the comment
    at the check itself.

    ``_opener`` is a test seam (``callable(url) -> file-like``); production
    always uses ``urllib.request.urlopen``. It exists so the disclosure and
    failure-message contracts can be tested without network access.
    """
    dest = model_path(filename)
    if os.path.isfile(dest):
        return dest

    os.makedirs(model_root(), exist_ok = True)
    # Announced BEFORE the request opens, not after it finishes: the point is
    # that a user watching the log knows why the machine just started pulling
    # hundreds of megabytes.
    logger.info(
        "assist_vision: downloading model %s (%s) from %s -- first use only, "
        "cached in %s",
        filename, _human(size_bytes), url, model_root(),
    )
    open_url = _opener or urllib.request.urlopen
    tmp_fd, tmp_path = tempfile.mkstemp(prefix = f".{filename}.", dir = model_root())
    os.close(tmp_fd)
    try:
        with open_url(url) as response, open(tmp_path, "wb") as out:
            shutil.copyfileobj(response, out)
            written = out.tell()
            expected = _declared_length(response)
        # A dropped connection mid-transfer does NOT raise: copyfileobj asks for
        # an explicit amount, and http.client's read(amt) returns short and
        # closes rather than raising IncompleteRead ("it might break
        # compatibility", says its own source). Without this check a truncated
        # fetch gets renamed into place looking complete, and because the
        # isfile() short-circuit above never re-fetches, every later call loads
        # the corrupt file and dies inside the model runtime -- fixable only by
        # deleting a file the user does not know exists. On a ~176 MB download
        # that is not a rare event.
        if expected is not None and written != expected:
            raise OSError(f"transfer ended after {written} of {expected} bytes")
        os.replace(tmp_path, dest)
    except Exception as e:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        hint = (
            f" Alternatively, download it yourself and set {env_var} to its path."
            if env_var else
            f" Alternatively, download it yourself and place it at {dest}."
        )
        raise RuntimeError(
            f"could not download {filename} ({_human(size_bytes)}) from {url}: "
            f"{type(e).__name__}: {e}.{hint}"
        ) from e
    logger.info("assist_vision: downloaded %s to %s", filename, dest)
    return dest


def position(cx, cy, w, h):
    """Coarse 3x3 grid position of a box centre, as words.

    Shared by both detectors so ``detect_shapes`` and ``webcam_look`` describe
    the same picture the same way.
    """
    col = "left" if cx < w / 3 else ("right" if cx > 2 * w / 3 else "center")
    row = "top" if cy < h / 3 else ("bottom" if cy > 2 * h / 3 else "middle")
    if row == "middle" and col == "center":
        return "center"
    if row == "middle":
        return col
    if col == "center":
        return row
    return f"{row}-{col}"


def format_detections(raw, w, h, conf = 0.4):
    """Filter by confidence, round boxes, add grid position and per-label index.

    ``raw`` is an iterable of ``(label, confidence, x1, y1, x2, y2)`` or
    ``(label, confidence, x1, y1, x2, y2, mask)``; the mask is carried through
    only when present, which is what lets the box-only YOLO detector and the
    mask-producing Mask R-CNN detector share this. Pure.

    ``index`` numbers each detection within its own label (person #1, person
    #2, dog #1) rather than globally -- the shape a future "swap the 2nd
    person" capability needs to disambiguate without a breaking change.
    """
    dets = []
    counts = {}
    for item in raw:
        label, c, x1, y1, x2, y2 = item[:6]
        if float(c) < conf:
            continue
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        counts[label] = counts.get(label, 0) + 1
        det = {
            "label": label,
            "index": counts[label],
            "confidence": round(float(c), 2),
            "box": [round(x1), round(y1), round(x2), round(y2)],
            "position": position(cx, cy, w, h),
        }
        if len(item) > 6:
            det["mask"] = item[6]
        dets.append(det)
    return dets


def _plural(label):
    if label == "person":
        return "people"
    return label if label.endswith("s") else label + "s"


def summarize(dets):
    """Group by label with counts, confidences and positions -> one line.

    Zero detections is a normal result, never an error, for both detectors.
    """
    if not dets:
        return "No recognizable objects detected."
    groups = OrderedDict()
    for d in dets:
        groups.setdefault(d["label"], []).append(d)
    parts = []
    for label, items in groups.items():
        n = len(items)
        confs = ", ".join(f"{int(i['confidence'] * 100)}%" for i in items)
        pos = ", ".join(sorted({i["position"] for i in items}))
        name = label if n == 1 else _plural(label)
        parts.append(f"{n} {name} ({confs}; {pos})")
    return ", ".join(parts)
