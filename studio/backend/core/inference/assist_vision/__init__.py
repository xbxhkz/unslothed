# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Vision tools for the Studio agent loop.

Adds background removal, object/shape detection, webcam capture, prompt-based
image editing, and face swapping. Implementations live here rather than in
``tools.py`` so upstream merges stay cheap: ``tools.py`` carries exactly two
lines referring to this package (one entry in ``ALL_TOOLS``, one branch in
``execute_tool``).

Heavy model libraries (torch, onnxruntime, insightface, ultralytics, cv2) are
imported INSIDE functions, never at module scope -- a module-scope
``import torch`` stalls the event loop for seconds on first use.
"""
import os
import tempfile

from .schemas import ASSIST_VISION_TOOLS, ASSIST_VISION_TOOL_NAMES  # noqa: F401


def _write_png(data):
    """Write PNG bytes to a temp file and return its absolute path.

    Tools return a PATH, never an inline data: URI -- a data URI is re-sent to
    the model on every later turn and persisted into replayed history.
    """
    fd, path = tempfile.mkstemp(suffix = ".png", prefix = "unsloth_vision_")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


def _do_remove_background(arguments, session_id):
    from .bg_removal import remove_background
    from .paths import resolve_image_bytes

    data, err = resolve_image_bytes(arguments.get("image_path"), session_id = session_id)
    if err:
        return f"remove_background failed: {err}"
    out = remove_background(data)
    return f"Background removed. Transparent PNG written to: {_write_png(out)}"


def _describe(dets):
    """One line per label group, or a plain not-found line. Never 'error'."""
    if not dets:
        return "No recognizable objects detected."
    groups = {}
    for d in dets:
        groups.setdefault(d["label"], []).append(d)
    parts = []
    for label, items in groups.items():
        confs = ", ".join(f"{int(i['confidence'] * 100)}%" for i in items)
        positions = ", ".join(sorted({i["position"] for i in items}))
        parts.append(f"{len(items)} {label} ({confs}; {positions})")
    return ", ".join(parts)


def _do_detect_shapes(arguments, session_id):
    from . import shape_detect, yolo
    from .paths import resolve_image_bytes

    data, err = resolve_image_bytes(arguments.get("image_path"), session_id = session_id)
    if err:
        return f"detect_shapes failed: {err}"
    dets = shape_detect.detect(data)
    summary = _describe(dets)
    if not dets:
        return summary
    annotated = yolo.annotate(data, dets, ".png")
    return f"{summary}\nAnnotated image written to: {_write_png(annotated)}"


def _do_webcam_look(arguments, session_id):
    from . import webcam, yolo

    index = arguments.get("camera_index")
    frame = webcam.capture_frame_jpeg(index = index)
    dets = yolo.detect(frame)
    summary = yolo.summarize(dets)
    if not dets:
        return summary
    annotated = yolo.annotate(frame, dets, ".png")
    return f"{summary}\nAnnotated image written to: {_write_png(annotated)}"


def _do_edit_image_prompt(arguments, session_id):
    from .image_edit import edit_image
    from .paths import resolve_image_bytes

    data, err = resolve_image_bytes(arguments.get("image_path"), session_id = session_id)
    if err:
        return f"edit_image_prompt failed: {err}"
    strength = arguments.get("strength")
    kwargs = {} if strength is None else {"strength": float(strength)}
    out = edit_image(data, arguments.get("prompt", ""), **kwargs)
    return f"Image edited. Result written to: {_write_png(out)}"


def _do_face_swap(arguments, session_id):
    from .face_swap import LicenseNotAcceptedError, NoFaceDetectedError, swap_face
    from .paths import resolve_image_bytes

    source, err = resolve_image_bytes(arguments.get("source_face_path"), session_id = session_id)
    if err:
        return f"face_swap failed: {err}"
    target, err = resolve_image_bytes(arguments.get("target_image_path"), session_id = session_id)
    if err:
        return f"face_swap failed: {err}"
    try:
        out = swap_face(source, target)
    except LicenseNotAcceptedError:
        return (
            "face_swap is unavailable: InsightFace's face-detection and face-swap "
            "models are licensed for non-commercial, research-only use. Accepting "
            "that license is a decision only a person can make -- this assistant "
            "cannot make it on the user's behalf, and there is no way to complete "
            "it through this conversation. Ask the user to review InsightFace's "
            "published license for these models themselves, then take the "
            "acceptance step directly on this Unsloth Studio installation (there "
            "is no in-app settings page for this yet, so it must be done outside "
            "of chat). Once that is done, try again."
        )
    except NoFaceDetectedError as e:
        return f"face_swap failed: {e}"
    return (
        "Face swapped. Result written to: "
        f"{_write_png(out)} (carries metadata marking it AI-edited)"
    )


_HANDLERS = {
    "remove_background": _do_remove_background,
    "detect_shapes": _do_detect_shapes,
    "webcam_look": _do_webcam_look,
    "edit_image_prompt": _do_edit_image_prompt,
    "face_swap": _do_face_swap,
}


def execute(name, arguments, *, session_id = None):
    """Run a vision tool. Always returns str; never raises into the agent loop."""
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"unknown vision tool: {name}"
    try:
        return handler(arguments or {}, session_id)
    except Exception as e:  # noqa: BLE001 - the tool boundary must not raise
        return f"{name} failed: {type(e).__name__}: {e}"
