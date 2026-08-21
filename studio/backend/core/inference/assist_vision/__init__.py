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
import secrets
import time

from .schemas import ASSIST_VISION_TOOLS, ASSIST_VISION_TOOL_NAMES  # noqa: F401


class _Budget:
    """The caller's timeout and stop button, checked at stage boundaries.

    Every other long-running tool in ``tools.py`` receives ``timeout`` and
    ``cancel_event``; the vision branch forwarded neither, so a first
    ``detect_shapes`` call (which downloads ~170 MB) and every
    ``edit_image_prompt`` diffusion pass ran with no ceiling and ignored the
    user's stop button entirely.

    HONEST LIMIT: a single model inference -- one torch forward pass, one
    onnxruntime run, one diffusion pass, one InsightFace swap -- is an
    uninterruptible call inside a third-party library. This bounds the tool's
    participation at every boundary it controls (before starting, between
    stages, around a download) and stops the NEXT stage, but it cannot abort an
    inference already running. It does not pretend otherwise: nothing here
    claims to have killed work it merely stopped waiting on.
    """

    __slots__ = ("deadline", "cancel_event")

    def __init__(self, timeout = None, cancel_event = None):
        self.deadline = (
            time.monotonic() + timeout
            if timeout is not None and timeout > 0
            else None
        )
        self.cancel_event = cancel_event

    def check(self, name):
        """Error text if this call should stop here, else None."""
        if self.cancel_event is not None and self.cancel_event.is_set():
            return f"{name} cancelled."
        if self.deadline is not None and time.monotonic() >= self.deadline:
            return f"{name} timed out."
        return None


def _write_png(data, session_id, name):
    """Write PNG bytes into the SESSION WORKDIR and return the path.

    Tools return a PATH, never an inline data: URI -- a data URI is re-sent to
    the model on every later turn and persisted into replayed history.

    The workdir, not ``tempfile.mkstemp``: the system temp directory is outside
    the sandbox, so ``resolve_image_bytes`` refuses every path written there
    ("is outside this conversation's working directory"). That made a returned
    path useless to everything downstream -- ``remove_background`` ->
    ``detect_shapes`` chaining was structurally impossible, ``edit_file`` /
    ``python`` / ``terminal`` could not touch a result, and the user could not
    find one. Writing where Studio already lets this conversation read and write
    fixes the lifecycle too: sandbox contents are session-scoped and cleaned up
    with the session, where temp files just leaked a multi-MB PNG per call.

    ``_get_workdir`` is imported lazily for the same reason ``paths.py`` does
    it: ``tools`` imports this package, so a module-scope import is circular.
    """
    from core.inference import tools as _tools

    workdir = _tools._get_workdir(session_id)
    os.makedirs(workdir, exist_ok = True)
    # Random suffix, not a counter: two calls in one session must not collide,
    # or the second silently overwrites the result the model was just told about.
    path = os.path.join(workdir, f"{name}_{secrets.token_hex(4)}.png")
    with open(path, "wb") as f:
        f.write(data)
    return path


def _do_remove_background(arguments, session_id, budget):
    from .bg_removal import remove_background
    from .paths import resolve_image_bytes

    stop = budget.check("remove_background")
    if stop:
        return stop
    data, err = resolve_image_bytes(arguments.get("image_path"), session_id = session_id)
    if err:
        return f"remove_background failed: {err}"
    # Checked before the model runs: the first call here may download 176 MB.
    stop = budget.check("remove_background")
    if stop:
        return stop
    out = remove_background(data)
    path = _write_png(out, session_id, "remove_background")
    return f"Background removed. Transparent PNG written to: {path}"


def _annotated_line(image_bytes, dets, session_id, name, budget = None):
    """The 'Annotated image written to: ...' line, or nothing.

    ``annotate`` returns None when the image cannot be decoded or re-encoded.
    Reporting findings without an annotated copy is the honest outcome there;
    the detections themselves are still good.
    """
    from . import yolo

    if budget is not None and budget.check(name):
        # Findings are already in hand; skip only the extra rendering work.
        return "\n(Stopped before rendering an annotated copy.)"
    annotated = yolo.annotate(image_bytes, dets, ".png")
    if annotated is None:
        return "\n(Could not render an annotated copy of this image.)"
    return f"\nAnnotated image written to: {_write_png(annotated, session_id, name)}"


def _do_detect_shapes(arguments, session_id, budget):
    from . import shape_detect, yolo
    from .paths import resolve_image_bytes

    stop = budget.check("detect_shapes")
    if stop:
        return stop
    data, err = resolve_image_bytes(arguments.get("image_path"), session_id = session_id)
    if err:
        return f"detect_shapes failed: {err}"
    # The expensive boundary: the first call downloads ~170 MB of weights
    # before any inference starts.
    stop = budget.check("detect_shapes")
    if stop:
        return stop
    dets = shape_detect.detect(data)
    # yolo.summarize, not a near-copy of it: the local duplicate lacked
    # pluralisation, so detect_shapes said "2 person" where webcam_look said
    # "2 people" for the same picture. shape_detect's detections already carry
    # exactly the keys summarize reads.
    summary = yolo.summarize(dets)
    if not dets:
        return summary
    return summary + _annotated_line(data, dets, session_id, "detect_shapes", budget)


def _do_webcam_look(arguments, session_id, budget):
    from . import webcam, yolo

    stop = budget.check("webcam_look")
    if stop:
        return stop
    index = arguments.get("camera_index")
    frame = webcam.capture_frame_jpeg(index = index)
    stop = budget.check("webcam_look")
    if stop:
        return stop
    dets = yolo.detect(frame)
    summary = yolo.summarize(dets)
    if not dets:
        return summary
    return summary + _annotated_line(frame, dets, session_id, "webcam_look", budget)


def _do_edit_image_prompt(arguments, session_id, budget):
    from .image_edit import edit_image
    from .paths import resolve_image_bytes

    stop = budget.check("edit_image_prompt")
    if stop:
        return stop
    data, err = resolve_image_bytes(arguments.get("image_path"), session_id = session_id)
    if err:
        return f"edit_image_prompt failed: {err}"
    # Last boundary before a full diffusion pass, the longest-running stage
    # any of these tools has.
    stop = budget.check("edit_image_prompt")
    if stop:
        return stop
    strength = arguments.get("strength")
    kwargs = {} if strength is None else {"strength": float(strength)}
    out = edit_image(data, arguments.get("prompt", ""), **kwargs)
    path = _write_png(out, session_id, "edit_image_prompt")
    return f"Image edited. Result written to: {path}"


def _do_face_swap(arguments, session_id, budget):
    from .face_swap import LicenseNotAcceptedError, NoFaceDetectedError, swap_face
    from .paths import resolve_image_bytes

    stop = budget.check("face_swap")
    if stop:
        return stop
    source, err = resolve_image_bytes(arguments.get("source_face_path"), session_id = session_id)
    if err:
        return f"face_swap failed: {err}"
    target, err = resolve_image_bytes(arguments.get("target_image_path"), session_id = session_id)
    if err:
        return f"face_swap failed: {err}"
    stop = budget.check("face_swap")
    if stop:
        return stop
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
    path = _write_png(out, session_id, "face_swap")
    return (
        f"Face swapped. Result written to: {path} "
        "(carries metadata marking it AI-edited)"
    )


_HANDLERS = {
    "remove_background": _do_remove_background,
    "detect_shapes": _do_detect_shapes,
    "webcam_look": _do_webcam_look,
    "edit_image_prompt": _do_edit_image_prompt,
    "face_swap": _do_face_swap,
}


def execute(name, arguments, *, session_id = None, timeout = None, cancel_event = None):
    """Run a vision tool. Always returns str; never raises into the agent loop.

    ``timeout`` and ``cancel_event`` come from ``execute_tool`` exactly as they
    do for every other long-running tool, and are honoured at each stage
    boundary -- see ``_Budget`` for what that can and cannot interrupt.
    """
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"unknown vision tool: {name}"
    try:
        return handler(arguments or {}, session_id, _Budget(timeout, cancel_event))
    except Exception as e:  # noqa: BLE001 - the tool boundary must not raise
        return f"{name} failed: {type(e).__name__}: {e}"
