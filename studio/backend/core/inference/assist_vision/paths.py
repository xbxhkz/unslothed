# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Turn a tool's ``image_path`` argument into image bytes, or into error text.

Returns ``(bytes, None)`` or ``(None, error_text)`` and never raises: the tool
boundary returns strings, so a raised exception would escape into the agent
loop instead of becoming something the model can read and retry.

Confinement, when a ``session_id`` is given, reuses Studio's own sandbox
machinery from ``core.inference.tools`` (``_get_workdir`` /
``_is_outside_workdir`` -- the same pair ``edit_file`` resolves paths
through) rather than inventing a separate home-directory-plus-extra-roots
model. That import is deferred into the function body: ``tools`` is a large
module that will eventually import this package back (to register the vision
tools), so importing it at module scope here would be circular.
"""
import io
import os

_DEFAULT_MAX_BYTES = 26214400  # 25 MiB


def resolve_image_bytes(path_value, *, session_id = None, max_bytes = _DEFAULT_MAX_BYTES):
    """Resolve ``path_value`` to image bytes. Never raises.

    When ``session_id`` is given, the path is confined to that session's
    sandbox workdir the same way ``edit_file`` confines writes -- an
    unconfined image reader feeding a model that echoes content back would
    otherwise be a file-exfiltration primitive. Callers that omit
    ``session_id`` get no confinement, which is why every real tool call
    passes it.
    """
    if not path_value or not str(path_value).strip():
        return None, "image_path is required"

    candidate = os.path.abspath(os.path.expanduser(str(path_value).strip()))

    if session_id is not None:
        try:
            from core.inference import tools as _tools
            workdir = _tools._get_workdir(session_id)
            outside = _tools._is_outside_workdir(candidate, workdir)
        except Exception as e:
            return None, f"could not confine image_path: {e}"
        if outside:
            return None, (
                f"image_path '{path_value}' is outside this conversation's working "
                "directory, which is the only place vision tools can read images "
                "from. Use a path under the working directory."
            )

    if not os.path.exists(candidate):
        return None, f"image_path not found: {path_value}"
    if not os.path.isfile(candidate):
        return None, f"image_path is not a file: {path_value}"

    try:
        size = os.path.getsize(candidate)
    except OSError as e:
        return None, f"could not read {path_value}: {e}"
    if size > max_bytes:
        return None, (
            f"image is too large ({size} bytes, max {max_bytes}). "
            "Resize it or point at a smaller file."
        )

    try:
        with open(candidate, "rb") as f:
            data = f.read()
    except OSError as e:
        return None, f"could not read {path_value}: {e}"

    # Decode-verify here so every tool gets the same clear message rather than
    # each one failing differently deep inside its own model library.
    try:
        from PIL import Image
        Image.open(io.BytesIO(data)).verify()
    except Exception:
        return None, f"not a readable image file: {path_value}"

    return data, None
