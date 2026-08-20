# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Background removal via a bundled U2Net ONNX model.

No rembg/transformers dependency: rembg pulls in a large extra dependency
tree this app avoids, and a frozen build has no pip available at runtime to
install anything at all. Mirrors the lazy-singleton + injectable-session
pattern used by the sibling ``core/inference/assist_vision/yolo.py`` module
so tests exercise the real pre/post-processing without needing the 168 MB
weight file, which is fetched separately and does not ship with the repo.
"""
import io
import os

import numpy as np
from PIL import Image

_MODEL_INPUT_SIZE = 320  # U2Net's standard input resolution
_session = None


def _model_path() -> str:
    override = os.environ.get("UNSLOTH_U2NET_PATH")
    if override:
        return override
    return os.path.join(os.path.dirname(__file__), "weights", "u2net.onnx")


def _get_session():
    global _session
    if _session is None:
        path = _model_path()
        # Checked BEFORE importing onnxruntime / constructing the session so a
        # dev environment where the weight file was never downloaded gets a
        # specific, actionable error instead of a raw onnxruntime "No such
        # file" (or an ImportError that hides the real problem).
        if not os.path.isfile(path):
            raise RuntimeError(
                f"U2Net model not found at {path}. Download u2net.onnx into "
                "that folder or set UNSLOTH_U2NET_PATH."
            )
        import onnxruntime
        _session = onnxruntime.InferenceSession(path)
    return _session


def _preprocess(img: Image.Image) -> np.ndarray:
    resized = img.convert("RGB").resize((_MODEL_INPUT_SIZE, _MODEL_INPUT_SIZE), Image.LANCZOS)
    arr = np.array(resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return np.expand_dims(arr, axis=0).astype(np.float32)


def remove_background(image_bytes: bytes, *, session=None) -> bytes:
    """Return RGBA PNG bytes with the background made transparent."""
    img = Image.open(io.BytesIO(image_bytes))
    original_size = img.size

    sess = session or _get_session()
    input_name = sess.get_inputs()[0].name
    output = sess.run(None, {input_name: _preprocess(img)})[0]

    mask = output[0][0]
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(original_size, Image.LANCZOS)

    rgba = img.convert("RGBA")
    rgba.putalpha(mask_img)

    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    return buf.getvalue()
