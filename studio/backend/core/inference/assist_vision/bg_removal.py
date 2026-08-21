# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Background removal via a U2Net ONNX model, downloaded on first use.

No rembg/transformers dependency: rembg pulls in a large extra dependency
tree this app avoids, and a frozen build has no pip available at runtime to
install anything at all. Mirrors the lazy-singleton + injectable-session
pattern used by the sibling ``core/inference/assist_vision/yolo.py`` module
so tests exercise the real pre/post-processing without needing the weight
file at all.

The weights are NOT bundled. The spec once claimed they were, but no
``weights/`` directory ever existed and nothing created one, so this tool
failed on every install with "U2Net model not found". Shipping a 168 MB
binary inside a fork that must absorb every upstream release is the wrong
trade, so it is fetched on first use instead -- the same thing
``detect_shapes`` already does for its torchvision weights -- and the fetch
is announced in the log before it starts, never silent.
"""
import io
import os

import numpy as np
from PIL import Image

from .models import download_model, model_path

_MODEL_INPUT_SIZE = 320  # U2Net's standard input resolution
_MODEL_FILENAME = "u2net.onnx"
_MODEL_SIZE_BYTES = 176 * 1024 * 1024
# U2Net's canonical public distribution: the release assets rembg itself
# pulls from, which is where this weight file is published for general use.
_MODEL_URL = (
    "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
)
_PATH_ENV = "UNSLOTH_U2NET_PATH"

_session = None


def _model_path() -> str:
    """An explicit override wins; otherwise the shared model cache."""
    override = os.environ.get(_PATH_ENV)
    if override:
        return override
    return model_path(_MODEL_FILENAME)


def _get_session():
    global _session
    if _session is None:
        path = _model_path()
        if not os.path.isfile(path):
            if os.environ.get(_PATH_ENV):
                # An override that points at nothing is a user mistake, not a
                # cue to download somewhere they did not ask for.
                raise RuntimeError(
                    f"{_PATH_ENV} is set to {path}, but no file is there. "
                    "Point it at u2net.onnx or unset it to download the model "
                    "automatically."
                )
            # Raises with the file, the URL and the env var named on failure.
            path = download_model(
                _MODEL_URL, _MODEL_FILENAME,
                size_bytes = _MODEL_SIZE_BYTES, env_var = _PATH_ENV,
            )
        # Imported only once a real file exists, so a missing weight reports
        # itself rather than surfacing as a raw onnxruntime "No such file".
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
