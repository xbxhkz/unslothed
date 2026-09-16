# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Face-swap via a locally-run InsightFace pipeline (face detection/
alignment + the pretrained InSwapper model). Both the detection model pack
(buffalo_l) and the swap model (inswapper_128.onnx) are licensed for
non-commercial research use only by InsightFace -- neither is bundled in
Studio's installer or fetched automatically. This module never lets
InsightFace's own implicit downloader run without the user's explicit,
recorded acceptance -- a file-backed marker under the model cache directory,
since Studio has no settings system to back this the way the source project
did -- see _ensure_models_available(). Output PNGs carry embedded provenance
metadata identifying them as AI-face-swapped (metadata only, no visible
watermark).

API verified directly against the installed insightface==1.0.1 package
(site-packages, not web search / training-data memory) before writing this
module:
  - FaceAnalysis.__init__(self, name='buffalo_l', root='~/.insightface',
    allowed_modules=None, **kwargs) -- root= is confirmed the right kwarg.
  - model_zoo.get_model(name, **kwargs) reads root= via kwargs (confirmed:
    `root = kwargs.get('root', '~/.insightface')`), mirroring
    FaceAnalysis(root=...). _get_swapper() passes the bare filename (not an
    absolute path) as `name`, since get_model()'s internal download_onnx()
    interpolates `name` directly into the fetch URL -- an absolute local
    path there would build a malformed, non-working download URL.
  - FaceAnalysis.get(img) returns a plain Python list ([] or list[Face]) --
    confirmed via source, matches the truthiness checks below.
"""
import io
import os

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from .models import model_root as _model_root

_MODEL_PACK_NAME = "buffalo_l"
_SWAP_MODEL_FILENAME = "inswapper_128.onnx"

_analyzer = None
_swapper = None


class LicenseNotAcceptedError(Exception):
    """Raised when a swap is attempted before the InsightFace model
    license (covering both buffalo_l and inswapper_128.onnx) has been
    explicitly accepted via record_licence_acceptance()."""


class NoFaceDetectedError(Exception):
    """Raised when face detection finds no usable face in an input image."""


def _licence_marker_path():
    return os.path.join(_model_root(), "INSIGHTFACE_LICENCE_ACCEPTED")


def licence_accepted():
    """True only once the user has explicitly accepted InsightFace's terms."""
    return os.path.isfile(_licence_marker_path())


def _insightface_models_dir():
    """Where InsightFace puts what it downloads under ``root=_model_root()``.

    Both call sites below pass ``root = _model_root()``, and InsightFace joins a
    ``models`` sub_dir onto it before the pack name or the onnx filename --
    ``utils/storage.py``'s ``download()``/``download_onnx()`` do
    ``os.path.join(expanduser(root), sub_dir, name)``, and
    ``model_zoo.get_model`` does the same. That one join is the only thing here
    not already named by this module; the pack and file names come from the
    constants above rather than being written out a second time.
    """
    return os.path.join(os.path.expanduser(_model_root()), "models")


def models_present():
    """True when both InsightFace models are already on disk.

    ``licence_accepted()`` says only that the user agreed to let the downloader
    run -- it writes a marker, not ~300 MB of weights. Asking that question
    instead of this one is how a readiness check came to answer "ready" for an
    install that had never downloaded anything.

    Deliberately cheap: two stat calls, no import of insightface, no network.
    The pack is a DIRECTORY that the zip is extracted into; the swap model is a
    single file.
    """
    root = _insightface_models_dir()
    return (
        os.path.isdir(os.path.join(root, _MODEL_PACK_NAME))
        and os.path.isfile(os.path.join(root, _SWAP_MODEL_FILENAME))
    )


def record_licence_acceptance():
    """Record explicit acceptance. Called only from an explicit user action."""
    os.makedirs(_model_root(), exist_ok=True)
    with open(_licence_marker_path(), "w", encoding="utf-8") as f:
        f.write("InsightFace models are licensed for non-commercial research use only.\n")


def _ensure_models_available():
    """Gate: the InsightFace license must be explicitly accepted before
    InsightFace's own downloader is allowed to run. Its primary caller is
    swap_face() itself, unconditionally, so the gate applies even when
    analyzer/swapper are injected (see swap_face()'s docstring for why). It
    is also called from _get_analyzer() and _get_swapper() so neither
    singleton can be constructed (and neither model implicitly downloaded)
    on its own without passing this check first -- do not remove either
    call site as a "redundant cleanup"."""
    if not licence_accepted():
        raise LicenseNotAcceptedError(
            "Face-swap requires accepting InsightFace's model license first "
            "-- call record_licence_acceptance() after the user has explicitly "
            "reviewed and accepted it."
        )


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        _ensure_models_available()
        from insightface.app import FaceAnalysis
        _analyzer = FaceAnalysis(name=_MODEL_PACK_NAME, root=_model_root())
        _analyzer.prepare(ctx_id=0, det_size=(640, 640))
    return _analyzer


def _get_swapper():
    global _swapper
    if _swapper is None:
        _ensure_models_available()
        from insightface.model_zoo import get_model
        # Pass the bare filename (not a full path) as `name`, with root=
        # telling InsightFace where to look/save it -- mirroring
        # _get_analyzer()'s FaceAnalysis(root=...) pattern. get_model()
        # interpolates `name` directly into its download URL when it needs
        # to fetch the file; passing our absolute local path there would
        # build a malformed, non-working URL instead of a real download.
        _swapper = get_model(_SWAP_MODEL_FILENAME, download=True, root=_model_root())
    return _swapper


def _provenance_metadata() -> PngInfo:
    info = PngInfo()
    info.add_text("assist:ai-edited", "face-swap")
    return info


def swap_face(source_face_bytes: bytes, target_image_bytes: bytes, *,
              analyzer=None, swapper=None) -> bytes:
    """Swap the face from source_face_bytes into target_image_bytes.
    Returns PNG bytes with embedded provenance metadata. Raises
    LicenseNotAcceptedError, NoFaceDetectedError, or lets underlying
    inference errors propagate -- callers apply the never-raises discipline
    at their own boundary, matching this package's other vision tools'
    convention.

    The license gate is checked here unconditionally -- not only inside
    _get_analyzer()/_get_swapper() -- so that license acceptance is required
    for every swap regardless of whether analyzer/swapper are injected.
    Gating only the getters would let a caller that injects its own
    analyzer/swapper bypass the licence entirely; see this module's Step 5
    negative control in the task report for confirmation that the gate test
    actually catches that regression."""
    _ensure_models_available()
    analyzer = analyzer if analyzer is not None else _get_analyzer()
    swapper = swapper if swapper is not None else _get_swapper()

    source_img = np.array(Image.open(io.BytesIO(source_face_bytes)).convert("RGB"))[:, :, ::-1]
    target_img = np.array(Image.open(io.BytesIO(target_image_bytes)).convert("RGB"))[:, :, ::-1]

    source_faces = analyzer.get(source_img)
    if not source_faces:
        raise NoFaceDetectedError("No face detected in the source image")
    if source_faces[0].kps is None:
        # The detector found a face-shaped region but couldn't produce
        # usable landmarks for it (confirmed against the installed
        # insightface==1.0.1 source: FaceAnalysis.get() sets kps=None
        # whenever its detector's kpss batch comes back None). Unguarded,
        # INSwapper.get() below passes this straight to
        # face_align.norm_crop2 -> estimate_norm's `assert lmk.shape ==
        # (5, 2)`, which crashes on None with a bare, unactionable
        # "'NoneType' object has no attribute 'shape'".
        raise NoFaceDetectedError(
            "A face was detected in the source image but its landmarks "
            "couldn't be aligned -- try a clearer, more front-facing photo"
        )
    target_faces = analyzer.get(target_img)
    if not target_faces:
        raise NoFaceDetectedError("No face detected in the target image")
    if target_faces[0].kps is None:
        raise NoFaceDetectedError(
            "A face was detected in the target image but its landmarks "
            "couldn't be aligned -- try a clearer, more front-facing photo"
        )

    result = swapper.get(target_img, target_faces[0], source_faces[0], paste_back=True)

    out = Image.fromarray(result[:, :, ::-1])
    buf = io.BytesIO()
    out.save(buf, format="PNG", pnginfo=_provenance_metadata())
    return buf.getvalue()
