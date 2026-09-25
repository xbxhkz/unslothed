# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Concrete readiness probes.

Each one DELEGATES to a helper that already exists rather than reimplementing
its check -- assist_code.servers for language servers, assist_vision.yolo for
the weight, face_swap.licence_accepted for the gate. Reimplementing is how the
audit log's redaction layer drifted from the upstream classifier it was meant to
reuse.

The module-level `_`-prefixed indirections exist so tests can substitute them
without touching the real filesystem, PATH or licence state.
"""

from __future__ import annotations

import importlib.util
import os
import sys

from core.inference.tool_readiness import MISSING, READY, UNKNOWN, Readiness, register_default

# No external dependency: if the process is running, these work. Verified against
# each one's dispatch path in tools.execute_tool rather than assumed -- python
# runs sys.executable (the interpreter already executing this), terminal runs the
# platform shell, edit_file and render_html touch only the filesystem and a
# string. None of the four has an availability gate; their early returns are all
# per-argument validation, which readiness does not model.
#
# search_conversation was here and was WRONG: it is gated on
# conversation_archive.enabled(), so with CONVERSATION_ARCHIVE off, or with
# sqlite_vec's vec0 native library missing from a frozen build, this reported
# "ready" for a tool that answers "Searching earlier conversation is unavailable
# on this server." It has its own probe below.
ALWAYS_READY_TOOLS = frozenset({
    "terminal",
    "python",
    "edit_file",
    "render_html",
    # Self-evident rather than unchecked: this is the tool producing the report,
    # so if the row is being rendered at all, it works. Leaving it unprobed would
    # make the full report say "unknown -- nobody checked" about the checker.
    "check_tool_readiness",
    # Same reason: it is a discovery tool reading the same probes. Without this,
    # the full readiness report would call it "unknown -- nobody checked".
    "find_capability",
})

_CODE_TOOL_NAMES = (
    "code_definition",
    "code_references",
    "code_hover",
    "code_symbols",
    "code_diagnostics",
)

# Pinned: reading MaskRCNN_ResNet50_FPN_Weights.DEFAULT.url at runtime imports
# torchvision's detection stack. A test asserts it still matches torchvision.
_MASKRCNN_WEIGHT = "maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth"


def _yolo_weight_present() -> bool:
    """weights_path() returns the cached file when it exists, else the bare
    filename (ultralytics' cue to download). isfile() therefore distinguishes
    'already here' from 'would be fetched on first use'."""
    from core.inference.assist_vision import yolo

    return os.path.isfile(yolo.weights_path())


def _face_swap_licence_accepted() -> bool:
    from core.inference.assist_vision.face_swap import licence_accepted

    return bool(licence_accepted())


def _face_swap_models_present() -> bool:
    """Delegates to face_swap.models_present(), which owns the filenames and the
    root convention. The licence marker says the user agreed to let InsightFace
    download ~300 MB; it does not say the download happened."""
    from core.inference.assist_vision.face_swap import models_present

    return bool(models_present())


def _conversation_archive_enabled() -> bool:
    """Delegates to the archive's own gate rather than re-deriving it.

    enabled() is `config.CONVERSATION_ARCHIVE and rag_db.rag_available()`, and
    rag_available() is deliberately not the RAG_AVAILABLE flag: the flag only
    records that `import sqlite_vec` worked, while the vec0 native library is a
    separate file a venv (or a frozen build) can be missing. Reimplementing the
    two halves here is exactly the drift this package's docstring warns about, so
    the helper is called instead. Imported lazily, as the tool itself does."""
    from core.rag import conversation_archive

    return bool(conversation_archive.enabled())


def _module_present(name: str) -> bool:
    """find_spec locates a module WITHOUT importing it -- ~2 ms, no side effects.
    It raises for a submodule whose parent is absent, so it is guarded here as
    well as at the registry level."""
    try:
        return importlib.util.find_spec(name) is not None
    except BaseException:  # noqa: BLE001
        return False


def _language_servers() -> dict[str, bool]:
    """{language: installed}. Binary names come from servers._SERVERS, never
    hardcoded here, so a rename upstream cannot leave this silently wrong."""
    from core.inference.assist_code import servers

    out: dict[str, bool] = {}
    for language in servers.SUPPORTED:
        spec = servers._SERVERS.get(language)
        out[language] = bool(spec and servers._which(spec["binary"]))
    return out


def _bg_removal_weight_present() -> bool:
    """Delegates to bg_removal._model_path(), which owns the override env var and
    the shared model cache root."""
    from core.inference.assist_vision import bg_removal

    return os.path.isfile(bg_removal._model_path())


def _maskrcnn_weight_present() -> bool:
    """Delegates to assist_vision.models.model_root(), which owns the vision
    model cache root. shape_detect._get_model() redirects torch's hub dir
    there with torch.hub.set_dir(model_root()) before downloading, so that is
    where the weight actually lands -- not torch's default ~/.cache/torch/hub.
    model_root() is a plain os/env function with no torch dependency, so this
    needs no already-imported torch and has no unknown state."""
    from core.inference.assist_vision.models import model_root

    return os.path.isfile(os.path.join(model_root(), "checkpoints", _MASKRCNN_WEIGHT))


def _diffusion_state() -> str:
    """'unknown' | 'sd_cpp' | 'not_loaded' | 'loaded', read from modules already
    in memory. Never get_active_diffusion_engine() / get_diffusion_backend():
    both construct the engine they return."""
    router = sys.modules.get("core.inference.diffusion_engine_router")
    if router is None:
        return "unknown"
    if getattr(router, "_active_engine_name", None) == getattr(router, "ENGINE_SD_CPP", "sd_cpp"):
        return "sd_cpp"
    diffusion = sys.modules.get("core.inference.diffusion")
    if diffusion is None:
        return "unknown"
    backend = getattr(diffusion, "_diffusion_backend", None)
    if backend is None or not backend.is_loaded:
        return "not_loaded"
    return "loaded"


def _probe_always_ready() -> Readiness:
    return Readiness(READY, "no external dependency")


def _probe_unknown_kb() -> Readiness:
    return Readiness(
        UNKNOWN,
        "no cheap knowledge-base check available; call it to find out",
    )


def _probe_webcam_look() -> Readiness:
    # The package check comes first: a frozen build missing a lazily-imported
    # module is this project's most common defect class, and "the weight is
    # cached" is a useless answer when the code that loads it cannot import.
    # _do_webcam_look reaches both -- webcam.capture_frame_jpeg imports cv2,
    # yolo.detect imports ultralytics.
    absent = [name for name in ("ultralytics", "cv2") if not _module_present(name)]
    if absent:
        return Readiness(
            MISSING,
            f"{', '.join(absent)} not importable in this build",
            missing = ", ".join(absent),
            remedy = "pip install ultralytics opencv-python (or declare them in the frozen build)",
        )
    if _yolo_weight_present():
        return Readiness(READY, "yolov8n.pt present; ultralytics and cv2 importable")
    return Readiness(
        MISSING,
        "yolov8n.pt not in the model cache",
        missing = "yolov8n.pt",
        remedy = "ultralytics downloads it (~6 MB) on first use",
    )


def _probe_face_swap() -> Readiness:
    # Licence first, and deliberately: an unaccepted licence is a different and
    # far more actionable "missing" than absent files, because nothing can be
    # downloaded until the user accepts, and accepting is a thing they can do.
    if not _face_swap_licence_accepted():
        return Readiness(
            MISSING,
            "InsightFace licence not accepted, so its models cannot be downloaded",
            missing = "InsightFace model licence acceptance",
            remedy = "accept the InsightFace licence before using face swap",
        )
    if not _module_present("insightface"):
        return Readiness(
            MISSING,
            "licence accepted, but the insightface package is not importable in this build",
            missing = "insightface",
            remedy = "pip install insightface (or declare it in the frozen build)",
        )
    # The licence marker is written by an explicit user action and downloads
    # nothing. Answering "ready" on it alone told the model a ~300 MB pack was
    # present when the cache could be empty -- and contradicted webcam_look,
    # which calls the identical situation "missing".
    if not _face_swap_models_present():
        return Readiness(
            MISSING,
            "InsightFace models not downloaded",
            missing = "buffalo_l and/or inswapper_128.onnx",
            remedy = (
                "InsightFace fetches them (~300 MB) on first use, now that the "
                "licence is accepted"
            ),
        )
    return Readiness(
        READY, "InsightFace licence accepted, models downloaded, package importable"
    )


def _probe_search_conversation() -> Readiness:
    if _conversation_archive_enabled():
        return Readiness(
            READY, "the conversation archive is enabled and its vector store is usable"
        )
    return Readiness(
        MISSING,
        "the conversation archive is disabled, or its sqlite_vec vec0 library is not loadable",
        missing = "an enabled conversation archive with a working vector store",
        remedy = (
            "enable the conversation archive in settings, and make sure sqlite_vec's "
            "vec0 native library ships with this build"
        ),
    )


def _probe_web_search() -> Readiness:
    if _module_present("ddgs"):
        return Readiness(READY, "ddgs present; network reachability not checked")
    return Readiness(
        MISSING,
        "the ddgs package is not importable; network reachability not checked either",
        missing = "ddgs",
        remedy = "pip install ddgs (or declare it in the frozen build)",
    )


def _probe_remove_background() -> Readiness:
    if not _module_present("onnxruntime"):
        return Readiness(
            MISSING,
            "onnxruntime not importable in this build",
            missing = "onnxruntime",
            remedy = "pip install onnxruntime (or declare it in the frozen build)",
        )
    if _bg_removal_weight_present():
        return Readiness(READY, "u2net.onnx present; onnxruntime importable")
    return Readiness(
        MISSING,
        "u2net.onnx not in the model cache",
        missing = "u2net.onnx",
        remedy = "it is downloaded on first use",
    )


def _probe_detect_shapes() -> Readiness:
    absent = [name for name in ("torch", "torchvision") if not _module_present(name)]
    if absent:
        return Readiness(
            MISSING,
            f"{', '.join(absent)} not importable in this build",
            missing = ", ".join(absent),
            remedy = "pip install torch torchvision (or declare them in the frozen build)",
        )
    present = _maskrcnn_weight_present()
    if present:
        return Readiness(READY, "Mask R-CNN weights present; torch and torchvision importable")
    return Readiness(
        MISSING,
        "Mask R-CNN weights not downloaded",
        missing = _MASKRCNN_WEIGHT,
        remedy = "torchvision downloads them (~170 MB) on first use",
    )


def _probe_edit_image_prompt() -> Readiness:
    state = _diffusion_state()
    if state == "sd_cpp":
        return Readiness(
            MISSING,
            "the native sd.cpp engine is active, and it does not support image-to-image",
            missing = "an image model on the diffusers engine",
            remedy = "load the image model in Studio on the diffusers engine",
        )
    if state == "not_loaded":
        return Readiness(
            MISSING,
            "no image model loaded in Studio",
            missing = "a loaded image model",
            remedy = "load an image model in Studio",
        )
    if state == "loaded":
        return Readiness(READY, "an image model is loaded on the diffusers engine")
    return Readiness(
        UNKNOWN,
        "image generation has not been used this session, so no engine is in memory to check",
    )


def _probe_code_tool() -> Readiness:
    servers = _language_servers()
    have = [lang for lang, ok in servers.items() if ok]
    detail = ", ".join(f"{lang} {'OK' if ok else 'not installed'}" for lang, ok in sorted(servers.items()))
    if have:
        return Readiness(READY, detail)
    return Readiness(
        MISSING,
        detail,
        missing = "a language server for any supported language",
        remedy = "installed automatically on first use, or install manually (see the tool description)",
    )


def install_default_probes() -> None:
    """Register every v1 probe.

    Called on EVERY readiness query and every enriched failure, so it must be
    safe to re-run -- hence register_default, which leaves an already-registered
    probe alone. Using register() here would silently overwrite a caller's
    explicit registration the moment the feature was exercised.
    """
    for name in ALWAYS_READY_TOOLS:
        register_default(name, _probe_always_ready)
    register_default("search_knowledge_base", _probe_unknown_kb)
    register_default("search_conversation", _probe_search_conversation)
    register_default("web_search", _probe_web_search)
    register_default("webcam_look", _probe_webcam_look)
    register_default("face_swap", _probe_face_swap)
    register_default("remove_background", _probe_remove_background)
    register_default("detect_shapes", _probe_detect_shapes)
    register_default("edit_image_prompt", _probe_edit_image_prompt)
    for name in _CODE_TOOL_NAMES:
        register_default(name, _probe_code_tool)
