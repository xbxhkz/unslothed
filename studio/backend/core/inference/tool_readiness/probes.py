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

from core.inference.tool_readiness import MISSING, READY, UNKNOWN, Readiness, register_default

# No external dependency: if the process is running, these work.
ALWAYS_READY_TOOLS = frozenset({
    "terminal",
    "python",
    "edit_file",
    "render_html",
    "search_conversation",
})

_CODE_TOOL_NAMES = (
    "code_definition",
    "code_references",
    "code_hover",
    "code_symbols",
    "code_diagnostics",
)


def _yolo_weight_present() -> bool:
    """weights_path() returns the cached file when it exists, else the bare
    filename (ultralytics' cue to download). isfile() therefore distinguishes
    'already here' from 'would be fetched on first use'."""
    from core.inference.assist_vision import yolo

    return os.path.isfile(yolo.weights_path())


def _face_swap_licence_accepted() -> bool:
    from core.inference.assist_vision.face_swap import licence_accepted

    return bool(licence_accepted())


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


def _probe_always_ready() -> Readiness:
    return Readiness(READY, "no external dependency")


def _probe_unknown_kb() -> Readiness:
    return Readiness(
        UNKNOWN,
        "no cheap knowledge-base check available; call it to find out",
    )


def _probe_webcam_look() -> Readiness:
    if _yolo_weight_present():
        return Readiness(READY, "yolov8n.pt present")
    return Readiness(
        MISSING,
        "yolov8n.pt not in the model cache",
        missing = "yolov8n.pt",
        remedy = "ultralytics downloads it (~6 MB) on first use",
    )


def _probe_face_swap() -> Readiness:
    if _face_swap_licence_accepted():
        return Readiness(READY, "InsightFace licence accepted")
    return Readiness(
        MISSING,
        "InsightFace licence not accepted, so its models cannot be downloaded",
        missing = "InsightFace model licence acceptance",
        remedy = "accept the InsightFace licence before using face swap",
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
    register_default("web_search", _probe_web_search)
    register_default("webcam_look", _probe_webcam_look)
    register_default("face_swap", _probe_face_swap)
    for name in _CODE_TOOL_NAMES:
        register_default(name, _probe_code_tool)
