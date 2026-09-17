# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Requirement checks for the capability map.

Every check is CHEAP -- an import-spec lookup, a PATH lookup, a file stat, or a
read of state already in memory -- and never imports, loads or creates what it
asks about.

Dispatch is BY NAME, looked up at call time (_CHECK_NAMES -> globals()). A table
of function objects captured at import would make monkeypatching a check
silently ineffective, which is the exact shape of this project's inert controls.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import Optional

from core.inference.tool_readiness import MISSING, READY, UNKNOWN, Readiness

_CHECK_NAMES = {
    "tool": "_check_tool",
    "module": "_check_module",
    "binary": "_check_binary",
    "model": "_check_model",
    "cached_model": "_check_cached_model",
}


def _which(name: str):
    return shutil.which(name)


def _check_tool(name: str) -> Readiness:
    from core.inference import tool_readiness
    from core.inference.tool_readiness.probes import install_default_probes

    # Cheap and idempotent (register_default never clobbers). Without it the
    # registry can be empty and every tool provider would read "unknown".
    install_default_probes()
    readiness = tool_readiness.resolve(name)
    # Detail only. Readiness remedies can be install commands ("pip install
    # ddgs"); the map is a descriptive surface and must not hand those on.
    return Readiness(readiness.state, f"tool {name}: {readiness.detail}")


def _check_module(name: str) -> Readiness:
    from core.inference.tool_readiness import probes

    if probes._module_present(name):
        return Readiness(READY, f"{name} importable")
    return Readiness(MISSING, f"{name} not installed")


def _check_binary(name: str) -> Readiness:
    if _which(name):
        return Readiness(READY, f"{name} on PATH")
    return Readiness(
        MISSING,
        f"{name} not on PATH (a program installed after Unslothed started "
        "is not seen until Unslothed restarts)",
    )


def _llama_backend():
    """routes.inference's resident llama.cpp backend WITHOUT importing the route
    module (a ~20,000-line router core code must not depend on) and without
    get_llama_cpp_backend(). None when the route is not loaded."""
    module = sys.modules.get("routes.inference")
    return getattr(module, "_llama_cpp_backend", None) if module is not None else None


def _orchestrator():
    """The transformers orchestrator through peek_inference_backend(), which the
    orchestrator documents as never constructing one. get_inference_backend()
    DOES construct one, blocking on the torch import."""
    module = sys.modules.get("core.inference.orchestrator")
    if module is None:
        return None
    peek = getattr(module, "peek_inference_backend", None)
    return peek() if callable(peek) else None


def _loaded_models() -> Optional[list[tuple[str, bool]]]:
    """[(model name, has vision)] for locally loaded models, or None when no
    backend module is in memory at all."""
    if "routes.inference" not in sys.modules and "core.inference.orchestrator" not in sys.modules:
        return None
    loaded: list[tuple[str, bool]] = []
    llama = _llama_backend()
    if llama is not None and llama.is_loaded:
        loaded.append((llama.model_identifier or "a GGUF model", bool(llama.is_vision)))
    orchestrator = _orchestrator()
    if orchestrator is not None:
        active = getattr(orchestrator, "active_model_name", None)
        entry = (getattr(orchestrator, "models", None) or {}).get(active) if active else None
        if entry is not None:
            loaded.append((entry.get("display_name") or active, bool(entry.get("is_vision"))))
    return loaded


def _check_model(name: str) -> Readiness:
    if name != "vision":
        # Audio is deliberately absent: llama.cpp's only audio flag routes TTS
        # OUTPUT, and reading it as audio understanding would be a false ready.
        return Readiness(UNKNOWN, f"no check for model ability {name!r}")
    loaded = _loaded_models()
    if loaded is None:
        return Readiness(UNKNOWN, "no local model backend is running in this process")
    if not loaded:
        return Readiness(UNKNOWN, "no local model is loaded; an external model may be serving")
    with_vision = [model for model, has_vision in loaded if has_vision]
    if with_vision:
        return Readiness(READY, f"{with_vision[0]} is loaded")
    names = ", ".join(model for model, _ in loaded)
    return Readiness(MISSING, f"{names} is loaded, without vision")


def _whisper_cache_dir() -> str:
    """Where whisper.load_model downloads models. PINNED rather than asked, because
    importing whisper loads torch; a test asserts whisper still builds it this way."""
    default = os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")


def _check_cached_model(name: str) -> Readiness:
    """A library that downloads its model on first use is MISSING until one is
    cached -- the same convention readiness applies to webcam_look's weight."""
    if name != "whisper":
        return Readiness(UNKNOWN, f"no check for cached model {name!r}")
    try:
        cached = sorted(f for f in os.listdir(_whisper_cache_dir()) if f.endswith(".pt"))
    except OSError:
        cached = []
    if cached:
        return Readiness(READY, f"Whisper model cached ({', '.join(cached)})")
    return Readiness(
        MISSING, "no Whisper model downloaded yet; Whisper fetches one on first use"
    )


def check_requirement(req) -> Readiness:
    """Readiness for one requirement. Never raises."""
    try:
        function_name = _CHECK_NAMES.get(getattr(req, "kind", None))
        check = globals().get(function_name) if function_name else None
        if check is None:
            return Readiness(UNKNOWN, f"no check for requirement kind {getattr(req, 'kind', None)!r}")
        result = check(req.name)
        if not isinstance(result, Readiness):
            return Readiness(UNKNOWN, f"check returned {type(result).__name__}, not Readiness")
        return result
    except BaseException as exc:  # noqa: BLE001 - a capability query must never break a turn
        return Readiness(UNKNOWN, f"check failed: {exc}")
