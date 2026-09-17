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

import shutil

from core.inference.tool_readiness import MISSING, READY, UNKNOWN, Readiness

_CHECK_NAMES = {
    "tool": "_check_tool",
    "module": "_check_module",
    "binary": "_check_binary",
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
