# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whether a tool will actually work, asked cheaply.

The model already receives every tool's schema each turn, so it does not need to
be told which tools EXIST. What it cannot find out is whether one will work:
webcam_look needs a downloaded weight, the code tools need a language server on
PATH, an MCP tool needs its server connected. Today it discovers that by calling
the tool and failing.

THREE STATES, NOT A BOOLEAN. A tool with no probe -- every MCP tool, anything
added later -- reports "unknown", never "ready". Defaulting unknown to ready is
the optimistic lie that makes this feature worse than not having it: the model
would read "nobody checked" as "verified working".

Probes must be CHEAP: a file stat, a PATH lookup, an already-held connection
flag. No process spawning, no network calls. The model may ask on any turn.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

CACHE_TTL_SECONDS = 60.0

READY = "ready"
MISSING = "missing"
UNKNOWN = "unknown"


@dataclass(frozen = True)
class Readiness:
    state: str
    detail: str
    missing: Optional[str] = None
    remedy: Optional[str] = None


Probe = Callable[[], Readiness]

_probes: dict[str, Probe] = {}
_cache: dict[str, tuple[float, Readiness]] = {}
_lock = threading.Lock()


def reset_for_tests() -> None:
    with _lock:
        _probes.clear()
        _cache.clear()


def register(tool_name: str, probe: Probe) -> None:
    """Register or REPLACE a probe. An explicit registration always wins."""
    with _lock:
        _probes[tool_name] = probe
        _cache.pop(tool_name, None)


def register_default(tool_name: str, probe: Probe) -> None:
    """Register only if nothing is registered for this tool yet.

    install_default_probes() is called on every readiness query and every
    enriched failure, so it must be safe to re-run. Using register() there would
    silently overwrite any probe a caller -- or a test -- had registered
    explicitly, the moment the feature was exercised.
    """
    with _lock:
        if tool_name in _probes:
            return
        _probes[tool_name] = probe
        _cache.pop(tool_name, None)


def registered_names() -> list[str]:
    with _lock:
        return sorted(_probes)


def resolve(tool_name: str, *, refresh: bool = False) -> Readiness:
    """Readiness for one tool. Never raises."""
    now = time.monotonic()
    with _lock:
        probe = _probes.get(tool_name)
        if not refresh:
            hit = _cache.get(tool_name)
            if hit is not None and now - hit[0] < CACHE_TTL_SECONDS:
                return hit[1]
    if probe is None:
        return Readiness(UNKNOWN, "no readiness check defined for this tool")
    try:
        result = probe()
        if not isinstance(result, Readiness):  # a probe that returns junk is a bug, not a crash
            result = Readiness(UNKNOWN, f"probe returned {type(result).__name__}, not Readiness")
    except BaseException as exc:  # noqa: BLE001 - a probe must never break a caller
        result = Readiness(UNKNOWN, f"readiness check failed: {exc}")
    with _lock:
        # A concurrent register() may have replaced the probe while we were
        # running the old one outside the lock. Only cache this result if the
        # probe we ran is still the one registered -- otherwise we'd silently
        # overwrite a newer registration's future reading with a stale one,
        # which would make register()'s "an explicit registration always wins"
        # false under concurrency.
        if _probes.get(tool_name) is probe:
            _cache[tool_name] = (now, result)
    return result


def resolve_all(*, refresh: bool = False) -> dict[str, Readiness]:
    return {name: resolve(name, refresh = refresh) for name in registered_names()}
