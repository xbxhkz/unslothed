# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""RAM-tier budget and LRU eviction for parked (not destroyed) diffusion/video pipelines.

Deliberately ignorant of what a "pipeline" is -- every operation takes callables the caller
hands it, the same decoupling core.inference.gpu_arbiter._EVICTORS already uses. This module
only ever tracks owner -> (footprint_mib, parked_at, unload_callable); it never imports
DiffusionBackend or the video backend.

The tracked tuple stores each owner's UNLOAD callable, not a restore one: forced eviction of an
older parked owner (to make RAM room for a new one) always means a full teardown of that OTHER
owner, which only ITS OWN unload() can do -- this module has no other way to reach it. restore()
is never stored; it is supplied fresh by restore_owner's own caller, since only that caller
(the backend whose pipeline it is) ever has a reason to call it.

Contract: park_owner's caller (gpu_arbiter's evictors) never needs its own fallback logic --
on return, the named owner is EITHER parked (tracked here) OR fully unloaded. It never raises
for any of the anticipated fallback paths (unsizeable footprint, over budget with a failed
forced eviction, park() inapplicable or failing); a genuinely unexpected exception from park()
or unload() still propagates, since that is a bug, not a known degrade path.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

from loggers import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
# owner -> (footprint_mib, parked_at_monotonic, unload_callable)
_parked: dict[str, tuple[int, float, Callable[[], None]]] = {}


def is_parked(owner: str) -> bool:
    with _lock:
        return owner in _parked


def parked_footprint_mib() -> int:
    with _lock:
        return sum(footprint for footprint, _, _ in _parked.values())


def forget_parked(owner: str) -> None:
    """Clears owner's parked bookkeeping without calling restore() or unload() -- the caller has
    already decided what to do with the actual pipeline."""
    with _lock:
        _parked.pop(owner, None)


def _oldest_parked_excluding(exclude: set) -> Optional[str]:
    with _lock:
        candidates = [(parked_at, o) for o, (_, parked_at, _) in _parked.items() if o not in exclude]
    if not candidates:
        return None
    candidates.sort(key = lambda pair: pair[0])
    return candidates[0][1]


def park_owner(
    owner: str,
    park: Callable[[], bool],
    unload: Callable[[], None],
    footprint_mib: Optional[int],
    protect: Optional[str] = None,
) -> None:
    """Park ``owner`` (or fully unload it), evicting older parked owners to fit the RAM budget.

    ``protect`` names an owner that must never be chosen as a forced-eviction victim: the owner
    whose acquire is causing this park (gpu_arbiter passes the NEW owner). If it is parked, its
    begin_load() is about to restore it, so tearing it down to make room would throw away the very
    pipeline the switch-back exists to reuse. With nothing else evictable, ``owner`` itself is
    unloaded instead -- the same fallback as any other over-budget park. ``None`` keeps the
    original behavior exactly."""
    if footprint_mib is None:
        logger.info("memory_residency: %s has no resident footprint estimate, unloading directly", owner)
        unload()
        forget_parked(owner)
        return

    from utils.memory_park_settings import get_ram_park_budget_mib

    budget = get_ram_park_budget_mib()
    exclude = {owner} if protect is None else {owner, protect}
    while parked_footprint_mib() + footprint_mib > budget:
        victim = _oldest_parked_excluding(exclude)
        if victim is None:
            logger.info(
                "memory_residency: parking %s (%d MiB) would exceed the %d MiB budget with "
                "nothing left to evict; unloading directly", owner, footprint_mib, budget,
            )
            unload()
            forget_parked(owner)
            return
        with _lock:
            _, _, victim_unload = _parked.get(victim, (0, 0.0, None))
        try:
            if victim_unload is not None:
                victim_unload()
        except Exception:
            logger.exception(
                "memory_residency: forced eviction of %s failed; unloading %s directly instead "
                "of parking it", victim, owner,
            )
            unload()
            forget_parked(owner)
            forget_parked(victim)
            return
        forget_parked(victim)

    try:
        park_ok = park()
    except Exception:
        logger.exception("memory_residency: park() raised for %s, unloading directly", owner)
        park_ok = False
    if not park_ok:
        unload()
        forget_parked(owner)
        return
    with _lock:
        _parked[owner] = (footprint_mib, time.monotonic(), unload)


def restore_owner(owner: str, restore: Callable[[], None]) -> None:
    with _lock:
        if owner not in _parked:
            raise KeyError(f"{owner!r} is not parked")
    restore()
    forget_parked(owner)
