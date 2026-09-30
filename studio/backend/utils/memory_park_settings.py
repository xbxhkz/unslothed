# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted RAM budget for parked (evicted-to-RAM, not destroyed) diffusion/video pipelines.

Same precedence and caching shape as utils/vram_budget_settings.py: a stored value wins, the
environment is a standalone startup default, the constant is the last resort.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Optional

RAM_PARK_BUDGET_SETTING_KEY = "diffusion_ram_park_budget_mib"
RAM_PARK_BUDGET_ENV_VAR = "UNSLOTH_RAM_PARK_BUDGET_MIB"

# 64 GiB total on this machine; 28 GiB leaves headroom for Studio's own process, the OS, and
# whatever else is running. See spec section 4.2 for the reasoning.
RAM_PARK_BUDGET_MIN_MIB = 0
RAM_PARK_BUDGET_MAX_MIB = 61440  # 60 GiB -- never the whole 64GB, always some floor reserved
RAM_PARK_BUDGET_DEFAULT_MIB = 28672  # 28 GiB

_CACHE_TTL_S = 2.0
_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, Any]] = {}
_generation: dict[str, int] = {}
_MAX_REREADS = 3


def _cached_setting(key: str) -> Any:
    stored = None
    for _attempt in range(_MAX_REREADS):
        with _cache_lock:
            hit = _cache.get(key)
            if hit is not None and time.monotonic() - hit[0] < _CACHE_TTL_S:
                return hit[1]
            generation = _generation.get(key, 0)
        try:
            from storage.studio_db import get_app_setting
            stored = get_app_setting(key, None)
        except Exception:
            return None
        with _cache_lock:
            if _generation.get(key, 0) == generation:
                _cache[key] = (time.monotonic(), stored)
                return stored
    return stored


def _invalidate(key: str) -> None:
    with _cache_lock:
        _cache.pop(key, None)
        _generation[key] = _generation.get(key, 0) + 1


def coerce_ram_budget_mib(value: Any) -> Optional[int]:
    """A RAM park budget in MiB, in [RAM_PARK_BUDGET_MIN_MIB, RAM_PARK_BUDGET_MAX_MIB], else None."""
    if isinstance(value, bool):
        return None
    try:
        mib = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not RAM_PARK_BUDGET_MIN_MIB <= mib <= RAM_PARK_BUDGET_MAX_MIB:
        return None
    return mib


def _env_budget_mib() -> Optional[int]:
    return coerce_ram_budget_mib(os.environ.get(RAM_PARK_BUDGET_ENV_VAR))


def get_ram_park_budget_mib() -> int:
    """Never raises and never returns a value outside the supported range."""
    stored = coerce_ram_budget_mib(_cached_setting(RAM_PARK_BUDGET_SETTING_KEY))
    if stored is not None:
        return stored
    from_env = _env_budget_mib()
    if from_env is not None:
        return from_env
    return RAM_PARK_BUDGET_DEFAULT_MIB


def set_ram_park_budget_mib(value: Any = None) -> int:
    """Store a budget, or clear it with None so env/default applies again."""
    from storage.studio_db import upsert_app_settings

    if value is None:
        upsert_app_settings({RAM_PARK_BUDGET_SETTING_KEY: None})
        _invalidate(RAM_PARK_BUDGET_SETTING_KEY)
        return get_ram_park_budget_mib()
    coerced = coerce_ram_budget_mib(value)
    if coerced is None:
        raise ValueError(
            f"RAM park budget must be an integer MiB in "
            f"[{RAM_PARK_BUDGET_MIN_MIB}, {RAM_PARK_BUDGET_MAX_MIB}], got {value!r}"
        )
    upsert_app_settings({RAM_PARK_BUDGET_SETTING_KEY: coerced})
    _invalidate(RAM_PARK_BUDGET_SETTING_KEY)
    return coerced
