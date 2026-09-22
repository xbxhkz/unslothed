# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Roles: a name like "coding" bound to a concrete model.

Availability reuses tool readiness's three states on purpose. A role nobody
bound is "unknown", never "ready" -- the same rule the rest of this fork's
discovery work rests on, so a later piece can list models beside tools instead
of maintaining a second vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.inference.tool_readiness import MISSING, READY, UNKNOWN, Readiness

# Offered by the API as a starting point. Nothing behavioural is keyed to these;
# any custom role name works.
DEFAULT_ROLE_NAMES = ("primary", "coding", "vision", "image", "video")


@dataclass(frozen = True)
class RoleBinding:
    role: str
    model: str
    overrides: dict = field(default_factory = dict)


def normalize_role(name) -> str:
    return str(name).strip().lower()


def _resolve_local(model_id: str):
    """(load_path, gguf_variant, loader_id) when the model is downloaded, else None.

    Delegates to the resolver auto-switch uses, so "downloaded" means the same
    thing here as it does when a request names a model.

    Calls resolve_local_gguf with the default allow_scan = True, which on a
    stale index (5s TTL) runs a synchronous walk over ./models, the HF caches,
    LM Studio folders and user scan folders. That is deliberate: allow_scan =
    False answers only from the last built index and never rebuilds, so on a
    cold index it would report a downloaded model as missing and refuse the
    first delegation of a session. This function must never be called on the
    event loop; an async caller must wrap it in asyncio.to_thread."""
    from core.inference.local_model_resolver import resolve_local_gguf

    return resolve_local_gguf(model_id)


def bindings() -> dict[str, RoleBinding]:
    from core.inference.model_roles import storage

    out: dict[str, RoleBinding] = {}
    for name, raw in (storage.get_role_bindings() or {}).items():
        if not isinstance(raw, dict):
            continue
        model = raw.get("model")
        if not isinstance(model, str) or not model.strip():
            continue
        overrides = raw.get("overrides")
        out[normalize_role(name)] = RoleBinding(
            role = normalize_role(name),
            model = model.strip(),
            overrides = overrides if isinstance(overrides, dict) else {},
        )
    return out


def resolve(role) -> Optional[RoleBinding]:
    return bindings().get(normalize_role(role))


def availability(role) -> Readiness:
    """Readiness for a role. Never raises.

    May run a synchronous directory scan (see _resolve_local); never call this
    on the event loop, wrap it in asyncio.to_thread from an async caller."""
    binding = resolve(role)
    if binding is None:
        return Readiness(UNKNOWN, f"no model is bound to the role {normalize_role(role)!r}")
    try:
        found = _resolve_local(binding.model)
    except BaseException as exc:  # noqa: BLE001 - a role check must not break a turn
        return Readiness(UNKNOWN, f"could not check {binding.model}: {exc}")
    if found:
        return Readiness(READY, f"{binding.model} is downloaded")
    return Readiness(
        MISSING,
        f"{binding.model} is bound to this role but is not downloaded",
        missing = binding.model,
        remedy = "download the model, or bind the role to one you have",
    )
