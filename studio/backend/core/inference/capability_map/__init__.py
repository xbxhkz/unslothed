# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which tools, installed software or model abilities can do something, which is
best, and whether it works right now (master spec §37/§38).

A CAPABILITY ("ocr") lists PROVIDERS in curated preference order. A provider is a
tool, software reached through the python/terminal tools, or an ability of the
loaded model, and it declares REQUIREMENTS. Every requirement resolves to tool
readiness's three states, and these rules roll them up:

  provider   -- ready: every requirement ready (and there is at least one)
               missing: any requirement missing
               unknown: otherwise
  capability -- ready: some provider ready; BEST = the first ready one in order
               missing: every provider missing, or there are none
               unknown: otherwise

An `unknown` never makes anything `ready`, at either level. That is readiness's
central claim carried up: the model must never read "nobody checked" as
"verified working".

This package depends on tool_readiness; tool_readiness must never depend on it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

from core.inference.tool_readiness import MISSING, READY, UNKNOWN


@dataclass(frozen = True)
class Requirement:
    kind: str  # "tool" | "module" | "binary" | "model" | "cached_model"
    name: str


@dataclass(frozen = True)
class Provider:
    kind: str  # "tool" | "software" | "model" -- for display
    name: str
    via: Optional[str]  # "python" / "terminal" for software; None otherwise
    requires: tuple[Requirement, ...]
    reason: str


@dataclass(frozen = True)
class Capability:
    name: str
    title: str
    aliases: tuple[str, ...]
    providers: tuple[Provider, ...]  # curated preference order; may be empty
    acquire: Optional[str]  # §4 hook -- descriptive text, never a runnable command


@dataclass(frozen = True)
class ProviderResult:
    provider: Provider
    state: str
    detail: str


@dataclass(frozen = True)
class CapabilityResult:
    capability: Capability
    state: str
    best: Optional[ProviderResult]
    providers: tuple[ProviderResult, ...]


_SEPARATORS = re.compile(r"[\s_\-]+")


def normalize(text) -> str:
    """Case-insensitive, and spaces, underscores and hyphens are the same thing,
    so "video editing", "video_editing" and "Video-Editing" all match. Still an
    exact match after that -- no fuzzy or model-based guessing."""
    return _SEPARATORS.sub(" ", str(text)).strip().lower()


def find(query, capabilities: Sequence[Capability]) -> Optional[Capability]:
    key = normalize(query)
    if not key:
        return None
    for capability in capabilities:
        if key == normalize(capability.name):
            return capability
        if any(key == normalize(alias) for alias in capability.aliases):
            return capability
    return None


def requirements_for(provider: Provider) -> tuple[Requirement, ...]:
    """A software provider's `via` tool is an implicit first requirement:
    PyMuPDF is only usable if the python tool is."""
    implicit = (Requirement("tool", provider.via),) if provider.via else ()
    return implicit + tuple(provider.requires)


def resolve_provider(provider: Provider) -> ProviderResult:
    # Looked up at call time, so tests can substitute check_requirement.
    from core.inference.capability_map import providers as _providers

    requirements = requirements_for(provider)
    if not requirements:
        return ProviderResult(provider, UNKNOWN, "no requirement is declared for this provider")
    checked = [(req, _providers.check_requirement(req)) for req in requirements]
    for _req, readiness in checked:
        if readiness.state == MISSING:
            return ProviderResult(provider, MISSING, readiness.detail)
    for _req, readiness in checked:
        if readiness.state != READY:
            return ProviderResult(provider, UNKNOWN, readiness.detail)
    own = [readiness.detail for req, readiness in checked if req in provider.requires]
    return ProviderResult(provider, READY, "; ".join(own) if own else checked[-1][1].detail)


def resolve_capability(capability: Capability) -> CapabilityResult:
    results = tuple(resolve_provider(p) for p in capability.providers)
    for result in results:
        if result.state == READY:
            return CapabilityResult(capability, READY, result, results)
    # all() over no providers is True: a capability nothing provides is missing.
    if all(result.state == MISSING for result in results):
        return CapabilityResult(capability, MISSING, None, results)
    return CapabilityResult(capability, UNKNOWN, None, results)
