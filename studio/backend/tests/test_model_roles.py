# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Role bindings: a name like "coding" resolved to a concrete model.

Availability reuses tool readiness's three states deliberately. A role nobody
bound reports "unknown", never "ready" -- the same rule the readiness and
capability work rests on, for the same reason: the model must never read
"nobody checked" as "verified working".
"""

from __future__ import annotations

import pytest

from core.inference import model_roles
from core.inference.model_roles import storage
from core.inference.tool_readiness import MISSING, READY, UNKNOWN


@pytest.fixture(autouse = True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    yield


def _bind(monkeypatch, raw):
    monkeypatch.setattr(storage, "get_role_bindings", lambda: raw)


def test_a_binding_resolves_to_its_model(monkeypatch):
    _bind(monkeypatch, {"coding": {"model": "repo/Coder-GGUF:Q4_K_M"}})
    binding = model_roles.resolve("coding")
    assert binding is not None
    assert binding.model == "repo/Coder-GGUF:Q4_K_M"
    assert binding.overrides == {}


def test_overrides_are_carried_through(monkeypatch):
    _bind(monkeypatch, {"coding": {"model": "m", "overrides": {"n_ctx": 16384}}})
    assert model_roles.resolve("coding").overrides == {"n_ctx": 16384}


def test_role_names_are_normalized(monkeypatch):
    _bind(monkeypatch, {"coding": {"model": "m"}})
    assert model_roles.resolve("  Coding ") is not None
    assert model_roles.normalize_role("  Coding ") == "coding"


def test_an_unbound_role_resolves_to_none(monkeypatch):
    _bind(monkeypatch, {})
    assert model_roles.resolve("coding") is None


def test_a_malformed_binding_is_ignored_rather_than_raising(monkeypatch):
    _bind(monkeypatch, {"coding": "not-a-dict", "vision": {"no_model_key": 1}})
    assert model_roles.resolve("coding") is None
    assert model_roles.resolve("vision") is None


def test_an_unbound_role_is_unknown_not_ready(monkeypatch):
    """The central rule, at the role level."""
    _bind(monkeypatch, {})
    assert model_roles.availability("coding").state == UNKNOWN


def test_a_bound_downloaded_model_is_ready(monkeypatch):
    _bind(monkeypatch, {"coding": {"model": "m"}})
    monkeypatch.setattr(model_roles, "_resolve_local", lambda model: ("/models/m.gguf", None, "m"))
    result = model_roles.availability("coding")
    assert result.state == READY
    assert "m" in result.detail


def test_a_bound_model_that_is_not_downloaded_is_missing(monkeypatch):
    """Control for the test above."""
    _bind(monkeypatch, {"coding": {"model": "m"}})
    monkeypatch.setattr(model_roles, "_resolve_local", lambda model: None)
    result = model_roles.availability("coding")
    assert result.state == MISSING
    assert result.missing == "m"


def test_a_resolver_failure_is_unknown_and_never_raises(monkeypatch):
    _bind(monkeypatch, {"coding": {"model": "m"}})

    def boom(model):
        raise OSError("index unavailable")

    monkeypatch.setattr(model_roles, "_resolve_local", boom)
    result = model_roles.availability("coding")
    assert result.state == UNKNOWN
    assert "index unavailable" in result.detail


def test_bindings_returns_every_valid_role(monkeypatch):
    _bind(monkeypatch, {"coding": {"model": "a"}, "vision": {"model": "b"}, "bad": 3})
    assert set(model_roles.bindings()) == {"coding", "vision"}


def test_the_default_role_names_are_offered_but_not_required(monkeypatch):
    _bind(monkeypatch, {"my_custom_role": {"model": "m"}})
    assert "primary" in model_roles.DEFAULT_ROLE_NAMES
    assert model_roles.resolve("my_custom_role") is not None, "custom roles must work"


def test_settings_round_trip_through_the_app_setting(monkeypatch):
    saved = {}
    monkeypatch.setattr(storage, "_read_setting", lambda: saved.get("v"))
    monkeypatch.setattr(storage, "_write_setting", lambda value: saved.__setitem__("v", value))
    storage.set_role_bindings({"coding": {"model": "m", "overrides": {}}})
    assert storage.get_role_bindings() == {"coding": {"model": "m", "overrides": {}}}
