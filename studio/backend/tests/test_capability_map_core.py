# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Data model, matching and status rules for the capability map.

The rule everything rests on is carried up from tool readiness: an `unknown`
never makes anything `ready`. A capability is ready only when some provider was
actually checked and found ready.
"""

from __future__ import annotations

import pytest

from core.inference import capability_map as cm
from core.inference import tool_readiness as tr
from core.inference.capability_map import providers
from core.inference.tool_readiness import MISSING, READY, UNKNOWN, Readiness, probes


@pytest.fixture(autouse = True)
def _clean():
    tr.reset_for_tests()
    yield
    tr.reset_for_tests()


def _req(kind, name):
    return cm.Requirement(kind, name)


def _fake_checks(monkeypatch, states):
    """Every requirement resolves to states[req.name]."""

    def fake(req):
        return Readiness(states[req.name], f"{req.name} is {states[req.name]}")

    monkeypatch.setattr(providers, "check_requirement", fake)


def _provider(name, *module_names, via = None):
    return cm.Provider(
        "software" if via else "tool",
        name,
        via,
        tuple(_req("module", n) for n in module_names),
        f"reason for {name}",
    )


def _cap(*provs, acquire = None):
    return cm.Capability("c", "C", (), tuple(provs), acquire)


# --- matching ---------------------------------------------------------------


def test_normalize_ignores_case_and_separators():
    assert cm.normalize("  Video_Editing ") == "video editing"
    assert cm.normalize("speech-to-text") == "speech to text"


def test_find_matches_a_name_or_an_alias():
    caps = (cm.Capability("ocr", "Read text", ("text recognition",), (), None),)
    assert cm.find("OCR", caps) is caps[0]
    assert cm.find("Text-Recognition", caps) is caps[0]


def test_find_returns_none_for_no_match_or_an_empty_query():
    caps = (cm.Capability("ocr", "Read text", (), (), None),)
    assert cm.find("video", caps) is None
    assert cm.find("   ", caps) is None


# --- provider status --------------------------------------------------------


def test_a_provider_is_ready_only_when_every_requirement_is(monkeypatch):
    _fake_checks(monkeypatch, {"a": READY, "b": READY})
    assert cm.resolve_provider(_provider("p", "a", "b")).state == READY


def test_one_missing_requirement_makes_the_provider_missing_even_beside_an_unknown(monkeypatch):
    _fake_checks(monkeypatch, {"a": UNKNOWN, "b": MISSING})
    result = cm.resolve_provider(_provider("p", "a", "b"))
    assert result.state == MISSING
    assert "b is missing" in result.detail


def test_an_unknown_requirement_without_a_missing_one_is_unknown(monkeypatch):
    _fake_checks(monkeypatch, {"a": READY, "b": UNKNOWN})
    assert cm.resolve_provider(_provider("p", "a", "b")).state == UNKNOWN


def test_a_provider_with_no_requirements_is_unknown_not_vacuously_ready():
    """all() over nothing is True. Reporting 'ready' for a provider nobody
    declared any requirement for is the optimistic lie in its purest form."""
    assert cm.resolve_provider(_provider("p")).state == UNKNOWN


def test_a_software_provider_needs_its_via_tool(monkeypatch):
    _fake_checks(monkeypatch, {"python": MISSING, "fitz": READY})
    assert cm.resolve_provider(_provider("PyMuPDF", "fitz", via = "python")).state == MISSING


def test_the_via_tool_being_ready_lets_the_software_provider_be_ready(monkeypatch):
    """Control for the test above."""
    _fake_checks(monkeypatch, {"python": READY, "fitz": READY})
    result = cm.resolve_provider(_provider("PyMuPDF", "fitz", via = "python"))
    assert result.state == READY
    assert result.detail == "fitz is ready", (
        "a ready provider reports its OWN requirements, not its via tool's"
    )


# --- capability status ------------------------------------------------------


def test_best_option_is_the_first_READY_provider_not_the_first_listed(monkeypatch):
    _fake_checks(monkeypatch, {"a": UNKNOWN, "b": READY, "c": READY})
    result = cm.resolve_capability(_cap(_provider("A", "a"), _provider("B", "b"), _provider("C", "c")))
    assert result.state == READY
    assert result.best.provider.name == "B"


def test_an_unknown_provider_never_makes_a_capability_ready(monkeypatch):
    _fake_checks(monkeypatch, {"a": UNKNOWN, "b": MISSING})
    result = cm.resolve_capability(_cap(_provider("A", "a"), _provider("B", "b")))
    assert result.state == UNKNOWN
    assert result.best is None


def test_every_provider_missing_makes_the_capability_missing(monkeypatch):
    _fake_checks(monkeypatch, {"a": MISSING, "b": MISSING})
    result = cm.resolve_capability(_cap(_provider("A", "a"), _provider("B", "b")))
    assert result.state == MISSING


def test_a_capability_with_no_providers_is_missing():
    assert cm.resolve_capability(_cap()).state == MISSING


# --- requirement checks -----------------------------------------------------


def test_a_module_requirement_delegates_to_the_readiness_module_check(monkeypatch):
    monkeypatch.setattr(probes, "_module_present", lambda name: name == "fitz")
    assert providers.check_requirement(_req("module", "fitz")).state == READY
    assert providers.check_requirement(_req("module", "easyocr")).state == MISSING


def test_a_binary_requirement_reads_PATH_and_explains_a_restart(monkeypatch):
    monkeypatch.setattr(
        providers, "_which", lambda name: "C:/bin/ffmpeg.exe" if name == "ffmpeg" else None
    )
    assert providers.check_requirement(_req("binary", "ffmpeg")).state == READY
    result = providers.check_requirement(_req("binary", "tesseract"))
    assert result.state == MISSING
    assert "restart" in result.detail, (
        "a program installed after startup is invisible until restart; say so"
    )


def test_a_tool_requirement_uses_tool_readiness_and_an_explicit_probe_wins():
    """install_default_probes() runs inside the check; register_default must not
    clobber this explicit registration (3a's Ruling 1)."""
    tr.register("web_search", lambda: Readiness(MISSING, "ddgs gone", remedy = "pip install ddgs"))
    result = providers.check_requirement(_req("tool", "web_search"))
    assert result.state == MISSING
    assert "ddgs gone" in result.detail


def test_a_tool_requirements_remedy_is_not_carried_into_the_map():
    """§4 boundary. 3a's remedies include install commands; the map never repeats
    them, so a model reading the map is never handed one."""
    tr.register("web_search", lambda: Readiness(MISSING, "ddgs gone", remedy = "pip install ddgs"))
    result = providers.check_requirement(_req("tool", "web_search"))
    assert result.remedy is None
    assert "pip install" not in result.detail


def test_an_unrecognised_requirement_kind_is_unknown():
    assert providers.check_requirement(_req("telepathy", "x")).state == UNKNOWN


def test_a_raising_check_reports_unknown_and_never_raises(monkeypatch):
    """Also proves dispatch looks checks up BY NAME at call time: if the dispatch
    table held function objects captured at import, this monkeypatch would not
    take effect and the real PATH lookup would run instead."""

    def boom(name):
        raise OverflowError("nope")

    monkeypatch.setattr(providers, "_check_binary", boom)
    result = providers.check_requirement(_req("binary", "ffmpeg"))
    assert result.state == UNKNOWN
    assert "nope" in result.detail
