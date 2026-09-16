# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The concrete probes.

Every probe delegates to a helper that already exists. Reimplementing those
checks is how the audit log's redaction layer drifted from the upstream
classifier it was supposed to reuse.
"""

from __future__ import annotations

import pytest

from core.inference import tool_readiness as tr
from core.inference.tool_readiness import probes


@pytest.fixture(autouse = True)
def _clean():
    tr.reset_for_tests()
    probes.install_default_probes()
    yield
    tr.reset_for_tests()


def test_trivially_ready_tools_are_ready():
    for name in ("terminal", "python", "edit_file", "render_html", "search_conversation"):
        assert tr.resolve(name).state == tr.READY, name


def test_search_knowledge_base_is_unknown_in_v1():
    """Deliberate: storage/ exposes no cheap KB-count helper, so we do not guess."""
    assert tr.resolve("search_knowledge_base").state == tr.UNKNOWN


def test_web_search_missing_when_ddgs_is_absent(monkeypatch):
    monkeypatch.setattr(probes, "_module_present", lambda name: False)
    r = tr.resolve("web_search", refresh = True)
    assert r.state == tr.MISSING
    assert r.missing == "ddgs"


def test_web_search_ready_when_ddgs_is_present(monkeypatch):
    """Control, and the real state on a correctly built install."""
    monkeypatch.setattr(probes, "_module_present", lambda name: True)
    assert tr.resolve("web_search", refresh = True).state == tr.READY


def test_web_search_does_not_claim_network_reachability():
    """ready means 'the package is present'. Saying more would be a lie the
    spec's no-network-calls rule makes unavoidable -- so say it out loud."""
    r = probes._probe_web_search()
    assert "network" in r.detail.lower()


def test_webcam_look_missing_when_the_weight_is_absent(monkeypatch):
    monkeypatch.setattr(probes, "_yolo_weight_present", lambda: False)
    r = tr.resolve("webcam_look", refresh = True)
    assert r.state == tr.MISSING
    assert r.missing and "yolov8n" in r.missing


def test_webcam_look_ready_when_the_weight_is_present(monkeypatch):
    """Control: without this, a probe hardcoded to 'missing' would pass above."""
    monkeypatch.setattr(probes, "_yolo_weight_present", lambda: True)
    assert tr.resolve("webcam_look", refresh = True).state == tr.READY


def test_face_swap_missing_until_the_licence_is_accepted(monkeypatch):
    monkeypatch.setattr(probes, "_face_swap_licence_accepted", lambda: False)
    r = tr.resolve("face_swap", refresh = True)
    assert r.state == tr.MISSING
    assert r.remedy and "licence" in r.remedy.lower()


def test_face_swap_ready_once_accepted(monkeypatch):
    monkeypatch.setattr(probes, "_face_swap_licence_accepted", lambda: True)
    assert tr.resolve("face_swap", refresh = True).state == tr.READY


def test_code_tool_reports_per_language_breakdown(monkeypatch):
    monkeypatch.setattr(probes, "_language_servers", lambda: {"typescript": True, "csharp": False})
    r = tr.resolve("code_definition", refresh = True)
    assert r.state == tr.READY, "usable for at least one language"
    assert "typescript" in r.detail and "csharp" in r.detail


def test_code_tool_missing_when_no_language_server_is_installed(monkeypatch):
    monkeypatch.setattr(probes, "_language_servers", lambda: {"typescript": False, "csharp": False})
    r = tr.resolve("code_definition", refresh = True)
    assert r.state == tr.MISSING


def test_probe_names_match_real_tools():
    """A probe registered under a name no tool has is dead code that looks alive."""
    from core.inference.tools import ALL_TOOLS

    real = {t["function"]["name"] for t in ALL_TOOLS}
    for name in tr.registered_names():
        assert name in real, f"probe registered for unknown tool {name!r}"
