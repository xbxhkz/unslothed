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
    for name in ("terminal", "python", "edit_file", "render_html", "check_tool_readiness"):
        assert tr.resolve(name).state == tr.READY, name


def test_search_conversation_is_not_asserted_ready_without_a_check():
    """It used to be in ALWAYS_READY_TOOLS, which is a bare `return READY`.

    The tool is gated on conversation_archive.enabled(), so that was a false
    'ready' -- and the tool's own refusal string carries no 'Error:' prefix, so
    the enrichment path could not correct it either. Both answers were wrong at
    once."""
    assert "search_conversation" not in probes.ALWAYS_READY_TOOLS


def test_search_conversation_missing_when_the_archive_is_disabled(monkeypatch):
    monkeypatch.setattr(probes, "_conversation_archive_enabled", lambda: False)
    r = tr.resolve("search_conversation", refresh = True)
    assert r.state == tr.MISSING
    assert r.remedy and "archive" in r.remedy.lower()


def test_search_conversation_ready_when_the_archive_is_enabled(monkeypatch):
    """Control: without this, a probe hardcoded to 'missing' would pass above."""
    monkeypatch.setattr(probes, "_conversation_archive_enabled", lambda: True)
    assert tr.resolve("search_conversation", refresh = True).state == tr.READY


def test_search_conversation_probe_delegates_to_the_archives_own_gate(monkeypatch):
    """Reimplementing `CONVERSATION_ARCHIVE and rag_available()` here is how the
    audit log's redaction layer drifted from the classifier it was meant to
    reuse. Assert the delegation itself, not a value that a copy would also
    produce: patching conversation_archive.enabled must change the answer."""
    from core.rag import conversation_archive

    monkeypatch.setattr(conversation_archive, "enabled", lambda: False)
    assert tr.resolve("search_conversation", refresh = True).state == tr.MISSING
    monkeypatch.setattr(conversation_archive, "enabled", lambda: True)
    assert tr.resolve("search_conversation", refresh = True).state == tr.READY


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


def test_web_search_does_not_claim_network_reachability(monkeypatch):
    """ready means 'the package is present'. Saying more would be a lie the
    spec's no-network-calls rule makes unavoidable -- so say it out loud.

    BOTH branches, monkeypatched. Unpatched this only ever walked the READY
    branch on an install that has ddgs, so the MISSING branch's wording was
    untested: deleting the phrase there left every test passing."""
    monkeypatch.setattr(probes, "_module_present", lambda name: True)
    assert "network" in probes._probe_web_search().detail.lower()
    monkeypatch.setattr(probes, "_module_present", lambda name: False)
    assert "network" in probes._probe_web_search().detail.lower()


def test_webcam_look_missing_when_the_weight_is_absent(monkeypatch):
    # _module_present pinned True so this asserts the WEIGHT branch on any
    # machine, rather than passing for the wrong reason where ultralytics or cv2
    # happen to be uninstalled.
    monkeypatch.setattr(probes, "_module_present", lambda name: True)
    monkeypatch.setattr(probes, "_yolo_weight_present", lambda: False)
    r = tr.resolve("webcam_look", refresh = True)
    assert r.state == tr.MISSING
    assert r.missing and "yolov8n" in r.missing


def test_webcam_look_ready_when_the_weight_is_present(monkeypatch):
    """Control: without this, a probe hardcoded to 'missing' would pass above."""
    monkeypatch.setattr(probes, "_module_present", lambda name: True)
    monkeypatch.setattr(probes, "_yolo_weight_present", lambda: True)
    assert tr.resolve("webcam_look", refresh = True).state == tr.READY


@pytest.fixture
def _face_swap_all_present(monkeypatch):
    """The everything-installed baseline, so each test below flips one thing."""
    monkeypatch.setattr(probes, "_face_swap_licence_accepted", lambda: True)
    monkeypatch.setattr(probes, "_face_swap_models_present", lambda: True)
    monkeypatch.setattr(probes, "_module_present", lambda name: True)


def test_face_swap_missing_until_the_licence_is_accepted(monkeypatch, _face_swap_all_present):
    monkeypatch.setattr(probes, "_face_swap_licence_accepted", lambda: False)
    r = tr.resolve("face_swap", refresh = True)
    assert r.state == tr.MISSING
    assert r.remedy and "licence" in r.remedy.lower()


def test_face_swap_missing_when_the_models_are_not_downloaded(monkeypatch, _face_swap_all_present):
    """THE finding: licence_accepted() writes a marker and downloads nothing, so
    answering 'ready' on it alone claimed ~300 MB of weights were present on an
    install that had never fetched them -- and contradicted webcam_look, which
    calls the identical situation 'missing'. The spec's own worked example line
    'face_swap  missing  InsightFace models not downloaded' could not be produced
    by the old probe at all."""
    monkeypatch.setattr(probes, "_face_swap_models_present", lambda: False)
    r = tr.resolve("face_swap", refresh = True)
    assert r.state == tr.MISSING
    assert "not downloaded" in r.detail
    assert r.missing and "inswapper" in r.missing
    assert r.remedy and "300 MB" in r.remedy


def test_face_swap_missing_when_insightface_is_not_importable(
    monkeypatch, _face_swap_all_present
):
    """A frozen build missing a lazily-imported module is this project's most
    common defect class; _do_face_swap reaches `from insightface.app import
    FaceAnalysis`."""
    monkeypatch.setattr(probes, "_module_present", lambda name: name != "insightface")
    r = tr.resolve("face_swap", refresh = True)
    assert r.state == tr.MISSING
    assert r.missing == "insightface"


def test_face_swap_reports_the_licence_before_the_models(monkeypatch, _face_swap_all_present):
    """Order is deliberate: an unaccepted licence is the more actionable answer,
    because nothing can be downloaded until the user accepts."""
    monkeypatch.setattr(probes, "_face_swap_licence_accepted", lambda: False)
    monkeypatch.setattr(probes, "_face_swap_models_present", lambda: False)
    assert "licence" in tr.resolve("face_swap", refresh = True).detail.lower()


def test_face_swap_ready_only_when_licence_models_and_package_are_all_present(
    _face_swap_all_present,
):
    """Control: proves the four 'missing' assertions above are not passing
    because the probe now always says missing."""
    assert tr.resolve("face_swap", refresh = True).state == tr.READY


def test_face_swap_model_check_delegates_to_face_swap_itself(monkeypatch):
    """The filenames and the InsightFace root convention are face_swap.py's, not
    this package's. Patching the owner must change the answer."""
    from core.inference.assist_vision import face_swap

    monkeypatch.setattr(face_swap, "models_present", lambda: False)
    assert probes._face_swap_models_present() is False
    monkeypatch.setattr(face_swap, "models_present", lambda: True)
    assert probes._face_swap_models_present() is True


def test_webcam_look_missing_when_a_lazily_imported_package_is_absent(monkeypatch):
    """_do_webcam_look imports cv2 (webcam.capture_frame_jpeg) and ultralytics
    (yolo.detect). A cached weight is a useless 'ready' if neither can load."""
    monkeypatch.setattr(probes, "_yolo_weight_present", lambda: True)
    for absent in ("ultralytics", "cv2"):
        monkeypatch.setattr(probes, "_module_present", lambda name, a = absent: name != a)
        r = tr.resolve("webcam_look", refresh = True)
        assert r.state == tr.MISSING, absent
        assert r.missing == absent


def test_code_tool_reports_per_language_breakdown(monkeypatch):
    monkeypatch.setattr(probes, "_language_servers", lambda: {"typescript": True, "csharp": False})
    r = tr.resolve("code_definition", refresh = True)
    assert r.state == tr.READY, "usable for at least one language"
    assert "typescript" in r.detail and "csharp" in r.detail


def test_code_tool_missing_when_no_language_server_is_installed(monkeypatch):
    monkeypatch.setattr(probes, "_language_servers", lambda: {"typescript": False, "csharp": False})
    r = tr.resolve("code_definition", refresh = True)
    assert r.state == tr.MISSING


def test_the_no_argument_report_covers_every_real_tool():
    """THE finding: resolve_all() walks the REGISTRY, so the report rendered 14
    rows for 18 tools and detect_shapes, edit_image_prompt and remove_background
    were invisible. Absence from a table is not read as 'unknown' -- it is read
    as nothing, which is exactly the optimistic lie the three-state contract
    exists to prevent."""
    from core.inference.tools import ALL_TOOLS

    report = tr.execute("check_tool_readiness", {})
    lines = {line.split()[0] for line in report.splitlines() if not line.startswith(" ")}
    for tool in ALL_TOOLS:
        assert tool["function"]["name"] in lines, tool["function"]["name"]


def test_unprobed_tools_are_reported_as_unknown_rows_not_omitted():
    """The row must actually say 'unknown'; merely appearing is not enough."""
    report = tr.execute("check_tool_readiness", {})
    rows = {
        line.split()[0]: line
        for line in report.splitlines()
        if not line.startswith(" ")
    }
    for name in ("detect_shapes", "edit_image_prompt", "remove_background"):
        assert "unknown" in rows[name], rows[name]


def test_probe_names_match_real_tools():
    """A probe registered under a name no tool has is dead code that looks alive."""
    from core.inference.tools import ALL_TOOLS

    real = {t["function"]["name"] for t in ALL_TOOLS}
    for name in tr.registered_names():
        assert name in real, f"probe registered for unknown tool {name!r}"
