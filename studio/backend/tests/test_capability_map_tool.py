# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What find_capability says to the model."""

from __future__ import annotations

import re

import pytest

from core.inference import capability_map as cm
from core.inference import tool_readiness as tr
from core.inference.capability_map import providers
from core.inference.capability_map.vocabulary import CAPABILITIES
from core.inference.tool_readiness import MISSING, READY, UNKNOWN, Readiness


@pytest.fixture(autouse = True)
def _clean():
    tr.reset_for_tests()
    yield
    tr.reset_for_tests()


def _by_requirement(monkeypatch, states, default):
    """states keys are 'kind:name'; anything else resolves to `default`."""

    def fake(req):
        state = states.get(f"{req.kind}:{req.name}", default)
        return Readiness(state, f"{req.kind}:{req.name} is {state}")

    monkeypatch.setattr(providers, "check_requirement", fake)


def _row_index(report, capability_name):
    for index, line in enumerate(report.splitlines()):
        parts = line.split()
        if parts and parts[0] == capability_name:
            return index
    raise AssertionError(f"{capability_name} not in report")


def test_one_capability_renders_in_the_master_spec_format(monkeypatch):
    _by_requirement(monkeypatch, {}, READY)
    out = cm.execute("find_capability", {"capability": "ocr"})
    assert "CAPABILITY: ocr — Read text in images" in out
    assert "Status:      ready" in out
    assert "Best option: vision model" in out
    assert "Reason:      understands layout; nothing to install" in out
    assert "  1. vision model" in out
    assert "  2. Tesseract via terminal" in out


def test_a_tool_provider_renders_as_via_tool_name(monkeypatch):
    """Load-bearing for finding 1's mitigation: the model matches "via tool X"
    against its own tool list, so the exact phrasing must not drift."""
    _by_requirement(monkeypatch, {}, READY)
    out = cm.execute("find_capability", {"capability": "run_python"})
    assert "via tool python" in out


def test_the_best_option_skips_an_unknown_first_provider(monkeypatch):
    _by_requirement(
        monkeypatch,
        {"model:vision": UNKNOWN, "tool:terminal": READY, "binary:tesseract": READY},
        MISSING,
    )
    out = cm.execute("find_capability", {"capability": "ocr"})
    assert "Best option: Tesseract via terminal" in out


def test_a_missing_capability_says_how_to_get_it(monkeypatch):
    _by_requirement(monkeypatch, {}, MISSING)
    out = cm.execute("find_capability", {"capability": "video editing"})
    assert "Status:      missing" in out
    assert "To get it:   Needs FFmpeg" in out


def test_an_alias_finds_its_capability(monkeypatch):
    _by_requirement(monkeypatch, {}, MISSING)
    assert "CAPABILITY: speech_to_text" in cm.execute("find_capability", {"capability": "transcribe"})


def test_a_capability_with_no_providers_says_so(monkeypatch):
    _by_requirement(monkeypatch, {}, READY)
    out = cm.execute("find_capability", {"capability": "computer_control"})
    assert "Status:      missing" in out
    assert "Providers:   none yet" in out


def test_the_whole_map_lists_ready_then_unknown_then_missing(monkeypatch):
    _by_requirement(
        monkeypatch,
        {"tool:terminal": READY, "binary:ffmpeg": READY, "model:vision": UNKNOWN},
        MISSING,
    )
    out = cm.execute("find_capability", {})
    ready = _row_index(out, "video_editing")
    unknown = _row_index(out, "image_understanding")
    missing = _row_index(out, "computer_control")
    assert ready < unknown < missing, out


def test_the_whole_map_covers_every_capability_and_notes_mcp(monkeypatch):
    _by_requirement(monkeypatch, {}, MISSING)
    out = cm.execute("find_capability", {})
    for capability in CAPABILITIES:
        _row_index(out, capability.name)
    assert "MCP tools are not in this map" in out


def test_an_unmatched_query_lists_the_capabilities_instead_of_failing(monkeypatch):
    _by_requirement(monkeypatch, {}, MISSING)
    out = cm.execute("find_capability", {"capability": "teleportation"})
    assert "No capability matches 'teleportation'" in out
    for capability in CAPABILITIES:
        assert capability.name in out


def test_execute_never_raises_on_bad_arguments(monkeypatch):
    _by_requirement(monkeypatch, {}, MISSING)

    class Unprintable:
        def __str__(self):
            raise RuntimeError("cannot render")

    for arguments in (None, [], "ocr", {"capability": ["ocr"]}, {"capability": 5},
                      {"capability": {"x": 1}}, {"capability": Unprintable()}):
        out = cm.execute("find_capability", arguments)
        assert isinstance(out, str) and out, arguments


def test_output_never_carries_an_install_command_even_when_readiness_does():
    """§4 boundary, end to end: a real readiness probe whose remedy IS an install
    command must not leak into anything the map renders."""
    tr.register("web_search", lambda: Readiness(MISSING, "ddgs gone", remedy = "pip install ddgs"))
    single = cm.execute("find_capability", {"capability": "internet_search"})
    whole = cm.execute("find_capability", {})
    for out in (single, whole):
        assert not re.search(r"pip3?\s+install", out), out


def test_the_schema_matches_what_execute_accepts():
    from core.inference.capability_map.schemas import (
        CAPABILITY_TOOL_NAMES,
        CAPABILITY_TOOLS,
        FIND_CAPABILITY_TOOL,
    )

    function = FIND_CAPABILITY_TOOL["function"]
    assert function["name"] == "find_capability"
    assert set(function["parameters"]["properties"]) == {"capability"}
    assert function["parameters"]["required"] == []
    assert CAPABILITY_TOOLS == [FIND_CAPABILITY_TOOL]
    assert CAPABILITY_TOOL_NAMES == frozenset({"find_capability"})
