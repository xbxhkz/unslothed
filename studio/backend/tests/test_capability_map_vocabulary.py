# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The curated vocabulary must stay honest as the codebase grows.

A new tool without a capability fails here -- the intended forcing function. An
acquire hint that turns into an install command fails here too, because master
spec §4 says downloads must be identified, verified and presented to the user,
and a paste-ready command would let the model install through `terminal`.
"""

from __future__ import annotations

import re

from core.inference import capability_map as cm
from core.inference.capability_map.vocabulary import CAPABILITIES

_DISCOVERY_TOOLS = {"check_tool_readiness", "find_capability"}

_INSTALL_COMMAND = re.compile(
    r"pip3?\s+install|\bwinget\b|\bchoco\b|\bnpm\s+i(nstall)?\b|\bapt(-get)?\s+install|"
    r"\bapt-get\b|\bbrew\s+install|\bconda\s+install|\bcurl\b|dotnet\s+tool\s+install",
    re.IGNORECASE,
)

_CHECKABLE = {
    "tool": None,  # any real tool name
    "module": None,  # any module name
    "binary": None,  # any program name
    "model": {"vision"},
    "cached_model": {"whisper"},
}


def _tool_names_required():
    for capability in CAPABILITIES:
        for provider in capability.providers:
            for requirement in cm.requirements_for(provider):
                if requirement.kind == "tool":
                    yield requirement.name


def test_the_vocabulary_is_the_specified_capabilities_in_order():
    """The spec pinned 22. delegate_to_another_model is the 23rd, added when the
    delegation branch landed: the map's promise is that EVERY tool is findable by
    capability, so registering ask_model without an entry breaks it -- which is
    exactly how this was caught, by running both suites on the merged tree rather
    than each branch's own."""
    assert [c.name for c in CAPABILITIES] == [
        "internet_search", "run_python", "run_commands", "filesystem", "pdf_analysis",
        "ocr", "image_understanding", "object_detection", "camera", "image_processing",
        "image_editing", "background_removal", "face_swap", "image_generation",
        "video_editing", "speech_to_text", "word_documents", "document_search",
        "conversation_recall", "code_intelligence", "visual_output",
        "delegate_to_another_model", "computer_control",
    ]


def test_every_tool_requirement_names_a_real_tool():
    from core.inference.tools import ALL_TOOLS

    real = {t["function"]["name"] for t in ALL_TOOLS}
    for name in _tool_names_required():
        assert name in real, f"the vocabulary requires tool {name!r}, which does not exist"


def test_every_current_tool_is_findable_by_capability():
    """The vocabulary's promise: every tool maps to some capability."""
    from core.inference.tools import ALL_TOOLS

    required = set(_tool_names_required())
    for tool in ALL_TOOLS:
        name = tool["function"]["name"]
        if name in _DISCOVERY_TOOLS:
            continue
        assert name in required, f"tool {name!r} is in no capability"


def test_no_name_or_alias_belongs_to_two_capabilities():
    seen = {}
    for capability in CAPABILITIES:
        for key in (capability.name, *capability.aliases):
            normalized = cm.normalize(key)
            assert normalized not in seen or seen[normalized] == capability.name, (
                f"{key!r} matches both {seen.get(normalized)!r} and {capability.name!r}"
            )
            seen[normalized] = capability.name


def test_no_acquire_hint_contains_an_install_command():
    for capability in CAPABILITIES:
        if capability.acquire:
            assert not _INSTALL_COMMAND.search(capability.acquire), (
                f"{capability.name}: acquire hint is a runnable command: {capability.acquire!r}"
            )


def test_a_capability_nothing_provides_says_what_would_provide_it():
    for capability in CAPABILITIES:
        if not capability.providers:
            assert capability.acquire, f"{capability.name} has no providers and no acquire hint"


def test_every_provider_has_a_reason_and_something_to_check():
    for capability in CAPABILITIES:
        for provider in capability.providers:
            assert provider.reason.strip(), f"{capability.name}/{provider.name} has no reason"
            assert cm.requirements_for(provider), f"{capability.name}/{provider.name} checks nothing"


def test_every_requirement_is_one_the_checks_can_answer():
    """A requirement no check handles is 'unknown' forever -- silently."""
    for capability in CAPABILITIES:
        for provider in capability.providers:
            for requirement in provider.requires:
                assert requirement.kind in _CHECKABLE, (capability.name, requirement)
                allowed = _CHECKABLE[requirement.kind]
                if allowed is not None:
                    assert requirement.name in allowed, (capability.name, requirement)


def test_image_understanding_does_not_offer_detect_shapes_as_a_provider():
    """detect_shapes only returns object labels and confidences; it cannot answer
    an open question about an image, so it must not be listed here -- the vision
    model is image_understanding's only provider. detect_shapes stays reachable
    as object_detection's first-preference provider."""
    capability = cm.find("image_understanding", CAPABILITIES)
    tool_names = {p.name for p in capability.providers if p.kind == "tool"}
    assert "detect_shapes" not in tool_names


def test_speech_to_text_requires_whisper_ffmpeg_and_a_cached_model():
    """Whisper imports cleanly without ffmpeg, then fails on every file."""
    capability = cm.find("speech_to_text", CAPABILITIES)
    requires = {(r.kind, r.name) for p in capability.providers for r in p.requires}
    assert {("module", "whisper"), ("binary", "ffmpeg"), ("cached_model", "whisper")} <= requires


def test_the_master_spec_s37_questions_find_a_capability():
    for question, expected in (
        ("manipulate images", "image_processing"),
        ("edit videos", "video_editing"),
        ("access the filesystem", "filesystem"),
        ("execute python", "run_python"),
        ("search the internet", "internet_search"),
        ("control the computer", "computer_control"),
        ("generate images", "image_generation"),
        ("analyze pdfs", "pdf_analysis"),
        ("OCR", "ocr"),
    ):
        found = cm.find(question, CAPABILITIES)
        assert found is not None and found.name == expected, question
