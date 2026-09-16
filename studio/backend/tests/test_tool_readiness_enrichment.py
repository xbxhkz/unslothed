# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A failed call whose dependency is missing should say so.

Scoped tightly: only when the call FAILED and the probe reports MISSING. The
original text is never altered, only appended to. Successful calls are never
touched -- an audit/readiness layer that rewrites successful results would be a
behaviour change on a path that works.
"""

from __future__ import annotations

import pytest

from core.inference import tool_audit
from core.inference import tool_readiness as tr
from storage import tool_audit_db


@pytest.fixture(autouse = True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    tool_audit_db.reset_for_tests()
    tool_audit.reset_degraded_for_tests()
    tr.reset_for_tests()
    yield
    tr.reset_for_tests()


def _failing(name, arguments, **kwargs):
    return "Error: something went wrong"


def _succeeding(name, arguments, **kwargs):
    return "all good"


def test_a_failure_with_a_missing_dependency_is_explained():
    """The explicit tr.register below must survive the enricher's own
    install_default_probes() call -- that is what register_default (Task 1)
    guarantees. If defaults clobbered it, this test would resolve the REAL
    face_swap probe and fail on text it never wrote."""
    tr.register("face_swap", lambda: tr.Readiness(
        tr.MISSING, "licence not accepted",
        missing = "InsightFace licence", remedy = "accept it in settings",
    ))
    out = tool_audit.around(_failing, "face_swap", {})
    assert out.startswith("Error: something went wrong"), "original text must survive intact"
    assert "[readiness]" in out
    assert "InsightFace licence" in out
    assert "accept it in settings" in out


def test_a_SUCCESSFUL_call_is_never_enriched():
    """Control: enrichment must not touch a working path."""
    tr.register("face_swap", lambda: tr.Readiness(tr.MISSING, "x", missing = "y"))
    out = tool_audit.around(_succeeding, "face_swap", {})
    assert out == "all good"


def test_a_failure_whose_probe_says_ready_is_not_enriched():
    """Control: enrichment keys off MISSING, not off failure alone."""
    tr.register("terminal", lambda: tr.Readiness(tr.READY, "fine"))
    out = tool_audit.around(_failing, "terminal", {})
    assert out == "Error: something went wrong"


def test_an_unknown_readiness_does_not_enrich():
    out = tool_audit.around(_failing, "no_probe_tool", {})
    assert out == "Error: something went wrong"


def test_enrichment_failure_cannot_break_the_call(monkeypatch):
    def explode(*a, **k):
        raise OSError("probe subsystem down")

    monkeypatch.setattr(tr, "resolve", explode)
    out = tool_audit.around(_failing, "face_swap", {})
    assert out == "Error: something went wrong", "a broken enricher must not cost the caller its result"
