# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The check_tool_readiness tool, its dispatch, and the _ALWAYS_SAFE_TOOLS rebind.

The rebind test is behavioural on purpose. It depends on _ALWAYS_SAFE_TOOLS being
read as a module global at CALL time; if that were wrong the rebind would
silently do nothing, every readiness query would prompt the user for approval,
and no other test would notice. Asserting the rebind's presence rather than its
effect is exactly the inert control this project has produced nine times.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from core.inference import tool_audit
from core.inference import tools
from storage import tool_audit_db


@pytest.fixture(autouse = True)
def _isolated(tmp_path, monkeypatch):
    """REQUIRED, and not optional here.

    execute_tool is the audit shadow, so calling it writes an audit row. Without
    this redirect those rows land in the real ~/.unsloth/studio/studio.db -- and
    because the recorder is guarded to never raise, it would do so silently.
    The existing test_tool_audit_seam.py needs no fixture only because its checks
    are static; these actually invoke the tool.
    """
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    tool_audit_db.reset_for_tests()
    tool_audit.reset_degraded_for_tests()
    yield


def test_the_tool_is_registered_in_ALL_TOOLS():
    names = {t["function"]["name"] for t in tools.ALL_TOOLS}
    assert "check_tool_readiness" in names


def test_execute_tool_dispatches_to_the_readiness_handler():
    out = tools.execute_tool("check_tool_readiness", {})
    assert isinstance(out, str) and out.strip(), "should return a non-empty report"
    assert "terminal" in out, "the report should list known tools"


def test_asking_about_one_tool_reports_only_that_tool():
    out = tools.execute_tool("check_tool_readiness", {"tool": "terminal"})
    assert "terminal" in out
    assert "webcam_look" not in out


def test_an_unknown_tool_name_is_reported_not_an_error():
    out = tools.execute_tool("check_tool_readiness", {"tool": "no_such_tool"})
    assert "unknown" in out.lower()


def test_readiness_is_not_treated_as_high_risk():
    """THE load-bearing test. Without the additive _ALWAYS_SAFE_TOOLS rebind,
    unknown tools fail closed and every readiness query prompts the user."""
    assert tools.is_high_risk_tool_call("check_tool_readiness", {}) is False


def test_a_tool_outside_the_safe_set_IS_still_high_risk():
    """Control: proves the assertion above is not passing because everything is
    considered safe."""
    assert tools.is_high_risk_tool_call("zz_not_a_real_tool", {}) is True


def test_the_seam_stays_additive():
    repo = Path(tools.__file__).parents[4]
    base = subprocess.run(
        ["git", "merge-base", "origin/main", "HEAD"],
        cwd = repo, capture_output = True, text = True, check = True,
    ).stdout.strip()
    stat = subprocess.run(
        ["git", "diff", "--numstat", base, "--", "studio/backend/core/inference/tools.py"],
        cwd = repo, capture_output = True, text = True, check = True,
    ).stdout.split()
    assert stat, "no diff recorded for tools.py"
    insertions, deletions = int(stat[0]), int(stat[1])
    assert deletions == 0, f"tools.py lost {deletions} line(s); the seam must be additive"
    assert insertions <= 90, f"seam grew to {insertions}; budget is ~84"
