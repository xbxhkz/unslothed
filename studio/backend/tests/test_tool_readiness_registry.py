# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The readiness registry.

Three states, not a boolean. A tool nobody probed reports "unknown" -- treating
it as "ready" would let the model read "nobody checked" as "verified working",
which is the one outcome that makes this feature worse than not having it.
"""

from __future__ import annotations

import pytest

from core.inference import tool_readiness as tr


@pytest.fixture(autouse = True)
def _clean():
    tr.reset_for_tests()
    yield
    tr.reset_for_tests()


def test_unregistered_tool_is_unknown_not_ready():
    r = tr.resolve("no_such_tool")
    assert r.state == "unknown"
    assert r.missing is None


def test_a_registered_tool_is_NOT_unknown():
    """Control for the test above: if everything returned unknown it would pass."""
    tr.register("t", lambda: tr.Readiness("ready", "fine", None, None))
    assert tr.resolve("t").state == "ready"


def test_missing_carries_what_is_missing_and_how_to_get_it():
    tr.register(
        "t",
        lambda: tr.Readiness("missing", "weights absent", "yolov8n.pt", "it downloads on first use"),
    )
    r = tr.resolve("t")
    assert r.state == "missing"
    assert r.missing == "yolov8n.pt"
    assert r.remedy == "it downloads on first use"


def test_a_raising_probe_reports_unknown_and_does_not_propagate():
    def boom():
        raise OSError("disk gone")

    tr.register("t", boom)
    r = tr.resolve("t")
    assert r.state == "unknown"
    assert "disk gone" in r.detail


def test_results_are_cached_within_the_ttl():
    calls = []

    def probe():
        calls.append(1)
        return tr.Readiness("ready", "ok", None, None)

    tr.register("t", probe)
    tr.resolve("t")
    tr.resolve("t")
    assert len(calls) == 1, "second resolve inside the TTL must not re-probe"


def test_refresh_bypasses_the_cache():
    calls = []

    def probe():
        calls.append(1)
        return tr.Readiness("ready", "ok", None, None)

    tr.register("t", probe)
    tr.resolve("t")
    tr.resolve("t", refresh = True)
    assert len(calls) == 2


def test_register_default_does_NOT_overwrite_an_existing_probe():
    """Load-bearing. install_default_probes() runs on every readiness query and
    every enriched failure, so if defaults clobbered explicit registrations, any
    caller-registered probe -- including a test's -- would be silently replaced
    the moment the feature was exercised."""
    tr.register("t", lambda: tr.Readiness("missing", "explicit", "x", None))
    tr.register_default("t", lambda: tr.Readiness("ready", "default", None, None))
    assert tr.resolve("t").detail == "explicit"


def test_register_default_DOES_register_an_absent_probe():
    """Control: without this, a register_default that did nothing at all would
    pass the test above."""
    tr.register_default("t", lambda: tr.Readiness("ready", "default", None, None))
    assert tr.resolve("t").detail == "default"


def test_resolve_all_returns_every_registered_tool():
    tr.register("a", lambda: tr.Readiness("ready", "", None, None))
    tr.register("b", lambda: tr.Readiness("missing", "", "x", None))
    out = tr.resolve_all()
    assert set(out) == {"a", "b"}
    assert out["b"].state == "missing"
