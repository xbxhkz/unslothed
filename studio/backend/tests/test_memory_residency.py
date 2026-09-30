# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the RAM-tier park/restore coordinator. No real backend, GPU, or model --
every owner is a fake with recorder callables, matching tests/test_gpu_arbiter.py's pattern."""

from __future__ import annotations

import pytest

import core.inference.memory_residency as mr


@pytest.fixture(autouse = True)
def _reset(monkeypatch):
    monkeypatch.setattr(mr, "_parked", {})
    monkeypatch.setattr("utils.memory_park_settings.get_ram_park_budget_mib", lambda: 10000)


def _fake_owner(name, calls, park_ok = True, footprint = 1000):
    def park():
        calls.append(f"park-{name}")
        return park_ok

    def unload():
        calls.append(f"unload-{name}")

    def restore():
        calls.append(f"restore-{name}")

    return park, unload, restore, footprint


def test_park_then_is_parked_true():
    calls = []
    park, unload, restore, footprint = _fake_owner("a", calls)
    mr.park_owner("a", park, unload, footprint)
    assert mr.is_parked("a")
    assert calls == ["park-a"]


def test_restore_clears_parked_state():
    calls = []
    park, unload, restore, footprint = _fake_owner("a", calls)
    mr.park_owner("a", park, unload, footprint)
    mr.restore_owner("a", restore)
    assert not mr.is_parked("a")
    assert calls == ["park-a", "restore-a"]


def test_restore_raises_if_not_parked():
    with pytest.raises(KeyError):
        mr.restore_owner("nope", lambda: None)


def test_none_footprint_unloads_directly_never_parks():
    calls = []
    park, unload, restore, _ = _fake_owner("a", calls)
    mr.park_owner("a", park, unload, None)
    assert not mr.is_parked("a")
    assert calls == ["unload-a"]  # park() never called


def test_park_returning_false_falls_back_to_unload():
    calls = []
    park, unload, restore, footprint = _fake_owner("a", calls, park_ok = False)
    mr.park_owner("a", park, unload, footprint)
    assert not mr.is_parked("a")
    assert calls == ["park-a", "unload-a"]


def test_park_raising_falls_back_to_unload():
    calls = []
    def raising_park():
        calls.append("park-a")
        raise RuntimeError("boom")
    def unload():
        calls.append("unload-a")
    mr.park_owner("a", raising_park, unload, 1000)
    assert not mr.is_parked("a")
    assert calls == ["park-a", "unload-a"]


def test_exceeding_budget_evicts_the_oldest_parked_other_owner(monkeypatch):
    monkeypatch.setattr("utils.memory_park_settings.get_ram_park_budget_mib", lambda: 1500)
    calls = []
    park_a, unload_a, _, _ = _fake_owner("a", calls, footprint = 1000)
    mr.park_owner("a", park_a, unload_a, 1000)
    park_b, unload_b, _, _ = _fake_owner("b", calls, footprint = 1000)
    mr.park_owner("b", park_b, unload_b, 1000)  # 1000 + 1000 > 1500 -> evict a first
    assert calls == ["park-a", "unload-a", "park-b"]
    assert not mr.is_parked("a")
    assert mr.is_parked("b")


def test_failed_forced_eviction_falls_back_to_unloading_the_new_owner(monkeypatch):
    monkeypatch.setattr("utils.memory_park_settings.get_ram_park_budget_mib", lambda: 1500)
    calls = []
    def raising_unload_a():
        raise RuntimeError("boom")
    park_a, _, _, _ = _fake_owner("a", calls, footprint = 1000)
    mr.park_owner("a", park_a, raising_unload_a, 1000)
    park_b, unload_b, _, _ = _fake_owner("b", calls, footprint = 1000)
    mr.park_owner("b", park_b, unload_b, 1000)
    assert calls == ["park-a", "unload-b"]  # b's own park() never even attempted
    assert not mr.is_parked("a")
    assert not mr.is_parked("b")


def test_new_owner_alone_exceeds_budget_falls_back_to_direct_unload(monkeypatch):
    monkeypatch.setattr("utils.memory_park_settings.get_ram_park_budget_mib", lambda: 500)
    calls = []
    park, unload, _, _ = _fake_owner("a", calls, footprint = 1000)
    mr.park_owner("a", park, unload, 1000)  # nothing parked to evict, still over budget alone
    assert calls == ["unload-a"]  # park() never even attempted
    assert not mr.is_parked("a")


def test_protect_keeps_the_acquiring_owner_out_of_forced_eviction(monkeypatch):
    # Review I2: "a" is parked and is the owner about to take the GPU back (and restore). Parking
    # "b" over budget must NOT evict "a" -- even though it is the oldest (the ONLY) other parked
    # owner -- so with nothing else to evict, "b" unloads directly instead.
    monkeypatch.setattr("utils.memory_park_settings.get_ram_park_budget_mib", lambda: 1500)
    calls = []
    park_a, unload_a, _, _ = _fake_owner("a", calls)
    mr.park_owner("a", park_a, unload_a, 1000)
    park_b, unload_b, _, _ = _fake_owner("b", calls)
    mr.park_owner("b", park_b, unload_b, 1000, protect = "a")
    assert calls == ["park-a", "unload-b"]  # a untouched; b's own park() never attempted
    assert mr.is_parked("a")
    assert not mr.is_parked("b")


def test_protect_still_evicts_an_unprotected_older_owner(monkeypatch):
    # protect narrows the victim set, it does not switch eviction off: the oldest parked owner is
    # the protected one, so the NEXT oldest goes instead and the new owner still parks.
    monkeypatch.setattr("utils.memory_park_settings.get_ram_park_budget_mib", lambda: 2500)
    calls = []
    for name in ("a", "b"):
        park, unload, _, _ = _fake_owner(name, calls)
        mr.park_owner(name, park, unload, 1000)
    park_c, unload_c, _, _ = _fake_owner("c", calls)
    mr.park_owner("c", park_c, unload_c, 1000, protect = "a")  # 3000 > 2500
    assert calls == ["park-a", "park-b", "unload-b", "park-c"]
    assert mr.is_parked("a") and mr.is_parked("c")
    assert not mr.is_parked("b")


def test_parked_footprint_mib_sums_all_parked_owners():
    calls = []
    park_a, unload_a, _, _ = _fake_owner("a", calls, footprint = 1000)
    mr.park_owner("a", park_a, unload_a, 1000)
    park_b, unload_b, _, _ = _fake_owner("b", calls, footprint = 2000)
    mr.park_owner("b", park_b, unload_b, 2000)
    assert mr.parked_footprint_mib() == 3000


def test_a_fresh_module_state_reports_nothing_parked():
    assert mr.parked_footprint_mib() == 0
    assert not mr.is_parked("anything")
