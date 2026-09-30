# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import pytest

import utils.memory_park_settings as mp


@pytest.fixture(autouse = True)
def _isolate(monkeypatch):
    monkeypatch.delenv(mp.RAM_PARK_BUDGET_ENV_VAR, raising = False)
    monkeypatch.setattr(mp, "_cached_setting", lambda _key: None)


class TestCoerceRamBudgetMib:
    @pytest.mark.parametrize("value", [0, 1024, 28672, 61440, "16384"])
    def test_accepts_in_range(self, value):
        assert mp.coerce_ram_budget_mib(value) == int(value)

    @pytest.mark.parametrize(
        "value", [-1, 61441, "nan", float("nan"), "inf", float("inf"), "", None, True, "abc"],
    )
    def test_rejects_out_of_range_or_unusable(self, value):
        assert mp.coerce_ram_budget_mib(value) is None


def test_get_budget_falls_back_to_default_with_nothing_set():
    assert mp.get_ram_park_budget_mib() == mp.RAM_PARK_BUDGET_DEFAULT_MIB


def test_get_budget_reads_the_environment_when_nothing_stored(monkeypatch):
    monkeypatch.setenv(mp.RAM_PARK_BUDGET_ENV_VAR, "16384")
    assert mp.get_ram_park_budget_mib() == 16384


def test_stored_value_wins_over_environment(monkeypatch):
    monkeypatch.setenv(mp.RAM_PARK_BUDGET_ENV_VAR, "16384")
    monkeypatch.setattr(mp, "_cached_setting", lambda _key: 8192)
    assert mp.get_ram_park_budget_mib() == 8192


def test_set_then_clear_round_trips(monkeypatch, tmp_path):
    calls = {"upserted": None}

    def fake_upsert(settings):
        calls["upserted"] = settings
        return settings

    monkeypatch.setattr("storage.studio_db.upsert_app_settings", fake_upsert)
    result = mp.set_ram_park_budget_mib(4096)
    assert result == 4096
    assert calls["upserted"] == {mp.RAM_PARK_BUDGET_SETTING_KEY: 4096}

    mp.set_ram_park_budget_mib(None)
    assert calls["upserted"] == {mp.RAM_PARK_BUDGET_SETTING_KEY: None}


def test_set_rejects_an_out_of_range_value():
    with pytest.raises(ValueError):
        mp.set_ram_park_budget_mib(99999999)
