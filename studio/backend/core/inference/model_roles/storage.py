# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where role bindings live: one app setting, read and written through the
existing settings helpers rather than a new table.

The `_read_setting` / `_write_setting` indirections exist so tests can swap the
storage without a database.
"""

from __future__ import annotations

from typing import Any

ROLES_SETTING_KEY = "model_roles"


def _read_setting() -> Any:
    from storage.studio_db import get_app_setting

    return get_app_setting(ROLES_SETTING_KEY, None)


def _write_setting(value: dict) -> None:
    from storage.studio_db import upsert_app_settings

    upsert_app_settings({ROLES_SETTING_KEY: value})


def get_role_bindings() -> dict[str, dict]:
    """The raw stored mapping. Never raises; a corrupt setting reads as empty."""
    try:
        raw = _read_setting()
    except BaseException:  # noqa: BLE001 - a settings read must not break a turn
        return {}
    return raw if isinstance(raw, dict) else {}


def set_role_bindings(raw: dict) -> dict[str, dict]:
    """Replace the stored mapping. Returns what was stored."""
    value = raw if isinstance(raw, dict) else {}
    _write_setting(value)
    return value
