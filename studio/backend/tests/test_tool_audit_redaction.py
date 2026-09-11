# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Redaction for the tool audit log.

The log must not become the highest-value file on disk. It must also not
over-redact: a log that scrubs ordinary arguments is quietly useless, which is
why every positive case here has a negative partner.
"""

from __future__ import annotations

import json

from core.inference.tool_audit import redaction


def test_key_named_secret_is_redacted():
    out, hit = redaction.redact_arguments({"api_key": "hunter2", "path": "/tmp/x"})
    assert hit is True
    data = json.loads(out)
    assert data["api_key"] == redaction.REDACTED
    assert data["path"] == "/tmp/x", "only the secret-named key should change"


def test_ordinary_arguments_are_NOT_redacted():
    """The control that stops over-redaction gutting the log."""
    out, hit = redaction.redact_arguments({"command": "ls -la /var/log", "timeout": 30})
    assert hit is False
    assert json.loads(out) == {"command": "ls -la /var/log", "timeout": 30}


def test_token_shaped_values_are_redacted_by_pattern():
    for secret in (
        "sk-abcdefghijklmnopqrstuvwxyz0123",
        "ghp_abcdefghijklmnopqrstuvwxyz0123456789",
        "AKIAIOSFODNN7EXAMPLE",
    ):
        out, hit = redaction.redact_text(f"export TOKEN={secret}")
        assert hit is True, f"{secret!r} should have been redacted"
        assert secret not in out


def test_plain_prose_is_NOT_redacted():
    """Second over-redaction control, on the text path."""
    text = "Compiled 42 files in 3.2 seconds with no errors."
    out, hit = redaction.redact_text(text)
    assert hit is False
    assert out == text


def test_nested_values_are_reached():
    out, hit = redaction.redact_arguments({"env": {"AUTHORIZATION": "Bearer abc123def456"}})
    assert hit is True
    assert "abc123def456" not in out


def test_extract_paths_finds_path_arguments():
    paths = json.loads(redaction.extract_paths({"path": "/a/b.txt", "count": 3}))
    assert paths == ["/a/b.txt"]


def test_extract_paths_is_empty_when_there_are_none():
    assert json.loads(redaction.extract_paths({"query": "hello"})) == []
