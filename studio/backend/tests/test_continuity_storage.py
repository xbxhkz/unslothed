# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The storage primitives every continuity write builds on: atomic writes and a
read that tells a missing file apart from a corrupt one.

No test here touches a model, a backend, or a real tool -- this is pure file I/O
against tmp_path.
"""

from __future__ import annotations

import json
import os

import pytest

from core.continuity.schemas import ContinuityError
from core.continuity import storage


def test_write_json_atomic_writes_valid_json(tmp_path):
    target = str(tmp_path / "state.json")
    storage.write_json_atomic(target, {"a": 1})
    with open(target, encoding = "utf-8") as f:
        assert json.load(f) == {"a": 1}


def test_write_json_atomic_leaves_no_temp_file_on_success(tmp_path):
    target = str(tmp_path / "state.json")
    storage.write_json_atomic(target, {"a": 1})
    leftovers = [p for p in os.listdir(tmp_path) if p != "state.json"]
    assert leftovers == [], f"temp file(s) left behind: {leftovers}"


def test_read_json_missing_file_returns_none(tmp_path):
    assert storage.read_json(str(tmp_path / "nope.json")) is None


def test_read_json_corrupt_file_raises(tmp_path):
    target = tmp_path / "bad.json"
    target.write_text("{not valid json", encoding = "utf-8")
    with pytest.raises(ContinuityError):
        storage.read_json(str(target))


def test_ai_dir_creates_the_directory_tree(tmp_path):
    project_dir = str(tmp_path)
    result = storage.ai_dir(project_dir)
    assert result == os.path.join(project_dir, ".ai")
    assert os.path.isdir(os.path.join(result, "logs"))
    assert os.path.isdir(os.path.join(result, "checkpoints"))


def test_write_json_atomic_wraps_write_failure_in_continuity_error(tmp_path, monkeypatch):
    target = str(tmp_path / "state.json")

    def raise_on_dump(*args, **kwargs):
        raise ValueError("simulated json.dump failure")

    monkeypatch.setattr(json, "dump", raise_on_dump)
    with pytest.raises(ContinuityError) as exc_info:
        storage.write_json_atomic(target, {"a": 1})
    assert "could not write" in exc_info.value.message


def test_write_json_atomic_lets_keyboard_interrupt_propagate(tmp_path, monkeypatch):
    """A process-control signal during a write is not a write failure -- it
    must not be masked as a ContinuityError the way a genuine I/O error is."""
    target = str(tmp_path / "state.json")

    def raise_keyboard_interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(json, "dump", raise_keyboard_interrupt)
    with pytest.raises(KeyboardInterrupt):
        storage.write_json_atomic(target, {"a": 1})


def test_read_json_non_dict_json_raises(tmp_path):
    """A bare scalar is valid JSON but not a valid continuity file -- must
    raise ContinuityError, not let a caller's from_dict crash on it three
    calls later with a raw TypeError."""
    target = tmp_path / "state.json"
    target.write_text("42", encoding = "utf-8")
    with pytest.raises(ContinuityError):
        storage.read_json(str(target))


def test_read_json_wrong_encoding_raises_continuity_error(tmp_path):
    target = tmp_path / "state.json"
    target.write_text('{"a": 1}', encoding = "utf-16")
    with pytest.raises(ContinuityError):
        storage.read_json(str(target))


def test_ai_dir_path_never_touches_the_filesystem(tmp_path):
    project_dir = str(tmp_path)
    result = storage.ai_dir_path(project_dir)
    assert result == os.path.join(project_dir, ".ai")
    assert not os.path.exists(result)


def test_ai_dir_raises_cleanly_when_dot_ai_is_a_plain_file(tmp_path):
    project_dir = str(tmp_path)
    (tmp_path / ".ai").write_text("not a directory", encoding = "utf-8")
    with pytest.raises(ContinuityError):
        storage.ai_dir(project_dir)
