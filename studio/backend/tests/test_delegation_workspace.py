# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where a delegation's files live.

The location is forced, not chosen: assist_vision's _write_png records that
tools return a path and that resolve_image_bytes refuses any path outside the
conversation's working directory. A delegation folder written anywhere else
could not be read back by the primary's own file tools.
"""

from __future__ import annotations

import os

import pytest

from core.inference.delegation import workspace


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "_workdir_for", lambda session_id: str(tmp_path))
    return tmp_path


def test_a_delegation_folder_is_created_under_the_session_workdir(workdir):
    files = workspace.create("session-1", "coding")
    assert os.path.isdir(files.root)
    assert str(workdir) in files.root
    assert "delegations" in files.root


def test_two_delegations_do_not_collide(workdir):
    first = workspace.create("session-1", "coding")
    second = workspace.create("session-1", "coding")
    assert first.root != second.root
    assert first.delegation_id != second.delegation_id


def test_the_brief_records_the_role_task_and_context(workdir):
    files = workspace.create("session-1", "coding")
    workspace.write_brief(files, role = "coding", task = "rewrite the parser", context = "it is recursive descent")
    text = open(files.brief, encoding = "utf-8").read()
    assert "coding" in text
    assert "rewrite the parser" in text
    assert "recursive descent" in text


def test_the_work_file_and_transcript_are_written(workdir):
    files = workspace.create("session-1", "coding")
    workspace.write_work(files, "def parse(): ...")
    workspace.append_transcript(files, "round 1")
    workspace.append_transcript(files, "round 2")
    assert "def parse()" in open(files.work, encoding = "utf-8").read()
    transcript = open(files.transcript, encoding = "utf-8").read()
    assert "round 1" in transcript and "round 2" in transcript, "transcript appends, never truncates"


def test_paths_returned_to_the_model_are_relative_to_the_workdir(workdir):
    files = workspace.create("session-1", "coding")
    paths = workspace.relative_paths(files)
    for value in paths.values():
        assert not os.path.isabs(value), f"{value} is absolute; the model gets workdir-relative paths"
        assert value.startswith("delegations/")


def test_writing_never_raises_on_an_unwritable_root(workdir, monkeypatch):
    files = workspace.create("session-1", "coding")
    monkeypatch.setattr(workspace, "_write_text", lambda path, text, append = False: (_ for _ in ()).throw(OSError("read-only")))
    workspace.write_work(files, "x")
    workspace.append_transcript(files, "y")


def test_the_real_workdir_helper_is_the_tools_one():
    """Delegation must not invent a second workspace rule."""
    import inspect

    assert "_get_workdir" in inspect.getsource(workspace._workdir_for)
