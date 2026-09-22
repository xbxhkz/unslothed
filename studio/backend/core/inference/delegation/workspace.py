# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A delegation's files, under the conversation's own working directory.

That location is forced, not chosen. assist_vision/__init__.py::_write_png
records the rule: tools return a path, and resolve_image_bytes refuses any path
outside the conversation's working directory -- so a delegation folder written
elsewhere could not be read back by the primary's own file tools.

Every write is guarded: losing a transcript must not break a delegation.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass

_FOLDER = "delegations"


@dataclass(frozen = True)
class DelegationFiles:
    delegation_id: str
    root: str
    brief: str
    work: str
    transcript: str


def _workdir_for(session_id) -> str:
    """The conversation's working directory, from tools.py's own helper.

    Imported lazily: tools.py imports the delegation schemas at module level, so
    a module-level import back would be a cycle."""
    from core.inference.tools import _get_workdir

    return _get_workdir(session_id)


def _write_text(path: str, text: str, append: bool = False) -> None:
    with open(path, "a" if append else "w", encoding = "utf-8") as handle:
        handle.write(text)


def create(session_id, role) -> DelegationFiles:
    delegation_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    root = os.path.join(_workdir_for(session_id), _FOLDER, delegation_id)
    os.makedirs(root, exist_ok = True)
    return DelegationFiles(
        delegation_id = delegation_id,
        root = root,
        brief = os.path.join(root, "brief.md"),
        work = os.path.join(root, "work.md"),
        transcript = os.path.join(root, "transcript.md"),
    )


def write_brief(files: DelegationFiles, *, role, task, context = "") -> None:
    body = (
        f"# Delegation {files.delegation_id}\n\n"
        f"**Role:** {role}\n\n"
        f"## Task\n\n{task}\n\n"
        f"## What the primary already established\n\n{context or '(nothing supplied)'}\n\n"
        f"## Done means\n\n"
        f"The answer is written to `work.md`, and the short reply says what was done.\n"
    )
    try:
        _write_text(files.brief, body)
    except BaseException:  # noqa: BLE001 - a delegation must not die writing a file
        pass


def write_work(files: DelegationFiles, text: str) -> None:
    try:
        _write_text(files.work, text)
    except BaseException:  # noqa: BLE001
        pass


def append_transcript(files: DelegationFiles, text: str) -> None:
    try:
        _write_text(files.transcript, text.rstrip() + "\n\n", append = True)
    except BaseException:  # noqa: BLE001
        pass


def relative_paths(files: DelegationFiles) -> dict[str, str]:
    """Workdir-relative paths, which is what the model can act on."""
    base = f"{_FOLDER}/{files.delegation_id}"
    return {"brief": f"{base}/brief.md", "work": f"{base}/work.md", "transcript": f"{base}/transcript.md"}
