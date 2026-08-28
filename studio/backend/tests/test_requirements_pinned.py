# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every runtime dependency must be pinned.

A bare package name resolves to whatever is newest at install time. This
project has already been bitten once: a bare ``mcp`` in Assist's requirements
resolved to 2.0.0 on a fresh install and broke all four built-in MCP servers.
The failure did not look like a version problem, which is what made it
expensive.
"""
import pathlib
import re

REQ_DIR = pathlib.Path(__file__).resolve().parents[1] / "requirements"

# A bare name: letters/digits/._- and nothing else. No comparator, no extras,
# no environment marker, no URL.
_BARE = re.compile(r"^[A-Za-z][A-Za-z0-9._-]*$")


def _entries(path):
    for raw in path.read_text(encoding = "utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        yield line


def test_no_requirement_is_left_unpinned():
    unpinned = []
    for path in sorted(REQ_DIR.glob("*.txt")):
        for entry in _entries(path):
            if _BARE.match(entry):
                unpinned.append(f"{path.name}: {entry}")
    assert not unpinned, (
        "unpinned requirements resolve to whatever is newest at install time:\n  "
        + "\n  ".join(unpinned)
    )
