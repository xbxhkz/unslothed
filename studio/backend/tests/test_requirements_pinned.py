# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every runtime dependency must be pinned.

A bare package name resolves to whatever is newest at install time. This
project has already been bitten once: a bare ``mcp`` in Assist's requirements
resolved to 2.0.0 on a fresh install and broke all four built-in MCP servers.
The failure did not look like a version problem, which is what made it
expensive.

NOTE: This test scopes to studio/backend/requirements/*.txt only. pyproject.toml
is deliberately out of scope: it declares library metadata where bare names are
conventional and correct (libraries do not pin, applications do). The pyproject
extra entries like ``rich``, ``pydantic``, ``numpy`` are correct as-is.
"""
import pathlib
import re

REQ_DIR = pathlib.Path(__file__).resolve().parents[1] / "requirements"

# A bare name: letters/digits/._- and nothing else. No comparator, no extras,
# no environment marker, no URL.
_BARE = re.compile(r"^[A-Za-z][A-Za-z0-9._-]*$")


def _normalize_for_bare_check(entry):
    """Strip extras [...] and environment markers (;...) before checking if bare.

    This allows detecting bare package names that appear in one of these forms:
    - requests
    - uvicorn[standard]
    - foo; sys_platform == "win32"

    All three resolve to whatever is newest at install time, which is the defect
    this test prevents.
    """
    # Remove extras: foo[bar,baz] -> foo
    entry = re.sub(r"\[.*?\]", "", entry)
    # Remove environment markers and everything after: foo; sys_platform -> foo
    entry = re.split(r"\s*;", entry, maxsplit=1)[0].strip()
    return entry


def _entries(path):
    """Parse requirements.txt file, stripping comments with pip's syntax (whitespace before #).

    pip requires whitespace before # for a comment; plain # in URLs like
    #subdirectory= is part of the spec. This matches pip's actual parsing.
    """
    for raw in path.read_text(encoding="utf-8").splitlines():
        # Split on whitespace followed by #, not just # alone
        line = re.split(r"\s#", raw, maxsplit=1)[0].strip()
        if not line or line.startswith("-"):
            continue
        yield line


def test_no_requirement_is_left_unpinned():
    unpinned = []
    for path in sorted(REQ_DIR.glob("*.txt")):
        for entry in _entries(path):
            normalized = _normalize_for_bare_check(entry)
            if _BARE.match(normalized):
                unpinned.append(f"{path.name}: {entry}")
    assert not unpinned, (
        "unpinned requirements resolve to whatever is newest at install time:\n  "
        + "\n  ".join(unpinned)
    )


class TestRequirementShapes:
    """Negative control tests: verify the detector catches each unpinned shape."""

    def test_bare_name_detected(self, tmp_path):
        """Shape 1: bare package name like 'requests'."""
        req_file = tmp_path / "test.txt"
        req_file.write_text("requests\n", encoding="utf-8")

        entries = list(_entries(req_file))
        assert entries == ["requests"], "bare name should be parsed"

        normalized = _normalize_for_bare_check(entries[0])
        assert _BARE.match(normalized), "bare name should match bare pattern"

    def test_extras_syntax_detected(self, tmp_path):
        """Shape 2: extras syntax like 'uvicorn[standard]' is also unpinned."""
        req_file = tmp_path / "test.txt"
        req_file.write_text("uvicorn[standard]\n", encoding="utf-8")

        entries = list(_entries(req_file))
        assert entries == ["uvicorn[standard]"], "extras syntax should be parsed"

        normalized = _normalize_for_bare_check(entries[0])
        assert _BARE.match(normalized), "extras syntax should normalize to bare name and match"

    def test_environment_marker_detected(self, tmp_path):
        """Shape 3: environment marker like 'foo; sys_platform == \"win32\"' is also unpinned."""
        req_file = tmp_path / "test.txt"
        req_file.write_text('foo; sys_platform == "win32"\n', encoding="utf-8")

        entries = list(_entries(req_file))
        assert entries == ['foo; sys_platform == "win32"'], "env marker should be parsed"

        normalized = _normalize_for_bare_check(entries[0])
        assert _BARE.match(normalized), "env marker should normalize to bare name and match"

    def test_vcs_url_not_bare(self, tmp_path):
        """VCS URL with # fragment should parse correctly and not match bare pattern."""
        req_file = tmp_path / "test.txt"
        url = "triton_kernels @ git+https://github.com/triton-lang/triton.git@release/3.6.x#subdirectory=python/triton_kernels"
        req_file.write_text(url + "\n", encoding="utf-8")

        entries = list(_entries(req_file))
        assert entries == [url], "VCS URL with # fragment should be parsed intact (not treated as comment)"

        normalized = _normalize_for_bare_check(entries[0])
        assert not _BARE.match(normalized), "VCS URL should not match bare pattern"
