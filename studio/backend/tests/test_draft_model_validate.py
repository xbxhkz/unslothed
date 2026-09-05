# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Validation of a pinned drafter.

Auto-discovery is safe by construction: it only selects colocated,
name-matched sidecars. Pinning removes that protection, so these are the
checks that replace it. Each has a negative control proving it is the check
that fires, not a neighbour.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from routes.draft_model import (
    VERDICT_MISSING,
    VERDICT_OK,
    VERDICT_OUTSIDE,
    validate_choice,
)


def _fake_gguf(path: Path, size: int = 2048) -> Path:
    """A file that is not a real GGUF. Enough for existence/size/confinement,
    which run before any header parse."""
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"\0" * size)
    return path


class TestExistence:
    def test_a_missing_local_drafter_is_rejected_with_a_stated_reason(self, tmp_path):
        v = validate_choice(str(tmp_path / "target.gguf"), ("local", str(tmp_path / "nope.gguf")))
        assert not v.ok
        assert v.reason == VERDICT_MISSING
        assert "nope.gguf" in v.detail, "the reason must name the file, not just fail"

    def test_an_existing_local_drafter_passes_existence(self, tmp_path):
        target = _fake_gguf(tmp_path / "target.gguf")
        draft = _fake_gguf(tmp_path / "draft.gguf", size = 4096)
        v = validate_choice(str(target), ("local", str(draft)))
        assert v.reason != VERDICT_MISSING
        assert v.size_bytes == 4096, "size must be reported so the UI can warn before a 409"


class TestConfinement:
    def test_a_drafter_outside_the_target_tree_is_rejected(self, tmp_path):
        target = _fake_gguf(tmp_path / "models" / "target.gguf")
        outside = _fake_gguf(tmp_path / "elsewhere" / "secret.gguf")
        v = validate_choice(str(target), ("local", str(outside)))
        assert not v.ok
        assert v.reason == VERDICT_OUTSIDE

    def test_a_sibling_of_the_target_is_allowed(self, tmp_path):
        target = _fake_gguf(tmp_path / "models" / "target.gguf")
        sibling = _fake_gguf(tmp_path / "models" / "draft.gguf")
        v = validate_choice(str(target), ("local", str(sibling)))
        assert v.reason != VERDICT_OUTSIDE

    # --- negative control --------------------------------------------------
    def test_control_a_symlink_escape_is_rejected(self, tmp_path):
        """A path INSIDE the tree that resolves outside it. A confinement check
        using string prefixes on the unresolved path admits this."""
        target = _fake_gguf(tmp_path / "models" / "target.gguf")
        outside = _fake_gguf(tmp_path / "elsewhere" / "secret.gguf")
        link = tmp_path / "models" / "innocent.gguf"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable (Windows without developer mode)")
        v = validate_choice(str(target), ("local", str(link)))
        assert not v.ok, "a symlink out of the tree must not be admitted"
        assert v.reason == VERDICT_OUTSIDE


class TestRemote:
    def test_an_hf_repo_is_not_subjected_to_local_path_checks(self, tmp_path):
        target = _fake_gguf(tmp_path / "target.gguf")
        v = validate_choice(str(target), ("hf", "unsloth/Qwen3-0.6B-GGUF"))
        assert v.reason not in (VERDICT_MISSING, VERDICT_OUTSIDE)
