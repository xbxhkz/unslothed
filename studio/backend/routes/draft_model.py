# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pinning a specific speculative draft model.

Studio already resolves a drafter automatically: `auto` ranks colocated
MTP/DSpark/DFlash sidecars through utils/models/drafters/preference.py and
launches the winner. That is safe by construction -- only name-matched
neighbours are eligible -- but it is invisible and not reproducible, and there
is no way to name a drafter that lives elsewhere.

This module lets the user pin one, by composing the drafter flags into the
caller's existing ``llama_extra_args``. The load path in routes/inference.py
already honours a caller-named drafter completely (VRAM charged, last-wins
leaving exactly one resident, native path rules applied to split shards), so
nothing there changes.

Composition lives here rather than in the frontend on purpose. The flag
vocabulary is seven spellings across two families, each accepting ``-f v`` and
``-f=v``, and _extra_args_mtp_draft_source already parses exactly that set.
Rebuilding it in TypeScript would mean two parsers with nothing binding them
together -- the shape of the unpinned-dependency failure this project has
already paid for once. The sets are imported, never retyped; tests/
test_draft_model_compose.py pins their names so an upstream rename fails there.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from core.inference.llama_cpp import _HF_DRAFT_FLAGS, _LOCAL_DRAFT_FLAGS
from core.inference.llama_server_args import _flag_name

# ("local", path) or ("hf", repo_id).
DraftChoice = Tuple[str, str]

# The spelling written when pinning. Reading accepts all seven; writing picks
# the canonical long form of each family so the raw-args box stays legible.
_CANONICAL_LOCAL_FLAG = "--model-draft"
_CANONICAL_HF_FLAG = "--spec-draft-hf"

_ALL_DRAFT_FLAGS = _HF_DRAFT_FLAGS | _LOCAL_DRAFT_FLAGS


def _strip_draft_flags(args: Sequence[str]) -> list[str]:
    """Every drafter-naming flag and its value removed; everything else kept in
    order.

    Matching is on the parsed flag NAME, never on a substring: an ``--alias``
    value of ``my-md-model`` contains "-md" and must survive.
    """
    out: list[str] = []
    skip_next = False
    for i, raw in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        token = str(raw)
        flag = _flag_name(token)
        if flag in _ALL_DRAFT_FLAGS:
            _, eq, _inline = token.partition("=")
            if not eq:
                # Separate value form: drop the following token too, unless it
                # is itself a flag (a malformed trailing "-md" owns no value).
                nxt = str(args[i + 1]) if i + 1 < len(args) else ""
                if nxt and not nxt.startswith("-"):
                    skip_next = True
            continue
        out.append(token)
    return out


def compose_draft_args(
    existing: Optional[Sequence[str]], choice: Optional[DraftChoice]
) -> list[str]:
    """``existing`` with any pinned drafter replaced by ``choice``.

    ``choice = None`` clears the pin, which restores auto-discovery: with no
    drafter flag present the load path falls back to its own sidecar ranking.
    """
    args = _strip_draft_flags(existing or [])
    if choice is None:
        return args
    kind, ref = choice
    flag = _CANONICAL_HF_FLAG if kind == "hf" else _CANONICAL_LOCAL_FLAG
    return args + [flag, str(ref)]


import os
from dataclasses import dataclass
from pathlib import Path

VERDICT_OK = "ok"
VERDICT_MISSING = "missing"
VERDICT_OUTSIDE = "outside_permitted_directory"
VERDICT_VOCAB_MISMATCH = "vocab_mismatch"
VERDICT_VOCAB_UNKNOWN = "vocab_unknown"


@dataclass
class DraftVerdict:
    ok: bool
    reason: str
    detail: str = ""
    size_bytes: Optional[int] = None
    vocab_target: Optional[int] = None
    vocab_draft: Optional[int] = None


def _resolve_real(p: str) -> Path:
    """Fully resolved path. `.resolve()` is what makes the confinement check
    survive a symlink pointing out of the tree; a string-prefix test on the
    unresolved path admits exactly that escape."""
    return Path(p).resolve()


def _is_confined(target: Path, draft: Path) -> bool:
    """A pinned local drafter must live in the target's directory tree.

    Same rule auto-discovery obeys implicitly by only ever considering
    colocated files, made explicit here because pinning can name anything.
    """
    root = target.parent
    try:
        draft.relative_to(root)
        return True
    except ValueError:
        return False


def validate_choice(target_path: Optional[str], choice: DraftChoice) -> DraftVerdict:
    """Whether ``choice`` is a usable drafter for ``target_path``.

    Remote repositories skip the local filesystem checks: there is no path to
    resolve and the load path prices them from their own listing.
    """
    kind, ref = choice
    if kind == "hf":
        return DraftVerdict(ok = True, reason = VERDICT_OK, detail = str(ref))

    draft = _resolve_real(str(ref))
    if not draft.is_file():
        return DraftVerdict(
            ok = False, reason = VERDICT_MISSING,
            detail = f"no file at {ref}",
        )
    if target_path:
        target = _resolve_real(target_path)
        if not _is_confined(target, draft):
            return DraftVerdict(
                ok = False, reason = VERDICT_OUTSIDE,
                detail = f"{draft.name} is outside {target.parent}",
            )
    return DraftVerdict(
        ok = True, reason = VERDICT_OK, detail = draft.name,
        size_bytes = draft.stat().st_size,
    )
