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
VERDICT_NO_TARGET = "no_target"
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

    if not target_path:
        return DraftVerdict(
            ok = False, reason = VERDICT_NO_TARGET,
            detail = "a model must be loaded first",
        )

    draft = _resolve_real(str(ref))
    if not draft.is_file():
        return DraftVerdict(
            ok = False, reason = VERDICT_MISSING,
            detail = f"no file at {ref}",
        )
    target = _resolve_real(target_path)
    if not _is_confined(target, draft):
        return DraftVerdict(
            ok = False, reason = VERDICT_OUTSIDE,
            detail = f"{draft.name} is outside {target.parent}",
        )
    size_bytes = draft.stat().st_size
    v_target = read_gguf_vocab_size(str(target))
    v_draft = read_gguf_vocab_size(str(draft))
    if v_target is None or v_draft is None:
        return DraftVerdict(
            ok = False, reason = VERDICT_VOCAB_UNKNOWN,
            detail = "could not read the vocabulary from one of the files",
            size_bytes = size_bytes, vocab_target = v_target, vocab_draft = v_draft,
        )
    if v_target != v_draft:
        return DraftVerdict(
            ok = False, reason = VERDICT_VOCAB_MISMATCH,
            detail = f"target vocabulary is {v_target}, drafter is {v_draft}",
            size_bytes = size_bytes, vocab_target = v_target, vocab_draft = v_draft,
        )
    return DraftVerdict(
        ok = True, reason = VERDICT_OK, detail = draft.name,
        size_bytes = size_bytes, vocab_target = v_target, vocab_draft = v_draft,
    )


import struct

_GGUF_MAGIC = 0x46554747  # b"GGUF" LE u32
_VOCAB_KEY = b"tokenizer.ggml.tokens"
_TYPE_ARRAY = 9

# Byte widths of the fixed-size GGUF value types, indexed by type id. Mirrors
# _FIXED_VTYPE_SIZES in utils/models/gguf_metadata.py; kept local so this module
# depends on one upstream private set (the draft flags) rather than two.
_FIXED_VTYPE_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_gguf_vocab_size(path: str) -> Optional[int]:
    """Token count from ``tokenizer.ggml.tokens``, or None when unreadable.

    The array LENGTH is the vocabulary size: many GGUFs carry no vocab_size
    key, which is the same approach the loader itself takes. Only the length is
    read -- the tokens are skipped without being materialised, so a six-figure
    vocabulary costs no memory.

    Returns None rather than 0 for "could not determine". Callers must treat
    None as unknown and say so; a 0 would compare equal to another 0 and
    silently admit a mismatched pair.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(24)
            if len(head) < 24:
                return None
            magic, _version, _tcount, kv_count = struct.unpack("<IIQQ", head)
            if magic != _GGUF_MAGIC:
                return None
            for _ in range(kv_count):
                raw = f.read(8)
                if len(raw) < 8:
                    return None
                (klen,) = struct.unpack("<Q", raw)
                key = f.read(klen)
                vtype_raw = f.read(4)
                if len(vtype_raw) < 4:
                    return None
                (vtype,) = struct.unpack("<I", vtype_raw)
                if key == _VOCAB_KEY and vtype == _TYPE_ARRAY:
                    atype_raw = f.read(4)
                    alen_raw = f.read(8)
                    if len(atype_raw) < 4 or len(alen_raw) < 8:
                        return None
                    (alen,) = struct.unpack("<Q", alen_raw)
                    return int(alen)
                if not _skip_value(f, vtype):
                    return None
    except (OSError, struct.error, ValueError):
        return None
    return None


def _skip_value(f, vtype: int) -> bool:
    """Advance past one GGUF value. False when the type is unknown, which ends
    the walk rather than misreading the rest of the header."""
    if vtype in _FIXED_VTYPE_SIZES:
        return len(f.read(_FIXED_VTYPE_SIZES[vtype])) == _FIXED_VTYPE_SIZES[vtype]
    if vtype == 8:  # string
        raw = f.read(8)
        if len(raw) < 8:
            return False
        (n,) = struct.unpack("<Q", raw)
        f.seek(n, os.SEEK_CUR)
        return True
    if vtype == _TYPE_ARRAY:
        head = f.read(12)
        if len(head) < 12:
            return False
        atype, alen = struct.unpack("<IQ", head)
        for _ in range(alen):
            if not _skip_value(f, atype):
                return False
        return True
    return False
