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

READING a pin back is the same problem, so it is answered the same way: the
/current endpoint below hands the frontend the pinned choice as structured
data. The picker never inspects ``llama_extra_args`` itself -- an earlier
revision did, recognised two spellings out of seven, and reported a pinned
model as "Automatic" for the other five. One parser, here, is the whole point
of the module.

Known limitation 1: a pinned drafter that is later deleted fails OPEN at load
time. routes/inference.py treats a local --model-draft that is not on disk as
"no drafter loads and none is charged", so speculation silently reverts to
none. The picker re-validates when it opens and marks a missing pin, but that
only covers the UI path. Closing this properly means editing the upstream load
path, which this sub-project's seam budget (one router-registration line)
deliberately declines. Revisit if the seam constraint is ever relaxed.

Known limitation 2, and it is MORE reachable than the one above: a saved pin is
silently STRIPPED on two of the three load paths. Upstream treats every
drafter-naming flag as shadowing the first-class ``speculative_type`` field --
_SPEC_FLAGS at core/inference/llama_server_args.py:514-546 lists all seven --
and ``strip_shadowing_flags(..., strip_spec = "speculative_type" in ...)`` is
applied to stored/inherited extras at utils/openai_auto_switch_settings.py:686
and routes/inference.py:7993. So a pin survives a Run Settings load that sends
no speculative_type, but an auto-switch load, an idle reload, or a chat-settings
Apply -- each of which does send one -- drops the pin and silently reverts to
auto-discovery. The picker still shows the pin, because it is still in the
saved arguments; only that particular load ignored it. Upstream's reasoning is
sound for ITS drafters (an inherited copy must not last-wins-override the
auto-detected one); it simply predates a user-chosen pin being a thing. Closing
it means teaching those two call sites to distinguish a UI-managed pin from an
inherited leftover -- an edit to upstream load paths, which the same seam
budget declines. Recorded here rather than hidden.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from core.inference.llama_cpp import (
    _HF_DRAFT_FLAGS,
    _LOCAL_DRAFT_FLAGS,
    cached_gguf_for_load,
)
from core.inference.llama_server_args import _flag_name
from hub.utils.paths import is_valid_repo_id
from utils.models.model_config import colocated_split_shards
from utils.paths import is_local_path

# ("local", path) or ("hf", repo_id).
DraftChoice = Tuple[str, str]

# The spelling written when pinning. Reading accepts all seven; writing picks
# the canonical long form of each family so the raw-args box stays legible.
_CANONICAL_LOCAL_FLAG = "--model-draft"
_CANONICAL_HF_FLAG = "--spec-draft-hf"

_ALL_DRAFT_FLAGS = _HF_DRAFT_FLAGS | _LOCAL_DRAFT_FLAGS


def _draft_flag_spans(args: Sequence[str]) -> list[Tuple[int, int, str, str]]:
    """Every drafter-naming flag in ``args`` as ``(start, stop, kind, value)``.

    ``stop`` is exclusive and covers the flag's value token when the ``-f v``
    spelling is used, so a caller can remove the whole span. ``kind`` is
    "hf" or "local"; ``value`` is "" for a malformed trailing flag that owns
    no value.

    THE single scanner. Both stripping (compose) and reading a pin back
    (/current) go through it, so the write side and the read side cannot
    recognise different subsets of the vocabulary -- the failure this module's
    docstring exists to forbid, which a hand-rolled reader reintroduced once
    already.

    Matching is on the parsed flag NAME, never on a substring: an ``--alias``
    value of ``my-md-model`` contains "-md" and must survive. ``_flag_name``
    also peels ``--flag=value`` and normalises ``_`` to ``-``, so the
    ``--model_draft`` spelling is covered without being listed.
    """
    spans: list[Tuple[int, int, str, str]] = []
    consumed = -1
    for i, raw in enumerate(args):
        if i <= consumed:
            continue
        token = str(raw)
        flag = _flag_name(token)
        if flag not in _ALL_DRAFT_FLAGS:
            continue
        kind = "hf" if flag in _HF_DRAFT_FLAGS else "local"
        _, eq, inline = token.partition("=")
        if eq:
            spans.append((i, i + 1, kind, inline))
            continue
        # Separate value form: the following token belongs to the flag, unless
        # it is itself a flag (a malformed trailing "-md" owns no value).
        nxt = str(args[i + 1]) if i + 1 < len(args) else ""
        if nxt and not nxt.startswith("-"):
            consumed = i + 1
            spans.append((i, i + 2, kind, nxt))
        else:
            spans.append((i, i + 1, kind, ""))
    return spans


def _strip_draft_flags(args: Sequence[str]) -> list[str]:
    """Every drafter-naming flag and its value removed; everything else kept in
    order."""
    dropped = {
        i for start, stop, _kind, _value in _draft_flag_spans(args)
        for i in range(start, stop)
    }
    return [str(raw) for i, raw in enumerate(args) if i not in dropped]


def current_draft_pin(args: Optional[Sequence[str]]) -> Optional[DraftChoice]:
    """The drafter ``args`` currently pins, or None for auto-discovery.

    Last-wins, matching llama-server's own argument ordering and what
    compose_draft_args guarantees it leaves behind. A flag with no value pins
    nothing, so it is skipped rather than reported as a pin on "".
    """
    pinned: Optional[DraftChoice] = None
    for _start, _stop, kind, value in _draft_flag_spans(args or []):
        if value:
            pinned = (kind, value)
    return pinned


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
# Accepted, but nothing about the repository's CONTENT was checked -- only that
# the identifier is shaped like a repo id. ok is True (the pin is written) and
# the reason travels with it so the UI says "could not verify" rather than
# implying the drafter was validated the way a local one is. The spec's rule for
# a remote drafter is "an explicit could-not-verify state shown to the user,
# never a silent pass"; an affirmative "ok" with no further word is that silent
# pass, which is what this verdict exists to prevent.
VERDICT_UNVERIFIED = "unverified"
# The identifier is not a Hugging Face repo id at all. A shape failure IS a
# rejection: nothing downstream can do anything useful with it, and letting it
# through would put a path or a typo in --spec-draft-hf.
VERDICT_INVALID_REPO_ID = "invalid_repo_id"
# Given, but the identifier could not be resolved to a real local file --
# distinct from VERDICT_NO_TARGET ("nothing was given at all"). Returned by the
# routes below, before validate_choice ever runs: it takes a real path, not an
# identifier, and has no way to tell these two cases apart itself.
VERDICT_UNRESOLVED = "unresolved"


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


def _shard_paths(draft: Path) -> list[Path]:
    """Every file llama-server will open for ``draft``, fully resolved.

    A split GGUF names only its first shard on the command line; the loader
    opens the siblings implicitly. Confining just the launch path would leave
    those siblings unchecked, which is exactly the escape the spec names -- a
    split drafter whose second shard points outside the permitted directory.

    A non-split path is its own complete one-file set, so this degrades to
    ``[draft]``; so does any enumeration failure, which keeps a permissions
    error on the directory from turning into a crash while still validating
    the one path we do know about.
    """
    try:
        shards, _complete = colocated_split_shards(draft)
    except Exception:
        logger.debug("Could not enumerate shards for %s", draft, exc_info = True)
        return [draft]
    resolved = [_resolve_real(str(s)) for s in shards]
    return resolved or [draft]


def validate_choice(target_path: Optional[str], choice: DraftChoice) -> DraftVerdict:
    """Whether ``choice`` is a usable drafter for ``target_path``.

    Remote repositories skip the local filesystem checks: there is no path to
    resolve and the load path prices them from their own listing.
    """
    kind, ref = choice
    if kind == "hf":
        if not is_valid_repo_id(str(ref)):
            return DraftVerdict(
                ok = False, reason = VERDICT_INVALID_REPO_ID,
                detail = f"'{ref}' is not a Hugging Face repository id",
            )
        # Shape only. Confirming the repo EXISTS, or reading its GGUF header for
        # a vocabulary comparison, means Hub network I/O on a settings control --
        # the same cost that got ModelConfig.from_identifier rejected for the
        # resolver above. So the pin is written and the user is told plainly
        # that it was not checked, rather than being handed a bare "ok".
        return DraftVerdict(
            ok = True, reason = VERDICT_UNVERIFIED,
            detail = (
                f"'{ref}' is shaped like a repository id, but it was not "
                "contacted: existence and vocabulary are unverified"
            ),
        )

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
    # EVERY shard, not just the launch path. The loader opens the siblings of a
    # split drafter without them ever appearing on the command line, so a check
    # that stops at the named file confines nothing for a split set.
    for shard in _shard_paths(draft):
        if not _is_confined(target, shard):
            return DraftVerdict(
                ok = False, reason = VERDICT_OUTSIDE,
                detail = f"shard {shard.name} is outside {target.parent}",
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

# Sanity bounds mirroring utils/models/gguf_metadata.py's _parse_gguf_header /
# _skip_gguf_value: the path is user-named, so a hostile file is in scope. A
# key length or array/string length beyond these is never a real GGUF -- it is
# either corruption or an attacker forcing a huge f.read() -- so the walk
# aborts (returns None / False) rather than acting on the value.
_MAX_KEY_LEN = 1 << 20  # 1 MB
_MAX_LEN = 1 << 30  # 1 GB, applied to array element counts and string lengths


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
                if klen > _MAX_KEY_LEN:
                    return None
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
                    if alen > _MAX_LEN:
                        # Bounded like its four siblings, and for a sharper
                        # reason than theirs: this value is RETURNED, not just
                        # read past. Unbounded, a crafted header yields an
                        # absurd "vocabulary size" that is not None, so two such
                        # files compare equal and validate_choice admits the
                        # pair -- the silent pass the None contract exists to
                        # prevent.
                        return None
                    return int(alen)
                if not _skip_value(f, vtype):
                    return None
    except (OSError, struct.error, ValueError, OverflowError, MemoryError, RecursionError):
        return None
    return None


def _skip_value(f, vtype: int) -> bool:
    """Advance past one GGUF value. False when the type is unknown or a length
    exceeds a sanity bound, which ends the walk rather than misreading the
    rest of the header or acting on an attacker-controlled size.

    Arrays are skipped WITHOUT recursion, mirroring _skip_gguf_value in
    utils/models/gguf_metadata.py: a fixed-size element array is skipped with
    one seek, a string array is skipped element-by-element inline, and a
    nested array (or any other non-fixed-size element type) is not descended
    into at all -- it cannot be tokenizer.ggml.tokens, so refusing to
    recurse into it costs nothing real and a bounded number of stack frames
    is spent regardless of how deeply a hostile file nests arrays.
    """
    if vtype in _FIXED_VTYPE_SIZES:
        return len(f.read(_FIXED_VTYPE_SIZES[vtype])) == _FIXED_VTYPE_SIZES[vtype]
    if vtype == 8:  # string
        raw = f.read(8)
        if len(raw) < 8:
            return False
        (n,) = struct.unpack("<Q", raw)
        if n > _MAX_LEN:
            return False
        f.seek(n, os.SEEK_CUR)
        return True
    if vtype == _TYPE_ARRAY:
        head = f.read(12)
        if len(head) < 12:
            return False
        atype, alen = struct.unpack("<IQ", head)
        if alen > _MAX_LEN:
            return False
        if atype == 8:  # array of strings: each has its own length to bound
            for _ in range(alen):
                raw = f.read(8)
                if len(raw) < 8:
                    return False
                (n,) = struct.unpack("<Q", raw)
                if n > _MAX_LEN:
                    return False
                f.seek(n, os.SEEK_CUR)
            return True
        sz = _FIXED_VTYPE_SIZES.get(atype)
        if sz is None:
            return False  # nested array or unknown element type: abort, don't descend
        f.seek(sz * alen, os.SEEK_CUR)
        return True
    return False


from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from auth.authentication import get_current_subject
from loggers import get_logger

router = APIRouter()
logger = get_logger(__name__)


def _resolve_model_path(
    model_id: Optional[str], gguf_variant: Optional[str]
) -> Optional[Path]:
    """The real local .gguf path behind ``model_id``, or None if it cannot be
    found on this machine.

    ``model_id`` is a clean identifier the frontend already has on hand: either
    a local file/directory path, or a bare HF repo id like the ones the Hub tab
    hands out (a repo id is NOT a filesystem path, and treating it as one --
    the bug this resolver replaces -- makes ``/candidates`` glob a nonsense
    directory and makes ``/select`` reject a legitimate local drafter as
    outside the permitted directory, both silently).

    A repo id is resolved against what THIS install already has cached --
    filesystem only; ``cached_gguf_for_load(..., verify_sizes=False)`` never
    reaches the Hub, since the size-verification branch it would otherwise take
    is the only network-touching part of it -- because this module's whole job
    is checking siblings of a model already on disk, not fetching fresh
    information about one that might not be.

    ``ModelConfig.from_identifier`` performs the equivalent resolution for the
    LOAD path, but was rejected here on cost: its remote branch is a genuine
    network + detection pipeline (a Hub listing, vision detection, an optional
    transformers import -- see the comment at routes/inference.py's own call
    site), and that belongs to a one-time model load, not a value read on every
    candidate list and every drafter pick in a settings sidebar.

    A bare repo id with no ``gguf_variant`` is deliberately unresolvable: with
    more than one quant cached there would be no way to know which one is
    meant, and with none cached there is nothing to find either way.
    """
    if not model_id:
        return None
    try:
        if is_local_path(model_id):
            return _resolve_real(model_id)
        cached = cached_gguf_for_load(model_id, gguf_variant, verify_sizes = False)
    except Exception:
        logger.debug(
            "Could not resolve model id %r for drafter lookup", model_id, exc_info = True
        )
        return None
    if not cached:
        return None
    return _resolve_real(cached)


class ChoiceIn(BaseModel):
    kind: str
    ref: str


class SelectIn(BaseModel):
    model_id: Optional[str] = None
    gguf_variant: Optional[str] = None
    existing_args: list[str] = []
    choice: Optional[ChoiceIn] = None


class CurrentIn(BaseModel):
    existing_args: list[str] = []


@router.post("/current")
def current_draft_model(
    payload: CurrentIn,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    """The drafter ``existing_args`` currently pins, or ``null``.

    The frontend's single source of truth for what is pinned: it seeds the
    picker's controls from this and re-validates from this, and never inspects
    the argument list itself. That is not a stylistic preference -- the
    vocabulary is seven spellings in two value forms with ``_``/``-``
    normalisation on top, and a picker that reimplemented a subset of it
    displayed "Automatic" for a model that was in fact pinned.

    POST rather than GET because the input is an argument LIST; encoding one in
    a query string would be a third place for the two sides to disagree about
    how arguments are spelled.
    """
    pin = current_draft_pin(payload.existing_args)
    if pin is None:
        return {"pin": None}
    kind, ref = pin
    return {"pin": {"kind": kind, "ref": ref}}


@router.get("/candidates")
def list_candidates(
    model_id: str = Query("", max_length = 4096),
    gguf_variant: Optional[str] = Query(None, max_length = 256),
    current_subject: str = Depends(get_current_subject),
) -> dict:
    """Drafters selectable for ``model_id``.

    Colocated GGUFs only. The target itself is excluded: a model cannot draft
    for itself, and offering it invites a load that wastes VRAM on a second
    copy of the same weights.

    ``resolved`` is false when ``model_id`` could not be resolved to a real
    local file at all, as distinct from resolving fine and finding no
    siblings. The two must never collapse into the same empty list: that would
    have a resolution failure read to the UI as "this model genuinely has no
    drafters nearby" rather than "this model could not even be checked".
    """
    if not model_id:
        return {"candidates": [], "resolved": True}
    target = _resolve_model_path(model_id, gguf_variant)
    if target is None:
        return {"candidates": [], "resolved": False}
    out: list[dict] = []
    if target.parent.is_dir():
        for f in sorted(target.parent.glob("*.gguf")):
            if _resolve_real(str(f)) == target:
                continue
            out.append({
                "kind": "local",
                "ref": str(f),
                "label": f.name,
                "source": "sidecar" if _looks_like_sidecar(f.name) else "local",
            })
    return {"candidates": out, "resolved": True}


def _looks_like_sidecar(name: str) -> bool:
    """Whether auto-discovery would treat this as a drafter sidecar. Labelling
    only -- it changes how the entry is presented, never whether it is offered,
    so a naming convention that drifts upstream cannot hide a valid choice."""
    low = name.lower()
    return low.startswith(("mtp-", "dspark-", "dflash-")) or "-mtp-" in low


@router.post("/select")
def select_draft_model(
    payload: SelectIn,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    """Validate a choice and return the llama_extra_args that pin it.

    A rejected choice is a 200 carrying a verdict, not an error status: the UI
    renders the reason inline next to the picker, and an HTTP error would make
    a normal, expected outcome look like a fault.
    """
    if payload.choice is None:
        return {
            "ok": True, "reason": VERDICT_OK, "detail": "",
            "size_bytes": None, "vocab_target": None, "vocab_draft": None,
            "llama_extra_args": compose_draft_args(payload.existing_args, None),
        }
    choice = (payload.choice.kind, payload.choice.ref)
    target_path: Optional[str] = None
    if choice[0] != "hf":
        # An "hf" choice never looks at the target's own path (validate_choice
        # returns before it gets there), so resolving one here would spend a
        # cache scan on a value that outcome can never use.
        if payload.model_id:
            resolved = _resolve_model_path(payload.model_id, payload.gguf_variant)
            if resolved is None:
                # Given, but unresolvable -- distinct from validate_choice's own
                # VERDICT_NO_TARGET, which means no model_id was given at all.
                return {
                    "ok": False, "reason": VERDICT_UNRESOLVED,
                    "detail": f"could not resolve '{payload.model_id}' to a local file",
                    "size_bytes": None, "vocab_target": None, "vocab_draft": None,
                    "llama_extra_args": None,
                }
            target_path = str(resolved)
    verdict = validate_choice(target_path, choice)
    args = compose_draft_args(payload.existing_args, choice) if verdict.ok else None
    return {
        "ok": verdict.ok, "reason": verdict.reason, "detail": verdict.detail,
        "size_bytes": verdict.size_bytes,
        "vocab_target": verdict.vocab_target, "vocab_draft": verdict.vocab_draft,
        "llama_extra_args": args,
    }
