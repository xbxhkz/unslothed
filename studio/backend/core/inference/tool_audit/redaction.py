# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Scrub secrets before anything reaches the audit log.

Two failure directions matter equally. Missing a secret puts a credential on
disk. Over-redacting mangles ordinary arguments and quietly makes the log
useless -- which is why the test file pairs every positive case with a negative
one.

Known gap, accepted for v1 (see the design spec): a secret matching no pattern
and sitting under an innocuous key -- a bare passphrase inside a terminal
command -- still gets through. Closing it would mean decrypting the credential
store to scan every record, widening exposure in order to reduce it.
"""

from __future__ import annotations

import json
import re
from typing import Any

REDACTED = "[redacted]"

# Keys whose VALUE is secret regardless of what it looks like.
_SECRET_KEY_RE = re.compile(
    r"(?i)(pass(word|wd|phrase)?|token|secret|api[_-]?key|authorization|credential|private[_-]?key)"
)

# Values that are secret regardless of the key they sit under.
_SECRET_VALUE_RES = (
    re.compile(r"sk-[A-Za-z0-9]{16,}"),                  # OpenAI-style
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),           # GitHub
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),                     # AWS access key id
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{12,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),         # Slack
)
# Deliberately NO generic "long high-entropy string" rule. It is the highest
# false-positive layer of the four -- git SHAs, UUIDs, file hashes and base64
# payloads are all legitimate tool arguments -- and over-redaction is the
# quieter failure: it guts the log while still passing a "secrets are removed"
# test. The spec's accepted v1 gap covers what this misses.

# Argument names that carry a filesystem path, for the searchable paths_json column.
_PATH_KEYS = frozenset({"path", "file", "file_path", "filepath", "target", "directory", "dir", "cwd"})


def redact_text(text: str) -> tuple[str, bool]:
    """Replace secret-shaped runs in free text. Returns (text, was_redacted)."""
    if not isinstance(text, str) or not text:
        return text if isinstance(text, str) else "", False
    hit = False
    out = text
    for rx in _SECRET_VALUE_RES:
        out, n = rx.subn(REDACTED, out)
        if n:
            hit = True
    return out, hit


def _walk(value: Any, key: str | None) -> tuple[Any, bool]:
    if key is not None and isinstance(key, str) and _SECRET_KEY_RE.search(key):
        return REDACTED, True
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        hit = False
        out: dict[Any, Any] = {}
        for k, v in value.items():
            out[k], h = _walk(v, k if isinstance(k, str) else None)
            hit = hit or h
        return out, hit
    if isinstance(value, list):
        hit = False
        out_list = []
        for item in value:
            red, h = _walk(item, None)
            out_list.append(red)
            hit = hit or h
        return out_list, hit
    return value, False


def redact_arguments(arguments: Any) -> tuple[str, bool]:
    """Redact an argument mapping and serialise it. Returns (json, was_redacted)."""
    try:
        redacted, hit = _walk(arguments, None)
        return json.dumps(redacted, default = str, ensure_ascii = False), hit
    except Exception:
        # Never let redaction failure escape; an unserialisable argument must not
        # cost the caller its tool call.
        return json.dumps({"_unserialisable": True}), True


def extract_paths(arguments: Any) -> str:
    """Best-effort list of path-valued arguments, for search only.

    Deliberately shallow and name-based. A general mechanism would mean teaching
    this module every tool's semantics -- the coupling a registry exists to
    remove. Full arguments are stored regardless, so nothing is lost; only the
    indexed search is approximate.
    """
    found: list[str] = []
    try:
        if isinstance(arguments, dict):
            for k, v in arguments.items():
                if isinstance(k, str) and k.lower() in _PATH_KEYS and isinstance(v, str) and v:
                    found.append(v)
    except Exception:
        pass
    return json.dumps(found)
