# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Diagnostics, by whichever mechanism the server actually supports.

Classic LSP has no request for "what is wrong in this file": the server emits
textDocument/publishDiagnostics unsolicited after didOpen. LSP 3.17 added a
pull request (textDocument/diagnostic) but support is uneven, so the path is
chosen from what the server advertised at handshake rather than per language.

An empty result is a CLEAN FILE, not a failure -- including when the wait times
out, because a server with nothing to report may simply publish nothing.

``collect`` deliberately does NOT catch ``session.DocumentReadError``. A file
that cannot be read (missing, unreadable, over the size cap) is a different
kind of answer than "no diagnostics": returning [] for it would report a
clean bill of health for a file nobody actually looked at, which is worse
than no answer at all. It is left to propagate so the caller (the tool layer,
Task 8) can turn it into user-facing text the same way it already handles
other session-level failures, rather than this module silently downgrading a
read failure into a false "no problems found".
"""
from . import jsonrpc
from .session import path_to_uri

SEVERITY = {1: "error", 2: "warning", 3: "info", 4: "hint"}


def _normalise(items):
    out = []
    for item in items or []:
        rng = (item.get("range") or {}).get("start") or {}
        out.append({
            "severity": SEVERITY.get(item.get("severity"), "info"),
            # LSP is 0-based; every compiler and human is 1-based.
            "line": int(rng.get("line", 0)) + 1,
            "column": int(rng.get("character", 0)) + 1,
            "message": str(item.get("message", "")).strip(),
        })
    out.sort(key = lambda d: (d["line"], d["column"]))
    return out


def collect(session, file_path, *, timeout = 15.0):
    """Diagnostics for ``file_path``. Returns a list; [] means clean.

    Raises ``session.DocumentReadError`` if ``file_path`` itself could not be
    opened -- see the module docstring for why that is not swallowed here.
    """
    session.open_document(file_path)
    uri = path_to_uri(file_path)

    if session.supports("diagnosticProvider"):
        try:
            result = session.request(
                "textDocument/diagnostic",
                {"textDocument": {"uri": uri}},
                timeout = timeout,
            )
        except (jsonrpc.LspTimeout, jsonrpc.LspClosed):
            return []
        return _normalise((result or {}).get("items"))

    note = session.wait_notification(
        "textDocument/publishDiagnostics",
        lambda p: p.get("uri") == uri,
        timeout = timeout,
    )
    if note is None:
        return []
    return _normalise(note.get("diagnostics"))
