# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Diagnostics, by whichever mechanism the server actually supports.

Classic LSP has no request for "what is wrong in this file": the server emits
textDocument/publishDiagnostics unsolicited after didOpen. LSP 3.17 added a
pull request (textDocument/diagnostic) but support is uneven, so the path is
chosen from what the server advertised at handshake rather than per language.

An empty result is a CLEAN FILE, not a failure -- including when the wait times
out, because a server with nothing to report may simply publish nothing. That
justification is specific to TIMEOUT: a request that simply hasn't answered
yet is genuinely ambiguous between "still indexing" and "nothing to report."
A dead transport (``jsonrpc.LspClosed``) or a request the server actively
rejected (``jsonrpc.LspError``) carry no such ambiguity -- the server told us
something is wrong, or told us nothing at all because it's gone. Folding
either into an empty result would report a false clean bill of health, so on
the pull path only ``jsonrpc.LspTimeout`` is caught; ``LspClosed``/``LspError``
propagate.

``collect`` deliberately does NOT catch ``session.DocumentReadError``. A file
that cannot be read (missing, unreadable, over the size cap) is a different
kind of answer than "no diagnostics": returning [] for it would report a
clean bill of health for a file nobody actually looked at, which is worse
than no answer at all. It is left to propagate so the caller (the tool layer,
Task 8) can turn it into user-facing text the same way it already handles
other session-level failures, rather than this module silently downgrading a
read failure into a false "no problems found".
"""
import os

from . import jsonrpc
from .session import is_same_file_uri, path_to_uri

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

    On the pull path, only ``jsonrpc.LspTimeout`` is folded into `[]`.
    ``jsonrpc.LspClosed`` (dead transport) and ``jsonrpc.LspError`` (the
    server answered but rejected the request) both propagate -- see the
    module docstring for why a timeout's ambiguity does not extend to them.
    """
    session.open_document(file_path)
    uri = path_to_uri(file_path)
    target = os.path.abspath(file_path)

    if session.supports("diagnosticProvider"):
        try:
            result = session.request(
                "textDocument/diagnostic",
                {"textDocument": {"uri": uri}},
                timeout = timeout,
            )
        except jsonrpc.LspTimeout:
            return []
        return _normalise((result or {}).get("items"))

    note = session.wait_notification(
        "textDocument/publishDiagnostics",
        lambda p: _is_target_uri(p.get("uri"), target),
        timeout = timeout,
    )
    if note is None:
        return []
    return _normalise(note.get("diagnostics"))


def _is_target_uri(candidate_uri, target_path):
    """Whether a pushed notification's URI names ``target_path``.

    Decodes and compares real paths rather than comparing URI strings, and
    is why this exists at all: a real server is not guaranteed to echo back
    exactly the URI string we sent it (see ``uri_to_path``'s own docstring --
    typescript-language-server's own URIs percent-encode the drive-letter
    colon, this module's ``path_to_uri`` does not), so a plain `==` between
    the two would never match and every diagnostics call against a real
    server would silently time out and read as "clean," which is precisely
    how this was found: a file with a genuine, deliberately-introduced type
    error reported "No problems found" against a real server while every
    fake-server test -- which builds its notifications with this SAME
    module's ``path_to_uri``, so the strings always matched themselves --
    stayed green. ``os.path.normcase`` on the comparison, not just
    ``os.path.abspath``, because Windows paths are case-insensitive and a
    real server is free to echo back a differently-cased drive letter or
    directory segment for the identical file, as this one did.

    The comparison itself now lives in ``session.is_same_file_uri`` so the
    project-readiness gate (``Session.await_project_ready``) matches pushed
    diagnostics by exactly the same rule; this stays as the name the module
    and its tests already use, and as the place the reasoning above is
    recorded.
    """
    return is_same_file_uri(candidate_uri, target_path)
