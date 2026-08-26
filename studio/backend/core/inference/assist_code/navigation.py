# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Definition, references, hover and symbol search.

Addressing is symbol-first. LSP speaks line/character, but an agent thinks in
names and usually has not read the file -- so a name is resolved to a position
by scanning the opened document, with explicit line/column as the escape hatch
when a name is ambiguous.

Every location returned is 1-based, matching compilers and humans; LSP's
0-based coordinates never leak past this module.

Only ``jsonrpc.LspTimeout`` is folded into an empty result, and only here --
see diagnostics.py's module docstring for the general argument (a timeout is
genuinely ambiguous between "still working" and "nothing to report";
``LspClosed``/``LspError`` are not). It matters more here than for
diagnostics: an empty *navigation* result is action-guiding, not just
informational. "No references found" reads as "nothing calls this, safe to
delete." Folding a dead transport or a rejected request into that same empty
list would let a caller -- ultimately the model, via Task 8's tool layer --
act on a server failure with the false confidence of a real answer. So
``jsonrpc.LspClosed`` and ``jsonrpc.LspError`` both propagate here, same as
``session.DocumentReadError`` from ``open_document``: a file that was never
actually read must not be reported as having nothing in it.
"""
import re

from . import jsonrpc
from .session import path_to_uri, uri_to_path


def _loc(item):
    location = item.get("location") or item
    uri = location.get("uri") or location.get("targetUri")
    rng = location.get("range") or location.get("targetSelectionRange") or {}
    start = rng.get("start") or {}
    return {
        "path": uri_to_path(uri) if uri else "",
        "line": int(start.get("line", 0)) + 1,
        "column": int(start.get("character", 0)) + 1,
    }


def _locations(result):
    if result is None:
        return []
    if isinstance(result, dict):
        result = [result]
    return [_loc(item) for item in result]


def _whole_number(value):
    """``int(value)``, but rejects a float with a nonzero fractional part.

    ``int(4.7)`` succeeds and silently truncates to 4 -- accepting a
    fractional line number as if it were exact, which would contradict the
    "line and column must be whole numbers" text a caller sees for every
    other malformed input (a string, ``None``, ...). Rejecting it here
    keeps the behaviour honest about the message.
    """
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{value} is not a whole number")
    return int(value)


def resolve_position(session, file_path, *, symbol = None, line = None, column = None):
    """Return an LSP position (0-based) or error text.

    Symbol resolution is a plain, lexically-unaware regex scan over the
    file's lines: it returns the first ``\\bname\\b`` match, with no notion
    of comments or string literals. A name that appears first inside a
    comment or a string literal wins over a later real declaration -- by
    design (there is no parser here, just a scan), not a bug, but worth
    knowing when a caller reports "symbol not found where expected."
    """
    if line is not None:
        try:
            zero_line = _whole_number(line) - 1
            zero_col = _whole_number(column) - 1 if column is not None else 0
        except (TypeError, ValueError):
            return None, "line and column must be whole numbers"
        if zero_line < 0 or zero_col < 0:
            return None, "line and column are 1-based, so they start at 1"
        return {"line": zero_line, "character": zero_col}, None

    if not symbol or not str(symbol).strip():
        return None, "give either 'symbol' (a name) or 'line' (1-based)"

    name = str(symbol).strip()
    try:
        with open(file_path, "r", encoding = "utf-8", errors = "replace") as fh:
            lines = fh.read().splitlines()
    except OSError as e:
        return None, f"could not read {file_path}: {e}"

    pattern = re.compile(r"\b%s\b" % re.escape(name))
    for index, text in enumerate(lines):
        match = pattern.search(text)
        if match:
            return {"line": index, "character": match.start()}, None
    return None, f"symbol {name!r} does not appear in {file_path}"


def definition(session, file_path, position, *, timeout = 30.0):
    """Locations of the definition(s) of the symbol at ``position``.

    ``session.open_document`` and ``session.request`` are deliberately left
    unguarded except for ``jsonrpc.LspTimeout``: ``session.DocumentReadError``
    (file missing/unreadable/oversized), ``jsonrpc.LspClosed`` (dead
    transport) and ``jsonrpc.LspError`` (the server rejected the request) all
    propagate. See the module docstring for why an empty list must not stand
    in for any of those here.
    """
    session.open_document(file_path)
    try:
        result = session.request("textDocument/definition", {
            "textDocument": {"uri": path_to_uri(file_path)}, "position": position,
        }, timeout = timeout)
    except jsonrpc.LspTimeout:
        return []
    return _locations(result)


def references(session, file_path, position, *, timeout = 30.0):
    """Every usage of the symbol at ``position``, including its declaration.

    Same propagation rule as ``definition`` -- see the module docstring. An
    empty result here in particular reads as "nothing calls this," so a
    server failure must never be allowed to look like that answer.
    """
    session.open_document(file_path)
    try:
        result = session.request("textDocument/references", {
            "textDocument": {"uri": path_to_uri(file_path)}, "position": position,
            "context": {"includeDeclaration": True},
        }, timeout = timeout)
    except jsonrpc.LspTimeout:
        return []
    return _locations(result)


def _hover_text(contents):
    """LSP allows a string, a {language,value} pair, a MarkupContent, or a list."""
    if contents is None:
        return ""
    if isinstance(contents, str):
        return contents
    if isinstance(contents, dict):
        return str(contents.get("value", ""))
    if isinstance(contents, list):
        return "\n".join(_hover_text(part) for part in contents)
    return str(contents)


def _hover_from_result(result):
    """Given a raw ``textDocument/hover`` result, return cleaned plain text.

    Split out from ``hover()`` so the four content shapes -- and the
    markdown-fence stripping applied on top of them -- can be exercised
    directly in tests without a live session/fake-server round trip.
    """
    text = _hover_text((result or {}).get("contents"))
    # Strip markdown fences: the model gets plain text, not rendering markup.
    cleaned = re.sub(r"```[a-zA-Z0-9_+-]*\n?", "", text).replace("```", "")
    return cleaned.strip()


def hover(session, file_path, position, *, timeout = 30.0):
    """Plain-text hover info at ``position``. Same propagation rule as above."""
    session.open_document(file_path)
    try:
        result = session.request("textDocument/hover", {
            "textDocument": {"uri": path_to_uri(file_path)}, "position": position,
        }, timeout = timeout)
    except jsonrpc.LspTimeout:
        return ""
    return _hover_from_result(result)


_SYMBOL_KINDS = {
    5: "class", 6: "method", 8: "field", 9: "constructor", 10: "enum",
    11: "interface", 12: "function", 13: "variable", 14: "constant",
    23: "struct",
}


def symbols(session, query, *, timeout = 30.0):
    """Workspace-wide symbol search by name (or substring, server-dependent).

    Same propagation rule as ``definition``/``references``: only a timeout
    folds into ``[]``. A genuinely empty match set (the server answered, and
    nothing matched) and a server that failed to answer at all must not look
    the same to the caller.
    """
    try:
        result = session.request("workspace/symbol", {"query": query}, timeout = timeout)
    except jsonrpc.LspTimeout:
        return []
    out = []
    for item in result or []:
        loc = _loc(item)
        loc["name"] = item.get("name", "")
        loc["kind"] = _SYMBOL_KINDS.get(item.get("kind"), "symbol")
        out.append(loc)
    return out
