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


def _open_ready(session, file_path):
    """Open ``file_path`` and make sure the project graph actually covers it.

    Every cross-file answer in this module depends on the server having
    loaded the project, and a server that has not finished loading does not
    say so -- it answers anyway, from what it has. See
    ``Session.await_project_ready`` for the measurement and the reasoning.
    The gate runs at most once per session and fails open, so the cost is a
    fraction of a second on a session's first navigation call and nothing
    afterwards.

    ``open_document`` stays first: the readiness signal is the server's
    diagnostics for THIS document, which ``didOpen`` is what triggers.
    """
    session.open_document(file_path)
    session.await_project_ready(file_path)


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
    locations, _timed_out = definition_origins(session, file_path, position, timeout = timeout)
    return locations


def definition_origins(session, file_path, position, *, timeout = 30.0):
    """``definition()``, but with a timeout the caller can tell apart.

    ``definition()`` folds ``LspTimeout`` into ``[]``, which is right for a
    caller rendering results -- a timeout is genuinely ambiguous between
    "still working" and "nothing to report."

    It is NOT right for a caller making a confinement decision. The tool
    layer resolves a hover's definition to decide whether the hover text
    describes a file outside the sandbox, and there "no definition found"
    (safe: nothing located to disclose) and "the lookup timed out" (unknown:
    could be anywhere) must not look the same. Collapsing them would make an
    unverified symbol fail OPEN, which is the wrong direction for a check
    whose entire purpose is to withhold.

    Returns ``(locations, timed_out)``.
    """
    _open_ready(session, file_path)
    try:
        result = session.request("textDocument/definition", {
            "textDocument": {"uri": path_to_uri(file_path)}, "position": position,
        }, timeout = timeout)
    except jsonrpc.LspTimeout:
        return [], True
    return _locations(result), False


def references(session, file_path, position, *, timeout = 30.0):
    """Every usage of the symbol at ``position``, including its declaration.

    Same propagation rule as ``definition`` -- see the module docstring. An
    empty result here in particular reads as "nothing calls this," so a
    server failure must never be allowed to look like that answer.
    """
    _open_ready(session, file_path)
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


_FENCE_LINE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def _strip_fences(text):
    """Remove markdown code-fence delimiters from ``text``, line-wise.

    Round-2 review of Task 7 found a real content-loss bug in the original
    implementation: an inline regex applied to the whole string at once,
    with no notion of "opening" vs "closing" and no line awareness, that
    deleted ordinary prose immediately following ANY fence marker. That was
    fixed by restoring the distinction CommonMark actually makes: an info
    string (language tag) belongs to an OPENING fence only, never a
    closing one, so text glued onto a closer is real content and must be
    kept.

    Round-3 review found that fix was itself a narrower special case of
    that rule rather than the general one: it tracked a single
    "have we seen the first fence line yet" flag, so only the very first
    fence line in the whole text was ever treated as an opener -- every
    later fence line, including a second block's own opening tag, was
    treated as a closer and its tag leaked into the output as content
    (``'```py\\ncode1\\n```\\nmore\\n```js\\ncode2\\n```'`` produced
    ``'code1\\nmore\\njs\\ncode2'``, with the second block's "js" tag
    surviving). Real language servers -- rust-analyzer, gopls,
    typescript-language-server -- routinely emit hover as a signature
    block plus a separate, separately-tagged example block, so this is a
    realistic shape, not a theoretical one.

    The general rule: TOGGLE at every fence line, independently per
    delimiter character (backtick vs tilde -- CommonMark treats ``~~~`` as
    an equal alternative to backtick fences, and a tilde fence only closes
    with tildes, a backtick fence only with backticks; tracking each
    delimiter's open/closed state separately gets this for free, with no
    extra matching logic). An odd occurrence of a given delimiter is an
    opening fence -- its whole line, tag included, is discarded. An even
    occurrence is a closing fence -- only the marker itself is removed,
    and anything remaining on that line is kept as real content. The one
    exception: an odd (would-be-opening) occurrence with no line
    following it at all (e.g. hover text that is just "```trailing")
    cannot be confirmed as opening a block with real content inside, so
    its text is kept rather than gambled away as a tag -- this is what
    keeps the round-2 table's bare-trailing-fence case correct under the
    general toggle rule.

    A fence delimiter may be indented up to 3 spaces (CommonMark); 4 or
    more spaces is an indented code block, a different construct entirely,
    and is deliberately left untouched -- ``_FENCE_LINE`` only matches 0-3
    leading spaces, so a 4-space-indented line simply never matches and
    falls through to being kept verbatim, fence characters and all.
    """
    lines = text.split("\n")
    out = []
    in_fence = {"`": False, "~": False}
    for index, line in enumerate(lines):
        match = _FENCE_LINE.match(line)
        if match is None:
            out.append(line)
            continue
        marker, rest = match.group(1), match.group(2)
        delim = marker[0]
        if not in_fence[delim] and index < len(lines) - 1:
            in_fence[delim] = True
            continue  # opening fence: whole line (marker + tag) dropped
        in_fence[delim] = False
        if rest:
            out.append(rest)
    return "\n".join(out)


def _hover_from_result(result):
    """Given a raw ``textDocument/hover`` result, return cleaned plain text.

    Split out from ``hover()`` so the four content shapes -- and the
    markdown-fence stripping applied on top of them -- can be exercised
    directly in tests without a live session/fake-server round trip.
    """
    text = _hover_text((result or {}).get("contents"))
    # Strip markdown fences: the model gets plain text, not rendering markup.
    return _strip_fences(text).strip()


def hover(session, file_path, position, *, timeout = 30.0):
    """Plain-text hover info at ``position``. Same propagation rule as above."""
    _open_ready(session, file_path)
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


def symbols(session, query, *, file_path = None, timeout = 30.0):
    """Workspace-wide symbol search by name (or substring, server-dependent).

    Same propagation rule as ``definition``/``references``: only a timeout
    folds into ``[]``. A genuinely empty match set (the server answered, and
    nothing matched) and a server that failed to answer at all must not look
    the same to the caller.

    ``file_path`` names any file in the project. Passing it is what makes
    this work as the FIRST call in a session: ``workspace/symbol`` reads like
    a request that needs no particular file, but tsserver has no project at
    all until a document is opened and rejects it outright with ``No
    Project`` until then. Opening through ``_open_ready`` also puts this
    request behind the same project-readiness gate as the other three, so a
    cold session cannot answer it from a half-loaded graph either.

    Optional, and defaulting to None, only so the fake-server unit tests can
    drive this function without a real file on disk; every production caller
    passes it.
    """
    if file_path is not None:
        _open_ready(session, file_path)
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
