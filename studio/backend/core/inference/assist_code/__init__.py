# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read-only language intelligence (LSP) tools for Studio's agent loop.

``execute`` is the single entry point ``execute_tool`` delegates to. It always
returns ``str`` and never raises: the tool boundary is a string, so an
escaping exception would land in the agent loop instead of becoming text the
model can read and act on.

Three exception types are allowed to reach ``execute`` from the handlers
below, deliberately, because Tasks 6 and 7's ``diagnostics.py`` and
``navigation.py`` do not swallow them (see their module docstrings for the
full argument): ``session.DocumentReadError`` (the file itself could not be
opened -- missing, unreadable, or over the 5 MB cap), ``jsonrpc.LspError``
(the server is alive and answered, but rejected the request) and
``jsonrpc.LspClosed`` (the transport died). ``execute`` is the layer that
renders each one distinctly instead of collapsing them into one generic
failure string -- and, just as important, never lets any of them render as
an *empty result*. An empty result from these tools is a legitimate answer
(a clean file, a symbol nothing references), so a failure that looked like
one would tell the model "no problems found" when the language server
actually crashed, and it would draw a confidently wrong conclusion from that.
A bare ``except Exception`` is kept too, but strictly as the last-resort
fallback beneath the three specific cases -- not the primary path.
"""
import contextlib
import os

from .schemas import ASSIST_CODE_TOOLS, ASSIST_CODE_TOOL_NAMES  # noqa: F401

_MAX_RESULTS = 50


@contextlib.contextmanager
def _session_for(path, session_id, start_timeout):
    """Resolve path, pick a language and workspace, and lease a pooled session.

    A context manager, not a bare accessor, and that is load-bearing:
    ``pool.lease()`` (not ``pool.acquire()``) marks the session in-flight for
    as long as it is held, so eviction and idle-reaping cannot close it out
    from under a request that is still using it. ``pool.acquire()`` alone
    gives no such protection -- see pool.py's own docstring for the failure
    this caused before ``lease()`` existed: an unrelated tool call against a
    different workspace could evict and close a session another call was
    mid-request against, which then saw an unrelated ``LspClosed`` instead of
    its own request completing.

    Each handler below opens exactly one of these per call and does all of
    its work for that call inside the ``with`` block, so there is never more
    than one lease open on the same (language, root) key at a time.

    Yields ``(session, resolved_path, None)`` on success, or
    ``(None, None, error_text)`` if the path could not be resolved, no
    server exists for its language, or the server could not be installed or
    started (``servers.ServerUnavailable`` / ``session.SessionStartFailed``).
    A caller sees no difference between those three failure modes; each is
    turned into user-facing text here rather than a raw exception.

    Once inside the ``with`` block, further failures against the *leased*
    session -- ``session.DocumentReadError``, ``jsonrpc.LspError``,
    ``jsonrpc.LspClosed`` -- are deliberately NOT caught here. They propagate
    through this generator (``pool.lease()``'s own ``finally`` still runs and
    releases the lease correctly on the way out -- see pool.py) up to
    ``execute()``, which is the one place that renders each distinctly. This
    function catching them early would erase that distinction a layer too
    soon.
    """
    from . import paths, pool, servers, session as sess

    resolved, err = paths.resolve_file(path, session_id = session_id)
    if err:
        yield None, None, err
        return

    language = servers.language_for(resolved)
    if language is None:
        yield None, None, (
            f"no language server for {os.path.basename(resolved)}. "
            f"Supported file types: .ts .tsx .js .jsx .cs"
        )
        return
    try:
        command = servers.server_command(language)
    except servers.ServerUnavailable as e:
        yield None, None, str(e)
        return

    root = paths.workspace_for(resolved, session_id = session_id)

    def factory(resolved_root):
        return sess.Session(command, resolved_root, language = language)

    # The except below must cover the ACQUISITION only, never the yield.
    #
    # With `yield` inside the `try`, a handler body that raised
    # SessionStartFailed would have it thrown back in at the yield by
    # contextlib, caught here as if the server had failed to start, and
    # answered with a second yield -- which makes contextlib raise
    # RuntimeError("generator didn't stop after throw()") and destroys the
    # real error on its way past. Inert today (no handler raises it), but
    # the failure mode is silent replacement of a genuine error, so it is
    # not a shape to leave lying around.
    #
    # ExitStack separates the two: enter_context() is the only thing the
    # except can see, and the yield then happens under `with stack`, which
    # still releases the lease on any exception -- including one thrown in
    # at the yield.
    stack = contextlib.ExitStack()
    try:
        session = stack.enter_context(
            pool.lease(language, root, factory, start_timeout = start_timeout))
    except sess.SessionStartFailed as e:
        stack.close()
        yield None, None, str(e)
        return
    with stack:
        yield session, resolved, None


_MAX_SERVER_MESSAGE = 200


def _short(message):
    """First line of a server message, capped to ``_MAX_SERVER_MESSAGE`` chars.

    A language server's rejection message is not written for a model. tsserver
    answers a request it cannot serve with a one-line summary followed by a
    full JavaScript stack trace -- a dozen-plus lines of node_modules paths
    and line numbers with no bearing on anything the model can act on.
    Interpolated whole, that displaced real context with noise.

    The first line is the part that carries the diagnosis ("No Project.",
    "Could not find source file"); everything after it is provenance for a
    JS developer debugging the server itself. Truncation is marked with an
    ellipsis rather than silent, so a genuinely long single-line message is
    never mistaken for a complete one.
    """
    text = str(message or "").strip()
    if not text:
        return ""
    first = text.splitlines()[0].strip()
    truncated = len(first) > _MAX_SERVER_MESSAGE or "\n" in text.strip()
    if len(first) > _MAX_SERVER_MESSAGE:
        first = first[:_MAX_SERVER_MESSAGE].rstrip()
    return f"{first}..." if truncated else first


def _rel(path, session_id):
    try:
        from core.inference import tools as _tools
        return os.path.relpath(path, _tools._get_workdir(session_id))
    except Exception:
        return path


def _outside_workdir(path, session_id):
    """Whether ``path`` lies outside this conversation's working directory.

    Fails CLOSED. If the check itself cannot be completed -- no workdir, an
    unreadable path, anything -- the location is treated as outside and
    withheld. A confinement check that fails open is not a confinement check;
    the cost of a false withhold is a missing line the caller is told about,
    and the cost of a false admit is the disclosure this exists to prevent.
    """
    if not path:
        return True
    try:
        from core.inference import tools as _tools
        return bool(_tools._is_outside_workdir(
            os.path.abspath(path), _tools._get_workdir(session_id)))
    except Exception:
        return True


def _withheld_note(count):
    return (
        f"{count} result(s) outside this conversation's working directory "
        f"were withheld."
    )


def _format_locations(locations, session_id, empty_message):
    """Render locations, dropping any that fall outside the sandbox.

    Confinement was enforced only on tool ARGUMENTS, which is not where a
    language server's answers come from. Its program graph is built from the
    code, not from us: an ordinary relative import or a tsconfig ``include``
    legitimately reaches outside the workspace root, and every location that
    came back was rendered for the model unfiltered. Demonstrated against the
    real server -- ``code_definition`` answering ``..\\wb-outside\\secret.ts``,
    ``code_symbols`` listing a constant declared there. The model can write
    the triggering file itself (``edit_file``, ``python`` and ``terminal``
    are all Studio tools), so this was reachable by argument choice plus one
    file write, and it bypassed the file-read confinement every other tool
    honours.

    The withheld COUNT is reported, and that is the load-bearing half. An
    escape must never render as a clean empty result: "No references found"
    is a legitimate answer that reads as "nothing calls this, safe to
    delete," and silently filtering every hit into that same text would turn
    a confinement action into a false fact about the code. When everything
    was withheld, the count is returned INSTEAD of the empty message, not
    alongside it -- saying "no references found" there would be untrue;
    references were found, they were just not ours to show.
    """
    kept, withheld = [], 0
    for loc in locations or []:
        if _outside_workdir(loc.get("path"), session_id):
            withheld += 1
        else:
            kept.append(loc)

    if not kept:
        return _withheld_note(withheld) if withheld else empty_message

    lines = []
    for loc in kept[:_MAX_RESULTS]:
        label = loc.get("name")
        prefix = f"{label} ({loc.get('kind', 'symbol')}) - " if label else ""
        lines.append(f"{prefix}{_rel(loc['path'], session_id)}:{loc['line']}:{loc['column']}")
    if len(kept) > _MAX_RESULTS:
        lines.append(f"... and {len(kept) - _MAX_RESULTS} more")
    if withheld:
        lines.append(_withheld_note(withheld))
    return "\n".join(lines)


def _position(session, resolved, arguments):
    from . import navigation
    return navigation.resolve_position(
        session, resolved,
        symbol = arguments.get("symbol"),
        line = arguments.get("line"),
        column = arguments.get("column"),
    )


def _do_diagnostics(arguments, session_id, budget):
    from . import diagnostics
    with _session_for(arguments.get("path"), session_id, budget) as (session, resolved, err):
        if err:
            return err
        found = diagnostics.collect(session, resolved, timeout = min(budget, 20.0))
        if not found:
            return f"No problems found in {_rel(resolved, session_id)}."
        lines = [f"{len(found)} problem(s) in {_rel(resolved, session_id)}:"]
        for d in found[:_MAX_RESULTS]:
            lines.append(f"  {d['line']}:{d['column']}  {d['severity']}: {d['message']}")
        if len(found) > _MAX_RESULTS:
            lines.append(f"  ... and {len(found) - _MAX_RESULTS} more")
        return "\n".join(lines)


def _do_definition(arguments, session_id, budget):
    from . import navigation
    with _session_for(arguments.get("path"), session_id, budget) as (session, resolved, err):
        if err:
            return err
        position, err = _position(session, resolved, arguments)
        if err:
            return err
        locs = navigation.definition(session, resolved, position, timeout = min(budget, 20.0))
        return _format_locations(locs, session_id, "No definition found.")


def _do_references(arguments, session_id, budget):
    from . import navigation
    with _session_for(arguments.get("path"), session_id, budget) as (session, resolved, err):
        if err:
            return err
        position, err = _position(session, resolved, arguments)
        if err:
            return err
        locs = navigation.references(session, resolved, position, timeout = min(budget, 20.0))
        return _format_locations(locs, session_id, "No references found.")


def _do_hover(arguments, session_id, budget):
    """Type information at a position, withheld if it describes an outside file.

    Hover is the worst of the five for disclosure, and unlike the location
    tools it cannot be fixed by filtering a path out of the output: the
    disclosure is the text itself. Measured against the real server, hovering
    a ``const`` imported from outside the sandbox returned

        (alias) const SECRET_VALUE: "sk-live-DEADBEEF"
        TOP SECRET: the deploy key is sk-live-DEADBEEF

    TypeScript's literal-type inference puts a const string's VALUE into the
    signature verbatim, and JSDoc comes through as-is. That is decisive
    between the two remedies on the table: "strip everything but the
    signature line" would have kept the first line, and the first line IS the
    secret. So the content is suppressed in full when its definition lands
    outside the working directory.

    The check is the definition's location, resolved before the hover text is
    ever fetched. ``any`` outside, not ``all``: a merged declaration can be
    part-inside and part-outside, and hover returns one blob with no way to
    attribute which half a given line came from -- so one outside origin
    taints the answer.

    Two non-obvious cases, both deliberate:

    * No definition located at all (``[]``) -- the hover is shown. Nothing was
      located, so there is no outside file whose content this could be
      disclosing; the text is inference from the file already open, which is
      inside by construction. Suppressing here would cost the tool most of
      its value on ordinary inferred types for no gain.
    * The definition lookup TIMED OUT -- the hover is withheld. This is why
      ``definition_origins`` exists rather than reusing ``definition()``,
      which folds a timeout into the same ``[]`` as "none found." Those must
      not be the same answer to a confinement check: unverified has to fail
      closed, or a slow server becomes the way through.

    The known cost, accepted: a symbol defined in TypeScript's own bundled
    ``lib.*.d.ts`` (``console``, ``Array``, ``Promise``) resolves outside the
    workdir and so is withheld, even though its content is public. The rule
    is unconditional by design -- it cannot know which outside file is
    harmless -- and the model is told the text was withheld rather than being
    given a wrong answer.
    """
    from . import navigation
    with _session_for(arguments.get("path"), session_id, budget) as (session, resolved, err):
        if err:
            return err
        position, err = _position(session, resolved, arguments)
        if err:
            return err
        per_request = min(budget, 20.0)
        origins, timed_out = navigation.definition_origins(
            session, resolved, position, timeout = per_request)
        if timed_out:
            return (
                "Type information was withheld: the definition lookup timed out, "
                "so it could not be confirmed that this symbol is defined inside "
                "this conversation's working directory."
            )
        if any(_outside_workdir(o.get("path"), session_id) for o in origins):
            return (
                "Type information was withheld: this symbol is defined outside "
                "this conversation's working directory."
            )
        text = navigation.hover(session, resolved, position, timeout = per_request)
        return text or "No type information available at that position."


def _do_symbols(arguments, session_id, budget):
    """Workspace-wide symbol search.

    Opening a document first is not optional here, and this was the one
    handler of the five that skipped it. ``workspace/symbol`` reads like a
    workspace-level request that needs no particular file, and the resolved
    path looks like it is only there to pick the workspace root -- which is
    why it was originally discarded. But tsserver has no project loaded until
    a document is opened, so a ``workspace/symbol`` sent first thing after the
    handshake is rejected outright with ``No Project`` (plus a dozen lines of
    JavaScript stack trace, which is what ``execute``'s ``_short`` now caps).
    Passing ``file_path`` to ``navigation.symbols`` is what opens it, and puts
    this request behind the same readiness gate as the other three.

    That made ``code_symbols`` fail on the FIRST call of every session and
    succeed on every call after, because any other tool opens a document on
    its way through. The ordering hid the bug from the suite: a test that runs
    after another tool passes whether or not this line is here. It also made
    the failure land on exactly the case the tool exists for -- "find this
    name, I don't know which file holds it" is what a model asks *before* it
    has opened anything.
    """
    from . import navigation
    query = arguments.get("query")
    if not query or not str(query).strip():
        return "query is required"
    with _session_for(arguments.get("path"), session_id, budget) as (session, resolved, err):
        if err:
            return err
        found = navigation.symbols(
            session, str(query).strip(), file_path = resolved, timeout = min(budget, 20.0))
        return _format_locations(found, session_id, f"No symbol matching {query!r} was found.")


_HANDLERS = {
    "code_diagnostics": _do_diagnostics,
    "code_definition": _do_definition,
    "code_references": _do_references,
    "code_hover": _do_hover,
    "code_symbols": _do_symbols,
}


def execute(name, arguments, *, session_id = None, timeout = None, cancel_event = None):
    """Run a code-intelligence tool. Always returns str; never raises.

    See the module docstring for why ``session.DocumentReadError``,
    ``jsonrpc.LspError`` and ``jsonrpc.LspClosed`` get their own ``except``
    clauses here instead of falling into the generic handler: each tells the
    model something different happened than "no results," and folding any of
    them into the same text as the others -- or worse, into an empty result
    a caller reads as a clean answer -- would erase a distinction the model
    needs to react correctly (retry a crashed server; don't retry a file that
    doesn't exist; don't treat a rejected request as "nothing found").
    """
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"unknown code tool: {name}"
    if cancel_event is not None and cancel_event.is_set():
        return f"{name} was cancelled before it started"
    try:
        budget = float(timeout) if timeout else 60.0
    except (TypeError, ValueError):
        budget = 60.0

    from . import jsonrpc
    from .session import DocumentReadError

    try:
        return handler(arguments or {}, session_id, budget)
    except DocumentReadError as e:
        return f"{name}: could not read the file -- {e}"
    except jsonrpc.LspError as e:
        return (
            f"{name}: the language server rejected this request "
            f"(code {e.code}): {_short(e.server_message or e)}"
        )
    except jsonrpc.LspClosed as e:
        return (
            f"{name}: the language server crashed or its connection closed "
            f"unexpectedly ({e}). This is a transport failure, not a "
            f"\"nothing found\" result -- retrying may work, since a fresh "
            f"session will be started for the next call."
        )
    except Exception as e:  # noqa: BLE001 - last-resort guard, not the primary path
        return f"{name} failed: {type(e).__name__}: {e}"
