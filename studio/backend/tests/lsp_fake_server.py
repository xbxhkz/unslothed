# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A real subprocess that speaks real LSP framing, for tests.

Not a mock. It parses Content-Length headers off stdin and writes framed JSON
to stdout, so the client's framing, correlation and notification handling are
genuinely exercised. Behaviour is driven by argv so one file covers every case.

Modes:
  normal      -- replies to every request; publishes diagnostics on didOpen
  wedged      -- comes up (answers initialize), then stops answering
  deaf        -- never comes up: reads messages but never writes anything,
                 not even a reply to initialize
  crash       -- exits immediately on the first request
  pull        -- advertises diagnosticProvider and answers textDocument/diagnostic
  pull_error  -- advertises diagnosticProvider like `pull`, but answers
                 textDocument/diagnostic with a JSON-RPC *error* object
                 instead of a result -- the server is alive and answered,
                 it rejected the request. Distinct from `wedged`/no answer:
                 a caller must not treat this the same as a clean file.
  refs_error  -- like `normal`, but answers textDocument/references with a
                 JSON-RPC *error* object instead of a result. Same shape as
                 `pull_error`, for navigation.references()'s negative
                 control: a caller must not fold this into an empty
                 "no references" result -- that reads as "safe to delete."
  def_error   -- like `normal`, but answers textDocument/definition with a
                 JSON-RPC *error* object instead of a result. Same shape
                 and purpose as `refs_error`, for navigation.definition()'s
                 own negative control -- each of the four navigation
                 functions needs its own error mode so a regression that
                 widens only *that* function's except clause back to
                 swallowing LspError is still caught (see Task 7's round-1
                 review: a single shared negative control only proved one
                 function out of four).
  hover_error -- like `normal`, but answers textDocument/hover with a
                 JSON-RPC *error* object instead of a result. Same purpose
                 as `def_error`/`refs_error`, for navigation.hover().
  symbols_error -- like `normal`, but answers workspace/symbol with a
                 JSON-RPC *error* object instead of a result. Same purpose
                 as `def_error`/`refs_error`, for navigation.symbols().
  init_error  -- answers `initialize` itself with a JSON-RPC error object
                 (e.g. rejected params) rather than a result. The server is
                 alive throughout and answers shutdown/exit normally
                 afterwards -- distinct from `crash` (process death) and
                 `deaf`/`wedged` (no answer at all).
  noisy       -- emits unsolicited notifications before each reply
  slow        -- answers initialize (and shutdown) immediately, but delays
                 SLOW_DELAY seconds before answering any other request --
                 a real server that is merely slow, not dead

Optional third argv overrides SLOW_DELAY for "slow" mode (default 0.8s if
omitted, so every existing caller is unaffected). Added for pool.py's
in-flight-eviction regression test, which needs the per-request delay to
exceed jsonrpc.Transport.close()'s hardcoded 2.0s graceful-shutdown timeout
(jsonrpc.py:245): at the default 0.8s, a close() racing an in-flight "slow"
request always loses gracefully -- the fake server processes messages
strictly FIFO in one thread, so it finishes writing the pending request's
real reply (queued ahead of the shutdown message close() sends) before ever
reading shutdown, regardless of whether eviction was supposed to protect
the session. That made an earlier version of the regression test's own
negative control inert: reverting the in-flight guard didn't make it fail,
because the close() race couldn't corrupt the pending request either way.
A delay past 2.0s forces close()'s own shutdown wait to time out and fall
through to an abrupt process kill while the request is still genuinely
unanswered, which is what actually distinguishes protected from
unprotected.
"""
import json
import sys
import time

MODE = sys.argv[1] if len(sys.argv) > 1 else "normal"
SLOW_DELAY = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8  # seconds


def _read():
    length = None
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            break
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1].strip())
    if length is None:
        return None
    return json.loads(sys.stdin.buffer.read(length).decode("utf-8"))


def _write(payload):
    body = json.dumps(payload).encode("utf-8")
    sys.stdout.buffer.write(b"Content-Length: %d\r\n\r\n" % len(body))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def _capabilities():
    caps = {
        "definitionProvider": True,
        "referencesProvider": True,
        "hoverProvider": True,
        "workspaceSymbolProvider": True,
    }
    if MODE in ("pull", "pull_error"):
        caps["diagnosticProvider"] = {"interFileDependencies": False, "workspaceDiagnostics": False}
    return caps


def main():
    while True:
        msg = _read()
        if msg is None:
            return
        method = msg.get("method")
        mid = msg.get("id")

        if MODE == "deaf":
            continue  # read it, never write anything -- not even to initialize

        if MODE == "crash" and method != "initialize":
            sys.exit(3)

        if MODE == "noisy":
            _write({"jsonrpc": "2.0", "method": "window/logMessage",
                    "params": {"type": 3, "message": "chatter"}})

        if method == "initialize":
            if MODE == "init_error":
                _write({"jsonrpc": "2.0", "id": mid, "error": {
                    "code": -32602, "message": "Invalid params: unsupported rootUri",
                }})
                continue
            _write({"jsonrpc": "2.0", "id": mid,
                    "result": {"capabilities": _capabilities()}})
            continue
        if method == "shutdown":
            _write({"jsonrpc": "2.0", "id": mid, "result": None})
            continue
        if method == "exit":
            return
        if method == "textDocument/didOpen":
            if MODE in ("normal", "noisy"):
                uri = msg["params"]["textDocument"]["uri"]
                _write({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
                        "params": {"uri": uri, "diagnostics": [{
                            "range": {"start": {"line": 2, "character": 6},
                                      "end": {"line": 2, "character": 7}},
                            "severity": 1, "message": "Type 'string' is not assignable to type 'number'.",
                        }]}})
            continue
        if mid is None:
            continue  # any other notification

        if MODE == "wedged":
            continue  # read it, never answer

        if MODE == "slow":
            time.sleep(SLOW_DELAY)

        if method == "textDocument/diagnostic":
            if MODE == "pull_error":
                _write({"jsonrpc": "2.0", "id": mid, "error": {
                    "code": -32001, "message": "diagnostics unavailable: project not yet indexed",
                }})
                continue
            _write({"jsonrpc": "2.0", "id": mid, "result": {"kind": "full", "items": [{
                "range": {"start": {"line": 4, "character": 0}, "end": {"line": 4, "character": 3}},
                "severity": 2, "message": "'x' is declared but never used.",
            }]}})
            continue
        if method == "textDocument/definition":
            if MODE == "def_error":
                _write({"jsonrpc": "2.0", "id": mid, "error": {
                    "code": -32001, "message": "definition unavailable: project not yet indexed",
                }})
                continue
            _write({"jsonrpc": "2.0", "id": mid, "result": [{
                "uri": msg["params"]["textDocument"]["uri"],
                "range": {"start": {"line": 0, "character": 6},
                          "end": {"line": 0, "character": 7}},
            }]})
            continue
        if method == "textDocument/references":
            uri = msg["params"]["textDocument"]["uri"]
            if MODE == "refs_error":
                _write({"jsonrpc": "2.0", "id": mid, "error": {
                    "code": -32001, "message": "references unavailable: project not yet indexed",
                }})
                continue
            _write({"jsonrpc": "2.0", "id": mid, "result": [
                {"uri": uri, "range": {"start": {"line": 0, "character": 6},
                                       "end": {"line": 0, "character": 7}}},
                {"uri": uri, "range": {"start": {"line": 3, "character": 2},
                                       "end": {"line": 3, "character": 3}}},
            ]})
            continue
        if method == "textDocument/hover":
            if MODE == "hover_error":
                _write({"jsonrpc": "2.0", "id": mid, "error": {
                    "code": -32001, "message": "hover unavailable: project not yet indexed",
                }})
                continue
            _write({"jsonrpc": "2.0", "id": mid, "result": {
                "contents": {"kind": "markdown", "value": "```ts\nconst a: number\n```"}}})
            continue
        if method == "workspace/symbol":
            if MODE == "symbols_error":
                _write({"jsonrpc": "2.0", "id": mid, "error": {
                    "code": -32001, "message": "workspace symbol search unavailable: project not yet indexed",
                }})
                continue
            query = (msg.get("params") or {}).get("query", "")
            if query == "nothingmatches":
                _write({"jsonrpc": "2.0", "id": mid, "result": []})
                continue
            _write({"jsonrpc": "2.0", "id": mid, "result": [{
                "name": query or "a", "kind": 13,
                "location": {"uri": "file:///%s/a.ts" % "ws",
                             "range": {"start": {"line": 0, "character": 6},
                                       "end": {"line": 0, "character": 7}}},
            }]})
            continue
        _write({"jsonrpc": "2.0", "id": mid, "result": None})


if __name__ == "__main__":
    main()
