# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A real subprocess that speaks real LSP framing, for tests.

Not a mock. It parses Content-Length headers off stdin and writes framed JSON
to stdout, so the client's framing, correlation and notification handling are
genuinely exercised. Behaviour is driven by argv so one file covers every case.

Modes:
  normal   -- replies to every request; publishes diagnostics on didOpen
  wedged   -- comes up (answers initialize), then stops answering
  deaf     -- never comes up: reads messages but never writes anything,
              not even a reply to initialize
  crash    -- exits immediately on the first request
  pull     -- advertises diagnosticProvider and answers textDocument/diagnostic
  noisy    -- emits unsolicited notifications before each reply
"""
import json
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "normal"


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
    if MODE == "pull":
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

        if method == "textDocument/diagnostic":
            _write({"jsonrpc": "2.0", "id": mid, "result": {"kind": "full", "items": [{
                "range": {"start": {"line": 4, "character": 0}, "end": {"line": 4, "character": 3}},
                "severity": 2, "message": "'x' is declared but never used.",
            }]}})
            continue
        _write({"jsonrpc": "2.0", "id": mid, "result": None})


if __name__ == "__main__":
    main()
