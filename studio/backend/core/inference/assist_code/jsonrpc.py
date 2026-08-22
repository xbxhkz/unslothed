# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Content-Length framed JSON-RPC over a subprocess's stdio.

Two sinks, not one. LSP interleaves replies to our requests with notifications
the server sends unbidden -- diagnostics arrive that way -- so the reader
thread feeds BOTH a response map keyed by request id AND a notification list.
A design that only models request/response cannot express diagnostics at all.

Timeout and closure are deliberately different exceptions: a slow server should
be waited on or reported, a dead one must be restarted, and collapsing them
loses the distinction the caller needs.
"""
import itertools
import json
import threading

_HEADER_SEP = b"\r\n\r\n"


class LspTimeout(Exception):
    """A request was not answered inside its deadline."""


class LspClosed(Exception):
    """The server exited or its pipes closed."""


def encode(payload: dict) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    return b"Content-Length: %d%s%s" % (len(body), _HEADER_SEP, body)


def read_message(stream):
    """Read one framed message. Returns the dict, or None at EOF.

    Counts BYTES, not characters: a multi-byte body read as characters
    truncates, which is the classic framing bug in hand-rolled LSP clients.
    """
    length = None
    while True:
        line = stream.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            break
        if line.lower().startswith(b"content-length:"):
            try:
                length = int(line.split(b":", 1)[1].strip())
            except ValueError:
                return None
    if length is None:
        return None
    body = b""
    while len(body) < length:
        chunk = stream.read(length - len(body))
        if not chunk:
            return None
        body += chunk
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        return None


class Transport:
    """Owns a server subprocess and its reader thread."""

    def __init__(self, proc):
        self._proc = proc
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._replies = {}                 # id -> payload
        self._events = {}                  # id -> Event
        self._notifications = []           # list[(method, params)]
        self._note_cv = threading.Condition()
        self._closed = threading.Event()
        self._reader = threading.Thread(
            target = self._read_loop, name = "lsp-reader", daemon = True
        )
        self._reader.start()

    def _read_loop(self):
        try:
            while True:
                msg = read_message(self._proc.stdout)
                if msg is None:
                    break
                if "id" in msg and ("result" in msg or "error" in msg):
                    mid = msg["id"]
                    with self._lock:
                        self._replies[mid] = msg
                        ev = self._events.get(mid)
                    if ev is not None:
                        ev.set()
                elif "method" in msg:
                    with self._note_cv:
                        self._notifications.append((msg["method"], msg.get("params") or {}))
                        self._note_cv.notify_all()
        except Exception:
            pass
        finally:
            self._closed.set()
            # Wake everyone waiting; they re-check _closed and raise LspClosed.
            with self._lock:
                events = list(self._events.values())
            for ev in events:
                ev.set()
            with self._note_cv:
                self._note_cv.notify_all()

    def is_alive(self):
        """Whether the transport still believes the server is live.

        Public: callers outside this module (Session.is_healthy in Task 3)
        need to ask this without reaching into a private attribute across
        the module boundary.
        """
        return not self._closed.is_set() and self._proc.poll() is None

    def notify(self, method, params = None):
        if not self.is_alive():
            raise LspClosed(f"server is not running (notify {method})")
        try:
            self._proc.stdin.write(encode(
                {"jsonrpc": "2.0", "method": method, "params": params or {}}))
            self._proc.stdin.flush()
        except Exception as e:
            raise LspClosed(f"server pipe closed: {e}") from e

    def request(self, method, params = None, timeout = 30.0):
        if not self.is_alive():
            raise LspClosed(f"server is not running (request {method})")
        mid = next(self._ids)
        ev = threading.Event()
        with self._lock:
            self._events[mid] = ev
        try:
            try:
                self._proc.stdin.write(encode(
                    {"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}}))
                self._proc.stdin.flush()
            except Exception as e:
                raise LspClosed(f"server pipe closed: {e}") from e

            if not ev.wait(timeout):
                raise LspTimeout(f"{method} did not answer within {timeout:.0f}s")
            with self._lock:
                payload = self._replies.pop(mid, None)
            if payload is None:
                # Woken by shutdown rather than by a reply.
                raise LspClosed(f"server exited while handling {method}")
            if "error" in payload:
                err = payload["error"] or {}
                raise LspClosed(f"{method} failed: {err.get('message', err)}")
            return payload.get("result")
        finally:
            with self._lock:
                self._events.pop(mid, None)

    def wait_notification(self, method, predicate, timeout = 10.0):
        """Wait for a matching pushed notification. None if it never comes.

        Scans already-buffered notifications first: the server may publish
        before we start waiting, and a pure wait would miss it.
        """
        import time
        deadline = time.monotonic() + timeout
        seen = 0
        with self._note_cv:
            while True:
                while seen < len(self._notifications):
                    m, p = self._notifications[seen]
                    seen += 1
                    if m == method:
                        try:
                            if predicate(p):
                                return p
                        except Exception:
                            pass
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._closed.is_set():
                    return None
                self._note_cv.wait(timeout = min(remaining, 0.25))

    def close(self):
        try:
            if self.is_alive():
                try:
                    self.request("shutdown", None, timeout = 2.0)
                except Exception:
                    pass
                try:
                    self.notify("exit")
                except Exception:
                    pass
        except Exception:
            pass
        for closer in (
            lambda: self._proc.stdin.close(),
            lambda: self._proc.terminate(),
        ):
            try:
                closer()
            except Exception:
                pass
        try:
            self._proc.wait(timeout = 3)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._closed.set()
