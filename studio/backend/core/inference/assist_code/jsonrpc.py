# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Content-Length framed JSON-RPC over a subprocess's stdio.

Two sinks, not one. LSP interleaves replies to our requests with notifications
the server sends unbidden -- diagnostics arrive that way -- so the reader
thread feeds BOTH a response map keyed by request id AND a notification list.
A design that only models request/response cannot express diagnostics at all.

Timeout, closure, and a genuine protocol error are deliberately three
different exceptions, not one: a slow server should be waited on or
reported, a dead one must be restarted, and one that answered but rejected
the request is neither -- it is alive and something specific went wrong.
Collapsing any of these into another loses a distinction a caller needs
(``diagnostics.collect()`` treats a timeout as "no answer yet, maybe
nothing to report" but must not treat a real error the same way).
``LspError`` is a SIBLING of ``LspClosed``, not a subclass, so existing
``except LspClosed`` sites do not silently start swallowing it too.
"""
import collections
import itertools
import json
import threading

_HEADER_SEP = b"\r\n\r\n"


class LspTimeout(Exception):
    """A request was not answered inside its deadline."""


class LspClosed(Exception):
    """The server exited or its pipes closed."""


class LspError(Exception):
    """The server answered a request with a JSON-RPC error object.

    A SIBLING of ``LspClosed``, not a subclass -- deliberately. Subclassing
    it would still be caught by every ``except LspClosed`` already written
    for "the transport is dead, give up or fall back," which is exactly the
    outcome this type exists to prevent. A JSON-RPC error means the server
    is alive and answered; something about the request itself failed (bad
    params, a method-specific failure, ...). That is not the same situation
    as a dead transport, and code that treats a dead transport as safe to
    fall back on (e.g. diagnostics.collect() returning [] on the pull path)
    must not fall back on a genuine error the same way -- a real error
    swallowed as "clean" is worse than surfacing nothing.

    Carries the server's own error code/message/data so a caller (e.g. the
    tool layer) can render the underlying failure instead of just this
    exception's string form.
    """

    def __init__(self, method, error):
        error = error or {}
        self.method = method
        self.code = error.get("code")
        self.data = error.get("data")
        self.server_message = error.get("message")
        super().__init__(f"{method} failed: {self.server_message or error}")


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
        # Bounded so a long-lived session against a chatty server (diagnostics
        # on every edit is the normal case, not the pathological one) can't
        # grow this without limit. Each entry carries its own monotonic `seq`
        # rather than relying on positional index, so `wait_notification`
        # can track "already scanned" correctly even as old entries fall off
        # the front -- a positional cursor would silently skip an unscanned
        # entry whose index shifted underneath it.
        self._note_seq = itertools.count(1)
        self._notifications = collections.deque(maxlen = 4096)  # (seq, method, params)
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
                        # Only park the reply if someone is still waiting on
                        # it. A request that already timed out has removed
                        # its Event (see `request`'s finally); storing the
                        # reply anyway would leave it in `_replies` forever,
                        # since nothing will ever come back to collect it.
                        ev = self._events.get(mid)
                        if ev is not None:
                            self._replies[mid] = msg
                    if ev is not None:
                        ev.set()
                elif "method" in msg:
                    with self._note_cv:
                        self._notifications.append(
                            (next(self._note_seq), msg["method"], msg.get("params") or {})
                        )
                        self._note_cv.notify_all()
        except Exception:
            pass
        finally:
            # Marking closed and snapshotting waiters must be one atomic step
            # under `self._lock`. `request` checks is_alive() and registers
            # its Event under the same lock, so a registration can never land
            # in the gap between "mark closed" and "snapshot" and be missed
            # by this wake pass -- it either lands before (and is included in
            # the snapshot below) or after (and `request`'s own is_alive()
            # check, made under the same lock, already sees us closed and
            # raises LspClosed without registering an Event at all).
            with self._lock:
                self._closed.set()
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
        mid = next(self._ids)
        ev = threading.Event()
        with self._lock:
            # Atomic with _read_loop's own "mark closed, then snapshot
            # waiters" step (same lock): either we observe closed here and
            # raise without ever registering, or we register while still
            # open and are guaranteed to be included in that snapshot if the
            # reader dies afterwards. Checking is_alive() and registering as
            # two separate critical sections left a gap where a reader death
            # in between was invisible to the wake pass, so the caller waited
            # out the full timeout and got LspTimeout instead of LspClosed.
            if not self.is_alive():
                raise LspClosed(f"server is not running (request {method})")
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
                # The server is alive and answered -- it rejected the
                # request. Not LspClosed: a dead transport and a live
                # server reporting a real failure are different situations
                # and callers must be able to react differently to them.
                raise LspError(method, payload.get("error"))
            return payload.get("result")
        finally:
            with self._lock:
                self._events.pop(mid, None)
                # Defensive cleanup for the narrow window where the reader
                # stored a reply for this id (its Event was still registered
                # at that instant) moments before this pop runs. Without it,
                # that reply -- for a request nobody is waiting on anymore --
                # would sit in `_replies` forever: unbounded growth over a
                # long-lived session against a server that is occasionally
                # slower than the caller's own timeout (cold-start indexing,
                # workspace-symbol search on a large repo), which is the
                # normal case for these servers, not the pathological one.
                self._replies.pop(mid, None)

    def wait_notification(self, method, predicate, timeout = 10.0):
        """Wait for a matching pushed notification. None if it never comes.

        Scans already-buffered notifications first: the server may publish
        before we start waiting, and a pure wait would miss it.

        Tracks progress by each notification's own monotonic `seq`, not by a
        positional index into `_notifications`. `_notifications` is a bounded
        deque (see __init__): once it is at capacity, an append silently
        evicts the oldest entry, which shifts every remaining entry's index.
        A positional cursor left mid-scan across a `.wait()` call would then
        resume at the wrong offset and silently skip an entry it had not
        actually looked at yet. Comparing `seq` directly is immune to that
        shift -- it only skips entries genuinely already seen, never ones
        still sitting in the buffer.
        """
        import time
        deadline = time.monotonic() + timeout
        last_seq = 0
        with self._note_cv:
            while True:
                for seq, m, p in self._notifications:
                    if seq <= last_seq:
                        continue
                    last_seq = seq
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
