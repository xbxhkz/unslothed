# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import io
import os
import subprocess
import sys
import time

import pytest

from core.inference.assist_code import jsonrpc

FAKE = os.path.join(os.path.dirname(__file__), "lsp_fake_server.py")


def _spawn(mode = "normal"):
    return subprocess.Popen(
        [sys.executable, FAKE, mode],
        stdin = subprocess.PIPE, stdout = subprocess.PIPE, stderr = subprocess.DEVNULL,
    )


class _ChunkedStream:
    """A stream whose .read(n) deliberately returns fewer than n bytes at a
    time -- a genuinely fragmenting stream, unlike the two things this test
    file otherwise reads from. io.BytesIO.read(n) always returns all n bytes
    in a single call, and subprocess.Popen's default-buffered stdout wraps
    the pipe in io.BufferedReader, whose .read(n) already retries internally
    until n bytes arrive or real EOF. Neither exercises the accumulation
    loop in jsonrpc.read_message, so a test driven only against those two
    cannot detect a regression in it (confirmed directly: see the Task 2
    report's negative control, and Minor 1 of the round-1 review)."""

    def __init__(self, data, chunk_size = 3):
        self._data = data
        self._chunk_size = chunk_size
        self._pos = 0

    def readline(self):
        nl = self._data.find(b"\n", self._pos)
        end = len(self._data) if nl == -1 else nl + 1
        chunk = self._data[self._pos:end]
        self._pos = end
        return chunk

    def read(self, n):
        end = min(self._pos + min(n, self._chunk_size), len(self._data))
        chunk = self._data[self._pos:end]
        self._pos = end
        return chunk


class TestFraming:
    def test_encode_emits_a_content_length_header_and_a_blank_line(self):
        raw = jsonrpc.encode({"a": 1})
        head, _, body = raw.partition(b"\r\n\r\n")
        assert head == b"Content-Length: %d" % len(body)
        assert body == b'{"a": 1}'

    def test_read_message_round_trips_what_encode_wrote(self):
        payload = {"jsonrpc": "2.0", "id": 7, "result": {"ok": True}}
        assert jsonrpc.read_message(io.BytesIO(jsonrpc.encode(payload))) == payload

    def test_read_message_returns_none_at_eof(self):
        assert jsonrpc.read_message(io.BytesIO(b"")) is None

    def test_a_body_split_across_reads_is_reassembled(self):
        """Byte counting, not line reading -- the classic framing bug.

        Driven against _ChunkedStream, not io.BytesIO: BytesIO.read(n) never
        fragments, so it cannot exercise the accumulation loop this test
        exists to protect (round-1 review, Minor 1).
        """
        payload = {"jsonrpc": "2.0", "id": 1, "result": "x" * 5000}
        stream = _ChunkedStream(jsonrpc.encode(payload), chunk_size = 3)
        assert jsonrpc.read_message(stream) == payload

    def test_utf8_multibyte_is_counted_in_bytes_not_characters(self):
        payload = {"jsonrpc": "2.0", "id": 1, "result": "héllo — ünïcode ✅"}
        assert jsonrpc.read_message(io.BytesIO(jsonrpc.encode(payload))) == payload


class TestTransport:
    def test_a_request_gets_its_own_reply(self):
        t = jsonrpc.Transport(_spawn())
        try:
            reply = t.request("initialize", {"processId": None}, timeout = 10)
            assert "capabilities" in reply
        finally:
            t.close()

    def test_replies_are_correlated_by_id_not_by_arrival_order(self):
        t = jsonrpc.Transport(_spawn("noisy"))
        try:
            for _ in range(5):
                assert t.request("initialize", {}, timeout = 10) is not None
        finally:
            t.close()

    def test_an_unsolicited_notification_does_not_satisfy_a_request(self):
        """The noisy server emits logMessage before every reply."""
        t = jsonrpc.Transport(_spawn("noisy"))
        try:
            reply = t.request("initialize", {}, timeout = 10)
            assert "capabilities" in reply
        finally:
            t.close()

    def test_wait_notification_returns_a_pushed_message(self):
        t = jsonrpc.Transport(_spawn())
        try:
            t.request("initialize", {}, timeout = 10)
            t.notify("textDocument/didOpen", {"textDocument": {"uri": "file:///a.ts"}})
            note = t.wait_notification(
                "textDocument/publishDiagnostics",
                lambda p: p.get("uri") == "file:///a.ts",
                timeout = 10,
            )
            assert note is not None
            assert note["diagnostics"][0]["severity"] == 1
        finally:
            t.close()

    def test_a_wedged_server_times_out_rather_than_hanging(self):
        t = jsonrpc.Transport(_spawn("wedged"))
        try:
            t.request("initialize", {}, timeout = 10)
            with pytest.raises(jsonrpc.LspTimeout):
                t.request("textDocument/definition", {}, timeout = 1)
        finally:
            t.close()

    def test_a_dead_server_raises_closed_not_timeout(self):
        """A crash must be distinguishable from slowness -- they need
        different responses (restart vs wait)."""
        t = jsonrpc.Transport(_spawn("crash"))
        try:
            t.request("initialize", {}, timeout = 10)
            with pytest.raises(jsonrpc.LspClosed):
                t.request("textDocument/definition", {}, timeout = 10)
        finally:
            t.close()

    def test_wait_notification_that_never_arrives_returns_none(self):
        t = jsonrpc.Transport(_spawn("wedged"))
        try:
            t.request("initialize", {}, timeout = 10)
            assert t.wait_notification("nope/never", lambda p: True, timeout = 1) is None
        finally:
            t.close()

    def test_close_terminates_the_process(self):
        proc = _spawn()
        t = jsonrpc.Transport(proc)
        t.request("initialize", {}, timeout = 10)
        t.close()
        assert proc.poll() is not None

    def test_a_deaf_server_times_out_on_the_first_initialize(self):
        """deaf never comes up at all -- unlike wedged, it never even answers
        the handshake. Task 3 needs this to test start failure."""
        t = jsonrpc.Transport(_spawn("deaf"))
        try:
            with pytest.raises(jsonrpc.LspTimeout):
                t.request("initialize", {}, timeout = 1)
        finally:
            t.close()

    def test_a_late_reply_after_client_timeout_is_not_leaked(self):
        """A slow-but-not-dead server answering after the caller already gave
        up must not leave its reply parked in `_replies` forever. Unbounded
        growth over a long-lived session against a server that is
        occasionally slower than a caller's timeout -- cold-start indexing,
        workspace-symbol search on a large repo -- is the normal case for
        these servers, not the pathological one (round-1 review, Important 2).
        """
        t = jsonrpc.Transport(_spawn("slow"))
        try:
            t.request("initialize", {}, timeout = 10)
            with pytest.raises(jsonrpc.LspTimeout):
                t.request("textDocument/definition", {}, timeout = 0.2)
            # The slow server answers ~0.8s after receiving the request (see
            # SLOW_DELAY in lsp_fake_server.py). Give its late reply time to
            # actually arrive and be processed by the reader thread.
            time.sleep(1.5)
            assert t._replies == {}
        finally:
            t.close()
