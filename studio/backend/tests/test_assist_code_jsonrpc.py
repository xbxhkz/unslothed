# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import io
import os
import subprocess
import sys

import pytest

from core.inference.assist_code import jsonrpc

FAKE = os.path.join(os.path.dirname(__file__), "lsp_fake_server.py")


def _spawn(mode = "normal"):
    return subprocess.Popen(
        [sys.executable, FAKE, mode],
        stdin = subprocess.PIPE, stdout = subprocess.PIPE, stderr = subprocess.DEVNULL,
    )


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
        """Byte counting, not line reading -- the classic framing bug."""
        payload = {"jsonrpc": "2.0", "id": 1, "result": "x" * 5000}
        assert jsonrpc.read_message(io.BytesIO(jsonrpc.encode(payload))) == payload

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
