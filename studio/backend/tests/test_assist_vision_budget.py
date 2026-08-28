# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Timeout and cancellation reach the vision tools.

The delegating branch in ``execute_tool`` used to sit ABOVE
``effective_timeout`` and forwarded neither ``timeout`` nor ``cancel_event``,
while every other long-running tool got both. A first ``detect_shapes`` call
downloads ~170 MB and ``edit_image_prompt`` runs a full diffusion pass; neither
could be stopped by the user's stop button and neither had any ceiling.

These tests go through the real ``execute_tool``, because forwarding is the
thing that broke -- a test calling ``assist_vision.execute`` directly would
pass even with the branch wired above the timeout again.

On what cancellation can actually mean here: a single model inference is an
uninterruptible call inside a third-party library, so these bound the tool at
its stage boundaries (before starting, between stages, around a download) and
stop the NEXT stage. They deliberately do not assert that a running inference
was killed, because it is not.
"""

import threading

import pytest
from PIL import Image

from core.inference.tools import execute_tool

_SESSION = "vision-budget-session"


def _png(path, size = (16, 16)):
    Image.new("RGB", size, (10, 20, 30)).save(path, format = "PNG")
    return str(path)


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    from core.inference import tools

    d = tmp_path / "workdir"
    d.mkdir()
    monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(d))
    return d


class TestCancellation:
    def test_an_already_set_cancel_event_stops_before_any_model_runs(
        self, workdir, monkeypatch
    ):
        """The stop button was pressed before this tool call started."""
        from core.inference.assist_vision import bg_removal

        monkeypatch.setattr(
            bg_removal, "remove_background",
            lambda *a, **k: pytest.fail("must not run the model after cancellation"),
        )
        cancel = threading.Event()
        cancel.set()
        out = execute_tool(
            "remove_background",
            {"image_path": _png(workdir / "a.png")},
            session_id = _SESSION,
            cancel_event = cancel,
        )
        assert "cancelled" in out.lower()

    def test_cancelling_mid_call_stops_the_next_stage(self, workdir, monkeypatch):
        """Set during the resolve stage: the model stage must not start."""
        from core.inference.assist_vision import bg_removal, paths

        cancel = threading.Event()
        real_resolve = paths.resolve_image_bytes

        def _resolve_then_cancel(*a, **k):
            result = real_resolve(*a, **k)
            cancel.set()  # the user hits stop while the file is being read
            return result

        monkeypatch.setattr(paths, "resolve_image_bytes", _resolve_then_cancel)
        monkeypatch.setattr(
            bg_removal, "remove_background",
            lambda *a, **k: pytest.fail("cancelled before this stage should run"),
        )
        out = execute_tool(
            "remove_background",
            {"image_path": _png(workdir / "b.png")},
            session_id = _SESSION,
            cancel_event = cancel,
        )
        assert "cancelled" in out.lower()

    @pytest.mark.parametrize(
        "tool, args",
        [
            ("remove_background", {"image_path": "x.png"}),
            ("detect_shapes", {"image_path": "x.png"}),
            ("webcam_look", {}),
            ("edit_image_prompt", {"image_path": "x.png", "prompt": "make it blue"}),
            ("face_swap", {"source_face_path": "x.png", "target_image_path": "x.png"}),
        ],
    )
    def test_every_vision_tool_honours_cancellation(self, workdir, tool, args):
        """All five, not just the one that was easy to wire."""
        if "image_path" in args:
            _png(workdir / "x.png")
        if "source_face_path" in args:
            _png(workdir / "x.png")
        cancel = threading.Event()
        cancel.set()
        out = execute_tool(tool, args, session_id = _SESSION, cancel_event = cancel)
        assert "cancelled" in out.lower(), f"{tool} ignored the stop button: {out}"


class TestTimeout:
    def test_an_expired_deadline_stops_before_the_model_runs(self, workdir, monkeypatch):
        from core.inference.assist_vision import bg_removal

        monkeypatch.setattr(
            bg_removal, "remove_background",
            lambda *a, **k: pytest.fail("must not run the model past the deadline"),
        )
        out = execute_tool(
            "remove_background",
            {"image_path": _png(workdir / "c.png")},
            session_id = _SESSION,
            timeout = 0.0001,
        )
        assert "timed out" in out.lower()

    def test_a_generous_timeout_does_not_interfere(self, workdir, monkeypatch):
        """The budget must not fire on a normal call."""
        import io

        from core.inference.assist_vision import bg_removal

        def _fake(data, **kw):
            buf = io.BytesIO()
            Image.new("RGBA", (16, 16), (1, 2, 3, 255)).save(buf, format = "PNG")
            return buf.getvalue()

        monkeypatch.setattr(bg_removal, "remove_background", _fake)
        out = execute_tool(
            "remove_background",
            {"image_path": _png(workdir / "d.png")},
            session_id = _SESSION,
            timeout = 300,
        )
        assert "written to:" in out
        assert "timed out" not in out.lower()


class TestForwarding:
    def test_execute_tool_actually_forwards_both_values(self, workdir, monkeypatch):
        """Pins the wiring itself: the branch must sit BELOW effective_timeout
        and pass both through. Moving it back above would leave timeout None."""
        import core.inference.assist_vision as av

        seen = {}

        def _spy(name, arguments, *, session_id = None, timeout = None, cancel_event = None):
            seen["timeout"] = timeout
            seen["cancel_event"] = cancel_event
            return "ok"

        monkeypatch.setattr(av, "execute", _spy)
        cancel = threading.Event()
        execute_tool(
            "remove_background", {"image_path": "x.png"},
            session_id = _SESSION, timeout = 42, cancel_event = cancel,
        )
        assert seen["timeout"] == 42
        assert seen["cancel_event"] is cancel

    def test_an_omitted_timeout_still_arrives_as_the_default_ceiling(
        self, workdir, monkeypatch
    ):
        """timeout is _TIMEOUT_UNSET by default; the tool must receive the
        resolved default rather than that sentinel or None."""
        import core.inference.assist_vision as av
        from core.inference.tools import _EXEC_TIMEOUT

        seen = {}

        def _spy(name, arguments, *, session_id = None, timeout = None, cancel_event = None):
            seen["timeout"] = timeout
            return "ok"

        monkeypatch.setattr(av, "execute", _spy)
        execute_tool("remove_background", {"image_path": "x.png"}, session_id = _SESSION)
        assert seen["timeout"] == _EXEC_TIMEOUT
