# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The licence gate, and the acceptance path that makes face_swap usable.

Two things are held here.

(1) THE CARRY-FORWARD INVARIANT, which nothing asserted before: no schema may
expose a licence-accepting parameter, and no tool handler may call
``record_licence_acceptance()``. It was satisfied only by reading, so someone
adding an ``accept_licence`` parameter later would have shipped green. The
check walks the real schemas and parses the real handler sources.

(2) THE ACCEPTANCE PATH itself. ``record_licence_acceptance()`` previously had
no route, no CLI, no UI and no documentation, so no user could satisfy the gate
through any supported path while the schema was sent to the model every turn.

On what these tests do NOT claim: the gate is advisory. An agent with
``terminal`` in the same process can write the marker file directly, and no
test here pretends otherwise. The guarantee asserted is the narrower real one
-- acceptance cannot be triggered through the conversation: not by a tool, not
by a schema parameter, not by an argument value.
"""

import ast
import io
import pathlib

import pytest

from core.inference.assist_vision import ASSIST_VISION_TOOLS, accept_licence, face_swap

_PKG = pathlib.Path(face_swap.__file__).parent
# Every module that implements a tool. accept_licence.py is the human-facing
# entry point and is SUPPOSED to call the recorder, so it is excluded by name.
_TOOL_MODULES = sorted(
    p for p in _PKG.glob("*.py") if p.name != "accept_licence.py"
)


def _calls_in(path):
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if name:
                yield name


class TestNothingAcceptsOnTheUsersBehalf:
    def test_no_schema_exposes_a_licence_accepting_parameter(self):
        """A parameter the model can set would let it accept by calling the
        tool, which is exactly what must not be possible."""
        banned = ("licence", "license", "accept", "consent", "agree", "terms")
        for tool in ASSIST_VISION_TOOLS:
            fn = tool["function"]
            for param in fn["parameters"].get("properties", {}):
                lowered = param.lower()
                for word in banned:
                    assert word not in lowered, (
                        f"{fn['name']} exposes '{param}', which lets the model "
                        "accept the licence by calling the tool"
                    )

    def test_no_tool_module_calls_the_recorder(self):
        """The Task-4 -> Task-6 carry-forward, finally asserted. Only the
        human-facing accept_licence entry point may call this."""
        offenders = [
            p.name for p in _TOOL_MODULES
            if "record_licence_acceptance" in set(_calls_in(p))
        ]
        assert not offenders, (
            f"{offenders} call record_licence_acceptance(); acceptance must "
            "come from the human-initiated CLI alone"
        )

    def test_the_recorder_is_still_only_defined_once(self):
        """Guards against a second, quieter implementation appearing."""
        definitions = []
        for p in _PKG.glob("*.py"):
            tree = ast.parse(p.read_text(encoding = "utf-8"))
            for node in ast.walk(tree):
                if (isinstance(node, ast.FunctionDef)
                        and node.name == "record_licence_acceptance"):
                    definitions.append(p.name)
        assert definitions == ["face_swap.py"], definitions

    def test_the_refusal_names_the_command_a_human_can_run(self, tmp_path, monkeypatch):
        """The message was vague because its referent did not exist. Now it
        does, so the message must name it -- withholding it would only strand
        the human, not stop a capable agent."""
        from core.inference import tools
        from core.inference.tools import execute_tool
        from PIL import Image

        d = tmp_path / "workdir"
        d.mkdir()
        monkeypatch.setattr(tools, "_get_workdir", lambda session_id = None: str(d))
        monkeypatch.setenv("UNSLOTH_VISION_MODEL_DIR", str(tmp_path / "models"))
        for name in ("s.png", "t.png"):
            Image.new("RGB", (8, 8), (1, 2, 3)).save(d / name, format = "PNG")

        out = execute_tool(
            "face_swap",
            {"source_face_path": "s.png", "target_image_path": "t.png"},
            session_id = "lic",
        )
        assert "accept_licence" in out
        assert "python -m core.inference.assist_vision.accept_licence" in out
        # Still says the assistant must not do it itself.
        assert "cannot make it on the user's behalf" in out


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_VISION_MODEL_DIR", str(tmp_path))
    return tmp_path


class _Tty(io.StringIO):
    """A stdin double that reports itself as an interactive terminal."""

    def __init__(self, text = "", tty = True):
        super().__init__(text)
        self._tty = tty

    def isatty(self):
        return self._tty


class TestAcceptanceCli:
    def test_the_exact_phrase_records_acceptance(self, model_dir):
        out = io.StringIO()
        rc = accept_licence.main([], stdin = _Tty("I ACCEPT\n"), stdout = out)
        assert rc == 0
        assert face_swap.licence_accepted() is True

    def test_the_terms_are_printed_before_the_question(self, model_dir):
        """A person cannot accept terms they were never shown."""
        out = io.StringIO()
        accept_licence.main([], stdin = _Tty("I ACCEPT\n"), stdout = out)
        text = out.getvalue().lower()
        assert "non-commercial" in text
        assert "research" in text
        assert "inswapper_128.onnx" in text
        assert "buffalo_l" in text
        assert text.index("non-commercial") < text.index("to accept, type exactly")

    @pytest.mark.parametrize(
        "answer",
        ["", "\n", "y\n", "yes\n", "i accept\n", "I ACCEPT PLEASE\n", "ACCEPT\n"],
    )
    def test_anything_but_the_exact_phrase_declines(self, model_dir, answer):
        out = io.StringIO()
        rc = accept_licence.main([], stdin = _Tty(answer), stdout = out)
        assert rc == 1
        assert face_swap.licence_accepted() is False
        assert "declined" in out.getvalue().lower()

    def test_a_non_interactive_stdin_is_refused(self, model_dir):
        """`echo "I ACCEPT" | python -m ...` must not count as a person
        reading the terms. Advisory friction, not enforcement."""
        out = io.StringIO()
        rc = accept_licence.main([], stdin = _Tty("I ACCEPT\n", tty = False), stdout = out)
        assert rc == 1
        assert face_swap.licence_accepted() is False
        assert "non-interactive" in out.getvalue().lower()

    def test_there_is_no_flag_that_skips_the_prompt(self, model_dir):
        """A --yes flag would turn this into a one-liner an agent can fire
        blind, which is precisely what it must not be."""
        for flag in ("--yes", "-y", "--accept", "--force"):
            out = io.StringIO()
            rc = accept_licence.main([flag], stdin = _Tty("I ACCEPT\n"), stdout = out)
            assert rc == 2, f"{flag} was accepted as an argument"
            assert face_swap.licence_accepted() is False

    def test_running_it_again_is_a_harmless_no_op(self, model_dir):
        accept_licence.main([], stdin = _Tty("I ACCEPT\n"), stdout = io.StringIO())
        out = io.StringIO()
        rc = accept_licence.main([], stdin = _Tty(""), stdout = out)
        assert rc == 0
        assert "already been accepted" in out.getvalue().lower()

    def test_the_module_documents_that_the_gate_is_advisory(self):
        """Overclaiming enforcement here would be worse than the gap it
        replaced, so the honesty is pinned by a test."""
        doc = accept_licence.__doc__.lower()
        assert "advisory" in doc
        assert "not, and cannot be, technically enforceable" in doc


class TestGateStillHolds:
    def test_acceptance_actually_opens_the_gate(self, model_dir):
        """End to end: refused, accept via the CLI, allowed."""
        assert face_swap.licence_accepted() is False
        with pytest.raises(face_swap.LicenseNotAcceptedError):
            face_swap._ensure_models_available()

        accept_licence.main([], stdin = _Tty("I ACCEPT\n"), stdout = io.StringIO())

        face_swap._ensure_models_available()  # must not raise

    def test_declining_leaves_the_gate_shut(self, model_dir):
        accept_licence.main([], stdin = _Tty("no\n"), stdout = io.StringIO())
        with pytest.raises(face_swap.LicenseNotAcceptedError):
            face_swap._ensure_models_available()
