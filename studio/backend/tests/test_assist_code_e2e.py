# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""End-to-end against the REAL typescript-language-server.

Skipped when the server is absent, but never silently: a skip reason names it.
These are the only tests here that prove the thing actually works -- the fake
server proves the client is correct about a protocol we defined, and this
proves we were right about the protocol a real server implements.

Install with a PINNED TypeScript, not just `npm install -g
typescript-language-server typescript`: that bare command currently
resolves `typescript` to the 7.x line, a native-port preview that ships no
`tsserver.js` -- the classic JS server `typescript-language-server` actually
talks to. Every test here then fails at the LSP handshake with "Could not
find a valid TypeScript installation," which reads like a missing-server
problem, not a wrong-version one, because `shutil.which` above only checks
that `typescript-language-server` itself is on PATH; it has no way to
inspect what TypeScript it will resolve at start time. There is no
`package.json` or CI step pinning this deliberately -- a manifest or a CI
edit would widen this branch's minimal upstream seam -- so this docstring
and the skip reason below are where a fresh machine actually hitting this
will look:

    npm install -g typescript-language-server typescript@5

(5.9.3 confirmed to ship `tsserver.js`; any 5.x should.)

Two real, protocol-level bugs were found and fixed by getting these to pass
for real rather than accepting a skip (see session.py's ``uri_to_path`` and
diagnostics.py's ``_is_target_uri``): typescript-language-server (via
vscode-uri) percent-encodes a Windows drive-letter colon in the URIs it
sends back ("file:///c%3A/..."), which this module's own ``path_to_uri``
never does, and every fake-server test builds its fixtures with THIS
module's own encoder -- so the round trip always matched itself and neither
divergence ever showed up until a real server was in the loop.

Two fixture-shaped findings, not implementation bugs, are worked around
below rather than "fixed": (1) typescript-language-server answers
``initialize`` before tsserver has finished loading the project graph, so a
cross-file request made immediately after start can get a genuinely-answered
but stale result (references that sees only the declaration file) rather
than a timeout -- ``_warm_up_project`` polls a real request until the
project is actually ready, the same way a real editor's "loading project"
indicator does. (2) main.ts's original fixture used a named import
(`import { addNumbers } from './lib'`), which put the literal text
"addNumbers" in the file TWICE -- once in the import, once at the call
site. Task 7's symbol-first addressing deliberately resolves a name to its
FIRST textual occurrence (documented, already covered by its own unit
tests), so asking for "addNumbers" in that file resolved to the import
line, not the call -- and asking a real server for the definition of an
import specifier, standing on the specifier itself, answered with the
specifier's own position rather than crossing to lib.ts. A namespace import
(`import * as lib from './lib'` / `lib.addNumbers(...)`) leaves exactly one
plain-text occurrence of the identifier, at the call site, which is what
"symbol-first addressing, against a real server" actually needs to exercise.
"""
import shutil
import time

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("typescript-language-server") is None,
    reason = (
        "typescript-language-server not installed; run: "
        "npm install -g typescript-language-server typescript@5 "
        "-- the bare 'typescript' (no @5) pulls the TypeScript 7.x native-port "
        "preview, which ships no tsserver.js and fails every test below at the "
        "LSP handshake instead of here"
    ),
)


def _warm_up_project(session_id):
    """Poll a real cross-file request until tsserver's project graph is
    actually loaded, instead of asserting on whatever the first request
    -- made the instant the LSP handshake completes -- happens to see.

    Confirmed by hand against this exact server/fixture shape: the very
    first ``references`` call after a cold start returned only the
    declaration in lib.ts; a second call roughly 2s later, against the
    SAME session, correctly included the call site in main.ts too. The
    server was alive and answered both times -- this is not the
    LspTimeout/LspClosed ambiguity diagnostics.py's docstring discusses,
    it is tsserver's own project load finishing asynchronously after the
    LSP handshake it is not otherwise gated on.
    """
    from core.inference import tools
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        out = tools.execute_tool(
            "code_references", {"path": "lib.ts", "symbol": "addNumbers"}, session_id = session_id)
        if "main.ts" in out:
            return
        time.sleep(1.0)
    # Not fatal here -- if the project never finishes loading in 30s
    # something is genuinely wrong, and the individual test that runs next
    # will fail with a real, informative assertion rather than this
    # warm-up failing silently.


@pytest.fixture
def ts_project(tmp_path, monkeypatch):
    from core.inference import tools
    monkeypatch.setattr(tools, "_get_workdir", lambda _sid = None: str(tmp_path))
    (tmp_path / "tsconfig.json").write_text('{"compilerOptions":{"strict":true}}')
    (tmp_path / "lib.ts").write_text(
        "export function addNumbers(a: number, b: number): number {\n"
        "  return a + b\n"
        "}\n"
    )
    (tmp_path / "main.ts").write_text(
        "import * as lib from './lib'\n"
        "const wrong: number = lib.addNumbers(1, 'two')\n"
        "console.log(wrong)\n"
    )
    _warm_up_project("e2e-warmup")
    yield tmp_path
    from core.inference.assist_code import pool
    pool.shutdown_all()


def test_a_real_type_error_is_reported_with_its_line(ts_project):
    from core.inference import tools
    out = tools.execute_tool("code_diagnostics", {"path": "main.ts"}, session_id = "e2e")
    assert "problem" in out.lower()
    assert "\n  2:" in out, out


def test_a_clean_file_reports_no_problems(ts_project):
    from core.inference import tools
    out = tools.execute_tool("code_diagnostics", {"path": "lib.ts"}, session_id = "e2e")
    assert "no problems" in out.lower(), out


def test_definition_crosses_files(ts_project):
    from core.inference import tools
    out = tools.execute_tool(
        "code_definition", {"path": "main.ts", "symbol": "addNumbers"}, session_id = "e2e")
    assert "lib.ts" in out, out


def test_hover_reports_the_real_signature(ts_project):
    from core.inference import tools
    out = tools.execute_tool(
        "code_hover", {"path": "lib.ts", "symbol": "addNumbers"}, session_id = "e2e")
    assert "addNumbers" in out and "number" in out, out


def test_references_finds_the_call_site(ts_project):
    from core.inference import tools
    out = tools.execute_tool(
        "code_references", {"path": "lib.ts", "symbol": "addNumbers"}, session_id = "e2e")
    assert "main.ts" in out, out
