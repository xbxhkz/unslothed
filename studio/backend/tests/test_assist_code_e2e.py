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

One finding was originally mis-classified here as fixture-shaped and worked
around rather than fixed, and the whole-branch review caught that:
typescript-language-server answers ``initialize`` before tsserver has
finished loading the project graph, so a cross-file request made immediately
after start gets a genuinely-answered but INCOMPLETE result (references
seeing only the declaration file) rather than a timeout. The fixture used to
poll until the project was ready and every test then ran warm -- which meant
the suite could not observe the defect, and production, which never warms up,
kept shipping it. The readiness gate now lives in the product
(``Session.await_project_ready``); the fixture helper is gone, and the tests
below run genuinely cold on purpose.

The remaining fixture-shaped finding is real: main.ts's original fixture used a named import
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
    # Deliberately NO warm-up. The fixture used to poll a real cross-file
    # request here until tsserver's project graph had loaded, which made
    # every test below run against a warm session -- and so made the suite
    # structurally unable to observe the cold-start defect the product now
    # fixes (Session.await_project_ready). Warming up in the fixture put the
    # remedy where only the tests benefited; production never warmed up.
    yield tmp_path
    from core.inference.assist_code import pool
    pool.shutdown_all()


@pytest.fixture
def escaping_project(tmp_path, monkeypatch):
    """A sandbox whose program graph legitimately reaches outside it.

    Confinement was enforced only on tool ARGUMENTS. A language server's
    program graph is not built from our arguments -- an ordinary relative
    import pulls in a file outside the workspace root, and every location and
    hover string the server then returns was passed to the model unfiltered.
    The model can create the triggering file itself (`edit_file`, `python`
    and `terminal` are all Studio tools), so this is reachable by argument
    choice plus one file write, and it bypasses the file-read confinement
    every other tool honours.

    `outside` is a SIBLING of the workdir, not a child, so it is genuinely
    outside `_get_workdir` while still being reachable by `../`.
    """
    from core.inference import tools
    workdir = tmp_path / "wb-inside"
    workdir.mkdir()
    outside = tmp_path / "wb-outside"
    outside.mkdir()
    monkeypatch.setattr(tools, "_get_workdir", lambda _sid = None: str(workdir))

    (outside / "secret.ts").write_text(
        "/** TOP SECRET: the deploy key is sk-live-DEADBEEF */\n"
        "export const SECRET_VALUE = 'sk-live-DEADBEEF'\n"
    )
    (workdir / "tsconfig.json").write_text('{"compilerOptions":{"strict":true}}')
    (workdir / "app.ts").write_text(
        "import { SECRET_VALUE } from '../wb-outside/secret'\n"
        "export function useIt(): string {\n"
        "  return SECRET_VALUE\n"
        "}\n"
    )
    yield workdir
    from core.inference.assist_code import pool
    pool.shutdown_all()


class TestResultSideConfinement:
    """Confinement must hold on what the server RETURNS, not just on what we
    send it. An escape must never render as a clean empty result either --
    that is this branch's governing principle, and the reason LspError exists
    as a distinct type.
    """

    def test_definition_withholds_an_outside_location_and_says_so(self, escaping_project):
        from core.inference import tools
        out = tools.execute_tool(
            "code_definition", {"path": "app.ts", "symbol": "SECRET_VALUE"},
            session_id = "esc")
        assert "wb-outside" not in out, out
        assert "secret.ts" not in out, out
        assert "withheld" in out.lower(), out

    def test_references_keeps_inside_hits_and_reports_the_withheld_one(self, escaping_project):
        from core.inference import tools
        out = tools.execute_tool(
            "code_references", {"path": "app.ts", "symbol": "SECRET_VALUE"},
            session_id = "esc")
        assert "wb-outside" not in out, out
        assert "app.ts" in out, out
        assert "withheld" in out.lower(), out

    def test_symbols_withholds_an_outside_match_and_says_so(self, escaping_project):
        from core.inference import tools
        out = tools.execute_tool(
            "code_symbols", {"path": "app.ts", "query": "SECRET_VALUE"},
            session_id = "esc")
        assert "wb-outside" not in out, out
        assert "secret.ts" not in out, out
        assert "withheld" in out.lower(), out

    def test_hover_does_not_disclose_the_outside_file_content(self, escaping_project):
        """The worst of the four. TypeScript's literal-type inference puts a
        const string's VALUE in the hover signature verbatim, and JSDoc text
        comes through as-is -- so stripping to the signature line would not
        have been enough: the signature IS the disclosure."""
        from core.inference import tools
        out = tools.execute_tool(
            "code_hover", {"path": "app.ts", "symbol": "SECRET_VALUE"},
            session_id = "esc")
        assert "sk-live-DEADBEEF" not in out, out
        assert "TOP SECRET" not in out, out
        assert "deploy key" not in out, out


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


def test_references_finds_the_call_site_on_a_cold_session(ts_project):
    """The FIRST references call against a cold server must be complete.

    This is the test the fixture's warm-up used to disarm. Without the
    product-side readiness gate, tsserver answers this request from a
    half-loaded project graph: it returns the declaration in lib.ts and
    silently omits the call site in main.ts. Nothing raises, and an empty or
    short references list is exactly what "nothing calls this, safe to
    delete" looks like.

    ``pool.shutdown_all()`` makes the coldness explicit rather than relying
    on this test happening to run first.
    """
    from core.inference import tools
    from core.inference.assist_code import pool
    pool.shutdown_all()
    out = tools.execute_tool(
        "code_references", {"path": "lib.ts", "symbol": "addNumbers"}, session_id = "e2e-cold")
    assert "main.ts" in out, out


def test_symbols_works_as_the_very_first_call_in_a_fresh_session(ts_project):
    """``code_symbols`` must work when NOTHING has been opened yet.

    This is the one ordering that matters and the one no other test covered.
    ``code_symbols`` is the tool a model reaches for precisely *before* it has
    opened anything -- "find this name, I don't know which file holds it" --
    so the first call in a session is its normal case, not an edge case.

    ``pool.shutdown_all()`` is what makes this test able to fail: the
    ``ts_project`` fixture has already warmed a session against this
    workspace, and any test that runs after another tool in the same session
    passes whether or not the bug is present. Discarding the pooled session
    forces a brand-new server process with no document ever opened against
    it, which is the state a real first call arrives in.
    """
    from core.inference import tools
    from core.inference.assist_code import pool
    pool.shutdown_all()
    out = tools.execute_tool(
        "code_symbols", {"path": "lib.ts", "query": "addNumbers"}, session_id = "e2e-fresh")
    assert "rejected this request" not in out, out
    assert "addNumbers" in out, out
