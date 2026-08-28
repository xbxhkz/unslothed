# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""End-to-end against a REAL csharp-ls, the second of the two v1 languages.

Why this file exists separately from test_assist_code_e2e.py: every defect the
TypeScript e2e suite caught was a *server behaviour* bug, invisible to the fake
server because the fake answers whatever we ask it. Two of them were only found
because a real server disagreed with our assumptions -- a percent-encoded
drive-letter colon, and a project that is not loaded when `initialize` returns.

Nothing entitles us to assume csharp-ls shares typescript-language-server's
behaviour, so the fixes for those defects are re-verified here against a
different real implementation rather than argued by analogy.

What running these actually established, stated plainly because two of the
eight tests turned out not to discriminate:

  * The result-side confinement (I1) holds for csharp-ls, and its control is
    genuine -- neutering the predicate fails those tests.
  * Diagnostics, definition, references and hover all work against a real C#
    project, which nothing had verified before.
  * The two Criticals found on TypeScript -- `No Project` on a first
    `workspace/symbol`, and an async project-load race -- DO NOT REPRODUCE
    here, because csharp-ls loads its project during the handshake rather than
    after it. Their tests below are kept, but they cannot currently fail and
    are labelled as such.

Requires:  dotnet tool install --global csharp-ls
Verified against csharp-ls 0.27.0 with .NET 8.0.424.

csharp-ls needs a project file to load anything at all -- a bare .cs file in a
directory gives an empty workspace and every query returns nothing. The fixture
therefore writes a .csproj, which is also what `paths.workspace_for` keys on.
"""
import os
import re
import shutil

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("csharp-ls") is None,
    reason = "csharp-ls not installed; run: dotnet tool install --global csharp-ls",
)

# csharp-ls loads a project through MSBuild on first use, which is slower than
# tsserver's cold start -- measured ~3.5s here. The budget is generous because a
# timeout would surface as an empty result, which is the exact failure mode this
# suite exists to catch.
_COLD_START_BUDGET = 90.0


def _write(path, text):
    with open(path, "w", encoding = "utf-8") as fh:
        fh.write(text)


@pytest.fixture
def cs_project(tmp_path, monkeypatch):
    """A two-file C# project with one real type error and one cross-file call.

    Deliberately NO warm-up. The readiness gate belongs to the product, not to
    this fixture -- a fixture that warmed the project would hide exactly the
    defect these tests are here to detect.
    """
    from core.inference import tools
    monkeypatch.setattr(tools, "_get_workdir", lambda _sid = None: str(tmp_path))

    _write(str(tmp_path / "Probe.csproj"),
           '<Project Sdk="Microsoft.NET.Sdk">\n'
           "  <PropertyGroup>\n"
           "    <OutputType>Library</OutputType>\n"
           "    <TargetFramework>net8.0</TargetFramework>\n"
           "  </PropertyGroup>\n"
           "</Project>\n")
    _write(str(tmp_path / "Lib.cs"),
           "namespace Probe;\n"
           "\n"
           "public static class Lib\n"
           "{\n"
           "    /// <summary>Adds two numbers together.</summary>\n"
           "    public static int AddNumbers(int a, int b)\n"
           "    {\n"
           "        return a + b;\n"
           "    }\n"
           "}\n")
    _write(str(tmp_path / "Main.cs"),
           "namespace Probe;\n"
           "\n"
           "public class Runner\n"
           "{\n"
           "    public void Go()\n"
           "    {\n"
           "        int wrong = Lib.AddNumbers(1, \"two\");\n"
           "        System.Console.WriteLine(wrong);\n"
           "    }\n"
           "}\n")

    from core.inference.assist_code import pool
    pool.shutdown_all()
    yield tmp_path
    pool.shutdown_all()


class TestDiagnostics:
    def test_a_real_type_error_is_reported_with_its_line(self, cs_project):
        from core.inference import tools
        out = tools.execute_tool(
            "code_diagnostics", {"path": "Main.cs"}, session_id = "cs-e2e",
            timeout = _COLD_START_BUDGET)
        assert "problem" in out.lower(), out
        # Format is "  <line>:<column>  <severity>: <message>". Pin the line but
        # not the column -- the column is a compiler detail we do not control.
        assert re.search(r"^\s*7:\d+\s", out, re.MULTILINE), out
        assert "cannot convert" in out.lower(), out

    def test_a_clean_file_reports_no_problems(self, cs_project):
        from core.inference import tools
        out = tools.execute_tool(
            "code_diagnostics", {"path": "Lib.cs"}, session_id = "cs-e2e",
            timeout = _COLD_START_BUDGET)
        assert "no problems" in out.lower(), out


class TestNavigation:
    def test_definition_crosses_files(self, cs_project):
        from core.inference import tools
        out = tools.execute_tool(
            "code_definition", {"path": "Main.cs", "symbol": "AddNumbers"},
            session_id = "cs-e2e", timeout = _COLD_START_BUDGET)
        assert "Lib.cs" in out, out

    def test_hover_reports_the_real_signature_and_doc(self, cs_project):
        from core.inference import tools
        out = tools.execute_tool(
            "code_hover", {"path": "Lib.cs", "symbol": "AddNumbers"},
            session_id = "cs-e2e", timeout = _COLD_START_BUDGET)
        assert "AddNumbers" in out and "int" in out, out
        # csharp-ls surfaces the XML doc summary; confirm it survives fence stripping.
        assert "adds two numbers" in out.lower(), out


class TestTheColdStartRace:
    """C2 against a different real server -- and it does NOT reproduce here.

    typescript-language-server answers `initialize` before its project graph is
    loaded, so the first cross-file query returned only the declaration: a
    clean, plausible, WRONG answer reading as "nothing calls this".

    csharp-ls does not do this. It loads the project through MSBuild *during*
    the handshake, which is why its `start()` is slow (~3.5s) where tsserver's
    is fast -- by the time `initialize` returns there is no half-loaded window
    to race.

    So this test CANNOT CURRENTLY FAIL. Measured: neutering the readiness gate
    (`Session.await_project_ready` returning immediately) leaves it green. It is
    therefore NOT evidence that the gate works -- `test_assist_code_e2e.py` is
    what proves that, against the server that actually exhibits the race.

    It is kept deliberately, for what it does do: pin the invariant that a first
    call must be complete, so that a future csharp-ls which moves project
    loading off the handshake is caught rather than silently returning
    half-answers. Labelled rather than deleted, and labelled rather than counted
    as coverage.
    """

    def test_references_is_complete_on_the_very_first_call(self, cs_project):
        from core.inference import tools
        out = tools.execute_tool(
            "code_references", {"path": "Lib.cs", "symbol": "AddNumbers"},
            session_id = "cs-cold", timeout = _COLD_START_BUDGET)
        assert "Lib.cs" in out, out
        assert "Main.cs" in out, (
            "the call site is missing on the first call -- the readiness gate did "
            "not hold for csharp-ls:\n" + out)


class TestSymbolsAsTheFirstCall:
    """C1 against a different real server -- and it does NOT reproduce here.

    tsserver rejects `workspace/symbol` with `No Project` until a document has
    been opened, so `code_symbols` failed on the first call of every session.
    `navigation.symbols` now opens one through `_open_ready` first.

    csharp-ls has a project loaded before it answers anything, so it serves
    `workspace/symbol` immediately. Measured: removing the `_open_ready` call
    from `symbols()` leaves this test green.

    Same standing as TestTheColdStartRace above -- it CANNOT CURRENTLY FAIL, so
    it is not evidence the fix works; `test_assist_code_e2e.py` carries that.
    Kept to pin the invariant against a csharp-ls that later defers project
    loading, and labelled so nobody mistakes it for proof.
    """

    def test_symbols_works_as_the_very_first_call_in_a_fresh_session(self, cs_project):
        from core.inference import tools
        from core.inference.assist_code import pool
        pool.shutdown_all()
        out = tools.execute_tool(
            "code_symbols", {"query": "AddNumbers", "path": "Lib.cs"},
            session_id = "cs-first", timeout = _COLD_START_BUDGET)
        assert "AddNumbers" in out, out
        assert "no project" not in out.lower(), out
        assert "error" not in out.lower(), out


class TestResultSideConfinement:
    """I1, re-verified against a different real server.

    The compiler's program graph reaches outside the workspace root by ordinary
    means -- here an explicit <Compile Include> pointing above it. Locations
    outside the sandbox must be withheld, and the withholding must be *stated*,
    because an escape rendering as a clean empty result is the failure this
    branch's whole design argues against.
    """

    @pytest.fixture
    def project_including_an_outside_file(self, cs_project, tmp_path_factory):
        outside = tmp_path_factory.mktemp("cs_outside")
        _write(str(outside / "Secret.cs"),
               "namespace Probe;\n"
               "public static class Secret\n"
               "{\n"
               "    /// <summary>SENSITIVE: the deploy key is sk-live-DEADBEEF</summary>\n"
               "    public const string DeployKey = \"sk-live-DEADBEEF\";\n"
               "}\n")
        _write(str(cs_project / "Probe.csproj"),
               '<Project Sdk="Microsoft.NET.Sdk">\n'
               "  <PropertyGroup>\n"
               "    <OutputType>Library</OutputType>\n"
               "    <TargetFramework>net8.0</TargetFramework>\n"
               "  </PropertyGroup>\n"
               "  <ItemGroup>\n"
               '    <Compile Include="%s" />\n' % str(outside / "Secret.cs").replace("\\", "/")
               + "  </ItemGroup>\n"
               "</Project>\n")
        _write(str(cs_project / "UsesSecret.cs"),
               "namespace Probe;\n"
               "public class UsesSecret\n"
               "{\n"
               "    public string Get() => Secret.DeployKey;\n"
               "}\n")
        from core.inference.assist_code import pool
        pool.shutdown_all()
        return cs_project, outside

    def test_definition_withholds_the_outside_location_and_says_so(
        self, project_including_an_outside_file
    ):
        from core.inference import tools
        out = tools.execute_tool(
            "code_definition", {"path": "UsesSecret.cs", "symbol": "DeployKey"},
            session_id = "cs-leak", timeout = _COLD_START_BUDGET)
        assert "withheld" in out.lower(), out
        assert "sk-live-DEADBEEF" not in out, out

    def test_hover_does_not_disclose_content_from_outside_the_sandbox(
        self, project_including_an_outside_file
    ):
        from core.inference import tools
        out = tools.execute_tool(
            "code_hover", {"path": "UsesSecret.cs", "symbol": "DeployKey"},
            session_id = "cs-leak", timeout = _COLD_START_BUDGET)
        # C# const strings carry their value in the signature exactly as TS does,
        # which is why hover is suppressed whole rather than trimmed to a signature.
        assert "sk-live-DEADBEEF" not in out, out
        assert "SENSITIVE" not in out, out
        assert "withheld" in out.lower(), out
