# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import pytest

from core.inference.assist_code import servers


class _RecordingLogger:
    """Records rendered lines; makes ORDERING observable, which is the point."""
    def __init__(self): self.lines = []
    def info(self, msg, *args): self.lines.append(msg % args if args else msg)
    def warning(self, msg, *args): self.lines.append(msg % args if args else msg)
    def text(self): return "\n".join(self.lines).lower()


@pytest.fixture
def spy(monkeypatch):
    s = _RecordingLogger()
    monkeypatch.setattr(servers, "logger", s)
    return s


class TestLanguageDetection:
    @pytest.mark.parametrize("name,expected", [
        ("a.ts", "typescript"), ("a.tsx", "typescript"), ("a.mts", "typescript"),
        ("a.js", "javascript"), ("a.jsx", "javascript"), ("a.cjs", "javascript"),
        ("a.cs", "csharp"),
        ("a.rs", None), ("a.cpp", None), ("a.py", None), ("noext", None),
    ])
    def test_extension_maps_to_language(self, name, expected):
        assert servers.language_for(name) == expected

    def test_rust_and_cpp_are_deliberately_absent_from_v1(self):
        assert "rust" not in servers.SUPPORTED
        assert "cpp" not in servers.SUPPORTED


class TestDiscovery:
    def test_an_already_installed_server_is_used_without_installing(self, spy, monkeypatch):
        monkeypatch.setattr(servers, "_which", lambda name: "C:/fake/typescript-language-server")
        def _boom(*a, **k):
            raise AssertionError("must not install a server that is already present")
        cmd = servers.server_command("typescript", installer = _boom)
        assert cmd[0] == "C:/fake/typescript-language-server"
        assert "installing" not in spy.text()

    def test_a_missing_server_is_announced_before_the_install_starts(self, spy, monkeypatch):
        monkeypatch.setattr(servers, "_which", lambda name: None)
        log_at_install_time = []
        def _installer(cmd):
            log_at_install_time.append(spy.text())
            return True
        calls = {"n": 0}
        def _which_after(name):
            calls["n"] += 1
            return None if calls["n"] == 1 else "C:/fake/typescript-language-server"
        monkeypatch.setattr(servers, "_which", _which_after)
        servers.server_command("typescript", installer = _installer)
        assert log_at_install_time, "installer was never called"
        announced = log_at_install_time[0]
        assert "installing" in announced
        assert "typescript-language-server" in announced

    def test_a_failed_install_names_the_manual_command(self, monkeypatch):
        monkeypatch.setattr(servers, "_which", lambda name: None)
        with pytest.raises(servers.ServerUnavailable) as e:
            servers.server_command("typescript", installer = lambda cmd: False)
        msg = str(e.value)
        assert "npm install" in msg
        assert "typescript-language-server" in msg

    def test_csharp_failure_names_the_dotnet_tool_command(self, monkeypatch):
        monkeypatch.setattr(servers, "_which", lambda name: None)
        with pytest.raises(servers.ServerUnavailable) as e:
            servers.server_command("csharp", installer = lambda cmd: False)
        msg = str(e.value)
        assert "dotnet tool install" in msg
        assert "csharp-ls" in msg

    def test_an_unsupported_language_names_the_supported_ones(self):
        with pytest.raises(servers.ServerUnavailable) as e:
            servers.server_command("rust")
        msg = str(e.value).lower()
        assert "rust" in msg
        assert "typescript" in msg and "csharp" in msg
