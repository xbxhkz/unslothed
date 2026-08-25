# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which language server serves which language, and how to get it.

No server is vendored. Each is installed on first use and the install is
announced BEFORE it starts -- a chat message that silently triggers a package
install is the surprise this avoids -- with failures naming the exact manual
command.

v1 is TypeScript/JavaScript and C# only. Rust and C++ are absent deliberately:
the dev machine has no cargo/rustc/g++/cl/cmake, and shipping a language that
cannot be exercised is how this project has repeatedly shipped a green suite
over broken code. Adding them later is a new entry in _SERVERS, not a new
mechanism.
"""
import logging
import shutil
import subprocess

try:
    from loggers import get_logger
    logger = get_logger(__name__)
except Exception:  # pragma: no cover - outside the app
    logger = logging.getLogger(__name__)

SUPPORTED = ("typescript", "javascript", "csharp")

_EXTENSIONS = {
    ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".cs": "csharp",
}

_SERVERS = {
    "typescript": {
        "binary": "typescript-language-server",
        "args": ["--stdio"],
        "install": ["npm", "install", "-g", "typescript-language-server", "typescript"],
        "manual": "npm install -g typescript-language-server typescript",
    },
    "javascript": {
        "binary": "typescript-language-server",
        "args": ["--stdio"],
        "install": ["npm", "install", "-g", "typescript-language-server", "typescript"],
        "manual": "npm install -g typescript-language-server typescript",
    },
    "csharp": {
        "binary": "csharp-ls",
        "args": [],
        "install": ["dotnet", "tool", "install", "--global", "csharp-ls"],
        "manual": "dotnet tool install --global csharp-ls",
    },
}


class ServerUnavailable(Exception):
    """No language server could be located or installed."""


def language_for(path):
    import os
    return _EXTENSIONS.get(os.path.splitext(str(path))[1].lower())


def _which(name):
    return shutil.which(name)


def _run_install(cmd):
    """Returns True on success. Never raises."""
    try:
        completed = subprocess.run(
            cmd, stdin = subprocess.DEVNULL,
            stdout = subprocess.PIPE, stderr = subprocess.STDOUT,
            timeout = 600,
        )
        return completed.returncode == 0
    except Exception:
        return False


def server_command(language, *, installer = None):
    """Locate the server for ``language``, installing it once if absent."""
    spec = _SERVERS.get(language)
    if spec is None:
        raise ServerUnavailable(
            f"no language server for {language!r}. Supported: {', '.join(SUPPORTED)}."
        )

    found = _which(spec["binary"])
    if found:
        return [found] + list(spec["args"])

    # Announced before the install runs, not after: a user watching the log
    # should know why the machine just started fetching packages.
    logger.info(
        "assist_code: installing %s for %s -- first use only (%s)",
        spec["binary"], language, spec["manual"],
    )
    run = installer or _run_install
    ok = run(spec["install"])
    if ok:
        found = _which(spec["binary"])
        if found:
            return [found] + list(spec["args"])
    raise ServerUnavailable(
        f"the {language} language server ({spec['binary']}) is not installed and could "
        f"not be installed automatically. Install it manually with: {spec['manual']}"
    )
