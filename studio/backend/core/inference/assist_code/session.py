# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One language server: spawn, handshake, capability negotiation, health.

Capabilities are negotiated, not assumed. Whether diagnostics are pulled
(``textDocument/diagnostic``, LSP 3.17) or pushed (``publishDiagnostics``)
depends on what the server advertises during ``initialize``, and support is
uneven across servers -- so the answer is read from the handshake rather than
hard-coded per language.
"""
import os
import subprocess
import sys
import urllib.parse
import urllib.request

from . import jsonrpc

_LANGUAGE_IDS = {
    "typescript": "typescript",
    "javascript": "javascript",
    "csharp": "csharp",
}

# 5 MB: generous for hand-written source, small enough to keep a minified
# bundle or other generated file out of a single JSON-RPC payload.
_MAX_DOCUMENT_BYTES = 5 * 1024 * 1024


class SessionStartFailed(Exception):
    """The server could not be spawned or did not complete the handshake."""


class DocumentReadError(Exception):
    """A file could not be read for ``open_document``.

    Deliberately NOT a ``SessionStartFailed``: the session may be perfectly
    healthy when this happens (a file vanished, a directory was passed by
    mistake, a file is too large). ``Session.request`` already keeps
    post-start failures (``LspTimeout``/``LspClosed``) distinct from startup
    failures for the same reason -- a caller that reacts to
    ``SessionStartFailed`` by restarting the session should never do so over
    a single bad file.
    """


def path_to_uri(path):
    # Base must be "file:///", not "file:": urljoin collapses the leading
    # "///" that pathname2url puts in front of a Windows drive letter (e.g.
    # "///C:/Users/...") down to a single slash when the base itself has no
    # authority component, producing "file:/C:/..." instead of the correct
    # "file:///C:/...". Anchoring the base with the triple slash already
    # present avoids that collapse, on both Windows and POSIX paths.
    return urllib.parse.urljoin("file:///", urllib.request.pathname2url(os.path.abspath(path)))


def uri_to_path(uri):
    parsed = urllib.parse.urlparse(uri)
    # A UNC/network URI (file://SERVER/share/...) puts the host in `netloc`,
    # not `path`. Reading only `parsed.path` silently drops it and returns a
    # plausible-looking but wrong local path (the current drive substituted
    # for the network host) instead of raising -- worse than an error, since
    # a caller has no signal anything went wrong. Folding netloc back in
    # front of path before url2pathname reconstructs the UNC form correctly.
    path = f"//{parsed.netloc}{parsed.path}" if parsed.netloc else parsed.path
    return os.path.abspath(urllib.request.url2pathname(path))


class Session:
    def __init__(self, command, root, *, language):
        self.command = list(command)
        self.root = os.path.abspath(root)
        self.language = language
        self.capabilities = {}
        self.opened_count = 0
        self._opened = set()
        self._transport = None
        self._proc = None

    def start(self, timeout = 60.0):
        try:
            self._proc = subprocess.Popen(
                self.command,
                stdin = subprocess.PIPE, stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                cwd = self.root,
            )
        except FileNotFoundError as e:
            raise SessionStartFailed(
                f"could not start the {self.language} language server: "
                f"{self.command[0]} was not found ({e})"
            ) from e
        except Exception as e:
            raise SessionStartFailed(
                f"could not start the {self.language} language server "
                f"({' '.join(self.command)}): {e}"
            ) from e

        self._transport = jsonrpc.Transport(self._proc)
        params = {
            "processId": os.getpid(),
            "rootUri": path_to_uri(self.root),
            "workspaceFolders": [{"uri": path_to_uri(self.root), "name": os.path.basename(self.root)}],
            "capabilities": {
                "textDocument": {
                    "publishDiagnostics": {"relatedInformation": False},
                    "diagnostic": {"dynamicRegistration": False},
                    "definition": {"linkSupport": False},
                    "references": {},
                    "hover": {"contentFormat": ["plaintext", "markdown"]},
                },
                "workspace": {"symbol": {}, "workspaceFolders": True},
            },
        }
        try:
            result = self._transport.request("initialize", params, timeout = timeout)
        except jsonrpc.LspTimeout as e:
            self._teardown_after_failed_start()
            raise SessionStartFailed(
                f"the {self.language} language server did not complete its handshake "
                f"within {timeout:.0f}s"
            ) from e
        except jsonrpc.LspClosed as e:
            self._teardown_after_failed_start()
            raise SessionStartFailed(
                f"the {self.language} language server exited during startup: {e}"
            ) from e
        self.capabilities = (result or {}).get("capabilities", {}) or {}
        try:
            self._transport.notify("initialized", {})
        except jsonrpc.LspClosed as e:
            self._teardown_after_failed_start()
            raise SessionStartFailed(f"server closed right after initialize: {e}") from e

    def _teardown_after_failed_start(self):
        """Reap the subprocess when ``start()`` fails after spawning it.

        Without this, a caller that lets ``SessionStartFailed`` propagate --
        Task 4's pool ``acquire()`` does exactly this -- strands a live
        language-server process. That's the worst time for it: cold-start
        failures are precisely when acquisition is retried repeatedly, so
        each retry would leak another process. ``self._proc`` is kept set
        (not reset to ``None``) so a caller inspecting the session after
        failure can still see that a process existed and was reaped.
        """
        if self._transport is not None:
            self._transport.close()
            self._transport = None
        elif self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout = 3)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass

    def supports(self, name):
        return bool(self.capabilities.get(name))

    def open_document(self, path):
        """didOpen once per file. Re-opening the same path is a no-op."""
        path = os.path.abspath(path)
        if path in self._opened:
            return
        try:
            size = os.path.getsize(path)
        except OSError as e:
            raise DocumentReadError(f"could not read {path}: {e}") from e
        if size > _MAX_DOCUMENT_BYTES:
            raise DocumentReadError(
                f"{path} is {size} bytes, over the {_MAX_DOCUMENT_BYTES}-byte "
                f"limit for opening a document (minified bundles and generated "
                f"files should not be sent to the language server whole)"
            )
        try:
            with open(path, "r", encoding = "utf-8", errors = "replace") as fh:
                text = fh.read()
        except OSError as e:
            raise DocumentReadError(f"could not read {path}: {e}") from e
        ext = os.path.splitext(path)[1].lower()
        language_id = _LANGUAGE_IDS.get(self.language, self.language)
        if ext in (".js", ".jsx", ".mjs", ".cjs"):
            language_id = "javascript"
        elif ext in (".ts", ".tsx", ".mts", ".cts"):
            language_id = "typescript"
        self._transport.notify("textDocument/didOpen", {"textDocument": {
            "uri": path_to_uri(path), "languageId": language_id,
            "version": 1, "text": text,
        }})
        self._opened.add(path)
        self.opened_count += 1

    def request(self, method, params = None, timeout = 30.0):
        if self._transport is None:
            raise jsonrpc.LspClosed("session was never started")
        return self._transport.request(method, params, timeout = timeout)

    def wait_notification(self, method, predicate, timeout = 10.0):
        if self._transport is None:
            raise jsonrpc.LspClosed("session was never started")
        return self._transport.wait_notification(method, predicate, timeout = timeout)

    def is_healthy(self):
        # Corrected from the brief: use the public Transport.is_alive() rather
        # than reaching into the private `_transport._closed` attribute across
        # the module boundary.
        return (
            self._transport is not None
            and self._proc is not None
            and self._proc.poll() is None
            and self._transport.is_alive()
        )

    def close(self):
        if self._transport is not None:
            self._transport.close()
            self._transport = None
        self._opened.clear()
