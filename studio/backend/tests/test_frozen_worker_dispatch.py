# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A frozen build must answer `-m <module>` the way an interpreter would.

Studio spawns every helper as [sys.executable, "-m", "<module>", ...]. Under
PyInstaller sys.executable is the app executable, so without run.py's shim those
arguments hit the app's own argparse and the child exits 2 -- surfacing as a
model download that fails instantly with a usage dump.

No exe is built here. run.py's shim returns before run.py's heavy imports, so
running the REAL file with sys.frozen faked exercises the real code path rather
than a copy of it. No network: the worker is asked for --help.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_RUN_PY = _BACKEND / "run.py"
_SPEC = _BACKEND.parent.parent / "packaging" / "Unslothed.spec"

# The worker whose absence started this: spawned by hub/services/download_lifecycle.py
# and imported by nothing, so only an explicit hiddenimports entry bundles it.
_WORKER = "hub.workers.hf_download"


def _drive(*, frozen: bool, argv: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    """Run run.py as __main__ in a child, optionally pretending to be frozen."""
    driver = textwrap.dedent(
        f"""
        import runpy, sys
        frozen = {frozen!r}
        if frozen:
            sys.frozen = True          # exactly what PyInstaller's bootloader sets
        sys.argv = {argv!r}
        sys.path.insert(0, {str(_BACKEND)!r})
        runpy.run_path({str(_RUN_PY)!r}, run_name="__main__")
        """
    )
    return subprocess.run(
        [sys.executable, "-c", driver],
        cwd = str(_BACKEND),
        capture_output = True,
        text = True,
        timeout = timeout,
    )


def test_frozen_dispatches_m_to_the_worker():
    """frozen + `-m <worker>` runs the WORKER, not the app."""
    proc = _drive(frozen = True, argv = ["run.py", "-m", _WORKER, "--help"])
    out = proc.stdout + proc.stderr
    assert "hf_download.py" in out, f"shim did not dispatch to the worker:\n{out[:2000]}"
    # The flags download_lifecycle.py actually passes; proves we reached the
    # worker's parser and not merely some other module that printed usage.
    for flag in ("--repo-id", "--variant", "--files-json", "--parent-pid", "--transport"):
        assert flag in out, f"worker help is missing {flag}:\n{out[:2000]}"


def test_not_frozen_does_not_dispatch():
    """The negative control: identical argv, no sys.frozen -> no dispatch.

    Proves sys.frozen is what gates the shim. Without this, a shim that fired
    unconditionally would pass the test above and silently hijack every
    source-tree `python run.py` invocation.
    """
    proc = _drive(frozen = False, argv = ["run.py", "-m", _WORKER, "--help"])
    out = proc.stdout + proc.stderr
    assert "hf_download.py" not in out, (
        "shim dispatched despite sys.frozen being unset -- the guard is inert:\n"
        f"{out[:2000]}"
    )
    # Fell through to the app's own parser, i.e. today's unfrozen behaviour.
    assert "Run Unsloth UI Backend server" in out or "usage: run.py" in out, out[:2000]


def test_frozen_normal_launch_is_untouched():
    """frozen WITHOUT `-m` must still start the app -- the regression that matters."""
    proc = _drive(frozen = True, argv = ["run.py", "--help"])
    out = proc.stdout + proc.stderr
    assert "hf_download.py" not in out, f"shim hijacked a normal launch:\n{out[:2000]}"
    assert "usage: run.py" in out, out[:2000]


def test_shim_precedes_the_heavy_imports():
    """Position is load-bearing: after sys.path setup, before the first import-from.

    A shim placed below them would work but make every short-lived worker pay for
    configure_cpu_threads, the torchao stub and structlog, and inherit their
    process-global side effects.
    """
    tree = ast.parse(_RUN_PY.read_text(encoding = "utf-8"))
    shim_line = None
    for node in tree.body:
        if isinstance(node, ast.If) and "frozen" in ast.dump(node.test):
            shim_line = node.lineno
            break
    assert shim_line is not None, "no module-level `sys.frozen` dispatch block in run.py"

    first_local_import = min(
        (n.lineno for n in tree.body if isinstance(n, ast.ImportFrom) and n.lineno > 100),
        default = None,
    )
    assert first_local_import is not None
    assert shim_line < first_local_import, (
        f"shim at line {shim_line} runs after the first heavy import at "
        f"{first_local_import}; a worker would pay for the app's startup"
    )


def test_frozen_intercepts_multiprocessing_fork():
    """A spawn child must reach spawn_main(), not the app's argparse.

    Frozen, multiprocessing launches children as
    [sys.executable, "--multiprocessing-fork", <kwds>]. Without
    multiprocessing.freeze_support() at the entry point those arguments hit
    argparse and every training/export/inference worker dies at exit 2.

    The handles below are deliberately bogus, so the child fails inside
    spawn_main -- reaching it at all is the proof of interception.
    """
    proc = _drive(
        frozen = True,
        argv = ["run.py", "--multiprocessing-fork", "tracker_fd=3", "pipe_handle=4"],
    )
    out = proc.stdout + proc.stderr
    assert "unrecognized arguments" not in out, (
        "the spawn child fell through to the app's argparse -- freeze_support() "
        f"is not being called early enough:\n{out[:2000]}"
    )
    assert "spawn_main" in out, f"never reached spawn_main:\n{out[:2000]}"


def test_not_frozen_ignores_multiprocessing_fork():
    """Negative control: without sys.frozen, freeze_support() must do nothing.

    CPython gates it on sys.frozen, so an unfrozen run falls through to argparse
    exactly as it does today. If this ever starts intercepting, the shim is
    firing on argv shape alone and would hijack a source-tree run.
    """
    proc = _drive(
        frozen = False,
        argv = ["run.py", "--multiprocessing-fork", "tracker_fd=3", "pipe_handle=4"],
    )
    out = proc.stdout + proc.stderr
    assert "spawn_main" not in out, (
        f"intercepted despite sys.frozen being unset -- the guard is inert:\n{out[:2000]}"
    )
    assert "unrecognized arguments" in out, out[:2000]


@pytest.mark.skipif(not _SPEC.is_file(), reason = "packaging spec not present")
def test_spec_copies_metadata_checked_at_import():
    """diffusers/transformers read importlib.metadata at IMPORT time.

    PyInstaller bundles modules by following imports but copies a package's
    .dist-info only when a hook asks, so `import requests` succeeded while
    importlib.metadata.version("requests") raised and selecting a diffusion
    model died with "The 'requests' distribution was not found".
    """
    spec = _SPEC.read_text(encoding = "utf-8")
    assert "copy_metadata" in spec, "spec no longer copies any distribution metadata"
    for dist in ("requests", "filelock", "numpy", "tokenizers", "safetensors"):
        assert dist in spec, (
            f"{dist} dropped from the metadata list; diffusers/transformers "
            "version-check it at import, so a frozen build would fail on model select"
        )


@pytest.mark.skipif(not _SPEC.is_file(), reason = "packaging spec not present")
def test_spec_collects_neighbouring_data_files():
    """Packages that open a data file beside their own source at import time.

    PyInstaller follows imports, so the module lands in the PYZ while the data
    file beside it is left behind: the import succeeds and the open then raises.
    kernels/deps.py reads python_depends.json this way, which surfaced as
    "Cannot load dependency data, is `kernels` correctly installed?" while
    loading a Wan pipeline -- naming the wrong culprit, since kernels was
    installed and merely unshipped.
    """
    spec = _SPEC.read_text(encoding = "utf-8")
    for pkg in ("kernels", "whisper"):
        assert f'collect_data_files("{pkg}")' in spec, (
            f"spec no longer collects {pkg}'s data files; it opens one relative "
            "to __file__ at import, so the frozen build raises on use"
        )


@pytest.mark.skipif(not _SPEC.is_file(), reason = "packaging spec not present")
def test_spec_bundles_the_spawned_worker():
    """The other half: dispatch is useless if the module is not in the bundle.

    hub.workers is imported by nothing, so PyInstaller's static walk misses it
    entirely -- it was absent from Analysis-00.toc while 20+ sibling hub.*
    modules were present.
    """
    spec = _SPEC.read_text(encoding = "utf-8")
    assert "hub.workers" in spec, (
        "Unslothed.spec no longer bundles hub.workers; a frozen download will "
        "fail with ModuleNotFoundError even though run.py dispatches correctly"
    )
