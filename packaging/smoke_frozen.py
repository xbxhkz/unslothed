# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Post-build smoke test, executed INSIDE the freshly built Unslothed.exe.
#
# build-installer.ps1 copies this into dist/Unslothed/_internal as
# _smoke_frozen.py, runs `Unslothed.exe -m _smoke_frozen`, and deletes it again
# before Inno Setup packages anything. It never ships; it exists to fail a build.
#
# Why it exists: every check below guards a defect that shipped in a build which
# looked completely fine. Each was found by a user hitting it, and each cost a
# ~50-minute rebuild. None is reachable from the Python test suite, because none
# is reachable except from a frozen executable -- PyInstaller bundles modules by
# following imports, and each of these is something the import graph cannot see:
# a spawn, a data file, a .dist-info, an entry point, a lazily imported
# submodule.
#
# The rule this encodes: "the file is in the bundle" is not evidence. Only "the
# exe does the thing" is. So every check RUNS the real code path.
#
# IMPORTANT structural note: checks are REGISTERED at import and only RUN under
# __main__. They must never run at import time. The multiprocessing check spawns
# a child, and that child re-imports this module to unpickle its target -- if
# importing ran the checks, each child would spawn another. Registry, not
# decorator side effects.
#
# Exit codes: 0 all passed, 1 one or more failed.

import importlib
import os
import sys
import traceback

_CHECKS: list[tuple[str, object]] = []


def check(name):
    """Register a check to be run by main(). Must not run it here -- see the note above."""

    def wrap(fn):
        _CHECKS.append((name, fn))
        return fn

    return wrap


def _mp_child(q):
    """Runs in the spawned child. Module level so the child can unpickle it."""
    try:
        import torch

        q.put({"ok": True, "pid": os.getpid(), "torch": torch.__version__})
    except Exception as exc:  # noqa: BLE001
        q.put({"ok": False, "error": f"{exc.__class__.__name__}: {exc}"})


# --- 347e2bf: `-m` dispatch --------------------------------------------------
@check("-m dispatch (run.py frozen shim)")
def _m_dispatch():
    # Reaching this file at all proves it: the exe was invoked as
    # `Unslothed.exe -m _smoke_frozen`, which before the shim died in argparse.
    assert getattr(sys, "frozen", False), "not frozen -- run this via the built exe"
    return "this module was reached via -m"


# --- 347e2bf: the spawned download worker ------------------------------------
@check("hub.workers.hf_download bundled")
def _worker_bundled():
    m = importlib.import_module("hub.workers.hf_download")
    assert hasattr(m, "main"), "worker has no main()"
    return m.__name__


# --- 0ae413a: torchvision's stdlib import ------------------------------------
@check("modulefinder (torchvision imports it; Analysis never walks torchvision)")
def _modulefinder():
    importlib.import_module("modulefinder")
    importlib.import_module("torchvision.transforms")
    return "torchvision.transforms imports"


# --- 9c15af9 + d035cbc: distribution metadata --------------------------------
@check("distribution metadata readable at runtime")
def _metadata():
    import importlib.metadata as im

    # requests: diffusers' declared runtime check. torchcodec: an UNGUARDED read
    # in transformers/audio_utils.py that surfaced as "Could not import module
    # 'UMT5EncoderModel'". gguf: read by diffusers, would have been next.
    got = {d: im.version(d) for d in ("requests", "torchcodec", "gguf", "transformers")}
    return ", ".join(f"{k}={v}" for k, v in got.items())


# --- fde8ac0: data files opened beside a module's own source -----------------
@check("kernels/whisper data files present")
def _data_files():
    import pathlib

    k = importlib.import_module("kernels.deps")
    assert getattr(k, "DEPENDENCY_DATA", None), "kernels python_depends.json did not load"

    import whisper

    w = pathlib.Path(whisper.__file__).parent
    for rel in ("normalizers/english.json", "assets/mel_filters.npz"):
        assert (w / rel).is_file(), f"whisper {rel} missing"
    return "kernels DEPENDENCY_DATA + whisper assets"


# --- d79cbc4: triton, complete -----------------------------------------------
@check("triton backend discovery")
def _triton_discovery():
    from triton.backends import backends
    from triton.runtime.driver import driver

    assert backends, "zero triton backends discovered (entry-point metadata missing?)"
    active = [n for n, b in backends.items() if b.driver.is_active()]
    assert active, f"no ACTIVE triton driver; discovered {list(backends)}"
    return f"{list(backends)} active={active} target={driver.active.get_current_target()}"


@check("inductor codegen + triton kernel execution")
def _triton_compile():
    import random

    import torch

    if not torch.cuda.is_available():
        return "SKIPPED (no CUDA device on the build machine)"

    # Defeat Inductor's on-disk cache deliberately. Caught while validating this
    # file: with triton's entry-point metadata removed -- so backend discovery
    # returned ZERO backends -- this check still passed, because Inductor served
    # a kernel compiled on an earlier run. It was therefore proving nothing.
    # A per-run constant is baked into the FX graph, so the cache key differs
    # every time and codegen must actually happen.
    k = float(random.randint(1, 10**6))

    def f(x):
        return (x * 2.0 + k).sin().sum()

    torch._dynamo.reset()  # drop in-process compiled-function caches too

    x = torch.randn(4096, device="cuda", dtype=torch.float32)
    eager = float(f(x))
    compiled = float(torch.compile(f, backend="inductor")(x))
    torch.cuda.synchronize()
    delta = abs(eager - compiled)
    assert delta < 1e-2, f"compiled result diverged from eager by {delta}"
    return f"fresh-codegen k={int(k)} |delta|={delta:.2e}"


# --- 8360e15: multiprocessing.freeze_support ---------------------------------
@check("multiprocessing spawn child completes")
def _mp_spawn():
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_mp_child, args=(q,))
    p.start()
    try:
        got = q.get(timeout=300)
    finally:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
    assert got.get("ok"), f"child failed: {got}"
    assert got["pid"] != os.getpid(), "target ran in the parent, not a spawned child"
    return f"child pid {got['pid']} returned torch {got['torch']}"


# --- the user-facing path all of the above exists to serve -------------------
@check("diffusers import-time version checks pass")
def _diffusers_checks():
    importlib.import_module("diffusers.dependency_versions_check")
    return "require_version_core all resolved"


@check("Wan video pipeline resolves")
def _wan():
    import diffusers

    cls = getattr(diffusers, "WanPipeline")
    return f"{cls.__module__}.{cls.__name__}"


def main() -> int:
    print("=" * 72)
    print("Unslothed frozen smoke test")
    print("=" * 72)
    failures = []
    for name, fn in _CHECKS:
        try:
            detail = fn()
            print(f"  PASS  {name}" + (f" -- {detail}" if detail else ""))
        except Exception as exc:  # noqa: BLE001 -- one failure must not hide the rest
            failures.append(name)
            print(f"  FAIL  {name}")
            print(f"          {exc.__class__.__name__}: {exc}")
            for ln in traceback.format_exc().strip().splitlines()[-6:]:
                print(f"          {ln}")
    print("-" * 72)
    print(f"{len(_CHECKS) - len(failures)} passed, {len(failures)} failed")
    if failures:
        print()
        print("A failure here means the built exe is broken in a way no unit test")
        print("can see. Do NOT ship it. Each check cites the commit that fixed the")
        print("defect it guards; read that commit message for the mechanism.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
