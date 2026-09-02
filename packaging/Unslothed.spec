# packaging/Unslothed.spec
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# PyInstaller spec for the self-contained Unslothed Windows build.
#
# Run via packaging/build-installer.ps1, not directly -- the frontend must be
# built first or the app ships with no UI.

import importlib.util
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

# SPECPATH is the directory holding this file: <repo>/packaging. One .parent
# reaches the repo root. A second .parent (as an earlier draft of this file
# had) overshoots to the folder *containing* the repo, which quietly turns
# every path below into a wrong one -- the build then fails with "not found"
# errors that look unrelated to this line. Print it so a future slip is
# visible immediately instead of as a confusing downstream failure.
ROOT = Path(SPECPATH).resolve().parent
BACKEND = ROOT / "studio" / "backend"
print(f"[Unslothed.spec] ROOT    = {ROOT}")
print(f"[Unslothed.spec] BACKEND = {BACKEND}")

# The real, runnable entry point. studio/backend/main.py only defines the
# FastAPI `app` object (no __main__ guard, no server start) -- it is meant to
# be imported, e.g. by `uvicorn main:app`, not executed. studio/backend/run.py
# is the self-contained script: it has the `if __name__ == "__main__":` block,
# does startup-order-sensitive setup (CPU thread caps, ROCm stubs, std-stream
# normalization before structlog binds them) ahead of any heavy import, and
# only then does `from main import app` inside run_server(). Freezing main.py
# instead would produce an exe that imports and exits without ever serving.
ENTRY = BACKEND / "run.py"
assert ENTRY.is_file(), (
    f"Entry point not found at {ENTRY}. Expected studio/backend/run.py -- "
    "if the backend was restructured, point this at whatever file now has "
    "the `if __name__ == \"__main__\":` block that calls run_server()."
)
print(f"[Unslothed.spec] entry   = {ENTRY}")

# core.inference.assist_vision / assist_code are imported at runtime as bare
# top-level names (`from core.inference.assist_vision import ...` in
# core/inference/tools.py), not as `studio.backend.core.inference...` --
# run.py puts BACKEND on sys.path itself (`sys.path.insert(0, backend_dir)`),
# never ROOT. The two dotted paths both happen to be importable (studio/ and
# studio/backend/ both have __init__.py), but they resolve to two different
# module objects; collect_submodules() on the studio.backend-prefixed name
# would bundle a parallel identity nothing ever imports, leaving the real
# bare-named submodules unprotected. Put BACKEND on sys.path here too so
# collect_submodules can resolve the same bare names the app actually uses.
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# --- data ------------------------------------------------------------------
# The frontend dist is the UI. Without it the app starts and serves nothing,
# which looks like a backend fault and is not. PyInstaller itself fails loudly
# at build time if any of these three source paths do not exist.
datas = [
    (str(ROOT / "studio" / "frontend" / "dist"), "studio/frontend/dist"),
    (str(BACKEND / "assets"), "studio/backend/assets"),
    (str(BACKEND / "requirements"), "studio/backend/requirements"),
]
for optional in ("vendor", "plugins"):
    src = BACKEND / optional
    if src.is_dir():
        datas.append((str(src), f"studio/backend/{optional}"))
    else:
        print(f"[Unslothed.spec] optional data dir {optional} not present at {src}, skipping")

# torch and its CUDA payload are copied wholesale rather than analysed. Torch
# loads extensions dynamically and ships CUDA DLLs that static analysis never
# sees; letting PyInstaller try produces a build that imports and then fails
# at the first real call.
#
# The install location is found from the package itself, not from
# site.getsitepackages() -- in a venv that list's last entry is not reliably
# the venv's own site-packages, and the whole torch strategy depends on
# finding the real directory. A wrong path here means src.is_dir() below is
# silently False and torch is silently left out of the bundle: it still
# builds, it still starts, and it dies with ModuleNotFoundError on the first
# real tool call -- the most expensive place to discover a packaging bug.
_torch_spec = importlib.util.find_spec("torch")
assert _torch_spec is not None and _torch_spec.origin, (
    "torch is not importable in the environment running this spec. Build "
    "with the same Python environment the app runs against -- one with "
    "torch installed -- not a bare system interpreter."
)
SITE = Path(_torch_spec.origin).parent.parent
assert SITE.is_dir(), (
    f"Derived site-packages={SITE} from torch's spec.origin="
    f"{_torch_spec.origin!r}, but that directory does not exist."
)
print(f"[Unslothed.spec] SITE (site-packages, via torch's spec.origin) = {SITE}")

# torch and torchvision are load-bearing -- hiddenimports below assumes both
# are on disk (assist_vision calls torchvision.models.detection and
# torchvision.transforms.functional). Missing either must fail the build, not
# ship silently broken. nvidia/functorch/torchgen are copied when present but
# are not asserted: nvidia is absent on a legitimate CPU-only torch install
# (this dev machine has one), and functorch/torchgen are torch-internal
# support trees whose layout has moved between torch versions.
_REQUIRED_HEAVY = {"torch", "torchvision"}
for heavy in ("torch", "torchvision", "nvidia", "functorch", "torchgen"):
    src = SITE / heavy
    if src.is_dir():
        print(f"[Unslothed.spec] copying {heavy} wholesale from {src}")
        datas.append((str(src), heavy))
    elif heavy in _REQUIRED_HEAVY:
        raise AssertionError(
            f"Required tree '{heavy}' not found at {src}. The frozen build "
            "would ship without it and fail at the first call that needs it, "
            "not at build time. Install it in this environment before "
            "building."
        )
    else:
        print(f"[Unslothed.spec] {heavy} not found at {src}, skipping (optional)")

# --- hidden imports --------------------------------------------------------
# Every entry below is imported INSIDE a function body somewhere in
# assist_vision or assist_code, deliberately, to keep startup fast. That is
# exactly what PyInstaller's static analysis cannot see. Without these the
# frozen build registers the tool and then raises ModuleNotFoundError on the
# first call -- a failure that looks like a tool bug, not a packaging bug.
hiddenimports = [
    # assist_vision
    "cv2",
    "numpy",
    "onnxruntime",
    "torch",
    "torchvision",
    "torchvision.models.detection",
    "torchvision.transforms.functional",
    "ultralytics",
    "insightface",
    "insightface.app",
    "insightface.model_zoo",
    # assist_code needs no third-party imports -- it is stdlib only, and its
    # language servers are external processes installed on first use.
]
hiddenimports += collect_submodules("core.inference.assist_vision")
hiddenimports += collect_submodules("core.inference.assist_code")

a = Analysis(
    [str(ENTRY)],
    pathex = [str(ROOT), str(BACKEND)],
    binaries = [],
    datas = datas,
    hiddenimports = hiddenimports,
    hookspath = [],
    runtime_hooks = [],
    # Excluded because they are copied in via `datas` above. Leaving them in
    # Analysis doubles the build time and produces a broken CUDA payload.
    excludes = ["torch", "torchvision", "nvidia", "functorch", "torchgen"],
    noarchive = False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries = True,
    name = "Unslothed",
    debug = False,
    strip = False,
    upx = False,          # UPX corrupts CUDA DLLs; never enable it here.
    console = True,       # the backend logs to stdout; a windowed build hides startup failures
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip = False,
    upx = False,
    name = "Unslothed",
)
