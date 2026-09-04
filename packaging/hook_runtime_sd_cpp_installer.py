# packaging/hook_runtime_sd_cpp_installer.py
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# PyInstaller runtime hook: PyInstaller runs this automatically at frozen-app
# startup, before the entry script (studio/backend/run.py) executes.
#
# Why this exists: core/inference/sd_cpp_backend.py's _installer_module()
# computes its own target directory as Path(__file__).resolve().parents[3]
# and does sys.path.insert(0, str(that)), then `import install_sd_cpp_prebuilt`.
# In the dev tree that correctly lands on studio/ (four levels up from
# studio/backend/core/inference/sd_cpp_backend.py). In a PyInstaller 6.x
# onedir frozen build it does not: PyInstaller gives frozen modules a
# synthetic __file__ rooted at sys._MEIPASS (the _internal/ folder), so
# parents[3] from _internal/core/inference/sd_cpp_backend.py lands one level
# ABOVE _internal -- the top-level onedir folder next to Unslothed.exe, not
# inside _internal at all. Unslothed.spec's own
# datas.append((..., "studio")) entry places install_sd_cpp_prebuilt.py at
# _internal/studio/install_sd_cpp_prebuilt.py, per PyInstaller's COLLECT
# convention that nothing but the exe itself can land outside _internal/ --
# so the two paths never meet, and no datas entry alone can fix this (there
# is no datas destination that reaches outside _internal/). Confirmed with a
# standalone PyInstaller 6.20.0 onedir probe build before writing this.
#
# The fix here does not need to correct _installer_module()'s own (wrong,
# when frozen) insert -- it only needs the CORRECT directory to already be on
# sys.path before that code runs, since Python's import system searches the
# whole of sys.path, not only index 0. _installer_module()'s own insert then
# becomes a harmless extra entry that simply doesn't contain the file.
import os
import sys

_meipass = getattr(sys, "_MEIPASS", None)
if _meipass:
    _studio_dir = os.path.join(_meipass, "studio")
    if _studio_dir not in sys.path:
        sys.path.insert(0, _studio_dir)
