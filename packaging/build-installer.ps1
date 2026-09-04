# packaging/build-installer.ps1
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Builds the self-contained Unslothed Windows installer.
#
#   powershell -ExecutionPolicy Bypass -File packaging\build-installer.ps1
#
# Stages: frontend -> PyInstaller -> Inno Setup. The frontend must come first;
# a build without studio/frontend/dist produces an app that starts and serves
# no UI, which reads as a backend fault and is not.
#
# PyInstaller is invoked via an EXPLICIT interpreter path, never via a bare
# `pyinstaller` resolved from PATH. On the machine this was written on, two
# interpreters exist and either could satisfy `pyinstaller` on PATH:
#
#   system Python  -> has PyInstaller,     torch 2.11.0+cpu
#   Studio venv     -> no PyInstaller,      torch 2.10.0+cu130
#
# Building from system Python would produce a complete, working-looking
# bundle that silently ships CPU-only torch under a CUDA promise --
# Unslothed.spec's own "+cu" assertion only protects against this if the
# Studio venv is actually the interpreter that runs it. Resolving the
# interpreter explicitly here makes that mistake structurally impossible
# instead of merely detected. Override the default via -VenvPython or
# $env:UNSLOTHED_BUILD_PYTHON if the Studio venv lives somewhere else.

param(
    [string]$VenvPython = $(
        if ($env:UNSLOTHED_BUILD_PYTHON) { $env:UNSLOTHED_BUILD_PYTHON }
        else { Join-Path $env:USERPROFILE ".unsloth\studio\unsloth_studio\Scripts\python.exe" }
    )
)

$ErrorActionPreference = "Stop"
function Write-Step($m) { Write-Host ""; Write-Host ("==> " + $m) -ForegroundColor Cyan }
function Fail($m) { Write-Host ""; Write-Host ("ERROR: " + $m) -ForegroundColor Red; exit 1 }

$Root = Split-Path -Parent $PSScriptRoot
$Packaging = Join-Path $Root "packaging"

Write-Step "Building the frontend"
Push-Location (Join-Path $Root "studio\frontend")
if (-not (Test-Path "node_modules")) { npm ci }
npm run build
Pop-Location
$dist = Join-Path $Root "studio\frontend\dist\index.html"
if (-not (Test-Path $dist)) { Fail "frontend build produced no dist/index.html" }

Write-Step "Resolving version"
# No `2>` here -- same reasoning as the `pip show` probes below: under
# $ErrorActionPreference = "Stop", redirecting a native command's stderr (even
# to $null) converts it into a terminating error, independent of exit code.
# `--always` means this practically never writes to stderr (it falls back to
# a short SHA instead of failing when there are no tags), but relying on that
# is exactly the kind of "doesn't reproduce today" gap that bit the probes
# below; leaving stderr unredirected is what actually makes this safe.
$Version = (git -C $Root describe --tags --always)
if (-not $Version) { $Version = "0.0.0" }
$Version = $Version -replace '[^0-9A-Za-z.\-]', ''
Write-Host "  version: $Version"

Write-Step "Resolving the build interpreter"
if (-not (Test-Path $VenvPython)) {
    Fail ("Studio venv Python not found at $VenvPython -- pass -VenvPython or set " +
          "`$env:UNSLOTHED_BUILD_PYTHON to the venv's python.exe. This must be the " +
          "interpreter with the CUDA torch build (2.10.0+cuXXX), not system Python, " +
          "which typically has PyInstaller but a CPU-only torch.")
}
Write-Host "  interpreter: $VenvPython"
& $VenvPython -c "import torch; print('  torch: ' + torch.__version__)"
if ($LASTEXITCODE -ne 0) { Fail "torch is not importable in $VenvPython -- this must be the Studio venv, not a bare interpreter." }

Write-Step "Ensuring PyInstaller is installed in the build venv"
# Build-time only: not added to studio/backend/requirements/, not present in
# the shipped app, and removable afterwards. ~5 MB. Installing here (rather
# than relying on a prior manual `pip install pyinstaller`) means this script
# is reproducible from a clean venv, not just from this machine's current state.
#
# The existence probes below redirect stdout only (`1>`), not stderr. That is
# deliberate, not an oversight: under $ErrorActionPreference = "Stop", Windows
# PowerShell converts a native command's stderr output into a terminating
# error the instant it is redirected with `2>` or `*>` -- even when the target
# is $null -- so `pip show <absent-package>` (which prints a WARNING to
# stderr and exits 1) aborts the whole script before the `if ($LASTEXITCODE)`
# check below ever runs, despite the redirection. Leaving stderr unredirected
# sidesteps that conversion entirely: the WARNING prints (harmless, and
# expected on a clean venv's first run) and $LASTEXITCODE comes through
# intact for the branch that follows.
& $VenvPython -m pip show pyinstaller 1> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  installing pyinstaller into the venv"
    & $VenvPython -m pip install pyinstaller
    if ($LASTEXITCODE -ne 0) { Fail "pip install pyinstaller failed" }
}
# cv2, onnxruntime and ultralytics ship no hook in PyInstaller core (only
# hook-PIL* and hook-numpy do) and read their own data files via
# __file__-relative paths at runtime, which Analysis's static import walk
# never follows. Without pyinstaller-hooks-contrib those data files are
# silently absent from the bundle -- the same missing-payload shape as the
# torch problem, one layer down.
& $VenvPython -m pip show pyinstaller-hooks-contrib 1> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  installing pyinstaller-hooks-contrib into the venv (hooks for cv2/onnxruntime/ultralytics)"
    & $VenvPython -m pip install pyinstaller-hooks-contrib
    if ($LASTEXITCODE -ne 0) { Fail "pip install pyinstaller-hooks-contrib failed" }
}

Write-Step "Running PyInstaller"
Push-Location $Packaging
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
& $VenvPython -m PyInstaller --noconfirm --clean "Unslothed.spec"
$pyinstallerExit = $LASTEXITCODE
Pop-Location
if ($pyinstallerExit -ne 0) { Fail "PyInstaller failed with exit code $pyinstallerExit" }
$appExe = Join-Path $Packaging "dist\Unslothed\Unslothed.exe"
if (-not (Test-Path $appExe)) { Fail "PyInstaller produced no Unslothed.exe" }

Write-Step "Checking warn-Unslothed.txt for missing modules"
# PyInstaller writes this file every build, listing every import its static
# analysis could not resolve (some tagged "(delayed)" for imports inside
# function bodies). This is a mechanical, far more complete check for a
# missing hidden import than manual auditing or the smoke test -- it runs
# every time, not just when a build fails.
$warnFile = Join-Path $Packaging "build\Unslothed\warn-Unslothed.txt"
if (Test-Path $warnFile) {
    $missing = Select-String -Path $warnFile -Pattern "missing module" -SimpleMatch
    if ($missing) {
        Write-Host ("  " + $missing.Count + " 'missing module' line(s) in warn-Unslothed.txt:") -ForegroundColor Yellow
        $missing | ForEach-Object { Write-Host ("    " + $_.Line.Trim()) -ForegroundColor Yellow }
        Write-Host "  review these -- a real one belongs in hiddenimports in Unslothed.spec." -ForegroundColor Yellow
    } else {
        Write-Host "  no 'missing module' entries found."
    }
} else {
    Write-Host "  WARNING: no warn-Unslothed.txt found at $warnFile (PyInstaller normally always writes one)." -ForegroundColor Yellow
}

Write-Step "Locating Inno Setup"
$iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
if (-not (Test-Path $iscc)) { $iscc = "$env:ProgramFiles\Inno Setup 6\ISCC.exe" }
if (-not (Test-Path $iscc)) { Fail "ISCC.exe not found -- install Inno Setup 6" }

Write-Step "Compiling the installer"
Push-Location $Packaging
& $iscc "/DMyAppVersion=$Version" "Unslothed.iss"
$isccExit = $LASTEXITCODE
Pop-Location
if ($isccExit -ne 0) { Fail "ISCC failed with exit code $isccExit" }

$setup = Join-Path $Packaging "Output\Unslothed-Setup.exe"
if (-not (Test-Path $setup)) { Fail "no installer at $setup" }
$sizeGb = [math]::Round((Get-Item $setup).Length / 1GB, 2)
Write-Step "Done: $setup ($sizeGb GB)"
