; packaging/Unslothed.iss
; SPDX-License-Identifier: AGPL-3.0-only
; Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
;
; Per-user install, no elevation. Studio keeps its data in %USERPROFILE%\.unsloth\studio
; and this installer must never write there -- installing must not disturb an
; existing setup's models, chats or auth.

#define MyAppName "Unslothed"
#define MyAppExeName "Unslothed.exe"

[Setup]
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={localappdata}\{#MyAppName}
DefaultGroupName={#MyAppName}
OutputBaseFilename=Unslothed-Setup
OutputDir=Output
Compression=lzma2/max
SolidCompression=yes
; lowest = install for this user only, no UAC prompt. Matches DefaultDirName.
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible
DisableProgramGroupPage=yes
UninstallDisplayName={#MyAppName}

[Files]
Source: "dist\Unslothed\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{userdesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
