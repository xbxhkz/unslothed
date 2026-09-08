// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { FolderBrowser } from "@/features/model-picker";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useCallback, useState } from "react";
import {
  loadWorkspaceRoot,
  previewWorkspaceRoot,
  saveWorkspaceRoot,
} from "./api/workspace-root";
import { SettingsRow } from "./components/settings-row";
import { useWorkspaceRootField } from "./hooks/use-workspace-root-field";

/**
 * The global workspace folder: where a chat that is not in a project works,
 * instead of its own sandbox. Mirrors the llama.cpp custom-path row this
 * section already has -- an editable path, a Save button, and the same
 * server-side folder browser the model picker uses -- but the state machine
 * behind it (preview-before-commit, warnings that never block) lives in
 * `useWorkspaceRootField` so the project-level version of this control reuses
 * it rather than re-implementing it.
 */
export function WorkspaceRootSetting() {
  const t = useT();
  const [browserOpen, setBrowserOpen] = useState(false);

  const messages = {
    loadFailed: t("settings.resources.storage.workspaceLoadError"),
    previewFailed: t("settings.resources.storage.workspacePreviewError"),
    notExist: t("settings.resources.storage.workspaceNotExist"),
    saveFailed: t("settings.resources.storage.workspaceSaveError"),
  };

  const { path, setPath, warnings, problem, saving, preview, save, clear } =
    useWorkspaceRootField({
      load: loadWorkspaceRoot,
      save: saveWorkspaceRoot,
      preview: previewWorkspaceRoot,
      messages,
    });

  const commit = useCallback(
    async (explicitPath?: string | null) => {
      const ok = await save(explicitPath);
      if (ok) {
        // Only close the browser on success, matching the llama.cpp path
        // row: a failed save leaves it open so the user can retry.
        setBrowserOpen(false);
        toast.success(t("settings.resources.storage.workspaceSaved"));
      } else {
        toast.error(t("settings.resources.storage.workspaceSaveError"));
      }
    },
    [save, t],
  );

  return (
    <SettingsRow
      label={t("settings.resources.storage.workspaceFolder")}
      description={t("settings.resources.storage.workspaceFolderDescription")}
      hint={t("settings.resources.storage.workspaceFolderHint")}
      className="max-[840px]:flex-col max-[840px]:items-stretch max-[840px]:gap-2"
      alignTop={warnings.length > 0 || Boolean(problem)}
    >
      <div className="grid w-[392px] min-w-0 grid-cols-[minmax(0,1fr)_auto_auto] gap-x-2 gap-y-1.5 max-[840px]:w-full">
        <Input
          id="workspace-root"
          aria-label={t("settings.resources.storage.workspaceFolder")}
          value={path}
          placeholder={t("settings.resources.storage.chooseWorkspaceTitle")}
          onChange={(event) => setPath(event.target.value)}
          onBlur={(event) => void preview(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              void commit();
            }
          }}
          title={path || undefined}
          className="h-8 min-w-0 font-mono text-xs"
          data-testid="workspace-root-input"
        />
        <Button
          variant="outline"
          size="sm"
          className="h-8"
          disabled={saving}
          onClick={() => void commit()}
          data-testid="workspace-root-save"
        >
          {saving ? t("common.saving") : t("common.save")}
        </Button>
        <Button
          variant="outline"
          size="sm"
          className="h-8"
          disabled={saving}
          onClick={() => setBrowserOpen(true)}
          data-testid="workspace-root-change"
        >
          {t("settings.resources.storage.changeAction")}
        </Button>

        {warnings.length > 0 || problem || path ? (
          <div className="col-span-3 flex min-w-0 flex-col gap-1 pl-3.5 pr-1">
            {warnings.map((w) => (
              <p
                key={w.code}
                role="alert"
                className="text-xs text-amber-600 dark:text-amber-400"
              >
                {w.message}
              </p>
            ))}
            {problem ? (
              <p
                role="alert"
                className="text-xs text-destructive"
                data-testid="workspace-root-error"
              >
                {problem}
              </p>
            ) : null}
            {path ? (
              // Only clears the draft field, matching the sandbox opt-out
              // being advisory until Save: the folder stays configured on
              // the backend until the user commits the blank value.
              <Button
                variant="link"
                size="xs"
                className="h-auto w-fit px-0 text-xs"
                disabled={saving}
                onClick={clear}
              >
                {t("settings.resources.storage.clearWorkspaceAction")}
              </Button>
            ) : null}
          </div>
        ) : null}
      </div>

      <FolderBrowser
        open={browserOpen}
        onOpenChange={setBrowserOpen}
        onSelect={(picked) => void commit(picked)}
        initialPath={path || undefined}
        title={t("settings.resources.storage.chooseWorkspaceTitle")}
        description={t("settings.resources.storage.workspaceFolderDescription")}
        confirmLabel={t("settings.resources.storage.chooseWorkspaceAction")}
        showModelHints={false}
      />
    </SettingsRow>
  );
}
