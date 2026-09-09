// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { FolderBrowser } from "@/features/model-picker";
import {
  type WorkspaceRootState,
  previewWorkspaceRoot,
  useWorkspaceRootField,
} from "@/features/settings";
import { toast } from "@/lib/toast";
import { useCallback, useEffect, useState } from "react";
import { updateChatProjectRootPath } from "../hooks/use-chat-projects";
import { getStoredChatProject } from "../utils/chat-history-storage";

const MESSAGES = {
  loadFailed: "Could not read this project's folder.",
  previewFailed: "Could not check that folder.",
  notExist: "That folder does not exist yet.",
  saveFailed: "Could not save this project's folder.",
};

async function loadProjectRoot(
  projectId: string,
): Promise<WorkspaceRootState | null> {
  try {
    const record = await getStoredChatProject(projectId);
    if (!record) return null;
    return { path: record.rootPath ?? null, warnings: [] };
  } catch {
    return null;
  }
}

async function saveProjectRoot(
  projectId: string,
  path: string | null,
): Promise<WorkspaceRootState | null> {
  try {
    const record = await updateChatProjectRootPath(projectId, path);
    const savedPath = record.rootPath ?? null;
    // The project PATCH route returns the plain project record, which
    // carries no `warnings` field (routes/chat_history.py's ChatProject
    // model has none) -- classify the path we just committed so a
    // Browse-picked folder still surfaces a warning, exactly as a typed one
    // does through `preview`. A failed classification here degrades to "no
    // warnings" rather than failing the save: the folder is already
    // committed, and warn-only means a missed warning is a lesser problem
    // than reporting a successful save as a failure.
    const warnings = savedPath
      ? ((await previewWorkspaceRoot(savedPath))?.warnings ?? [])
      : [];
    return { path: savedPath, warnings };
  } catch {
    return null;
  }
}

/**
 * A project's own folder: its chats work here instead of the global
 * workspace folder. Reuses `useWorkspaceRootField` -- the same
 * preview-before-commit state machine and warn-only policy behind the
 * global setting (Task 6) -- pointed at this project's PATCH endpoint
 * instead. The project PATCH route returns the full project record, not
 * `{path, warnings}`, so `saveProjectRoot` below folds in a classification
 * from the shared preview endpoint itself: without that, a folder picked
 * through Browse (which saves immediately, with no separate preview step)
 * would silently carry no warning at all, since nothing else in this
 * component's commit path would ever call `preview` for it.
 */
export function ProjectFolderDialog({
  projectId,
  projectName,
  open,
  onOpenChange,
}: {
  projectId: string;
  projectName: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [browserOpen, setBrowserOpen] = useState(false);

  const load = useCallback(() => loadProjectRoot(projectId), [projectId]);
  const save = useCallback(
    (path: string | null) => saveProjectRoot(projectId, path),
    [projectId],
  );

  const {
    path,
    setPath,
    warnings,
    problem,
    saving,
    preview,
    save: commit,
    clear,
    reload,
  } = useWorkspaceRootField({
    load,
    save,
    preview: previewWorkspaceRoot,
    messages: MESSAGES,
  });

  // This dialog stays mounted across opens (`open` only toggles Radix's
  // visibility, matching the project's Rename/Delete dialogs), so a fresh
  // open must re-fetch rather than show a draft abandoned on an earlier
  // cancel, or a value another tab changed meanwhile.
  useEffect(() => {
    if (open) reload();
  }, [open, reload]);

  const commitAndClose = useCallback(
    async (explicitPath?: string | null) => {
      const result = await commit(explicitPath);
      if (!result) {
        toast.error("Couldn't update the project folder");
        return;
      }
      // Only close on success, so a failed save leaves the dialog open to
      // retry rather than silently discarding the edit.
      setBrowserOpen(false);
      onOpenChange(false);
      // The dialog is closing (this stays true for both the typed-Save and
      // Browse-select paths), so its own inline warnings list is about to
      // disappear with it -- carry any warnings into the toast instead of
      // dropping them, using `result` (not hook state) since state set
      // inside `commit` is not guaranteed visible in this closure yet.
      if (result.warnings.length > 0) {
        toast.warning("Project folder updated", {
          description: result.warnings.map((w) => w.message).join(" "),
        });
      } else {
        toast.success("Project folder updated");
      }
    },
    [commit, onOpenChange],
  );

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="corner-squircle dialog-soft-surface sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Project folder</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-muted-foreground">
            Chats in "{projectName}" work in this folder instead of the global
            workspace folder (or the sandbox, if neither is set). Chats that
            already exist switch to it too -- files they wrote earlier stay
            where they are and are no longer listed in those chats.
          </p>
          <div className="flex items-center gap-2">
            <Input
              value={path}
              onChange={(event) => setPath(event.target.value)}
              onBlur={(event) => void preview(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") {
                  event.preventDefault();
                  void commitAndClose();
                }
              }}
              autoFocus={true}
              placeholder="e.g. C:\Users\you\project"
              aria-label="Project folder"
              className="font-mono text-xs"
            />
            <Button
              type="button"
              variant="outline"
              disabled={saving}
              onClick={() => setBrowserOpen(true)}
            >
              Browse
            </Button>
          </div>

          {warnings.length > 0 || problem || path ? (
            <div className="flex flex-col gap-1.5">
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
                <p role="alert" className="text-xs text-destructive">
                  {problem}
                </p>
              ) : null}
              {path ? (
                // Only clears the draft field; the project keeps its stored
                // folder until Save is pressed, matching the warn-only,
                // nothing-happens-until-committed policy.
                <Button
                  type="button"
                  variant="link"
                  size="xs"
                  className="h-auto w-fit px-0 text-xs"
                  disabled={saving}
                  onClick={clear}
                >
                  Clear (use the global folder)
                </Button>
              ) : null}
            </div>
          ) : null}

          <DialogFooter className="flex-wrap gap-2 sm:justify-end">
            <Button
              type="button"
              variant="ghost"
              onClick={() => onOpenChange(false)}
            >
              Cancel
            </Button>
            <Button
              type="button"
              onClick={() => void commitAndClose()}
              disabled={saving}
            >
              {saving ? "Saving..." : "Save"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <FolderBrowser
        open={browserOpen}
        onOpenChange={setBrowserOpen}
        onSelect={(picked) => void commitAndClose(picked)}
        initialPath={path || undefined}
        title="Choose project folder"
        description="Chats in this project will work in the selected folder."
        confirmLabel="Use this folder"
        showModelHints={false}
      />
    </>
  );
}
