// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useState } from "react";
import type { WorkspaceRootWarning } from "../api/workspace-root";

interface WorkspaceRootResult {
  path: string | null;
  warnings: WorkspaceRootWarning[];
}

interface WorkspaceRootPreviewResult extends WorkspaceRootResult {
  exists: boolean;
}

export interface UseWorkspaceRootFieldOptions {
  /** Reads the current stored value. `null` on any failure. */
  load: () => Promise<WorkspaceRootResult | null>;
  /** Commits `path` (already trimmed, or `null` to clear). `null` on any
   * failure -- the two callers store this differently (a global setting vs.
   * a project's `rootPath`), so only the endpoint differs, not the shape. */
  save: (path: string | null) => Promise<WorkspaceRootResult | null>;
  /** Classifies a candidate WITHOUT storing it. The same endpoint for every
   * caller: classification has nothing specific to what is being edited. */
  preview: (path: string) => Promise<WorkspaceRootPreviewResult | null>;
  /** Copy for the failure cases, supplied by the caller rather than
   * hardcoded here: the settings surface localizes through `useT()` while
   * the project dialog does not, so this hook must not pick a language
   * convention on either caller's behalf. */
  messages: {
    loadFailed: string;
    previewFailed: string;
    notExist: string;
    saveFailed: string;
  };
}

/**
 * State machine shared by every workspace-root field (the global setting,
 * and a project's own folder): load the current value, preview a candidate
 * so warnings appear before it is committed, and save it.
 *
 * Warnings are advisory only -- nothing here ever blocks a save. Every
 * network call the injected functions make is expected to resolve to `null`
 * on failure rather than throw or leave a rejection unhandled, so a problem
 * always surfaces as `problem` text instead of a silently blank field.
 */
export function useWorkspaceRootField({
  load,
  save,
  preview,
  messages,
}: UseWorkspaceRootFieldOptions) {
  const [path, setPath] = useState("");
  const [warnings, setWarnings] = useState<WorkspaceRootWarning[]>([]);
  const [problem, setProblem] = useState("");
  const [saved, setSaved] = useState(false);
  const [saving, setSaving] = useState(false);
  const [loaded, setLoaded] = useState(false);

  // Resolves to the loaded path (`""` when cleared, `null` when the load
  // failed or was superseded) so `reload` below can chain a preview off it.
  // Exposed as `reload` too: a caller that keeps this hook mounted across
  // opens (a dialog toggled by `open`, rather than mounted fresh each time)
  // needs a way to re-fetch instead of showing whatever was left over --
  // possibly an abandoned, never-saved draft -- from the previous open.
  const runLoad = useCallback(
    (signal: { cancelled: boolean }) => {
      setProblem("");
      setSaved(false);
      return load().then((result) => {
        if (signal.cancelled) return null;
        setLoaded(true);
        if (!result) {
          setProblem(messages.loadFailed);
          return null;
        }
        const loadedPath = result.path ?? "";
        setPath(loadedPath);
        setWarnings(result.warnings);
        return loadedPath;
      });
    },
    [load, messages.loadFailed],
  );

  useEffect(() => {
    const signal = { cancelled: false };
    void runLoad(signal);
    return () => {
      signal.cancelled = true;
    };
  }, [runLoad]);

  const runPreview = useCallback(
    async (candidate: string) => {
      setProblem("");
      setSaved(false);
      if (!candidate.trim()) {
        setWarnings([]);
        return;
      }
      const result = await preview(candidate);
      if (!result) {
        setProblem(messages.previewFailed);
        return;
      }
      setWarnings(result.warnings);
      if (!result.exists) setProblem(messages.notExist);
    },
    [preview, messages.previewFailed, messages.notExist],
  );

  // Re-fetches, then re-classifies the loaded path through `preview` when it
  // is non-empty. The plain load alone is not enough for a caller whose
  // `load` cannot itself carry authoritative warnings (a project's PATCH
  // route returns no `warnings` field, so its `load` hard-codes an empty
  // array) -- without this, reopening a dialog on an already-flagged root
  // would silently show no warning at all until the user re-typed the path.
  const reload = useCallback(async () => {
    const signal = { cancelled: false };
    const loadedPath = await runLoad(signal);
    if (signal.cancelled || !loadedPath) return;
    await runPreview(loadedPath);
  }, [runLoad, runPreview]);

  // `explicitPath` lets a caller (the folder browser) commit the path it just
  // picked directly, rather than relying on `path` state that a preceding
  // setState in the same handler has not applied yet. Returns the saved
  // `{path, warnings}` (not just a boolean) so a caller that closes its own
  // UI on success -- and so can no longer rely on this hook's state being
  // rendered anywhere -- still has the warnings in hand to show some other
  // way (e.g. a toast).
  const commit = useCallback(
    async (explicitPath?: string | null) => {
      const target =
        explicitPath !== undefined
          ? explicitPath?.trim() || null
          : path.trim() || null;
      setProblem("");
      setSaving(true);
      const result = await save(target);
      setSaving(false);
      if (!result) {
        setProblem(messages.saveFailed);
        return null;
      }
      setPath(result.path ?? "");
      setWarnings(result.warnings);
      setSaved(true);
      return result;
    },
    [save, path, messages.saveFailed],
  );

  const clear = useCallback(() => {
    setPath("");
    setWarnings([]);
    setProblem("");
    setSaved(false);
  }, []);

  return {
    path,
    setPath,
    warnings,
    problem,
    saved,
    saving,
    loaded,
    preview: runPreview,
    save: commit,
    clear,
    reload,
  };
}
