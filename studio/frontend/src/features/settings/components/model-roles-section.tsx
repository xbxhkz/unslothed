// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useT } from "@/i18n";
import { type ReactElement, useCallback, useEffect, useState } from "react";
import { loadModelRoles, saveModelRoles } from "../api/model-roles";
import { listOpenAIModels } from "../api/openai-models";
import {
  DELEGATABLE_ROLES,
  type RoleRow,
  mergeRoleRows,
} from "../lib/model-roles-binding";
import { SettingsRow } from "./settings-row";
import { SettingsSection } from "./settings-section";

const STATE_TONE: Record<string, string> = {
  ready: "text-muted-foreground",
  missing: "text-destructive",
  unknown: "text-muted-foreground",
};

function RoleRowControl({
  row,
  models,
  disabled,
  onChange,
}: {
  row: RoleRow;
  models: string[];
  disabled: boolean;
  onChange: (model: string) => void;
}): ReactElement {
  const t = useT();
  const delegatable = (DELEGATABLE_ROLES as readonly string[]).includes(row.role);
  // A model bound before it was deleted from disk still has to appear in its own
  // dropdown, or opening Settings would silently re-point the role at whatever
  // sorts first -- and the next save would persist that.
  const options = row.model && !models.includes(row.model) ? [row.model, ...models] : models;
  return (
    <SettingsRow
      label={row.role}
      description={
        delegatable
          ? row.detail || t("settings.modelRoles.unbound")
          : t("settings.modelRoles.notDelegatable")
      }
      alignTop
    >
      <div className="flex flex-col items-end gap-1">
        <select
          className="h-8 w-[260px] rounded-md border border-input bg-background px-2 text-sm"
          value={row.model}
          disabled={disabled}
          aria-label={t("settings.modelRoles.modelForRole", { role: row.role })}
          onChange={(event) => onChange(event.target.value)}
        >
          <option value="">{t("settings.modelRoles.noModel")}</option>
          {options.map((id) => (
            <option key={id} value={id}>
              {id}
            </option>
          ))}
        </select>
        {row.model ? (
          <span className={`text-xs ${STATE_TONE[row.state] ?? STATE_TONE.unknown}`}>
            {t(`settings.modelRoles.state.${row.state}`)}
          </span>
        ) : null}
      </div>
    </SettingsRow>
  );
}

export function ModelRolesSection(): ReactElement {
  const t = useT();
  const [rows, setRows] = useState<RoleRow[]>([]);
  const [models, setModels] = useState<string[]>([]);
  const [overridesNote, setOverridesNote] = useState("");
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const [read, available] = await Promise.all([
          loadModelRoles(),
          // A failure here must not take the section down with it: the bindings
          // are still readable and editable by typing, so the dropdown degrades
          // to whatever is already bound rather than the whole panel erroring.
          listOpenAIModels().catch(() => []),
        ]);
        if (cancelled) return;
        setRows(mergeRoleRows(read.rows));
        setOverridesNote(read.overridesNote);
        setModels(available.map((m) => m.id));
        setLoadError(null);
      } catch (err) {
        if (cancelled) return;
        // Never render a failed read as "nothing is bound": those are opposite
        // claims, and the second one invites a save that wipes real bindings,
        // because PUT replaces the whole set.
        console.error("model-roles: failed to load", err);
        setLoadError(err instanceof Error ? err.message : String(err));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const onSave = useCallback(async () => {
    setIsSaving(true);
    setSaveError(null);
    try {
      const read = await saveModelRoles(rows);
      setRows(mergeRoleRows(read.rows));
      setOverridesNote(read.overridesNote);
      setSaved(true);
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsSaving(false);
    }
  }, [rows]);

  const setModel = useCallback((role: string, model: string) => {
    setSaved(false);
    setRows((current) =>
      current.map((row) => (row.role === role ? { ...row, model } : row)),
    );
  }, []);

  return (
    <SettingsSection
      title={t("settings.modelRoles.title")}
      description={t("settings.modelRoles.description")}
    >
      {/* The Save button lives INSIDE this else-branch deliberately, not behind a
          disabled flag. PUT replaces the whole set, so saving after a failed read
          would send the empty rows a failed read leaves behind and delete every
          binding the user could not see. Structural beats conditional here: there
          is no state in which the button exists and the bindings are unknown.
          Keep it inside this branch if you refactor -- the node --test suite runs
          plain modules and cannot see JSX, so nothing will catch it moving out. */}
      {loadError ? (
        <p className="py-2 text-xs text-destructive">
          {t("settings.modelRoles.loadFailed")}
        </p>
      ) : (
        <>
          {rows.map((row) => (
            <RoleRowControl
              key={row.role}
              row={row}
              models={models}
              disabled={isSaving}
              onChange={(model) => setModel(row.role, model)}
            />
          ))}
          <div className="flex items-center justify-end gap-2 pt-2">
            {saveError ? (
              <span className="max-w-[320px] text-right text-xs text-destructive">
                {saveError}
              </span>
            ) : saved ? (
              <span className="text-xs text-muted-foreground">
                {t("settings.modelRoles.saved")}
              </span>
            ) : null}
            <Button variant="outline" size="sm" disabled={isSaving} onClick={onSave}>
              {isSaving ? t("common.saving") : t("common.save")}
            </Button>
          </div>
          {overridesNote ? (
            // The server's own sentence, rendered verbatim rather than
            // paraphrased: it is the only thing telling a user that a stored
            // override does nothing.
            <p className="pt-2 text-xs text-muted-foreground leading-relaxed">
              {overridesNote}
            </p>
          ) : null}
        </>
      )}
    </SettingsSection>
  );
}
