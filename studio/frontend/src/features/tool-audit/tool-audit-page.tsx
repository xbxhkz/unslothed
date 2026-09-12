// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ReactElement, useCallback, useEffect, useState } from "react";
import { useT } from "@/i18n";
import { authFetch } from "@/features/auth";

type AuditEntry = {
  id: number;
  ts: number;
  duration_ms: number | null;
  tool_name: string;
  arguments_json: string;
  outcome: string;
  redacted: boolean;
  disable_sandbox: boolean;
  result_head: string | null;
  result_tail: string | null;
  result_bytes: number | null;
  error_text: string | null;
};

export function ToolAuditPage(): ReactElement {
  const t = useT();
  const [entries, setEntries] = useState<AuditEntry[]>([]);
  const [expanded, setExpanded] = useState<number | null>(null);
  const [failedWrites, setFailedWrites] = useState(0);
  const [toolFilter, setToolFilter] = useState<string>("");

  const load = useCallback(async () => {
    const qs = toolFilter ? `?tool_name=${encodeURIComponent(toolFilter)}` : "";
    const [listRes, statusRes] = await Promise.all([
      authFetch(`/api/tool-audit/entries${qs}`),
      authFetch("/api/tool-audit/status"),
    ]);
    if (listRes.ok) setEntries((await listRes.json()).entries ?? []);
    if (statusRes.ok) setFailedWrites((await statusRes.json()).failed_writes ?? 0);
  }, [toolFilter]);

  useEffect(() => {
    void load();
  }, [load]);

  const tools = Array.from(new Set(entries.map((e) => e.tool_name))).sort();

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center gap-3">
        <h1 className="text-lg font-semibold">{t("toolAudit.title")}</h1>
        <select
          className="border rounded px-2 py-1 text-sm"
          value={toolFilter}
          onChange={(e) => setToolFilter(e.target.value)}
          aria-label={t("toolAudit.filterTool")}
        >
          <option value="">{t("toolAudit.filterAll")}</option>
          {tools.map((name) => (
            <option key={name} value={name}>{name}</option>
          ))}
        </select>
        <button className="border rounded px-2 py-1 text-sm" onClick={() => void load()}>
          {t("toolAudit.refresh")}
        </button>
      </div>

      {failedWrites > 0 && (
        <div role="alert" className="border border-amber-500 rounded p-2 text-sm">
          {t("toolAudit.degraded", { count: failedWrites })}
        </div>
      )}

      {entries.length === 0 ? (
        <p className="text-sm opacity-70">{t("toolAudit.empty")}</p>
      ) : (
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left">
              <th>{t("toolAudit.columnTime")}</th>
              <th>{t("toolAudit.columnTool")}</th>
              <th>{t("toolAudit.columnOutcome")}</th>
              <th>{t("toolAudit.columnDuration")}</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((e) => (
              <tr key={e.id} className="border-t align-top">
                <td>
                  <button onClick={() => setExpanded(expanded === e.id ? null : e.id)}>
                    {new Date(e.ts * 1000).toLocaleString()}
                  </button>
                  {expanded === e.id && (
                    <div className="my-2 space-y-2">
                      <div>
                        <strong>{t("toolAudit.argumentsHeading")}</strong>
                        <pre className="overflow-x-auto text-xs">{e.arguments_json}</pre>
                      </div>
                      {e.error_text ? (
                        <div>
                          <strong>{t("toolAudit.errorHeading")}</strong>
                          <pre className="overflow-x-auto text-xs">{e.error_text}</pre>
                        </div>
                      ) : (
                        <div>
                          <strong>{t("toolAudit.resultHeading")}</strong>
                          <pre className="overflow-x-auto text-xs">
                            {(e.result_head ?? "") + (e.result_tail ? `\n...\n${e.result_tail}` : "")}
                          </pre>
                          {e.result_tail ? (
                            <p className="text-xs opacity-70">
                              {t("toolAudit.truncatedNote", { bytes: e.result_bytes ?? 0 })}
                            </p>
                          ) : null}
                        </div>
                      )}
                    </div>
                  )}
                </td>
                <td>
                  {e.tool_name}
                  {e.redacted && <span className="ml-1 text-xs">[{t("toolAudit.redactedBadge")}]</span>}
                  {e.disable_sandbox && (
                    <span className="ml-1 text-xs">[{t("toolAudit.sandboxBypassBadge")}]</span>
                  )}
                </td>
                <td>{e.outcome}</td>
                <td>{e.duration_ms == null ? "-" : `${e.duration_ms} ms`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
