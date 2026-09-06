// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Client for the draft-model chooser.
 *
 * This module deliberately does NOT build llama-server flags. The drafter flag
 * vocabulary is seven spellings across two families, each accepting `-f v` and
 * `-f=v`, with last-wins ordering; the backend already parses that set, and a
 * second implementation here would be free to drift from it. We send a choice
 * and receive a finished argument array.
 */

import { authFetch } from "@/features/auth";

export type DraftChoice = { kind: "local" | "hf"; ref: string } | null;

/** Reasons the backend can send back that the picker treats specially. Kept as
 *  constants so a rename on either side is one grep, not a silent miss. */
export const VERDICT_UNVERIFIED = "unverified";

export type DraftCandidate = {
  kind: string;
  ref: string;
  label: string;
  source: "sidecar" | "local";
};

export type SelectResult = {
  ok: boolean;
  reason: string;
  detail: string;
  sizeBytes: number | null;
  vocabTarget: number | null;
  vocabDraft: number | null;
  llamaExtraArgs: string[] | null;
};

/**
 * `resolved` is separate from an empty `candidates` list on purpose: the backend
 * resolves `modelId` (an HF repo id or a local path) to an actual local GGUF file
 * before it can glob that file's directory for siblings, and that resolution can
 * fail (nothing cached locally under this id/variant yet). An empty list for that
 * reason reads as "no drafters were found" when the truth is "this model could not
 * even be checked" -- two different things the picker has to say differently.
 */
export type DraftCandidatesResult = {
  candidates: DraftCandidate[];
  resolved: boolean;
  /** False when the request itself failed (non-2xx, unparsable body). Distinct
   *  again from `resolved`: the backend never answered, so nothing at all is
   *  known about this model -- not even that it could not be resolved. */
  ok: boolean;
};

export async function fetchDraftCandidates(
  modelId: string,
  ggufVariant: string | null,
): Promise<DraftCandidatesResult> {
  const params = new URLSearchParams({
    // biome-ignore lint/style/useNamingConvention: API schema
    model_id: modelId,
  });
  if (ggufVariant) {
    params.set("gguf_variant", ggufVariant);
  }
  const url = `/api/draft-model/candidates?${params.toString()}`;
  const res = await authFetch(url);
  if (!res.ok) {
    // NOT `resolved: true`. That claimed the backend had looked and found
    // nothing, so every backend or proxy error rendered as the one message
    // this feature went out of its way to make distinguishable ("No colocated
    // drafters found") -- the third site in this branch with that defect.
    return { candidates: [], resolved: false, ok: false };
  }
  const body = await res.json().catch(() => null);
  if (body === null) {
    return { candidates: [], resolved: false, ok: false };
  }
  return {
    candidates: Array.isArray(body?.candidates) ? body.candidates : [],
    resolved: Boolean(body?.resolved),
    ok: true,
  };
}

/** What `existingArgs` currently pins, as the backend reads it. */
export type CurrentPinResult = {
  pin: { kind: "local" | "hf"; ref: string } | null;
  /** False when the request failed. The picker must not seed "Automatic" off a
   *  failed read -- that is indistinguishable from a genuine no-pin answer and
   *  is how a pinned model came to render as unpinned. */
  ok: boolean;
};

/**
 * The pinned drafter behind an argument list.
 *
 * This exists so the picker never parses llama-server flags itself. The
 * vocabulary is seven spellings in two families, each in `-f v` and `-f=v`
 * form, with `_` normalised to `-`; the backend already owns that parser, and
 * a second one here is exactly what this module's header says it will not do.
 * A hand-rolled reader was written once anyway, recognised two spellings, and
 * displayed "Automatic" for a model that was pinned.
 *
 * POST because the input is an argument list, not an identifier.
 */
export async function fetchCurrentDraftPin(
  existingArgs: string[],
): Promise<CurrentPinResult> {
  const res = await authFetch("/api/draft-model/current", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      // biome-ignore lint/style/useNamingConvention: API schema
      existing_args: existingArgs,
    }),
  });
  const body = await res.json().catch(() => null);
  if (!res.ok || body === null) {
    return { pin: null, ok: false };
  }
  const pin = body?.pin;
  if (
    pin &&
    (pin.kind === "local" || pin.kind === "hf") &&
    typeof pin.ref === "string" &&
    pin.ref !== ""
  ) {
    return { pin: { kind: pin.kind, ref: pin.ref }, ok: true };
  }
  return { pin: null, ok: true };
}

export async function selectDraftModel(
  modelId: string,
  ggufVariant: string | null,
  existingArgs: string[],
  choice: DraftChoice,
): Promise<SelectResult> {
  const res = await authFetch("/api/draft-model/select", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      // biome-ignore lint/style/useNamingConvention: API schema
      model_id: modelId,
      // biome-ignore lint/style/useNamingConvention: API schema
      gguf_variant: ggufVariant,
      // biome-ignore lint/style/useNamingConvention: API schema
      existing_args: existingArgs,
      choice: choice ? { kind: choice.kind, ref: choice.ref } : null,
    }),
  });
  const b = await res.json().catch(() => null);
  if (!res.ok || b === null) {
    // A non-2xx status or an unparsable body (dev-server HTML fallback, proxy
    // error page, a 500 with no JSON handler) is a real possibility, not a
    // theoretical one -- surface it as an ordinary rejection rather than
    // letting a thrown SyntaxError reach an unguarded caller and leave the
    // picker blank with no indication anything failed.
    return {
      ok: false,
      reason: "network",
      detail: `the backend could not be reached (HTTP ${res.status})`,
      sizeBytes: null,
      vocabTarget: null,
      vocabDraft: null,
      llamaExtraArgs: null,
    };
  }
  return {
    ok: Boolean(b?.ok),
    reason: String(b?.reason ?? ""),
    detail: String(b?.detail ?? ""),
    sizeBytes: b?.size_bytes ?? null,
    vocabTarget: b?.vocab_target ?? null,
    vocabDraft: b?.vocab_draft ?? null,
    llamaExtraArgs: Array.isArray(b?.llama_extra_args)
      ? b.llama_extra_args
      : null,
  };
}
