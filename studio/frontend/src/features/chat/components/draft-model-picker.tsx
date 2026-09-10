// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { InfoHint } from "@/components/ui/info-hint";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useT } from "@/i18n";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  type DraftCandidate,
  type DraftChoice,
  type SelectResult,
  VERDICT_UNVERIFIED,
  fetchCurrentDraftPin,
  fetchDraftCandidates,
  selectDraftModel,
} from "../api/draft-model-api";

/**
 * Copy for when the model behind `modelId` could not be resolved to a local file at
 * all, as distinct from resolving fine and simply having no colocated siblings.
 * Conflating the two would tell a user "no colocated drafters found" for a model the
 * backend never actually got to look next to.
 */
const UNRESOLVED_MODEL_MESSAGE =
  "Could not resolve this model to look for drafters.";
const NO_CANDIDATES_MESSAGE = "No colocated drafters found.";
/**
 * A third state, distinct from both of the above: the request itself failed, so
 * the backend never answered at all. Collapsing this into NO_CANDIDATES_MESSAGE
 * told the user a model definitely has no drafters on the strength of a proxy
 * error.
 */
const CANDIDATES_UNAVAILABLE_MESSAGE =
  "Could not reach the backend to list drafters.";

/** Modes that launch no separate drafter. A picker here would be a control
 *  with no effect, which is worse than no control. */
const NO_DRAFTER_MODES = new Set(["off", "ngram", "ngram-simple"]);

/**
 * Sentinel Select value standing in for "no pin". Radix's Select.Item
 * rejects an empty-string value (reserved to mean "nothing selected"), and
 * every real candidate ref is either a filesystem path or an HF repo id, so
 * this can never collide with one.
 */
const AUTOMATIC_VALUE = "__automatic__";

/** What an accepted verdict has to say for itself. Never empty: an "hf" pick
 *  carries no size and no vocabulary, so the size/vocab branches alone left the
 *  user with no feedback whatsoever for the one input they typed by hand. */
function describeVerdict(r: SelectResult): string {
  const parts: string[] = [];
  if (r.reason === VERDICT_UNVERIFIED) {
    // The backend checked the SHAPE of the repo id and nothing else -- no
    // network call, so no existence and no vocabulary comparison. Saying so is
    // the spec's requirement for a remote drafter: an explicit could-not-verify
    // state, never a silent pass.
    parts.push("pinned, but this repository could not be verified");
  }
  if (r.vocabTarget !== null && r.vocabDraft !== null) {
    // Deliberately "vocabulary matches", never "compatible": equal vocab size
    // is necessary, not sufficient, for a good speculative pair. A vocab
    // mismatch is a rejection reason (vocab_mismatch), so it never reaches
    // here -- it surfaces as `problem` via r.detail instead.
    parts.push(`vocabulary matches (${r.vocabTarget})`);
  }
  if (r.sizeBytes !== null) {
    parts.push(`${(r.sizeBytes / 1024 ** 3).toFixed(2)} GB`);
  }
  return parts.join(" · ");
}

/** Nothing at all is known about the pin: the read failed. */
const PIN_UNREADABLE_MESSAGE =
  "Could not reach the backend to read the pinned drafter.";
/** The read threw before any response existed. */
const PIN_UNCHECKABLE_MESSAGE =
  "Could not reach the backend to check the pinned draft model.";

/** What the picker should show for a list that came back empty. Three answers,
 *  in decreasing order of how much the backend actually knows. */
function emptyListMessage(candidatesOk: boolean, resolved: boolean): string {
  if (!candidatesOk) {
    return CANDIDATES_UNAVAILABLE_MESSAGE;
  }
  return resolved ? NO_CANDIDATES_MESSAGE : UNRESOLVED_MODEL_MESSAGE;
}

/** The pin, and what to say about it, as one value the effect below applies in
 *  a single pass. `null` means the read itself failed -- which must never be
 *  rendered as "no pin", since that looks exactly like a deliberate Automatic. */
type PinState = {
  pin: { kind: "local" | "hf"; ref: string } | null;
  problem: string;
  note: string;
};

/**
 * Read what is pinned, then re-validate it.
 *
 * Both halves go through the backend, which owns the only parser for the flag
 * vocabulary. An earlier revision matched `--model-draft` and `--spec-draft-hf`
 * inline in the component and so recognised two spellings out of seven: `-md`,
 * `--spec-draft-model`, `-hfd`, `-hfrd`, `--hf-repo-draft`, the `--flag=value`
 * form and the `--model_draft` underscore form all read as "no pin", and the
 * control rendered "Automatic" for a model that was pinned.
 *
 * Re-validation is here because a pinned drafter that has since been deleted
 * fails OPEN at load time (see routes/draft_model.py's module docstring), so
 * this is the only place a missing one becomes visible.
 */
async function readPinState(
  modelId: string,
  ggufVariant: string | null,
  existingArgs: string[],
): Promise<PinState | null> {
  const current = await fetchCurrentDraftPin(existingArgs);
  if (!current.ok) {
    return null;
  }
  const pin = current.pin;
  if (!pin) {
    return { pin: null, problem: "", note: "" };
  }
  const verdict = await selectDraftModel(
    modelId,
    ggufVariant,
    existingArgs,
    pin,
  );
  if (verdict.ok) {
    return { pin, problem: "", note: describeVerdict(verdict) };
  }
  return {
    pin,
    problem: `pinned drafter unusable: ${verdict.detail || verdict.reason}`,
    note: "",
  };
}

export type DraftModelPickerProps = {
  modelId: string;
  ggufVariant: string | null;
  speculativeType: string;
  existingArgs: string[];
  /** True while the stored llama_extra_args are still being read. The picker is
   *  inert until it settles: `existingArgs` is a placeholder `[]` in that
   *  window, so a pick would both be composed against the wrong baseline and
   *  flip llamaExtraArgs from undefined to an array, which is precisely the
   *  signal the page's hydration guards use to detect "the user typed while we
   *  were in flight" and abandon the stored list. */
  hydrating?: boolean;
  onArgsChange: (args: string[]) => void;
};

export function DraftModelPicker({
  modelId,
  ggufVariant,
  speculativeType,
  existingArgs,
  hydrating = false,
  onArgsChange,
}: DraftModelPickerProps) {
  const [candidates, setCandidates] = useState<DraftCandidate[]>([]);
  // Whether the backend could resolve modelId to an actual local file at all, as
  // opposed to resolving it fine and simply finding no siblings. Starts true so a
  // model that hasn't answered yet doesn't flash the unresolved copy.
  const [resolved, setResolved] = useState(true);
  // Whether the candidates request itself succeeded. Third state, see
  // CANDIDATES_UNAVAILABLE_MESSAGE.
  const [candidatesOk, setCandidatesOk] = useState(true);
  // Mirrors what an uncontrolled native <select> tracks for free: which
  // option reads as selected. Radix's Select is a controlled component, so
  // this state exists purely to reflect the current pin back into the
  // trigger -- it plays no part in validation or in what gets sent to
  // onArgsChange.
  const [selectedRef, setSelectedRef] = useState("");
  const [hfRepo, setHfRepo] = useState("");
  const [problem, setProblem] = useState("");
  const [note, setNote] = useState("");
  // Distinct from selectedRef/hfRepo: hfRepo also holds whatever the user is
  // mid-typing into the HF repo box, which is not yet a pin. This tracks only
  // a backend-confirmed pin (a successful apply, or a read-back that found
  // one), so it can gate the auto-load note below without a typing keystroke
  // flipping it on before anything is actually applied.
  const [hasPin, setHasPin] = useState(false);
  const t = useT();

  const hidden = NO_DRAFTER_MODES.has(speculativeType);
  const disabled = hydrating;
  // A stable identity for the effect below, so it re-runs when the CONTENT of
  // the argument list changes rather than on every re-render that hands down a
  // fresh array. `existingArgs` hydrates asynchronously: the previous effect
  // was mount-only, so on the common path (Advanced already open from
  // localStorage) it ran once against the placeholder `[]` and never again,
  // and the pin it was written to find had not arrived yet.
  const argsKey = JSON.stringify(existingArgs);
  // The last list this component itself wrote. Re-reading our own write would
  // be harmless but wasteful, and re-validating it would re-report a verdict
  // the user has already been shown.
  const selfWrittenKey = useRef<string | null>(null);

  useEffect(() => {
    if (hidden || !modelId) {
      return;
    }
    let cancelled = false;
    fetchDraftCandidates(modelId, ggufVariant)
      .then((r) => {
        if (!cancelled) {
          setCandidates(r.candidates);
          setResolved(r.resolved);
          setCandidatesOk(r.ok);
        }
      })
      .catch(() => {
        // authFetch's own fetch() throws outright when the backend is not
        // listening (auth/api.ts), before any Response exists for
        // fetchDraftCandidates to inspect. Without this the rejection is
        // unhandled and the list silently stays empty, which renders as
        // "No colocated drafters found" -- a confident, wrong answer.
        if (!cancelled) {
          setCandidates([]);
          setCandidatesOk(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [hidden, modelId, ggufVariant]);

  const apply = useCallback(
    async (choice: DraftChoice) => {
      setProblem("");
      setNote("");
      try {
        const r = await selectDraftModel(
          modelId,
          ggufVariant,
          existingArgs,
          choice,
        );
        if (!r.ok || r.llamaExtraArgs === null) {
          // Rejected: surface the reason and leave the caller's args untouched.
          // Covers the unresolved-model verdict the same way as any other
          // rejection -- the backend's reason/detail text is what's shown.
          setProblem(r.detail || r.reason);
          return;
        }
        setNote(
          choice === null
            ? "Automatic: the backend picks a colocated sidecar."
            : describeVerdict(r),
        );
        setHasPin(choice !== null);
        selfWrittenKey.current = JSON.stringify(r.llamaExtraArgs);
        onArgsChange(r.llamaExtraArgs);
      } catch {
        // Not merely a defensive backstop: selectDraftModel guards a bad
        // status or an unparsable body, but authFetch's own fetch call can
        // still throw outright (offline, backend not listening yet) before
        // any response exists to guard. Confirmed by exercising this path --
        // without this catch, that throw is an unhandled rejection and the
        // picker goes blank with no indication anything failed.
        setProblem("could not reach the backend to check this draft model");
      }
    },
    [modelId, ggufVariant, existingArgs, onArgsChange],
  );

  // Seed the controls from what is actually pinned, and re-validate it (see
  // readPinState above for why the backend answers both questions).
  //
  // Keyed on the CONTENT of existingArgs rather than on mount, because the
  // stored arguments hydrate asynchronously and the mount-time version ran
  // before they arrived. Our own writes are skipped by key, which is what the
  // mount-only dependency list was really trying to achieve.
  // biome-ignore lint/correctness/useExhaustiveDependencies: argsKey is the content identity of existingArgs; depending on the array itself would re-run on every render
  useEffect(() => {
    if (hidden || !modelId || hydrating) {
      return;
    }
    if (argsKey === selfWrittenKey.current) {
      return;
    }
    let cancelled = false;
    readPinState(modelId, ggufVariant, existingArgs)
      .then((state) => {
        if (cancelled) {
          return;
        }
        if (state === null) {
          // Never fall through to "no pin" here: that is indistinguishable from
          // a genuine Automatic and would show the user an unpinned control for
          // a pinned model.
          setProblem(PIN_UNREADABLE_MESSAGE);
          return;
        }
        setSelectedRef(state.pin?.kind === "local" ? state.pin.ref : "");
        setHfRepo(state.pin?.kind === "hf" ? state.pin.ref : "");
        setHasPin(state.pin !== null);
        setProblem(state.problem);
        setNote(state.note);
      })
      .catch(() => {
        // Same backstop as apply(): authFetch's own fetch() can throw before
        // any Response exists (offline, backend not listening yet), which the
        // api client's internal guards cannot catch. Without this, a pinned
        // drafter that can't be read back produces an unhandled rejection and
        // the user sees nothing at all.
        if (!cancelled) {
          setProblem(PIN_UNCHECKABLE_MESSAGE);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [hidden, modelId, ggufVariant, hydrating, argsKey]);

  if (hidden) {
    return null;
  }

  // A pin that is not among the offered candidates still has to read as
  // selected -- a hand-written --model-draft, or a file the candidates request
  // could not list. Radix renders an empty trigger for a value with no matching
  // item, which would look exactly like "Automatic".
  const pinnedIsListed = candidates.some((c) => c.ref === selectedRef);

  // Real gap, not a fault: llama_server_args.py's inherited-extras stripper
  // treats every drafter-selector flag as shadowing speculative_type (the
  // MTP-path comment at _SPEC_FLAGS explains why -- an inherited copy must
  // not last-wins-override Unsloth's own auto-detected drafter). That strip
  // only runs on a load that omits llama_extra_args and inherits it, and only
  // when speculative_type is explicitly set to a non-auto value -- exactly an
  // auto-switch load, an idle reload, or a chat-settings Apply for a model
  // with both a pin and a forced strategy. A Run Settings load always sends
  // llama_extra_args explicitly, so it is never stripped there.
  const showsAutoLoadFallbackNote =
    hasPin && speculativeType !== "" && speculativeType !== "auto";

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 items-center gap-1.5">
          <span className="min-w-0 text-ui-13 font-medium leading-[1.25] tracking-nav text-nav-fg">
            Draft model
          </span>
          <InfoHint>
            Pin a specific drafter for speculative decoding, or leave Automatic
            to let the backend choose. Equal vocabulary size is required for a
            drafter to load, but does not by itself guarantee good speculative
            acceptance.
          </InfoHint>
        </div>
        <Select
          value={selectedRef || AUTOMATIC_VALUE}
          disabled={disabled}
          onValueChange={(v) => {
            if (v === AUTOMATIC_VALUE) {
              setSelectedRef("");
              apply(null);
              return;
            }
            setSelectedRef(v);
            apply({ kind: "local", ref: v });
          }}
        >
          <SelectTrigger
            className="panel-select-trigger h-8 w-[200px] shrink-0"
            aria-label="Draft model"
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent className="menu-soft-surface ring-0 border-0 rounded-lg">
            <SelectItem value={AUTOMATIC_VALUE}>Automatic</SelectItem>
            {selectedRef && !pinnedIsListed && (
              <SelectItem value={selectedRef}>{selectedRef}</SelectItem>
            )}
            {candidates.map((c) => (
              <SelectItem key={c.ref} value={c.ref}>
                {c.label}
                {c.source === "sidecar" ? " (sidecar)" : ""}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {candidates.length === 0 && (
        <p className="text-ui-11 text-muted-foreground">
          {emptyListMessage(candidatesOk, resolved)}
        </p>
      )}

      <div className="flex items-center gap-2">
        <Input
          value={hfRepo}
          placeholder="Hugging Face repo, e.g. unsloth/Qwen3-0.6B-GGUF"
          aria-label="Hugging Face draft model repository"
          disabled={disabled}
          onChange={(e) => setHfRepo(e.target.value)}
          className="h-8 flex-1"
        />
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={disabled || !hfRepo.trim()}
          onClick={() => apply({ kind: "hf", ref: hfRepo.trim() })}
        >
          Use repo
        </Button>
      </div>

      <Button
        type="button"
        variant="ghost"
        size="sm"
        disabled={disabled}
        onClick={() => {
          setSelectedRef("");
          setHfRepo("");
          apply(null);
        }}
      >
        Clear (use automatic)
      </Button>

      {problem && (
        <p role="alert" className="text-ui-11 text-destructive">
          {problem}
        </p>
      )}
      {!problem && note && (
        <p className="text-ui-11 text-muted-foreground">{note}</p>
      )}
      {!problem && showsAutoLoadFallbackNote && (
        <p className="text-ui-11 text-muted-foreground">
          {t("chat.draftModelPicker.autoLoadFallbackNote")}
        </p>
      )}
    </div>
  );
}
