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
import { useCallback, useEffect, useState } from "react";
import {
  type DraftCandidate,
  type DraftChoice,
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

export type DraftModelPickerProps = {
  modelId: string;
  ggufVariant: string | null;
  speculativeType: string;
  existingArgs: string[];
  onArgsChange: (args: string[]) => void;
};

export function DraftModelPicker({
  modelId,
  ggufVariant,
  speculativeType,
  existingArgs,
  onArgsChange,
}: DraftModelPickerProps) {
  const [candidates, setCandidates] = useState<DraftCandidate[]>([]);
  // Whether the backend could resolve modelId to an actual local file at all, as
  // opposed to resolving it fine and simply finding no siblings. Starts true so a
  // model that hasn't answered yet doesn't flash the unresolved copy.
  const [resolved, setResolved] = useState(true);
  // Mirrors what an uncontrolled native <select> tracks for free: which
  // option reads as selected. Radix's Select is a controlled component, so
  // this state exists purely to reflect the last choice back into the
  // trigger -- it plays no part in validation or in what gets sent to
  // onArgsChange.
  const [selectedRef, setSelectedRef] = useState("");
  const [hfRepo, setHfRepo] = useState("");
  const [problem, setProblem] = useState("");
  const [note, setNote] = useState("");

  const hidden = NO_DRAFTER_MODES.has(speculativeType);

  useEffect(() => {
    if (hidden || !modelId) {
      return;
    }
    let cancelled = false;
    fetchDraftCandidates(modelId, ggufVariant).then((r) => {
      if (!cancelled) {
        setCandidates(r.candidates);
        setResolved(r.resolved);
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
        if (r.vocabTarget !== null && r.vocabDraft !== null) {
          // Deliberately "vocabulary matches", never "compatible": equal vocab
          // size is necessary, not sufficient, for a good speculative pair. A
          // vocab mismatch is a rejection reason (vocab_mismatch), so it never
          // reaches this branch -- it surfaces as `problem` via r.detail instead.
          setNote(`vocabulary matches (${r.vocabTarget})`);
        }
        if (r.sizeBytes !== null) {
          const gb = (r.sizeBytes / 1024 ** 3).toFixed(2);
          setNote((n) => (n ? `${n} · ${gb} GB` : `${gb} GB`));
        }
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

  // Re-validate an existing pin on mount (Task 8 extends this). A pinned
  // drafter that has since been deleted fails open at load time, so this is
  // the only place it becomes visible. Mount-only: re-running on every args
  // change would re-validate our own writes.
  // biome-ignore lint/correctness/useExhaustiveDependencies: mount-only re-validation of the pin baked into existingArgs at load time
  useEffect(() => {
    if (hidden || !modelId) {
      return;
    }
    const i = existingArgs.findIndex(
      (a) => a === "--model-draft" || a === "--spec-draft-hf",
    );
    if (i === -1 || i + 1 >= existingArgs.length) {
      return;
    }
    const kind = existingArgs[i] === "--spec-draft-hf" ? "hf" : "local";
    selectDraftModel(modelId, ggufVariant, existingArgs, {
      kind: kind as "local" | "hf",
      ref: existingArgs[i + 1],
    }).then((r) => {
      if (!r.ok) {
        setProblem(`pinned drafter unusable: ${r.detail || r.reason}`);
      }
    });
  }, [hidden, modelId, ggufVariant]);

  if (hidden) {
    return null;
  }

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
          {resolved ? NO_CANDIDATES_MESSAGE : UNRESOLVED_MODEL_MESSAGE}
        </p>
      )}

      <div className="flex items-center gap-2">
        <Input
          value={hfRepo}
          placeholder="Hugging Face repo, e.g. unsloth/Qwen3-0.6B-GGUF"
          aria-label="Hugging Face draft model repository"
          onChange={(e) => setHfRepo(e.target.value)}
          className="h-8 flex-1"
        />
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={!hfRepo.trim()}
          onClick={() => apply({ kind: "hf", ref: hfRepo.trim() })}
        >
          Use repo
        </Button>
      </div>

      <Button
        type="button"
        variant="ghost"
        size="sm"
        onClick={() => {
          setSelectedRef("");
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
    </div>
  );
}
