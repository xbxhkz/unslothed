# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The one supported way to accept InsightFace's model licence.

Run from ``studio/backend``::

    python -m core.inference.assist_vision.accept_licence

It prints InsightFace's terms for the models ``face_swap`` uses, waits for a
person to type the exact confirmation phrase, and only then writes the marker
file that ``face_swap.licence_accepted()`` looks for. There is deliberately no
``--yes`` / ``--accept`` flag: the whole point is that a human reads the terms
and answers, so a one-line non-interactive form would defeat it.

Before this existed, ``record_licence_acceptance()`` had no route, no CLI, no
UI and no documentation -- nothing called it anywhere in the backend or the
frontend. No user could satisfy the gate through any supported path, so
``face_swap`` shipped permanently unusable while advertising itself to the
model on every turn: the model would call it, be refused, and be unable to
help. The refusal message was vague because the thing it referred to did not
exist.

HONEST LIMITATION, stated plainly rather than overclaimed: this gate is
ADVISORY and always was. It is not, and cannot be, technically enforceable. In
an app whose agent already has ``terminal``, ``python`` and ``edit_file``, any
sufficiently capable agent in the same process can write this marker file
directly. What this module provides is a clear, documented, human-facing way to
make the decision deliberately -- not a security boundary. Nothing here should
be described as preventing an agent from accepting on a user's behalf; the
guarantee the code does make is narrower and real: no vision tool, no schema
parameter, and no tool argument value can trigger acceptance, so the assistant
cannot be talked into it through the conversation itself.

The interactive-terminal requirement below is friction of the same advisory
kind. It stops an accidental or piped acceptance (``echo ... | python -m ...``)
from counting as a person reading the terms. It is not a claim that acceptance
is impossible to automate.
"""
import sys

from .face_swap import (
    _MODEL_PACK_NAME,
    _SWAP_MODEL_FILENAME,
    licence_accepted,
    record_licence_acceptance,
)
from .models import model_root

_CONFIRMATION = "I ACCEPT"

_TERMS = f"""
InsightFace model licence -- non-commercial, research use only
==============================================================

The face_swap tool uses two pretrained InsightFace models:

  * {_MODEL_PACK_NAME}            (face detection and alignment)
  * {_SWAP_MODEL_FILENAME}   (the InSwapper face-swapping model)

InsightFace distributes these models for NON-COMMERCIAL, RESEARCH PURPOSES
ONLY. Using them in a commercial product or service is NOT permitted by that
licence. This is a restriction on the MODEL WEIGHTS specifically; it is
separate from, and stricter than, the licence covering Unsloth Studio itself.

By accepting you confirm that:

  1. You have reviewed InsightFace's published licence terms for these models
     yourself, from InsightFace's own project, and you accept them.
  2. Your use of face_swap will stay within non-commercial, research use.
  3. You accept responsibility for how face-swapped images produced here are
     used. Outputs carry metadata marking them as AI-edited, but metadata is
     easily stripped and is not a safeguard you should rely on. Creating or
     sharing images that misrepresent a real person can be harmful and, in many
     places, unlawful.

Accepting writes a marker file to:

  {{marker_dir}}

Nothing is downloaded now. The models are fetched by InsightFace the first time
face_swap actually runs.

To decline, press Enter or Ctrl-C. Nothing is written unless you type the
confirmation phrase exactly.
"""


def _print_terms(stream):
    stream.write(_TERMS.format(marker_dir = model_root()))
    stream.write("\n")


def main(argv = None, *, stdin = None, stdout = None) -> int:
    """Print the terms, ask, and record acceptance only on an exact match."""
    argv = sys.argv[1:] if argv is None else list(argv)
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout

    if argv:
        stdout.write(
            "This command takes no arguments. It is interactive by design: "
            "there is no flag that accepts the licence without a person "
            "reading it.\n"
        )
        return 2

    if licence_accepted():
        stdout.write(
            "InsightFace's model licence has already been accepted on this "
            f"installation.\nMarker directory: {model_root()}\n"
        )
        return 0

    _print_terms(stdout)

    # An interactive terminal, so a piped or redirected answer cannot stand in
    # for a person reading the terms. Advisory friction, not enforcement.
    isatty = getattr(stdin, "isatty", None)
    if not (callable(isatty) and isatty()):
        stdout.write(
            "Refusing to read the answer from a non-interactive input. Run this "
            "command directly in a terminal.\n"
        )
        return 1

    stdout.write(
        f"To accept, type exactly: {_CONFIRMATION}\n"
        "Anything else declines.\n> "
    )
    stdout.flush()

    try:
        answer = stdin.readline()
    except (KeyboardInterrupt, EOFError):
        stdout.write("\nDeclined. Nothing was written.\n")
        return 1

    if answer.strip() != _CONFIRMATION:
        stdout.write("Declined. Nothing was written.\n")
        return 1

    record_licence_acceptance()
    stdout.write(
        "\nAccepted. face_swap is now available on this installation.\n"
        f"Marker written under: {model_root()}\n"
        "To revoke, delete that marker file.\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
