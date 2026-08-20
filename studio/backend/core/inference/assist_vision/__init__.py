# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Vision tools for the Studio agent loop.

Adds background removal, object/shape detection, webcam capture, prompt-based
image editing, and face swapping. Implementations live here rather than in
``tools.py`` so upstream merges stay cheap: ``tools.py`` carries exactly two
lines referring to this package (one entry in ``ALL_TOOLS``, one branch in
``execute_tool``).

Heavy model libraries (torch, onnxruntime, insightface, ultralytics, cv2) are
imported INSIDE functions, never at module scope -- a module-scope
``import torch`` stalls the event loop for seconds on first use.
"""
