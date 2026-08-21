# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shape detection via a locally-run torchvision Mask R-CNN pipeline
(labeled, per-instance segmentation -- masks, not just boxes). Chosen over
this package's own ``yolo.py`` (ultralytics) pipeline specifically because
ultralytics is AGPL-3.0-licensed with a field-of-use-restricted weights
distribution, while torchvision is BSD-3-Clause with no such restriction on
its pretrained weights -- so this needs no license-acceptance gate, just a
plain download-if-missing step. Masks are produced (and returned in each
detection) even though this module's own callers currently only report
labels/boxes/positions in text -- a future shape-swap capability needs real
per-instance masks for a clean cutout, and building the right pipeline once
now avoids redoing this work later.

API verified directly against the installed torchvision package
(site-packages, not web search / training-data memory) before writing this
module:
  - maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
    weights_backbone=None) -- passing weights= makes the function skip the
    separate ImageNet backbone download entirely (confirmed via source:
    `if weights is not None: weights_backbone = None`), so only ONE download
    happens (the full COCO-pretrained state dict).
  - Inference (eval mode, no_grad) on a list of one [C,H,W] float tensor in
    0-1 range (via torchvision.transforms.functional.to_tensor) returns a
    list of one dict: {"boxes": FloatTensor[N,4], "labels": Int64Tensor[N],
    "scores": FloatTensor[N], "masks": FloatTensor[N,1,H,W] in 0-1 range,
    threshold at 0.5 per torchvision's own docs}. Masks come back at the
    SAME (H, W) as the input tensor -- no resizing needed.
  - MaskRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"] is a 91-entry
    COCO category list (index 0 is "__background__"; a few interior indices
    are the literal string "N/A", COCO's original non-contiguous numbering)
    -- readable without triggering any download, since it's plain Enum
    metadata, not part of the state dict fetch.
  - The weights download (torch.hub.load_state_dict_from_url under the
    hood) is cached under torch.hub.get_dir() -- explicitly redirected to
    the model cache dir via torch.hub.set_dir() before model construction,
    mirroring this package's own model-caching convention rather than
    leaving it to scatter into the user's global ~/.cache/torch.
"""
import io
import logging
import os

import numpy as np
from PIL import Image

from .models import format_detections as _format_detections, model_root

logger = logging.getLogger(__name__)

_model = None
_categories = None


def _get_model():
    global _model, _categories
    if _model is None:
        import torch
        os.makedirs(model_root(), exist_ok=True)
        # Disclosed before the fetch: torchvision pulls ~170 MB of COCO-
        # pretrained weights the first time this runs, triggered by a chat
        # message, and a user watching the log should know why.
        logger.info(
            "assist_vision: torchvision will download Mask R-CNN weights "
            "(~170 MB) on first use of detect_shapes, cached in %s",
            model_root(),
        )
        torch.hub.set_dir(model_root())
        from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        _categories = list(weights.meta["categories"])
        _model = maskrcnn_resnet50_fpn(weights=weights, weights_backbone=None)
        _model.eval()
    return _model


def _get_categories():
    if _categories is None:
        _get_model()  # populates _categories as a side effect
    return _categories


def _to_numpy(x):
    """Accepts either a real torch.Tensor (real model output) or a plain
    numpy array / list (test doubles) -- lets injected fakes skip
    constructing real torch tensors entirely."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def detect(image_bytes, *, model=None, categories=None, conf=0.4):
    """Detect labeled subjects (people, animals, other objects) in
    image_bytes -> list[Detection]. Accepts any PIL-readable format (PNG/
    JPEG/WEBP/...), matching this app's other image tools -- no format
    restriction. Never raises for "nothing detected" (returns []); model
    construction/inference errors propagate to the caller, which applies
    the never-raises discipline at its own boundary (matching every other
    image tool's convention).

    torch / torchvision.transforms.functional.to_tensor are imported here,
    lazily, rather than at module scope: importing torch costs ~2.3s, and
    this function is the ONLY code path (via the agent tool's own
    asyncio.to_thread(detector, image_bytes) call) where that cost is
    supposed to land -- off the ASGI event loop, inside the thread pool.
    A module-scope `import torch` would instead pay that cost synchronously
    on the event loop the first time this module is imported (e.g. by the
    agent tool's own `from core.inference.assist_vision.shape_detect import
    detect as detector` line, which runs BEFORE the to_thread call),
    stalling every concurrent request (including SSE chat streams) for
    ~2.5s on first use. Mirrors this package's ``bg_removal.py``'s own
    lazy-import-heavy-deps-inside-the-getter convention for the same
    reason."""
    import torch
    from torchvision.transforms.functional import to_tensor

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    tensor = to_tensor(img)

    m = model if model is not None else _get_model()
    cats = categories if categories is not None else _get_categories()

    with torch.no_grad():
        raw_out = m([tensor])[0]

    boxes = _to_numpy(raw_out["boxes"])
    labels = _to_numpy(raw_out["labels"])
    scores = _to_numpy(raw_out["scores"])
    masks = _to_numpy(raw_out["masks"])

    raw = []
    for i in range(len(boxes)):
        label_idx = int(labels[i])
        if label_idx <= 0 or label_idx >= len(cats):
            continue
        label = cats[label_idx]
        if not label or label == "N/A":
            continue
        x1, y1, x2, y2 = [float(v) for v in boxes[i]]
        mask = masks[i, 0] >= 0.5
        raw.append((label, float(scores[i]), x1, y1, x2, y2, mask))

    return _format_detections(raw, w, h, conf=conf)
