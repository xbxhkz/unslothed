# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""OpenAI-style schemas for the vision tools, shaped like Studio's own.

Two things these descriptions must get right, because the model has no other
way to learn them:

1. The PATH CONTRACT. Every image path is confined to the conversation's
   working directory. The old wording ("Absolute path to an image file on this
   machine") pointed the model at the one case most likely to be REFUSED, and
   never mentioned the bare filename that reliably works.
2. FIRST-USE DOWNLOADS. Three of these tools may fetch a model the first time
   they run. Saying so here is what makes the fetch disclosed to the user
   rather than a surprise the assistant triggered on their behalf.
"""

_PATH_CONTRACT = (
    "The file must be inside this conversation's working directory -- that is "
    "the only place these tools can read images from. A bare filename (e.g. "
    "'photo.png') resolves there and is the normal way to refer to a file you "
    "or the user just created; a path pointing outside it is refused."
)

_IMAGE_PATH = {
    "type": "string",
    "description": f"Path to an image file. {_PATH_CONTRACT}",
}

REMOVE_BACKGROUND_TOOL = {
    "type": "function",
    "function": {
        "name": "remove_background",
        "description": (
            "Remove the background from an image, returning a transparent PNG. "
            "Returns the path of the written file, inside the conversation's "
            "working directory. The first use may download a ~176 MB model."
        ),
        "parameters": {
            "type": "object",
            "properties": {"image_path": _IMAGE_PATH},
            "required": ["image_path"],
        },
    },
}

DETECT_SHAPES_TOOL = {
    "type": "function",
    "function": {
        "name": "detect_shapes",
        "description": (
            "Detect and identify subjects (people, animals, objects) in a photo. "
            "Returns what was found with confidence and rough position, plus the "
            "path of an annotated copy. Finding nothing is a normal result. "
            "The first use may download a ~170 MB model."
        ),
        "parameters": {
            "type": "object",
            "properties": {"image_path": _IMAGE_PATH},
            "required": ["image_path"],
        },
    },
}

WEBCAM_LOOK_TOOL = {
    "type": "function",
    "function": {
        "name": "webcam_look",
        "description": (
            "Capture a frame from the local webcam and identify objects in it. "
            "Returns what was found plus the path of an annotated image. "
            "The first use may download a ~6 MB model."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "camera_index": {
                    "type": "integer",
                    "description": "Camera index, default 0.",
                }
            },
        },
    },
}

EDIT_IMAGE_PROMPT_TOOL = {
    "type": "function",
    "function": {
        "name": "edit_image_prompt",
        "description": (
            "Edit an image by describing the change in natural language "
            "(e.g. 'make the sky sunset-coloured'). Returns the path of the edited image."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": _IMAGE_PATH,
                "prompt": {
                    "type": "string",
                    "description": "Natural-language description of the edit.",
                },
                "strength": {
                    "type": "number",
                    "description": "How much to change the image, 0-1. Default 0.6.",
                },
            },
            "required": ["image_path", "prompt"],
        },
    },
}

FACE_SWAP_TOOL = {
    "type": "function",
    "function": {
        "name": "face_swap",
        "description": (
            "Swap the face from a source image into a target image. Requires the user "
            "to have accepted the InsightFace model licence first (non-commercial "
            "research use only) -- you cannot accept it on their behalf. Outputs carry "
            "metadata marking them as AI-edited. Returns the path of the result."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source_face_path": {
                    "type": "string",
                    "description": (
                        f"Path to the image containing the source face. {_PATH_CONTRACT}"
                    ),
                },
                "target_image_path": {
                    "type": "string",
                    "description": (
                        f"Path to the image to swap the face into. {_PATH_CONTRACT}"
                    ),
                },
            },
            "required": ["source_face_path", "target_image_path"],
        },
    },
}

ASSIST_VISION_TOOLS = [
    REMOVE_BACKGROUND_TOOL,
    DETECT_SHAPES_TOOL,
    WEBCAM_LOOK_TOOL,
    EDIT_IMAGE_PROMPT_TOOL,
    FACE_SWAP_TOOL,
]

ASSIST_VISION_TOOL_NAMES = frozenset(t["function"]["name"] for t in ASSIST_VISION_TOOLS)
