# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The curated capability vocabulary.

Providers are in PREFERENCE ORDER; the capability's best option is the first one
that is actually ready, and its `reason` says why it ranks there. The order is a
judgement, not a measurement -- the reasons exist so the model can see the
trade-off and pick another listed provider when the task calls for it.

Acquire hints are DESCRIPTIVE. Never a runnable command: master spec §4 requires
downloads to be identified, verified and presented to the user, and acquisition
is a later piece with its own security design. A test enforces this.

MCP tools are not here: their names only exist at runtime.
"""

from __future__ import annotations

from core.inference.capability_map import Capability, Provider, Requirement


def _tool(name: str, reason: str) -> Provider:
    return Provider("tool", name, None, (Requirement("tool", name),), reason)


def _software(name: str, via: str, reason: str, *requires: Requirement) -> Provider:
    return Provider("software", name, via, tuple(requires), reason)


def _vision(reason: str) -> Provider:
    return Provider("model", "vision model", None, (Requirement("model", "vision"),), reason)


_CODE_TOOLS = ("code_definition", "code_references", "code_hover", "code_symbols", "code_diagnostics")

CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        "internet_search",
        "Search the web and read pages",
        ("web search", "search the internet", "search online", "browse the web"),
        (_tool("web_search", "searches and fetches pages in one tool, with no API key"),),
        "Needs the ddgs Python package.",
    ),
    Capability(
        "run_python",
        "Run Python code",
        ("python", "execute python", "run code"),
        (_tool("python", "runs in the Studio environment, so its installed packages are available"),),
        None,
    ),
    Capability(
        "run_commands",
        "Run terminal commands",
        ("terminal", "shell", "command line", "run a command"),
        (_tool("terminal", "runs the platform shell with the installed programs on PATH"),),
        None,
    ),
    Capability(
        "filesystem",
        "Read, write and edit files",
        ("files", "file access", "access the filesystem", "edit files", "read files", "write files"),
        (
            _tool("edit_file", "exact, reviewable edits to a single file"),
            _tool("terminal", "listing, moving and reading many files"),
            _tool("python", "bulk or structured file work"),
        ),
        None,
    ),
    Capability(
        "pdf_analysis",
        "Extract text and tables from PDFs",
        ("pdf", "pdfs", "read pdf", "analyze pdfs", "analyse pdfs", "pdf text"),
        (
            _software(
                "pymupdf4llm", "python",
                "keeps headings, tables and reading order as Markdown",
                Requirement("module", "pymupdf4llm"),
            ),
            _software(
                "PyMuPDF", "python",
                "page-level text, images and metadata",
                Requirement("module", "fitz"),
            ),
        ),
        "Needs the PyMuPDF Python package.",
    ),
    Capability(
        "ocr",
        "Read text in images",
        ("text recognition", "read text from image", "read text in images", "optical character recognition"),
        (
            _vision("understands layout; nothing to install"),
            _software(
                "Tesseract", "terminal",
                "fast on clean scans and large batches",
                Requirement("binary", "tesseract"),
            ),
            _software(
                "EasyOCR", "python",
                "handles photographed text and many languages",
                Requirement("module", "easyocr"),
            ),
        ),
        "Load a vision-capable model, or install Tesseract OCR (a free program).",
    ),
    Capability(
        "image_understanding",
        "Describe or answer questions about an image",
        ("describe image", "describe an image", "what is in this image", "image question", "vision"),
        (
            _vision("answers open questions about what an image shows"),
            _tool("detect_shapes", "labels the objects it finds, and nothing more"),
        ),
        "Load a vision-capable model.",
    ),
    Capability(
        "object_detection",
        "Find and label objects in a photo",
        ("detect objects", "find objects", "label objects", "object recognition"),
        (
            _tool("detect_shapes", "boxes and a confidence for each object"),
            _vision("describes the objects, without boxes"),
        ),
        "Needs torch, torchvision and the Mask R-CNN weights.",
    ),
    Capability(
        "camera",
        "See through the webcam",
        ("webcam", "take a photo", "see through the camera"),
        (_tool("webcam_look", "captures a frame and identifies what is in view"),),
        "Needs a webcam, ultralytics, OpenCV and the YOLO weights.",
    ),
    Capability(
        "image_processing",
        "Resize, crop, convert or adjust images precisely",
        ("manipulate images", "image manipulation", "resize image", "crop image", "convert image"),
        (
            _software(
                "Pillow", "python",
                "simple, exact resizing, cropping and conversion",
                Requirement("module", "PIL"),
            ),
            _software(
                "OpenCV", "python",
                "advanced filters and geometry",
                Requirement("module", "cv2"),
            ),
        ),
        "Needs the Pillow Python package.",
    ),
    Capability(
        "image_editing",
        "Change an image by describing the edit",
        ("edit image", "edit an image", "change an image", "photo editing"),
        (_tool("edit_image_prompt", "applies a described change with the loaded image model"),),
        "Load an image model in Studio on the diffusers engine.",
    ),
    Capability(
        "background_removal",
        "Remove an image's background",
        ("remove background", "transparent background", "cut out subject"),
        (_tool("remove_background", "returns a transparent PNG of the subject"),),
        "Needs onnxruntime and the u2net model.",
    ),
    Capability(
        "face_swap",
        "Swap a face between two images",
        ("swap faces", "face replacement"),
        (_tool("face_swap", "swaps a face from one image into another"),),
        "Needs the InsightFace licence accepted and its models downloaded.",
    ),
    Capability(
        "image_generation",
        "Create an image from a text description",
        ("generate images", "generate an image", "create image", "text to image", "draw a picture"),
        (),
        "Studio can generate images on its Images page, but no agent tool exposes text-to-image yet.",
    ),
    Capability(
        "video_editing",
        "Cut, convert or combine video",
        ("edit videos", "edit video", "cut video", "trim video", "convert video", "combine video"),
        (
            _software(
                "FFmpeg", "terminal",
                "cuts, converts and combines almost any format",
                Requirement("binary", "ffmpeg"),
            ),
        ),
        "Needs FFmpeg, a free command-line program.",
    ),
    Capability(
        "speech_to_text",
        "Transcribe audio or video speech",
        ("transcribe", "transcription", "speech recognition", "audio to text"),
        (
            _software(
                "Whisper", "python",
                "accurate local transcription in many languages",
                Requirement("module", "whisper"),
                Requirement("binary", "ffmpeg"),
                Requirement("cached_model", "whisper"),
            ),
        ),
        (
            "Needs the Whisper Python package, FFmpeg (a free command-line program), and a "
            "Whisper model, which Whisper downloads on first use (the smallest is about 75 MB)."
        ),
    ),
    Capability(
        "word_documents",
        "Read and write Word documents",
        ("word", "docx", "word document"),
        (
            _software(
                "python-docx", "python",
                "reads and writes .docx paragraphs, tables and styles",
                Requirement("module", "docx"),
            ),
        ),
        "Needs the python-docx Python package.",
    ),
    Capability(
        "document_search",
        "Search the user's uploaded documents",
        ("search documents", "knowledge base", "uploaded documents"),
        (_tool("search_knowledge_base", "searches the documents the user uploaded"),),
        "Upload documents to a knowledge base.",
    ),
    Capability(
        "conversation_recall",
        "Recall earlier parts of this conversation",
        ("recall conversation", "earlier in this conversation", "search conversation"),
        (_tool("search_conversation", "finds turns trimmed from this conversation's context"),),
        "Needs the conversation archive enabled with RAG available.",
    ),
    Capability(
        "code_intelligence",
        "Definitions, references, types and errors in code",
        ("go to definition", "find references", "type errors", "code navigation"),
        (
            Provider(
                "tool",
                "code tools",
                None,
                tuple(Requirement("tool", name) for name in _CODE_TOOLS),
                "language-server answers without running a build",
            ),
        ),
        "Needs a language server for the project's language.",
    ),
    Capability(
        "visual_output",
        "Show the user an interactive HTML canvas or chart",
        ("render html", "chart", "canvas", "show a visualization"),
        (_tool("render_html", "shows an interactive canvas or chart to the user"),),
        None,
    ),
    Capability(
        "computer_control",
        "Control the mouse, keyboard and applications",
        ("control the computer", "mouse and keyboard", "desktop control", "automate applications"),
        (),
        "No provider yet; desktop control is planned for a later piece.",
    ),
)
