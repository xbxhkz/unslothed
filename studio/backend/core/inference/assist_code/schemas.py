# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""OpenAI-style schemas for the five read-only code-intelligence tools."""

_PATH = (
    "Path to a source file. It must be inside the conversation's working "
    "directory; a bare filename resolves there. Supported: .ts .tsx .js .jsx .cs"
)
_WHERE = (
    "Give either 'symbol' (a name, easiest -- it is located for you) or an "
    "explicit 'line' (1-based) and optional 'column' (1-based)."
)
_FIRST_USE = (
    " On first use for a language this may install its language server "
    "(typescript-language-server via npm, csharp-ls via dotnet tool) and "
    "index the project, which can take a few seconds."
)


def _tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


_POSITION_PROPS = {
    "path": {"type": "string", "description": _PATH},
    "symbol": {"type": "string", "description": "Name of the symbol to locate. " + _WHERE},
    "line": {"type": "integer", "description": "1-based line number. Alternative to 'symbol'."},
    "column": {"type": "integer", "description": "1-based column number. Optional with 'line'."},
}

CODE_DIAGNOSTICS_TOOL = _tool(
    "code_diagnostics",
    "Type errors and warnings for a source file, from its language server, "
    "without running a build. Use after editing to check the change is sound. "
    "An empty result means the file is clean." + _FIRST_USE,
    {"path": {"type": "string", "description": _PATH}},
    ["path"],
)

CODE_DEFINITION_TOOL = _tool(
    "code_definition",
    "Where a symbol is defined. " + _WHERE + _FIRST_USE,
    _POSITION_PROPS, ["path"],
)

CODE_REFERENCES_TOOL = _tool(
    "code_references",
    "Every place a symbol is used across the project. Use before changing a "
    "shared function to see what depends on it. " + _WHERE + _FIRST_USE,
    _POSITION_PROPS, ["path"],
)

CODE_HOVER_TOOL = _tool(
    "code_hover",
    "Type, signature and documentation for a symbol. " + _WHERE + _FIRST_USE,
    _POSITION_PROPS, ["path"],
)

CODE_SYMBOLS_TOOL = _tool(
    "code_symbols",
    "Search the project for a symbol by name -- classes, functions, variables "
    "-- when you know a name but not which file holds it." + _FIRST_USE,
    {
        "query": {"type": "string", "description": "Symbol name or prefix to search for."},
        "path": {"type": "string", "description":
                 "Any file in the project to search, used to pick the workspace. " + _PATH},
    },
    ["query", "path"],
)

ASSIST_CODE_TOOLS = [
    CODE_DIAGNOSTICS_TOOL, CODE_DEFINITION_TOOL, CODE_REFERENCES_TOOL,
    CODE_HOVER_TOOL, CODE_SYMBOLS_TOOL,
]
ASSIST_CODE_TOOL_NAMES = {t["function"]["name"] for t in ASSIST_CODE_TOOLS}
