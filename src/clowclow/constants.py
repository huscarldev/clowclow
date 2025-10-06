"""Constants and configuration for clowclow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Tool name constants
MCP_TOOL_PREFIX = "mcp__pydantic-tools__"
MCP_SERVER_NAME = "pydantic-tools"
MCP_SERVER_VERSION = "1.0.0"

# Default system prompts
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_STRUCTURED_SYSTEM_SUFFIX = """
You must respond with valid JSON that exactly matches the provided schema.
Follow all constraints, required fields, and data types specified.
For nested objects (type: "object" with properties), create proper nested JSON objects with all their required fields.
Do not include any text outside of the JSON response."""

# Structured output instructions
STRUCTURED_OUTPUT_INSTRUCTIONS = """IMPORTANT: Respond with ONLY valid JSON that matches the schema above. No additional text or explanation.
- Follow the EXACT structure defined in the schema - include ONLY the fields listed in "properties"
- Do NOT add any extra fields not defined in the schema
- For nested objects (type: "object"), create a proper nested JSON object with ONLY the fields from that nested schema's properties
- Include ALL required fields (check the "required" array for each object and nested object)
- For optional fields with "default" values: you can either include them with appropriate values OR omit them (defaults will be used)
- Use the exact data types specified (string, number, object, array, etc.)
- RESPECT ALL CONSTRAINTS: "minimum", "maximum", "pattern", "minLength", "maxLength", etc.
- For fields with "pattern", ensure the value EXACTLY matches the regex pattern (e.g., "^[A-F]$" means a single letter A-F)
- For fields with "minimum"/"maximum", ensure the value is within the specified range"""

# Custom instructions for structured queries
STRUCTURED_QUERY_CUSTOM_INSTRUCTIONS = "Generate JSON that exactly matches the required schema. For list/array fields, use empty array [] instead of null if there are no items."
STRUCTURED_QUERY_WITH_TOOLS_INSTRUCTIONS = "Generate JSON that exactly matches the required schema based on the information provided from the tools. For list/array fields, use empty array [] instead of null if there are no items."
STRUCTURED_QUERY_FROM_RESPONSE_INSTRUCTIONS = "Generate JSON that exactly matches the required schema. For list/array fields, use empty array [] instead of null if there are no items. Extract the information from the previous response provided."

# Permission modes
PERMISSION_MODE_BYPASS = "bypassPermissions"
PERMISSION_MODE_ACCEPT_EDITS = "acceptEdits"

# Default turn limits
DEFAULT_MAX_TURNS_SIMPLE = 1
DEFAULT_MAX_TURNS_STRUCTURED = 1
DEFAULT_MAX_TURNS_TOOLS = 5

# Tool call ID
DEFAULT_TOOL_CALL_ID = "tool_call_1"


@dataclass
class ClaudeCodeConfig:
    """Configuration for Claude Code client."""

    workspace_dir: Path | None = None
    model: str | None = None
    api_key: str | None = None

    # Default turn limits
    max_turns_simple: int = DEFAULT_MAX_TURNS_SIMPLE
    max_turns_structured: int = DEFAULT_MAX_TURNS_STRUCTURED
    max_turns_tools: int = DEFAULT_MAX_TURNS_TOOLS

    # System prompts
    default_system_prompt: str = DEFAULT_SYSTEM_PROMPT

    # Permission settings
    # Note: bypassPermissions can enable default tools, use acceptEdits for simple queries
    permission_mode_simple: str = PERMISSION_MODE_ACCEPT_EDITS
    permission_mode_structured: str = PERMISSION_MODE_ACCEPT_EDITS
    permission_mode_tools: str = PERMISSION_MODE_BYPASS

    def __post_init__(self):
        """Initialize workspace directory."""
        if self.workspace_dir is None:
            import tempfile
            self.workspace_dir = Path(tempfile.gettempdir())
        self.workspace_dir.mkdir(exist_ok=True)
