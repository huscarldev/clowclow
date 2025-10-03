# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

**clowclow** is a Python package that makes Claude Code available to Pydantic AI. It provides a bridge between the `claude-agent-sdk` and `pydantic-ai`, allowing Pydantic AI agents to use Claude Code's capabilities.

### Key Components

1. **ClaudeCodeModel** ([src/clowclow/claude_code_model.py](src/clowclow/claude_code_model.py))
   - Implements the `pydantic_ai.models.Model` interface
   - Translates between Pydantic AI message formats and Claude Code SDK formats
   - Supports both simple text queries and structured output (via tool mode)
   - Handles multimodal content (text + images) by saving images to temp files

2. **CustomClaudeCodeClient** ([src/clowclow/claude_client.py](src/clowclow/claude_client.py))
   - Wraps `ClaudeSDKClient` for internal use
   - Provides `simple_query()` for text responses
   - Provides `structured_query()` for Pydantic model responses using `<schema>` tag method
   - Provides `tools_query()` for function tool calling via MCP integration
   - Handles multimodal inputs by converting base64 images to temp files

### Architecture Flow

1. Pydantic AI agent calls `ClaudeCodeModel.request()` or `request_stream()`
2. Model extracts messages, system prompts, and checks for images
3. For function tools (user-defined tools):
   - Checks for `function_tools` in model request parameters
   - Converts tools to MCP format and creates SDK MCP server
   - Uses PreToolUse hook to capture tool calls before execution
   - Calls `tools_query()` which returns tool call info
   - Converts tool calls to `ToolCallPart` for Pydantic AI to execute
   - Handles tool results from `ToolReturnPart` in subsequent turns
4. For structured output (output tool mode):
   - Extracts JSON schema from tool definition
   - Creates dynamic Pydantic model using `create_model()`
   - Calls `structured_query()` which embeds schema in `<schema>` tags
   - Converts result to `ToolCallPart` for Pydantic AI
5. For text output:
   - Calls `simple_query()`
   - Converts to `TextPart` for Pydantic AI
6. Client uses `ClaudeSDKClient` under the hood with configured options

## Development Commands

### Building & Installation
```bash
# Build the package (uses uv_build backend)
uv build

# Install in development mode
pip install -e .

# Run the CLI entry point
clowclow
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest src/clowclow/test_integration.py

# Run with verbose output
pytest -v

# Run async tests
pytest -v -k "asyncio"
```

### Environment Setup
- Python 3.13+ required
- No API key required (connects to Claude Code CLI)
- Dependencies: `claude-agent-sdk>=0.1.0`, `pydantic-ai>=1.0.5`

## Important Implementation Details

### Multimodal Content Handling
- Images are converted from base64 to temp files in workspace directory
- Claude Code is instructed to read the image file with absolute path
- Temp files are cleaned up after query completion
- Both `simple_query()` and `structured_query()` support multimodal input

### Structured Output Method
- Uses `<schema>` tag method: Embeds JSON schema in prompt within `<schema>` tags
- Works for both text-only and multimodal inputs (with images)
- Resolves `$ref` references: Inlines nested object schemas instead of using `$defs` for clearer structure
- Dynamic Pydantic models created from JSON schemas at runtime
- JSON is extracted from response text using pattern matching for robustness

### ClaudeAgentOptions Configuration
- `permission_mode="acceptEdits"`: Auto-accepts file edits
- `allowed_tools=["Read", "Write"]`: Limited tool access for safety
- `cwd`: Set to workspace directory for temp file operations
- `max_turns`: Defaults to 1 for simple queries, 5 for multimodal

### Message Extraction Logic
- Extracts most recent user message from message history
- Combines all system messages with newlines
- Handles `UserPromptPart`, `TextPart`, `BinaryContent`, `ImageUrl`, and `ToolReturnPart` types
- Detects multimodal content by checking for image types or list-based content

### Function Tool Calling (NEW)
- Supports Pydantic AI function tools via MCP (Model Context Protocol) integration
- Tools are registered as SDK MCP server tools with custom schemas
- Uses PreToolUse hooks to intercept tool calls before SDK executes them
- Returns tool call info to Pydantic AI for actual execution
- Handles multi-turn conversations with tool results
- Compatible with both sync and async tools
- Requires `permission_mode="bypassPermissions"` for automatic tool execution capture
