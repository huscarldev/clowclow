# Codebase Structure

## Directory Layout
```
clowclow/
├── src/clowclow/          # Main package code
│   ├── __init__.py        # Package exports & entry point
│   ├── claude_code_model.py  # ClaudeCodeModel implementation
│   └── claude_client.py   # CustomClaudeCodeClient wrapper
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── conftest.py       # Pytest configuration
├── pyproject.toml        # Project metadata & dependencies
├── uv.lock              # Dependency lock file
├── README.md            # Documentation
└── CLAUDE.md            # Claude Code instructions

```

## Core Components

### 1. ClaudeCodeModel (`src/clowclow/claude_code_model.py`)
- Implements `pydantic_ai.models.Model` interface
- Main entry point for Pydantic AI integration
- Translates between Pydantic AI and Claude Code SDK message formats
- Key methods:
  - `request()` - Handle non-streaming requests
  - `request_stream()` - Handle streaming requests
  - `_extract_user_message()` - Extract user prompt from messages
  - `_extract_multimodal_content()` - Handle images
  - `_get_type_from_schema()` - Convert JSON schema to Pydantic model

### 2. CustomClaudeCodeClient (`src/clowclow/claude_client.py`)
- Wraps `ClaudeSDKClient` for internal use
- Provides two query modes:
  - `simple_query()` - For text responses
  - `structured_query()` - For Pydantic model responses using `<schema>` tag method
- Handles multimodal inputs (base64 images → temp files)

### 3. Entry Point (`src/clowclow/__init__.py`)
- Exports `ClaudeCodeModel`
- Defines `main()` function as CLI entry point (`clowclow` command)

## Test Structure
- `tests/unit/` - Unit tests with mocking
- `tests/integration/` - Integration tests:
  - `test_agent_basic.py` - Basic text queries
  - `test_agent_structured.py` - Structured output with Pydantic models
  - `test_agent_streaming.py` - Streaming responses
  - `test_agent_multimodal.py` - Image/multimodal handling
  - `test_agent_tools.py` - Tool calling support
