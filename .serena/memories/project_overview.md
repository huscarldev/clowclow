# Project Overview: clowclow

## Purpose
**clowclow** is a Python package that bridges the Claude Code SDK with Pydantic AI. It allows developers to use Claude Code's powerful capabilities (like file operations and code execution) within Pydantic AI agents.

## Tech Stack
- **Python**: 3.13+ (requires modern Python)
- **Core Dependencies**:
  - `claude-code-sdk>=0.0.21` - Claude Code SDK integration
  - `pydantic-ai>=1.0.5` - Pydantic AI framework
- **Build System**: `uv` with `uv_build` backend (>=0.8.5,<0.9.0)
- **Dev Dependencies**:
  - `pytest>=8.4.2` - Testing framework
  - `pytest-asyncio>=0.24.0` - Async test support
  - `pytest-mock>=3.14.0` - Mocking utilities
  - `inline-snapshot>=0.13.0` - Snapshot testing
  - `dirty-equals>=0.8.0` - Flexible equality assertions
  - `coverage>=7.0.0` - Code coverage

## Key Capabilities
- Full Pydantic AI Integration
- Structured Output (Pydantic models)
- Tool Calling support
- Streaming responses
- Multimodal (images + text)
- Claude Code SDK's extended capabilities
