# clowclow

Make Claude Code available to Pydantic AI using the Claude Agent SDK.

## Overview

**clowclow** is a Python package that bridges the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk) with [Pydantic AI](https://ai.pydantic.dev/), allowing you to use Claude Code's powerful capabilities within Pydantic AI agents.

> ⚠️ **Early Release**: This package is in early development and was built with Claude Code (vibe coding with basic checks but no deep warranty). Many features may be broken or incomplete. Use at your own risk and please report issues!

## Installation

```bash
pip install clowclow
```

## Authentication

You can use clowclow in two ways:

### With Claude Code Subscription (No API Key Required)
If you have an active Claude Code subscription, the package will automatically connect to the Claude Code CLI without requiring an API key.

```python
from clowclow import ClaudeCodeModel

# No API key needed with subscription
model = ClaudeCodeModel()
```

### With Anthropic API Key
Alternatively, you can use your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

Or pass it explicitly:

```python
model = ClaudeCodeModel(api_key="your-api-key")
```

## Quick Start

```python
from pydantic_ai import Agent
from clowclow import ClaudeCodeModel

# Create a ClaudeCodeModel instance
model = ClaudeCodeModel()

# Use it with a Pydantic AI Agent
agent = Agent(model)

# Run queries
result = agent.run_sync("What is 2+2?")
print(result.output)
```

## Features

- **Full Pydantic AI Integration**: Works seamlessly with Pydantic AI's Agent interface
- **Structured Output**: Support for Pydantic models as output types using `<schema>` tag method
- **Function Tool Calling**: MCP-based integration for Pydantic AI function tools
- **Streaming**: Supports streaming responses
- **Multimodal**: Handle images and other content types
- **Claude Agent SDK**: Leverages the official Claude Agent SDK for extended capabilities

## Usage Examples

### Basic Text Query

```python
from pydantic_ai import Agent
from clowclow import ClaudeCodeModel

model = ClaudeCodeModel()
agent = Agent(model, system_prompt="You are a helpful assistant.")

result = agent.run_sync("Tell me a joke")
print(result.output)
```

### Structured Output

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from clowclow import ClaudeCodeModel

class CityInfo(BaseModel):
    city: str
    country: str
    population: int

model = ClaudeCodeModel()
agent = Agent(model, result_type=CityInfo)

result = agent.run_sync("What is the largest city in France?")
print(f"{result.output.city}, {result.output.country}")
```

### With Tools

```python
from pydantic_ai import Agent
from clowclow import ClaudeCodeModel

model = ClaudeCodeModel()
agent = Agent(model)

@agent.tool_plain
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 22°C"

result = agent.run_sync("What's the weather in Paris?")
print(result.output)
```

### Multimodal (Images)

```python
from pydantic_ai import Agent
from pydantic_ai.messages import ImageUrl
from clowclow import ClaudeCodeModel

model = ClaudeCodeModel()
agent = Agent(model)

result = agent.run_sync([
    "What's in this image?",
    ImageUrl(url="https://example.com/image.jpg")
])
print(result.output)
```

### Streaming

```python
import asyncio
from pydantic_ai import Agent
from clowclow import ClaudeCodeModel

async def stream_example():
    model = ClaudeCodeModel()
    agent = Agent(model)

    async with agent.run_stream("Write a short poem") as result:
        async for chunk in result.stream_text(debounce_by=None):
            print(chunk, end="", flush=True)

asyncio.run(stream_example())
```

## Configuration

### Workspace Directory

Specify a custom workspace directory for temporary files:

```python
from pathlib import Path

model = ClaudeCodeModel(workspace_dir=Path("/path/to/workspace"))
```

### Custom Model Name

```python
model = ClaudeCodeModel(model_name="custom-claude-code")
```

## Requirements

- Python 3.13+
- claude-agent-sdk >= 0.1.0
- pydantic-ai >= 1.0.5

## Development

### Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
```

### Run Tests with Coverage

```bash
uv run coverage run -m pytest && uv run coverage report
```

## License

MIT License

Copyright (c) 2025 clowclow contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Known Issues

As this is an early release, you may encounter:
- Incomplete feature implementations
- Edge cases that aren't handled properly
- Breaking changes in future versions
- Compatibility issues with certain Pydantic AI features

Please report any issues on the [GitHub issue tracker](https://github.com/penguindepot/clowclow/issues).
