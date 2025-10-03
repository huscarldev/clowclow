# clowclow

Make Claude Code available to Pydantic AI.

## Overview

**clowclow** is a Python package that bridges the [Claude Code SDK](https://github.com/anthropics/claude-code-sdk) with [Pydantic AI](https://ai.pydantic.dev/), allowing you to use Claude Code's powerful capabilities within Pydantic AI agents.

## Installation

```bash
pip install clowclow
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
- **Structured Output**: Support for Pydantic models as output types
- **Tool Calling**: Compatible with Pydantic AI's tool system
- **Streaming**: Supports streaming responses
- **Multimodal**: Handle images and other content types
- **Claude Code SDK**: Leverages Claude Code's extended capabilities

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

### API Key

Set your Anthropic API key via environment variable:

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

Or pass it explicitly:

```python
model = ClaudeCodeModel(api_key="your-api-key")
```

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
pytest --cov=clowclow --cov-report=html
```

## Requirements

- Python 3.13+
- claude-code-sdk >= 0.0.21
- pydantic-ai >= 1.0.5

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
