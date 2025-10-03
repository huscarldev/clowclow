# Pydantic AI Testing Best Practices (Applied to clowclow)

## Core Principles

### 1. Use TestModel for Fast Testing
TestModel is designed for unit testing without hitting actual LLMs:

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

# Fast, deterministic testing
agent = Agent(TestModel())
result = agent.run_sync("Test query")
assert result.output is not None
```

**Benefits:**
- No API calls → fast execution
- No costs
- Deterministic behavior
- Can configure tool calling behavior

### 2. Use Agent.override for Testing

Override model in tests without modifying production code:

```python
# Production code
agent = Agent('claude-sonnet-4')

# In tests
with agent.override(model=TestModel()):
    result = await agent.run("Test")
    # Uses TestModel instead of real API
```

### 3. Mock External Dependencies

For custom Model implementations, mock the underlying client:

```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_model_request():
    model = ClaudeCodeModel()
    
    with patch.object(model._client, 'simple_query', new_callable=AsyncMock) as mock:
        mock.return_value = "Mocked response"
        result = await model.request(messages, None, params)
        
        assert result.parts[0].content == "Mocked response"
        mock.assert_called_once()
```

### 4. Inspect Messages with capture_run_messages

Verify the agent-model exchange:

```python
from pydantic_ai import capture_run_messages

with capture_run_messages() as messages:
    result = await agent.run("Test query")

# Verify message flow
assert len(messages) >= 2
assert messages[0].kind == "request"
assert messages[-1].kind == "response"

# Check system prompt
request_msgs = [m for m in messages if m.kind == "request"]
assert any(m.instructions == "You are helpful." for m in request_msgs)
```

### 5. Strong Assertions

Always verify actual content, not just existence:

```python
# ❌ Weak assertion
assert result.output is not None

# ✅ Strong assertions
assert isinstance(result.output, str)
assert len(result.output) > 0
assert "expected content" in result.output

# For structured output
assert isinstance(result.output, MyModel)
assert result.output.field_name == "expected_value"
assert 0 <= result.output.score <= 100  # Verify constraints
```

### 6. Test Both Success and Failure Paths

```python
# Success path
@pytest.mark.asyncio
async def test_success():
    result = await agent.run("Valid query")
    assert result.output is not None

# Failure path
@pytest.mark.asyncio
async def test_error_handling():
    with patch.object(model._client, 'simple_query', side_effect=Exception("API Error")):
        with pytest.raises(RuntimeError, match="request failed"):
            await model.request(messages, None, params)
```

## Test Organization

### Separate Unit and Integration Tests

```
tests/
├── unit/           # Fast tests with mocks (no external dependencies)
├── integration/    # Tests with TestModel and live API
└── conftest.py    # Shared fixtures
```

### Use Markers for Slow Tests

```python
import pytest

@pytest.mark.live  # Requires live API
@pytest.mark.asyncio
async def test_with_real_api():
    agent = Agent(ClaudeCodeModel())
    result = await agent.run("Test")
    assert result.output is not None

# Run without live tests
# pytest -m "not live"
```

## TestModel Configuration

### Control Tool Calling

```python
# Call specific tools
agent = Agent(TestModel(call_tools=['tool_name']))

# Call all tools
agent = Agent(TestModel(call_tools='all'))

# Don't call any tools
agent = Agent(TestModel(call_tools=[]))
```

### Custom Output

```python
# Custom text output
agent = Agent(TestModel(custom_output_text="Fixed response"))

# Custom structured output
agent = Agent(
    TestModel(custom_output_args={'field': 'value'}),
    output_type=MyModel
)
```

## Structured Output Testing

### Test Schema Generation

```python
@pytest.mark.asyncio
async def test_structured_output():
    class Output(BaseModel):
        name: str
        age: int
    
    agent = Agent(TestModel(), output_type=Output)
    result = await agent.run("Get person info")
    
    # TestModel generates valid data matching schema
    assert isinstance(result.output, Output)
    assert isinstance(result.output.name, str)
    assert isinstance(result.output.age, int)
```

### Test Constraint Validation

```python
class ConstrainedOutput(BaseModel):
    score: int = Field(ge=0, le=100)
    rating: str = Field(pattern="^[A-F]$")

@pytest.mark.live
@pytest.mark.asyncio
async def test_constraints():
    agent = Agent(ClaudeCodeModel(), output_type=ConstrainedOutput)
    result = await agent.run("Rate this")
    
    # Verify constraints are enforced
    assert 0 <= result.output.score <= 100
    assert result.output.rating in ['A', 'B', 'C', 'D', 'E', 'F']
```

## Multimodal Testing

### Test Image Handling

```python
from pydantic_ai.messages import BinaryContent, ImageUrl

@pytest.mark.live
@pytest.mark.asyncio
async def test_image_input(test_image_data: bytes):
    agent = Agent(ClaudeCodeModel())
    
    result = await agent.run([
        "Describe this image:",
        BinaryContent(data=test_image_data, media_type="image/png")
    ])
    
    assert result.output is not None
    assert isinstance(result.output, str)
```

## Tool Testing

### Test Tool Registration

```python
def test_tool_registration():
    agent = Agent(TestModel())
    
    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Get weather for a city."""
        return f"Weather in {city}: Sunny"
    
    result = agent.run_sync("What's the weather in Paris?")
    assert result.output is not None
```

### Test Tool Retry

```python
def test_tool_retry():
    from pydantic_ai.exceptions import ModelRetry
    
    agent = Agent(TestModel())
    call_count = 0
    
    @agent.tool_plain
    def retry_tool() -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ModelRetry("First attempt failed")
        return "Success"
    
    result = agent.run_sync("Use the tool")
    assert call_count == 2  # Called twice
```

## Streaming Tests

### Test Streaming Response

```python
@pytest.mark.live
@pytest.mark.asyncio
async def test_streaming():
    agent = Agent(ClaudeCodeModel())
    
    async with agent.run_stream("Test") as result:
        chunks = []
        async for chunk in result.stream_text(debounce_by=None):
            chunks.append(chunk)
        
        # Verify streaming worked
        assert isinstance(chunks, list)
        
        # Get final output
        output = await result.get_output()
        assert output is not None
```

## Fixtures and Utilities

### Common Fixtures

```python
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

@pytest.fixture
def test_model_agent() -> Agent:
    """Baseline agent with TestModel."""
    return Agent(TestModel())

@pytest.fixture
def sample_messages():
    """Common message patterns."""
    from pydantic_ai.messages import ModelRequest, TextPart
    
    return {
        'simple': [ModelRequest(parts=[TextPart(content="Hello")], kind="request")],
        'with_system': [
            ModelRequest(
                parts=[TextPart(content="Query")],
                kind="request",
                instructions="You are helpful."
            )
        ]
    }
```

## CI/CD Best Practices

### Fast PR Checks

```bash
# Run only fast tests (no live API)
pytest -m "not live" --cov=src --cov-report=xml
```

### Full Test Suite

```bash
# Run everything on main branch
pytest --cov=src --cov-report=html
```

### Environment Safety

```python
# Optionally block live API calls in tests
import os
os.environ['PYDANTIC_AI_ALLOW_MODEL_REQUESTS'] = 'false'

# TestModel and FunctionModel are not affected by this setting
```

## Coverage Goals

- **Unit tests:** > 90% line coverage
- **Integration tests:** > 80% branch coverage
- **Edge cases:** Test empty inputs, None values, errors
- **Type coverage:** Test all JSON schema types, unions, nested models

## Common Patterns

### Baseline Comparison Pattern

```python
# Every live test should have a TestModel baseline
def test_feature_baseline():
    """Baseline: works with TestModel."""
    agent = Agent(TestModel())
    result = agent.run_sync("Test")
    assert result.output is not None

@pytest.mark.live
async def test_feature_live():
    """Live: works with real model."""
    agent = Agent(RealModel())
    result = await agent.run("Test")
    assert isinstance(result.output, str)
    assert len(result.output) > 0
```

### Error Testing Pattern

```python
@pytest.mark.asyncio
async def test_error_handling():
    model = CustomModel()
    
    with patch.object(model._client, 'query', side_effect=Exception("API Error")):
        with pytest.raises(RuntimeError, match="request failed"):
            await model.request(messages, None, params)
```

### Message Inspection Pattern

```python
@pytest.mark.live
@pytest.mark.asyncio
async def test_message_flow():
    agent = Agent(RealModel())
    
    with capture_run_messages() as messages:
        await agent.run("Test")
    
    assert messages[0].kind == "request"
    assert messages[-1].kind == "response"
```

## Resources

- [Pydantic AI Testing Docs](https://ai.pydantic.dev/testing/)
- [TestModel API Reference](https://ai.pydantic.dev/api/models/test/)
- [Pydantic AI Examples](https://ai.pydantic.dev/examples/)
