# clowclow Test Suite Architecture

## Overview
The clowclow package has a comprehensive test suite with 127 tests covering the ClaudeCodeModel implementation for Pydantic AI.

## Test Organization

### Directory Structure
```
tests/
├── unit/                   # Fast unit tests with mocks (42 tests)
│   └── test_claude_code_model.py
├── integration/            # Integration tests with TestModel and live API (85 tests)
│   ├── test_agent_basic.py           # Basic agent functionality
│   ├── test_agent_structured.py      # Structured output (Pydantic models)
│   ├── test_agent_streaming.py       # Streaming responses
│   ├── test_agent_multimodal.py      # Image handling (URL and binary)
│   └── test_agent_tools.py           # Tool calling
├── conftest.py            # Shared fixtures
├── pytest.ini             # Test configuration
└── README.md              # Testing documentation
```

## Running Tests

### Fast Development (< 0.3 seconds)
```bash
# Run only unit tests (no live API)
pytest tests/unit/ -v

# Run all non-live tests
pytest -m "not live" -v
```

### Full Test Suite
```bash
# Run everything including live API tests (~30-60 seconds)
pytest tests/ -v

# Run with coverage
pytest --cov=src/clowclow --cov-report=html
```

## Test Markers
- `@pytest.mark.live` - Requires live Claude Code API (55 tests)
- `@pytest.mark.slow` - Slow running tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.unit` - Unit tests

## Key Testing Patterns

### 1. Unit Tests with Mocks
All unit tests use mocks to avoid live API calls:

```python
@pytest.mark.asyncio
async def test_request_simple_text_query():
    model = ClaudeCodeModel()
    
    with patch.object(model._client, 'simple_query', new_callable=AsyncMock) as mock_query:
        mock_query.return_value = "Test response"
        result = await model.request(messages, None, ModelRequestParameters())
        
        assert result.parts[0].content == "Test response"
        assert isinstance(result.parts[0], TextPart)
```

### 2. TestModel Baselines
Every integration test has a TestModel baseline:

```python
def test_feature_with_testmodel():
    """Baseline: feature works with TestModel."""
    agent = Agent(TestModel())
    result = agent.run_sync("Test")
    assert result.output is not None

@pytest.mark.live
async def test_feature_with_claude_code():
    """Live: feature works with ClaudeCodeModel."""
    agent = Agent(ClaudeCodeModel())
    result = await agent.run("Test")
    # Verify actual content
    assert isinstance(result.output, str)
    assert len(result.output) > 0
```

### 3. Message Inspection
Use `capture_run_messages()` to verify message exchanges:

```python
@pytest.mark.live
@pytest.mark.asyncio
async def test_capture_messages():
    model = ClaudeCodeModel()
    agent = Agent(model)
    
    with capture_run_messages() as messages:
        result = await agent.run("Test")
    
    # Verify message flow
    assert messages[0].kind == "request"
    assert messages[-1].kind == "response"
```

## Shared Fixtures (conftest.py)

### Available Fixtures
- `test_workspace` - Temporary directory for file operations
- `test_image_data` - Sample 1x1 PNG image bytes
- `test_image_url` - Sample image URL string
- `mock_claude_client` - Mocked CustomClaudeCodeClient
- `test_model_agent` - Agent with TestModel for baselines
- `sample_messages` - Common message structures

### Usage Example
```python
def test_with_fixtures(test_workspace, mock_claude_client):
    model = ClaudeCodeModel(workspace_dir=test_workspace)
    # Use fixtures in test
```

## Critical Test Classes

### Unit Tests (tests/unit/test_claude_code_model.py)

1. **TestClaudeCodeModelInterface** - Model interface compliance
2. **TestModelMessageHandling** - Message extraction logic
3. **TestMessageExtractionEdgeCases** - Edge cases (empty, None, multimodal)
4. **TestSchemaTypeConversion** - JSON schema → Python type conversion
5. **TestRequestMethodWithMocks** - Mocked request/response testing
6. **TestStructuredOutputDynamicModels** - Dynamic Pydantic model creation

### Integration Tests (tests/integration/test_agent_basic.py)

1. **TestBasicAgentIntegration** - Agent creation and basic runs
2. **TestMessageInspection** - Message capture and verification
3. **TestAgentBehaviorConsistency** - ClaudeCodeModel vs TestModel consistency
4. **TestErrorHandling** - Error scenarios (empty queries, exceptions)
5. **TestResponseValidation** - Content verification (not just existence)

## Best Practices Implemented

### ✅ Assertion Quality
```python
# ❌ Weak - only checks existence
assert result.output is not None

# ✅ Strong - verifies type, content, correctness
assert isinstance(result.output, str)
assert len(result.output) > 0
assert "4" in result.output  # Verify actual answer
```

### ✅ Mocking Strategy
- **Unit tests**: 100% mocked, no live API calls
- **Integration tests**: Mix of TestModel (fast) and live API (marked with `@pytest.mark.live`)

### ✅ Test Speed
- Unit tests: < 0.3 seconds for all 42 tests
- Non-live tests: < 0.6 seconds for all 72 tests
- Live tests: ~30-60 seconds for 55 tests

## Configuration Files

### pytest.ini
- Test discovery patterns
- Marker definitions
- Asyncio configuration
- Coverage settings
- Output formatting

### Key Configuration
```ini
[pytest]
markers =
    live: mark test as requiring live Claude Code API
    slow: mark test as slow
    integration: integration test
    unit: unit test

asyncio_mode = auto
```

## Coverage Goals
- Unit tests: > 90% line coverage
- Integration tests: > 80% branch coverage
- Overall: > 85% coverage

## CI/CD Integration

### Recommended Pipeline
```yaml
# Fast PR checks (< 1 second)
- run: pytest tests/unit/ -v

# Standard PR checks with coverage
- run: pytest -m "not live" --cov=src/clowclow --cov-report=xml

# Full suite on main branch
- run: pytest --cov=src/clowclow --cov-report=html
```

## Common Issues & Solutions

### Tests fail with "No module named 'clowclow'"
```bash
pip install -e .  # Install in development mode
```

### Live tests fail with API errors
- Ensure Claude Code CLI is installed: `claude-code --version`
- Check active subscription
- Verify configuration

### Async tests fail
- Ensure pytest-asyncio installed: `pip install pytest-asyncio`
- Check pytest.ini has `asyncio_mode = auto`

## Documentation References
- [tests/README.md](tests/README.md) - Comprehensive testing guide
- [TEST_IMPROVEMENTS_SUMMARY.md](TEST_IMPROVEMENTS_SUMMARY.md) - Detailed improvements
- [Pydantic AI Testing Docs](https://ai.pydantic.dev/testing/) - Upstream best practices
