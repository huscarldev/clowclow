# clowclow Test Suite

This directory contains comprehensive tests for the clowclow package, which bridges Claude Code SDK with Pydantic AI.

## Test Organization

```
tests/
├── unit/                   # Fast unit tests with mocks (no live API)
│   └── test_claude_code_model.py
├── integration/            # Integration tests with TestModel and live API
│   ├── test_agent_basic.py
│   ├── test_agent_structured.py
│   ├── test_agent_streaming.py
│   ├── test_agent_multimodal.py
│   └── test_agent_tools.py
├── conftest.py            # Shared fixtures and configuration
└── README.md              # This file
```

## Running Tests

### Quick Start

```bash
# Run all unit tests (fast, no API calls)
uv run pytest tests/unit/ -v

# Run all non-live tests (recommended for development)
uv run pytest -m "not live" -v

# Run all tests including live API tests
uv run pytest tests/ -v

# Run only live API tests
uv run pytest -m live -v
```

### Common Test Commands

```bash
# Run tests with coverage
uv run pytest --cov=src/clowclow --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_claude_code_model.py -v

# Run specific test class
uv run pytest tests/unit/test_claude_code_model.py::TestSchemaTypeConversion -v

# Run specific test
uv run pytest tests/unit/test_claude_code_model.py::TestSchemaTypeConversion::test_get_type_from_schema_string -v

# Run tests matching a pattern
uv run pytest -k "test_schema" -v

# Run with verbose output and show print statements
uv run pytest -vv -s

# Stop at first failure
uv run pytest -x
```

## Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.live` - Requires live Claude Code API access (expensive, slow)
- `@pytest.mark.slow` - Takes longer to execute
- `@pytest.mark.integration` - Integration test
- `@pytest.mark.unit` - Unit test

Deselect tests by marker:
```bash
# Skip live API tests
pytest -m "not live"

# Skip slow tests
pytest -m "not slow"

# Run only unit tests
pytest -m unit
```

## Test Categories

### Unit Tests (`tests/unit/`)

**Purpose**: Test internal logic without external dependencies

**Characteristics**:
- Use mocks to avoid live API calls
- Fast execution (< 1 second total)
- Test edge cases, error handling, and type conversions
- Run on every code change

**Key test classes**:
- `TestClaudeCodeModelInterface` - Model interface compliance
- `TestMessageExtractionEdgeCases` - Message parsing edge cases
- `TestSchemaTypeConversion` - JSON schema to Python type conversion
- `TestRequestMethodWithMocks` - Mocked request testing
- `TestStructuredOutputDynamicModels` - Dynamic Pydantic model creation

### Integration Tests (`tests/integration/`)

**Purpose**: Test full integration with Pydantic AI Agent

**Characteristics**:
- Mix of TestModel (fast) and live API (slow) tests
- Test real-world usage patterns
- Verify behavior matches Pydantic AI expectations

**Test files**:
- `test_agent_basic.py` - Basic agent functionality, message inspection
- `test_agent_structured.py` - Structured output with Pydantic models
- `test_agent_streaming.py` - Streaming responses
- `test_agent_multimodal.py` - Image handling (URL and binary)
- `test_agent_tools.py` - Tool calling and function integration

## Best Practices

### Writing Tests

1. **Use TestModel for baselines**
   ```python
   def test_feature_with_testmodel():
       """Baseline: feature works with TestModel."""
       agent = Agent(TestModel())
       result = agent.run_sync("Test")
       assert result.output is not None
   ```

2. **Use mocks for unit tests**
   ```python
   @pytest.mark.asyncio
   async def test_with_mock():
       model = ClaudeCodeModel()
       with patch.object(model._client, 'simple_query', new_callable=AsyncMock) as mock:
           mock.return_value = "Mocked"
           result = await model.request(messages, None, ModelRequestParameters())
           assert result.parts[0].content == "Mocked"
   ```

3. **Use capture_run_messages for inspection**
   ```python
   @pytest.mark.live
   @pytest.mark.asyncio
   async def test_with_message_inspection():
       agent = Agent(ClaudeCodeModel())
       with capture_run_messages() as messages:
           await agent.run("Test")
       assert len(messages) >= 2
       assert messages[0].kind == "request"
   ```

4. **Verify actual content, not just existence**
   ```python
   # ❌ Weak assertion
   assert result.output is not None

   # ✅ Strong assertion
   assert isinstance(result.output, str)
   assert len(result.output) > 0
   assert "expected content" in result.output
   ```

5. **Test both success and failure paths**
   ```python
   @pytest.mark.asyncio
   async def test_error_handling():
       model = ClaudeCodeModel()
       with patch.object(model._client, 'simple_query', side_effect=Exception("API Error")):
           with pytest.raises(RuntimeError, match="Claude Code request failed"):
               await model.request(messages, None, ModelRequestParameters())
   ```

## Fixtures

Common fixtures available in all tests (defined in `conftest.py`):

- `test_workspace` - Temporary directory for file operations
- `test_image_data` - Sample 1x1 PNG image bytes
- `test_image_url` - Sample image URL string
- `mock_claude_client` - Mocked CustomClaudeCodeClient
- `test_model_agent` - Agent with TestModel for baselines
- `sample_messages` - Common message structures

## Continuous Integration

For CI/CD pipelines:

```bash
# Run fast tests only (for PR checks)
pytest -m "not live" --cov=src/clowclow --cov-report=xml

# Run full test suite (for main branch)
pytest --cov=src/clowclow --cov-report=html
```

## Troubleshooting

### Tests fail with "No module named 'clowclow'"
```bash
# Install package in development mode
pip install -e .
# Or use uv
uv pip install -e .
```

### Live tests fail with API errors
- Ensure Claude Code CLI is installed and configured
- Check that you have an active subscription
- Run `claude-code --version` to verify installation

### Async tests fail
- Ensure pytest-asyncio is installed: `uv pip install pytest-asyncio`
- Check that `pytest.ini` has `asyncio_mode = auto`

## Coverage

Generate coverage report:

```bash
# HTML report (recommended)
uv run pytest --cov=src/clowclow --cov-report=html
open htmlcov/index.html

# Terminal report
uv run pytest --cov=src/clowclow --cov-report=term-missing

# XML report (for CI)
uv run pytest --cov=src/clowclow --cov-report=xml
```

Coverage goals:
- Unit tests: > 90% line coverage
- Integration tests: > 80% branch coverage
- Overall: > 85% coverage

## Performance

Test suite performance goals:
- Unit tests: < 2 seconds total
- Non-live integration tests: < 5 seconds total
- Live integration tests: < 60 seconds total

Monitor performance:
```bash
# Show slowest tests
pytest --durations=10
```

## Contributing

When adding new features:

1. Write unit tests first (TDD)
2. Add integration tests with TestModel
3. Add live integration tests (marked with `@pytest.mark.live`)
4. Ensure all tests pass: `pytest -m "not live"`
5. Run full test suite: `pytest`
6. Check coverage: `pytest --cov`

## Resources

- [Pydantic AI Testing Docs](https://ai.pydantic.dev/testing/)
- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
