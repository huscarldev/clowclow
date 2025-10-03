# Task Completion Checklist

When completing a coding task in this project, follow these steps:

## 1. Testing
```bash
# Always run tests after making changes
pytest

# For async-related changes
pytest -v -k "asyncio"

# For specific feature areas, run relevant integration tests:
pytest tests/integration/test_agent_basic.py       # Basic functionality
pytest tests/integration/test_agent_structured.py  # Structured output
pytest tests/integration/test_agent_multimodal.py # Multimodal/images
pytest tests/integration/test_agent_streaming.py  # Streaming
pytest tests/integration/test_agent_tools.py      # Tool calling
```

## 2. Code Quality Checks
```bash
# Check test coverage
pytest --cov=clowclow --cov-report=term-missing

# Ensure coverage meets standards (review uncovered lines)
```

## 3. Type Checking
- Ensure all new functions/methods have type hints
- Use `from __future__ import annotations` for forward references
- Use modern Python 3.13+ type syntax (`str | None` instead of `Optional[str]`)

## 4. Documentation
- Add docstrings to new public classes and methods
- Update README.md if adding new features
- Update CLAUDE.md if changing architecture

## 5. Integration Verification
```bash
# If changes affect core model:
pytest tests/integration/

# If changes affect client:
pytest tests/unit/test_claude_code_model.py
```

## 6. Environment Requirements
- Set `ANTHROPIC_API_KEY` for integration tests
- Ensure Python 3.13+ is being used
- Verify dependencies are up to date in pyproject.toml

## 7. Build Verification (if needed)
```bash
# Verify package builds successfully
uv build

# Check that entry point works
clowclow
```

## Common Scenarios

### After Adding New Feature
1. Write unit tests
2. Write integration tests
3. Run `pytest`
4. Check coverage: `pytest --cov=clowclow`
5. Update documentation

### After Bug Fix
1. Write regression test
2. Run `pytest` to ensure fix works
3. Run full test suite to ensure no breakage

### Before Committing
1. `pytest` - All tests pass
2. `pytest --cov=clowclow` - Coverage acceptable
3. Review changes for type hints
4. Review changes for docstrings
