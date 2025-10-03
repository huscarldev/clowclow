# Code Style and Conventions

## General Python Style
- **Python Version**: 3.13+ (uses modern Python features)
- **Type Hints**: Extensive use throughout codebase
  - Uses `from __future__ import annotations` for forward references
  - Union types use `|` syntax (e.g., `str | None`)
  - Generic types properly annotated (e.g., `List[str]`, `Dict[str, Any]`)

## Naming Conventions
- **Classes**: PascalCase (e.g., `ClaudeCodeModel`, `CustomClaudeCodeClient`)
- **Functions/Methods**: snake_case (e.g., `simple_query`, `_extract_user_message`)
- **Private Methods**: Prefixed with `_` (e.g., `_extract_system_messages`, `_has_images`)
- **Constants**: UPPER_SNAKE_CASE (standard Python convention)
- **Variables**: snake_case

## Documentation
- **Docstrings**: Used for classes and public methods
- **Module Docstrings**: Present at top of test files explaining purpose
- **Comments**: Inline comments used sparingly for complex logic
- **Type Hints**: Serve as inline documentation

## Code Organization
- **Imports**: Grouped and ordered:
  1. Standard library imports
  2. Third-party imports
  3. Local imports
- **Method Order in Classes**:
  1. `__init__` constructor
  2. Properties
  3. Public methods
  4. Private helper methods

## Testing Conventions
- **Test Files**: Named `test_*.py`
- **Test Classes**: Named `Test*` (e.g., `TestStructuredOutputBasics`)
- **Test Functions**: Named `test_*` (e.g., `test_city_info_structured_output`)
- **Fixtures**: Defined in `conftest.py`
- **Async Tests**: Use `@pytest.mark.asyncio` decorator
- **Test Structure**: Integration tests organized by feature area

## Pydantic Models
- **Field Descriptions**: Use `Field(description="...")` for documentation
- **Optional Fields**: Use `| None = None` pattern
- **Default Values**: Provided where appropriate (e.g., `latitude: float = 0.0`)

## Architecture Patterns
- **Facade Pattern**: `ClaudeCodeModel` acts as facade to Claude SDK
- **Adapter Pattern**: Translates between Pydantic AI and Claude SDK formats
- **Factory Pattern**: Dynamic Pydantic model creation from JSON schemas
