# Suggested Development Commands

## Package Management (using uv)

### Installation
```bash
# Install package in development mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Using uv (preferred)
uv pip install -e ".[dev]"
```

### Building
```bash
# Build the package
uv build

# This creates distribution files in dist/
```

## Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/integration/test_agent_structured.py

# Run specific test class or function
pytest tests/integration/test_agent_structured.py::TestStructuredOutputBasics
pytest tests/integration/test_agent_structured.py::TestStructuredOutputBasics::test_city_info_structured_output

# Run tests matching pattern
pytest -k "structured"
pytest -k "asyncio"

# Run async tests specifically
pytest -v -k "asyncio"
```

### Coverage
```bash
# Run tests with coverage
pytest --cov=clowclow

# Generate HTML coverage report
pytest --cov=clowclow --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
```

## Running the Application

### CLI Entry Point
```bash
# Run the clowclow CLI (after installation)
clowclow

# Or run directly via Python
python -m clowclow
```

## Environment Setup

### API Key
```bash
# Set Anthropic API key (required for integration tests)
export ANTHROPIC_API_KEY="your-api-key"
```

## Common Development Workflows

### After Making Changes
1. Run tests: `pytest`
2. Check coverage: `pytest --cov=clowclow`
3. Build package: `uv build` (if needed)

### Running Integration Tests
```bash
# Make sure ANTHROPIC_API_KEY is set
export ANTHROPIC_API_KEY="your-key"

# Run integration tests
pytest tests/integration/

# Run specific integration test suite
pytest tests/integration/test_agent_basic.py
pytest tests/integration/test_agent_structured.py
pytest tests/integration/test_agent_multimodal.py
pytest tests/integration/test_agent_streaming.py
pytest tests/integration/test_agent_tools.py
```

## Useful System Commands (macOS/Darwin)

### File Operations
```bash
ls -la              # List files with details
find . -name "*.py" # Find Python files
grep -r "pattern"   # Search in files
```

### Git Operations
```bash
git status          # Check status
git add .           # Stage changes
git commit -m "msg" # Commit changes
git push            # Push to remote
```

### Process Management
```bash
ps aux | grep python    # Find Python processes
kill -9 <pid>          # Kill process
```
