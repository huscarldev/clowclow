"""Shared pytest fixtures for clowclow tests.

This module provides common fixtures and utilities for testing clowclow.

Fixtures:
- test_workspace: Temporary directory for file operations
- test_image_data: Sample image binary data for multimodal testing
- test_image_url: Sample image URL for multimodal testing
- mock_claude_client: Mocked CustomClaudeCodeClient for unit tests
- test_model_agent: Agent with TestModel for baseline testing
"""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


@pytest.fixture
def test_workspace(tmp_path: Path) -> Path:
    """Provide a temporary workspace directory for tests.

    Returns:
        Path to a clean temporary directory for test file operations.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace


@pytest.fixture
def test_image_data() -> bytes:
    """Provide test image data (1x1 PNG).

    Returns:
        Minimal valid PNG image as bytes.
    """
    # Minimal 1x1 red PNG
    png_data = base64.b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=='
    )
    return png_data


@pytest.fixture
def test_image_url() -> str:
    """Provide a test image URL.

    Returns:
        Sample image URL for testing multimodal inputs.
    """
    return 'https://example.com/test-image.png'


@pytest.fixture
def mock_claude_client(test_workspace: Path):
    """Provide a mocked CustomClaudeCodeClient for unit testing.

    Returns:
        Mock client with simple_query and structured_query async methods.
    """
    from clowclow.claude_client import CustomClaudeCodeClient

    client = Mock(spec=CustomClaudeCodeClient)
    client.workspace_dir = test_workspace
    client.simple_query = AsyncMock(return_value="Mocked response")
    client.structured_query = AsyncMock()

    return client


@pytest.fixture
def test_model_agent() -> Agent:
    """Provide an Agent with TestModel for baseline testing.

    Returns:
        Agent configured with TestModel for fast, deterministic tests.
    """
    return Agent(TestModel())


@pytest.fixture(scope="session")
def claude_model() -> str:
    """Provide Claude model name for live tests.

    Returns:
        Model identifier for Claude (currently Claude 3.5 Haiku).
        Change this fixture to test with different models.

    Available models (Claude 4 family - latest):
        - "claude-sonnet-4-5-20250929"  # Claude Sonnet 4.5 (best coding, highest intelligence)
        - "claude-opus-4-1-20250805"    # Claude Opus 4.1 (complex reasoning, instruction adherence)
        - "claude-opus-4-20250514"      # Claude Opus 4 (world's best coding model)
        - "claude-sonnet-4-20250514"    # Claude Sonnet 4 (hybrid: instant + extended thinking)

    Available models (Claude 3.7 family):
        - "claude-3-7-sonnet-20250219"  # Claude Sonnet 3.7 (hybrid AI reasoning model)

    Available models (Claude 3.5 family):
        - "claude-3-5-haiku-20241022"   # Claude Haiku 3.5 (fast, cost-effective)

    Available models (Claude 3 family - legacy):
        - "claude-3-haiku-20240307"     # Claude Haiku 3 (fast, older version)

    Note: Using different models may affect test results and performance.
          Higher-tier models (Opus 4.1, Sonnet 4.5) are more expensive but more capable.
          For testing, Claude 3.5 Haiku offers good balance of speed and cost.
    """
    return "claude-3-5-haiku-20241022"


@pytest.fixture(autouse=True)
def setup_claude_model(request, claude_model):
    """Automatically inject claude_model into test class instances.

    This fixture runs automatically for all tests and sets self.claude_model
    for test class methods, so they can use it without adding as a parameter.
    """
    if request.instance:  # Only for class-based tests
        request.instance.claude_model = claude_model


@pytest.fixture
def sample_messages():
    """Provide sample message structures for testing.

    Returns:
        Dict of common message patterns used in tests.
    """
    from pydantic_ai.messages import ModelRequest, TextPart, UserPromptPart, BinaryContent

    return {
        'simple_text': [
            ModelRequest(
                parts=[TextPart(content="Hello")],
                kind="request"
            )
        ],
        'with_system': [
            ModelRequest(
                parts=[TextPart(content="Query")],
                kind="request",
                instructions="You are helpful."
            )
        ],
        'multimodal': [
            ModelRequest(
                parts=[
                    TextPart(content="Analyze this"),
                    BinaryContent(data=b"image", media_type="image/png")
                ],
                kind="request"
            )
        ]
    }


# Configure pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "live: mark test as requiring live Claude Code API access (deselect with '-m \"not live\"')"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )


# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)


# Environment safety check (optional - uncomment to enforce)
# import os
# def pytest_sessionstart(session):
#     """Enforce safe testing environment."""
#     # Optionally block live API calls in tests unless explicitly enabled
#     if not os.environ.get('ALLOW_LIVE_TESTS'):
#         os.environ['PYDANTIC_AI_ALLOW_MODEL_REQUESTS'] = 'false'
