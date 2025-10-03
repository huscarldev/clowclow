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
