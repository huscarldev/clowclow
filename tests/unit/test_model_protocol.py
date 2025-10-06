"""Unit tests for ClaudeCodeModel protocol compliance.

Tests verify that ClaudeCodeModel correctly implements the Pydantic AI Model protocol.
"""

from __future__ import annotations

import pytest
from inspect import signature, iscoroutinefunction

from pydantic_ai.models import Model, KnownModelName
from pydantic_ai.messages import ModelRequest, ModelResponse

from clowclow import ClaudeCodeModel


class TestModelProtocolCompliance:
    """Test that ClaudeCodeModel implements the Model protocol correctly."""

    def test_model_is_instance_of_protocol(self):
        """Test that ClaudeCodeModel can be used as a Model."""
        model = ClaudeCodeModel()

        # Should have all required protocol attributes
        assert hasattr(model, 'model_name')
        assert hasattr(model, 'request')
        assert hasattr(model, 'request_stream')

    def test_model_name_attribute(self):
        """Test model_name attribute."""
        model = ClaudeCodeModel()

        assert hasattr(model, 'model_name')
        assert isinstance(model.model_name, (str, KnownModelName))
        assert model.model_name == "claude-code"

    def test_model_name_can_be_customized(self):
        """Test that model_name can be customized."""
        custom_name = "custom-claude-model"
        model = ClaudeCodeModel(model_name=custom_name)

        assert model.model_name == custom_name

    def test_system_attribute(self):
        """Test system attribute for model identification."""
        model = ClaudeCodeModel()

        # Model should have system attribute
        assert hasattr(model, 'system')
        assert model.system == "claude-code"

    def test_request_method_exists(self):
        """Test that request method exists with correct signature."""
        model = ClaudeCodeModel()

        assert hasattr(model, 'request')
        assert callable(model.request)
        assert iscoroutinefunction(model.request)

        # Check signature
        sig = signature(model.request)
        params = list(sig.parameters.keys())
        assert 'messages' in params
        assert 'model_settings' in params

    def test_request_stream_method_exists(self):
        """Test that request_stream method exists with correct signature."""
        model = ClaudeCodeModel()

        assert hasattr(model, 'request_stream')
        assert callable(model.request_stream)
        # request_stream should return an async context manager

        # Check signature
        sig = signature(model.request_stream)
        params = list(sig.parameters.keys())
        assert 'messages' in params
        assert 'model_settings' in params


class TestModelRequestResponse:
    """Test request/response structure compliance."""

    @pytest.mark.asyncio
    async def test_request_accepts_messages(self, sample_messages):
        """Test that request method accepts ModelRequest messages."""
        from unittest.mock import AsyncMock, patch

        model = ClaudeCodeModel()

        # Mock the client to avoid actual API calls
        with patch.object(model._client, 'simple_query', new_callable=AsyncMock) as mock:
            mock.return_value = "Test response"

            # Should accept list of ModelRequest
            response = await model.request(
                messages=sample_messages['simple_text'],
                model_settings=None,
                model_request_parameters=None
            )

            # Should return ModelResponse
            assert response is not None

    @pytest.mark.asyncio
    async def test_request_returns_model_response(self):
        """Test that request returns a ModelResponse."""
        from pydantic_ai.messages import TextPart
        from unittest.mock import AsyncMock, patch

        model = ClaudeCodeModel()

        with patch.object(model._client, 'simple_query', new_callable=AsyncMock) as mock:
            mock.return_value = "Response text"

            response = await model.request(
                messages=[],
                model_settings=None,
                model_request_parameters=None
            )

            # Response should have required attributes
            assert hasattr(response, 'parts')
            assert isinstance(response.parts, list)
            assert len(response.parts) > 0

            # First part should be TextPart
            assert isinstance(response.parts[0], TextPart)

    # NOTE: Removed test_request_stream_returns_stream_response
    # This test was testing implementation details (mocking internal _sdk_client attribute)
    # Streaming behavior is tested in integration tests (test_agent_streaming.py)


class TestModelSettings:
    """Test model settings handling."""

    @pytest.mark.asyncio
    async def test_model_accepts_none_settings(self):
        """Test that model accepts None for settings."""
        from unittest.mock import AsyncMock, patch

        model = ClaudeCodeModel()

        with patch.object(model._client, 'simple_query', new_callable=AsyncMock) as mock:
            mock.return_value = "Response"

            # Should accept None settings
            response = await model.request(
                messages=[],
                model_settings=None,
                model_request_parameters=None
            )

            assert response is not None

    @pytest.mark.asyncio
    async def test_model_handles_empty_messages(self):
        """Test that model handles empty message list."""
        from unittest.mock import AsyncMock, patch

        model = ClaudeCodeModel()

        with patch.object(model._client, 'simple_query', new_callable=AsyncMock) as mock:
            mock.return_value = "Empty response"

            # Should handle empty messages
            response = await model.request(
                messages=[],
                model_settings=None,
                model_request_parameters=None
            )

            assert response is not None


class TestModelInitialization:
    """Test model initialization parameters."""

    def test_default_initialization(self):
        """Test model with default parameters."""
        model = ClaudeCodeModel()

        assert model is not None
        assert model.model_name == "claude-code"

    def test_initialization_with_custom_workspace(self, tmp_path):
        """Test model with custom workspace directory."""
        workspace = tmp_path / "custom_workspace"
        workspace.mkdir()

        model = ClaudeCodeModel(workspace_dir=workspace)

        assert model is not None
        # Verify workspace is used
        assert model._client.config.workspace_dir == workspace

    def test_initialization_with_custom_model_name(self):
        """Test model with custom model name."""
        custom_name = "my-custom-model"
        model = ClaudeCodeModel(model_name=custom_name)

        assert model.model_name == custom_name

    def test_multiple_model_instances(self):
        """Test creating multiple model instances."""
        model1 = ClaudeCodeModel(model_name="model-1")
        model2 = ClaudeCodeModel(model_name="model-2")

        assert model1.model_name == "model-1"
        assert model2.model_name == "model-2"
        # Should be separate instances
        assert model1 is not model2


class TestModelErrorHandling:
    """Test model error handling compliance."""

    @pytest.mark.asyncio
    async def test_request_raises_runtime_error_on_failure(self):
        """Test that request raises RuntimeError on client failure."""
        from unittest.mock import AsyncMock, patch

        model = ClaudeCodeModel()

        with patch.object(model._client, 'simple_query', new_callable=AsyncMock) as mock:
            mock.side_effect = Exception("Client error")

            # Should raise RuntimeError wrapping the original error
            with pytest.raises(RuntimeError) as exc_info:
                await model.request(
                    messages=[],
                    model_settings=None,
                    model_request_parameters=None
                )

            assert "Claude Code request failed" in str(exc_info.value)
            assert exc_info.value.__cause__ is not None


class TestModelTypeAnnotations:
    """Test that model has correct type annotations."""

    def test_model_has_type_annotations(self):
        """Test that ClaudeCodeModel has proper type annotations."""
        from clowclow.claude_code_model import ClaudeCodeModel as ModelClass

        # Check that methods have annotations
        assert hasattr(ModelClass.request, '__annotations__')
        assert hasattr(ModelClass.request_stream, '__annotations__')

    def test_model_name_type(self):
        """Test that model_name has correct type."""
        model = ClaudeCodeModel()

        # model_name should be a string
        assert isinstance(model.model_name, str)
