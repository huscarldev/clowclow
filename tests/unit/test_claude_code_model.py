"""Unit tests for ClaudeCodeModel - testing Model interface compliance.

These tests use mocks to avoid hitting the live Claude Code API.
They verify the Model interface implementation and integration with the client.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, Mock, patch

from pydantic import BaseModel
from pydantic_ai.messages import (
    ModelRequest,
    UserPromptPart,
    TextPart,
    BinaryContent,
    ToolCallPart,
)
from pydantic_ai.models import ModelRequestParameters

from clowclow import ClaudeCodeModel


class TestClaudeCodeModelInterface:
    """Test that ClaudeCodeModel implements the Model interface correctly."""

    def test_model_name_property(self):
        """Test that model_name property returns correct value."""
        model = ClaudeCodeModel()
        assert model.model_name == "claude-code"

    def test_model_name_custom(self):
        """Test that custom model name is returned."""
        model = ClaudeCodeModel(model_name="custom-claude")
        assert model.model_name == "custom-claude"

    def test_system_property(self):
        """Test that system property returns 'claude-code'."""
        model = ClaudeCodeModel()
        assert model.system == "claude-code"

    def test_initialization_no_api_key_required(self):
        """Test model initializes without API key (subscription-based)."""
        model = ClaudeCodeModel()
        assert model.model_name == "claude-code"
        assert model.system == "claude-code"

    def test_initialization_with_workspace_dir(self, test_workspace):
        """Test model initializes with custom workspace directory."""
        model = ClaudeCodeModel(workspace_dir=test_workspace)
        assert model._client.config.workspace_dir == test_workspace


class TestModelPropertyCompliance:
    """Test Model properties comply with Pydantic AI interface."""

    def test_model_has_required_properties(self):
        """Test that model has all required properties."""
        model = ClaudeCodeModel()

        # Required properties from Model interface
        assert hasattr(model, 'model_name')
        assert hasattr(model, 'system')
        assert callable(getattr(model, 'request', None))
        assert callable(getattr(model, 'request_stream', None))

    def test_model_name_is_string(self):
        """Test that model_name returns a string."""
        model = ClaudeCodeModel()
        assert isinstance(model.model_name, str)
        assert len(model.model_name) > 0

    def test_system_is_literal(self):
        """Test that system returns the expected literal value."""
        model = ClaudeCodeModel()
        assert model.system == "claude-code"

    def test_workspace_directory_created(self, test_workspace):
        """Test that workspace directory is created if it doesn't exist."""
        workspace = test_workspace / "new_workspace"
        model = ClaudeCodeModel(workspace_dir=workspace)
        assert workspace.exists()
        assert workspace.is_dir()


class TestRequestMethodWithMocks:
    """Test the request method using mocks to avoid live API calls."""

    @pytest.mark.asyncio
    async def test_request_simple_text_query(self):
        """Test simple text query request."""
        model = ClaudeCodeModel()

        with patch.object(model._client, 'simple_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = "Test response"

            messages = [
                ModelRequest(
                    parts=[TextPart(content="Hello")],
                    kind="request"
                )
            ]

            result = await model.request(messages, None, ModelRequestParameters())

            assert len(result.parts) == 1
            assert result.parts[0].content == "Test response"
            assert isinstance(result.parts[0], TextPart)
            mock_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_with_system_prompt(self):
        """Test request with system prompt."""
        model = ClaudeCodeModel()

        with patch.object(model._client, 'simple_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = "Response"

            messages = [
                ModelRequest(
                    parts=[TextPart(content="Query")],
                    kind="request",
                    instructions="You are a helpful assistant."
                )
            ]

            await model.request(messages, None, ModelRequestParameters())

            # Verify system prompt was passed
            call_args = mock_query.call_args
            assert call_args.kwargs['system_prompt'] == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_request_structured_output(self):
        """Test structured output request (tool mode)."""
        model = ClaudeCodeModel()

        # Create a mock output tool
        class TestOutput(BaseModel):
            name: str
            age: int

        mock_tool = Mock()
        mock_tool.name = "test_output"
        mock_tool.parameters_json_schema = {
            "type": "object",
            "title": "TestOutput",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }

        params = ModelRequestParameters(
            output_mode='tool',
            output_tools=[mock_tool]
        )

        with patch.object(model._client, 'structured_query', new_callable=AsyncMock) as mock_query:
            # Mock the structured response
            mock_response = TestOutput(name="Alice", age=30)
            mock_query.return_value = mock_response

            messages = [
                ModelRequest(
                    parts=[TextPart(content="Get user info")],
                    kind="request"
                )
            ]

            result = await model.request(messages, None, params)

            # Should return a ToolCallPart
            assert len(result.parts) == 1
            assert isinstance(result.parts[0], ToolCallPart)
            assert result.parts[0].tool_name == "test_output"
            assert result.parts[0].args["name"] == "Alice"
            assert result.parts[0].args["age"] == 30

    @pytest.mark.asyncio
    async def test_request_error_handling(self):
        """Test error handling in request method."""
        model = ClaudeCodeModel()

        with patch.object(model._client, 'simple_query', new_callable=AsyncMock) as mock_query:
            mock_query.side_effect = Exception("API Error")

            messages = [
                ModelRequest(
                    parts=[TextPart(content="Test")],
                    kind="request"
                )
            ]

            with pytest.raises(RuntimeError) as exc_info:
                await model.request(messages, None, ModelRequestParameters())

            assert "Claude Code request failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_request_multimodal_uses_multimodal_extraction(self):
        """Test that multimodal messages use multimodal extraction."""
        model = ClaudeCodeModel()

        with patch.object(model._client, 'simple_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = "Image analyzed"

            messages = [
                ModelRequest(
                    parts=[
                        TextPart(content="Analyze this"),
                        BinaryContent(data=b"imagedata", media_type="image/png")
                    ],
                    kind="request"
                )
            ]

            await model.request(messages, None, ModelRequestParameters())

            # Verify simple_query was called with multimodal content (as list)
            assert mock_query.called
            call_args, call_kwargs = mock_query.call_args.args, mock_query.call_args.kwargs
            # Check if message was passed as positional or keyword arg
            if call_args:
                message_arg = call_args[0]
            else:
                message_arg = call_kwargs['message']

            assert isinstance(message_arg, list)
            assert any(block.get("type") == "image" for block in message_arg)

    @pytest.mark.asyncio
    async def test_request_with_function_tools(self):
        """Test request with function tools."""
        model = ClaudeCodeModel()

        # Create a mock function tool
        mock_tool = Mock()
        mock_tool.name = "get_weather"
        mock_tool.description = "Get weather for a location"
        mock_tool.parameters_json_schema = {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }

        params = ModelRequestParameters(
            function_tools=[mock_tool]
        )

        with patch.object(model._client, 'tools_query', new_callable=AsyncMock) as mock_query:
            # Mock a tool call response with correct format
            mock_query.return_value = {
                "tool_calls": [
                    {
                        "tool_name": "get_weather",
                        "args": {"location": "Paris"},
                        "tool_call_id": "call_123"
                    }
                ],
                "text": ""
            }

            messages = [
                ModelRequest(
                    parts=[TextPart(content="What's the weather in Paris?")],
                    kind="request"
                )
            ]

            result = await model.request(messages, None, params)

            # Should return a ToolCallPart
            assert len(result.parts) == 1
            assert isinstance(result.parts[0], ToolCallPart)
            assert result.parts[0].tool_name == "get_weather"
            assert result.parts[0].args["location"] == "Paris"


class TestStreamingResponse:
    """Test streaming response functionality."""

    @pytest.mark.asyncio
    async def test_request_stream_returns_response(self):
        """Test that request_stream returns a streaming response."""
        model = ClaudeCodeModel()

        messages = [
            ModelRequest(
                parts=[TextPart(content="Test")],
                kind="request"
            )
        ]

        # request_stream is an async context manager
        async with model.request_stream(messages, None, ModelRequestParameters()) as stream:
            # Should return a ClaudeCodeStreamedResponse
            assert stream is not None
            assert hasattr(stream, '__aiter__')


class TestToolConversion:
    """Test tool conversion to client format."""

    def test_convert_tools_to_client_format(self):
        """Test converting Pydantic AI tools to client format."""
        model = ClaudeCodeModel()

        # Create mock tools with proper attributes
        tool1 = Mock()
        tool1.name = "tool1"
        tool1.description = "First tool"
        tool1.parameters_json_schema = {
            "type": "object",
            "properties": {"arg1": {"type": "string"}}
        }

        tool2 = Mock()
        tool2.name = "tool2"
        tool2.description = "Second tool"
        tool2.parameters_json_schema = {
            "type": "object",
            "properties": {"arg2": {"type": "integer"}}
        }

        mock_tools = [tool1, tool2]

        converted = model._convert_tools_to_client_format(mock_tools)

        assert len(converted) == 2
        assert converted[0]["name"] == "tool1"
        assert converted[0]["description"] == "First tool"
        assert converted[0]["parameters_json_schema"]["properties"]["arg1"]["type"] == "string"
        assert converted[1]["name"] == "tool2"
        assert converted[1]["description"] == "Second tool"


class TestIntegrationWithRequestHandler:
    """Test integration with RequestHandler."""

    @pytest.mark.asyncio
    async def test_uses_request_handler_for_extraction(self):
        """Test that model uses RequestHandler for message extraction."""
        model = ClaudeCodeModel()

        with patch.object(model._client, 'simple_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = "Response"

            messages = [
                ModelRequest(
                    parts=[
                        UserPromptPart(content="Part 1"),
                        UserPromptPart(content="Part 2")
                    ],
                    kind="request"
                )
            ]

            await model.request(messages, None, ModelRequestParameters())

            # Verify that the extracted message contains both parts
            call_args = mock_query.call_args
            message = call_args.kwargs['message']
            assert "Part 1" in message
            assert "Part 2" in message
