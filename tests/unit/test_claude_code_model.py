"""Unit tests for ClaudeCodeModel - testing Model interface compliance.

These tests use mocks to avoid hitting the live Claude Code API.
They verify the Model interface implementation and internal logic.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

from pydantic import BaseModel
from pydantic_ai.messages import (
    ModelRequest,
    UserPromptPart,
    TextPart,
    BinaryContent,
    ImageUrl,
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
        assert model._client.workspace_dir == test_workspace


class TestModelMessageHandling:
    """Test message extraction and handling."""

    def test_extract_user_message_from_text_part(self):
        """Test extracting user message from TextPart."""
        model = ClaudeCodeModel()

        messages = [
            ModelRequest(
                parts=[TextPart(content="Hello world")],
                kind="request"
            )
        ]

        user_msg = model._extract_user_message(messages)
        assert user_msg == "Hello world"

    def test_extract_user_message_multiple_parts(self):
        """Test extracting user message with multiple parts."""
        model = ClaudeCodeModel()

        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(content="Part 1"),
                    UserPromptPart(content="Part 2")
                ],
                kind="request"
            )
        ]

        user_msg = model._extract_user_message(messages)
        assert "Part 1" in user_msg
        assert "Part 2" in user_msg

    def test_extract_system_messages(self):
        """Test extracting system messages from instructions."""
        model = ClaudeCodeModel()

        messages = [
            ModelRequest(
                parts=[UserPromptPart(content="User message")],
                kind="request",
                instructions="System instruction 1"
            ),
            ModelRequest(
                parts=[UserPromptPart(content="Another message")],
                kind="request",
                instructions="System instruction 2"
            )
        ]

        system_msg = model._extract_system_messages(messages)
        assert "System instruction 1" in system_msg
        assert "System instruction 2" in system_msg

    def test_has_images_detects_no_images(self):
        """Test that _has_images returns False for text-only messages."""
        model = ClaudeCodeModel()

        messages = [
            ModelRequest(
                parts=[UserPromptPart(content="Text only")],
                kind="request"
            )
        ]

        assert model._has_images(messages) is False


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


class TestMessageExtractionEdgeCases:
    """Test edge cases in message extraction logic."""

    def test_extract_user_message_empty_messages(self):
        """Test extracting from empty message list."""
        model = ClaudeCodeModel()
        user_msg = model._extract_user_message([])
        assert user_msg == ""

    def test_extract_user_message_no_user_parts(self):
        """Test extracting when no user parts exist."""
        model = ClaudeCodeModel()
        messages = [
            ModelRequest(
                parts=[],
                kind="request"
            )
        ]
        user_msg = model._extract_user_message(messages)
        assert user_msg == ""

    def test_extract_system_messages_empty(self):
        """Test extracting system messages when none exist."""
        model = ClaudeCodeModel()
        system_msg = model._extract_system_messages([])
        assert system_msg == ""

    def test_extract_system_messages_none_instructions(self):
        """Test extracting when instructions are None."""
        model = ClaudeCodeModel()
        messages = [
            ModelRequest(
                parts=[TextPart(content="test")],
                kind="request",
                instructions=None
            )
        ]
        system_msg = model._extract_system_messages(messages)
        assert system_msg == ""

    def test_has_images_detects_binary_content(self):
        """Test that _has_images detects BinaryContent."""
        model = ClaudeCodeModel()
        messages = [
            ModelRequest(
                parts=[BinaryContent(data=b"test", media_type="image/png")],
                kind="request"
            )
        ]
        assert model._has_images(messages) is True

    def test_has_images_detects_image_url(self):
        """Test that _has_images detects ImageUrl."""
        model = ClaudeCodeModel()
        messages = [
            ModelRequest(
                parts=[ImageUrl(url="https://example.com/image.png")],
                kind="request"
            )
        ]
        assert model._has_images(messages) is True

    def test_has_images_detects_list_content(self):
        """Test that _has_images detects list-based multimodal content."""
        model = ClaudeCodeModel()
        messages = [
            ModelRequest(
                parts=[UserPromptPart(content=["text", "more text"])],
                kind="request"
            )
        ]
        assert model._has_images(messages) is True

    def test_extract_multimodal_text_and_binary(self):
        """Test extracting multimodal content with text and binary image."""
        model = ClaudeCodeModel()
        messages = [
            ModelRequest(
                parts=[
                    TextPart(content="Look at this"),
                    BinaryContent(data=b"\x89PNG", media_type="image/png")
                ],
                kind="request"
            )
        ]
        content = model._extract_multimodal_content(messages)

        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Look at this"
        assert content[1]["type"] == "image"
        assert content[1]["source"]["type"] == "base64"

    def test_extract_multimodal_from_user_prompt_list(self):
        """Test extracting multimodal from UserPromptPart with list content."""
        model = ClaudeCodeModel()
        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(content=[
                        "Text content",
                        BinaryContent(data=b"imagedata", media_type="image/jpeg")
                    ])
                ],
                kind="request"
            )
        ]
        content = model._extract_multimodal_content(messages)

        assert len(content) >= 2
        assert any(block["type"] == "text" for block in content)
        assert any(block["type"] == "image" for block in content)


class TestSchemaTypeConversion:
    """Test JSON schema to Python type conversion."""

    def test_get_type_from_schema_string(self):
        """Test converting string type."""
        model = ClaudeCodeModel()
        schema = {"type": "string"}
        result = model._get_type_from_schema(schema)
        assert result == str

    def test_get_type_from_schema_integer(self):
        """Test converting integer type."""
        model = ClaudeCodeModel()
        schema = {"type": "integer"}
        result = model._get_type_from_schema(schema)
        assert result == int

    def test_get_type_from_schema_number(self):
        """Test converting number (float) type."""
        model = ClaudeCodeModel()
        schema = {"type": "number"}
        result = model._get_type_from_schema(schema)
        assert result == float

    def test_get_type_from_schema_boolean(self):
        """Test converting boolean type."""
        model = ClaudeCodeModel()
        schema = {"type": "boolean"}
        result = model._get_type_from_schema(schema)
        assert result == bool

    def test_get_type_from_schema_array_of_strings(self):
        """Test converting array type."""
        model = ClaudeCodeModel()
        schema = {"type": "array", "items": {"type": "string"}}
        result = model._get_type_from_schema(schema)
        assert result == list[str]

    def test_get_type_from_schema_array_of_integers(self):
        """Test converting array of integers."""
        model = ClaudeCodeModel()
        schema = {"type": "array", "items": {"type": "integer"}}
        result = model._get_type_from_schema(schema)
        assert result == list[int]

    def test_get_type_from_schema_object(self):
        """Test converting object type."""
        model = ClaudeCodeModel()
        schema = {"type": "object", "additionalProperties": {"type": "string"}}
        result = model._get_type_from_schema(schema)
        assert result == dict[str, str]

    def test_get_type_from_schema_object_with_int_values(self):
        """Test converting object with integer values."""
        model = ClaudeCodeModel()
        schema = {"type": "object", "additionalProperties": {"type": "integer"}}
        result = model._get_type_from_schema(schema)
        assert result == dict[str, int]

    def test_get_type_from_schema_anyof_with_null(self):
        """Test converting anyOf with null (optional field)."""
        model = ClaudeCodeModel()
        schema = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        result = model._get_type_from_schema(schema)
        assert result == str | None

    def test_get_type_from_schema_anyof_without_null(self):
        """Test converting anyOf without null."""
        model = ClaudeCodeModel()
        schema = {"anyOf": [{"type": "string"}]}
        result = model._get_type_from_schema(schema)
        assert result == str

    def test_get_type_from_schema_ref(self):
        """Test converting $ref reference."""
        model = ClaudeCodeModel()
        schema = {"$ref": "#/$defs/SomeModel"}
        result = model._get_type_from_schema(schema)
        assert result == dict

    def test_get_type_from_schema_unknown_defaults_to_string(self):
        """Test that unknown types default to string."""
        model = ClaudeCodeModel()
        schema = {"type": "unknown"}
        result = model._get_type_from_schema(schema)
        assert result == str


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


class TestStructuredOutputDynamicModels:
    """Test dynamic Pydantic model creation from JSON schemas."""

    @pytest.mark.asyncio
    async def test_structured_output_with_optional_fields(self):
        """Test that optional fields are handled correctly."""
        model = ClaudeCodeModel()

        mock_tool = Mock()
        mock_tool.name = "output"
        mock_tool.parameters_json_schema = {
            "type": "object",
            "properties": {
                "required_field": {"type": "string"},
                "optional_field": {"type": "string"}
            },
            "required": ["required_field"]
        }

        params = ModelRequestParameters(
            output_mode='tool',
            output_tools=[mock_tool]
        )

        # Mock BaseModel for the response
        class MockOutput(BaseModel):
            required_field: str
            optional_field: str | None = None

        with patch.object(model._client, 'structured_query', new_callable=AsyncMock) as mock_query:
            mock_query.return_value = MockOutput(required_field="value")

            messages = [ModelRequest(parts=[TextPart(content="test")], kind="request")]
            result = await model.request(messages, None, params)

            assert result.parts[0].args["required_field"] == "value"
            assert result.parts[0].args["optional_field"] is None

    @pytest.mark.asyncio
    async def test_structured_output_with_array_defaults(self):
        """Test that array fields get [] instead of None."""
        model = ClaudeCodeModel()

        mock_tool = Mock()
        mock_tool.name = "output"
        mock_tool.parameters_json_schema = {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}}
            },
            "required": []
        }

        params = ModelRequestParameters(
            output_mode='tool',
            output_tools=[mock_tool]
        )

        class MockOutput(BaseModel):
            items: list[str] = []

        with patch.object(model._client, 'structured_query', new_callable=AsyncMock) as mock_query:
            # Return model with None for items
            mock_output = MockOutput()
            mock_output.items = None  # type: ignore
            mock_query.return_value = mock_output

            messages = [ModelRequest(parts=[TextPart(content="test")], kind="request")]
            result = await model.request(messages, None, params)

            # Should post-process None to []
            assert result.parts[0].args["items"] == []

    @pytest.mark.asyncio
    async def test_structured_output_with_object_defaults(self):
        """Test that object fields get {} instead of None."""
        model = ClaudeCodeModel()

        mock_tool = Mock()
        mock_tool.name = "output"
        mock_tool.parameters_json_schema = {
            "type": "object",
            "properties": {
                "metadata": {"type": "object", "additionalProperties": {"type": "string"}}
            },
            "required": []
        }

        params = ModelRequestParameters(
            output_mode='tool',
            output_tools=[mock_tool]
        )

        class MockOutput(BaseModel):
            metadata: dict[str, str] = {}

        with patch.object(model._client, 'structured_query', new_callable=AsyncMock) as mock_query:
            mock_output = MockOutput()
            mock_output.metadata = None  # type: ignore
            mock_query.return_value = mock_output

            messages = [ModelRequest(parts=[TextPart(content="test")], kind="request")]
            result = await model.request(messages, None, params)

            # Should post-process None to {}
            assert result.parts[0].args["metadata"] == {}
