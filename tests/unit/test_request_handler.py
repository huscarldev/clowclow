"""Unit tests for RequestHandler - testing message extraction logic."""

from __future__ import annotations

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    UserPromptPart,
    TextPart,
    BinaryContent,
    ImageUrl,
    ToolReturnPart,
)

from clowclow.request_handler import RequestHandler


class TestMessageExtraction:
    """Test message extraction methods."""

    def test_extract_user_message_from_text_part(self):
        """Test extracting user message from TextPart."""
        messages = [
            ModelRequest(
                parts=[TextPart(content="Hello world")],
                kind="request"
            )
        ]

        user_msg = RequestHandler.extract_user_message(messages)
        assert user_msg == "Hello world"

    def test_extract_user_message_multiple_parts(self):
        """Test extracting user message with multiple parts."""
        messages = [
            ModelRequest(
                parts=[
                    UserPromptPart(content="Part 1"),
                    UserPromptPart(content="Part 2")
                ],
                kind="request"
            )
        ]

        user_msg = RequestHandler.extract_user_message(messages)
        assert "Part 1" in user_msg
        assert "Part 2" in user_msg

    def test_extract_system_messages(self):
        """Test extracting system messages from instructions."""
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

        system_msg = RequestHandler.extract_system_messages(messages)
        assert "System instruction 1" in system_msg
        assert "System instruction 2" in system_msg


class TestImageDetection:
    """Test image detection logic."""

    def test_has_images_detects_no_images(self):
        """Test that has_images returns False for text-only messages."""
        messages = [
            ModelRequest(
                parts=[UserPromptPart(content="Text only")],
                kind="request"
            )
        ]

        assert RequestHandler.has_images(messages) is False

    def test_has_images_detects_binary_content(self):
        """Test that has_images detects BinaryContent."""
        messages = [
            ModelRequest(
                parts=[BinaryContent(data=b"test", media_type="image/png")],
                kind="request"
            )
        ]
        assert RequestHandler.has_images(messages) is True

    def test_has_images_detects_image_url(self):
        """Test that has_images detects ImageUrl."""
        messages = [
            ModelRequest(
                parts=[ImageUrl(url="https://example.com/image.png")],
                kind="request"
            )
        ]
        assert RequestHandler.has_images(messages) is True

    def test_has_images_detects_list_content(self):
        """Test that has_images detects list-based multimodal content."""
        messages = [
            ModelRequest(
                parts=[UserPromptPart(content=["text", "more text"])],
                kind="request"
            )
        ]
        assert RequestHandler.has_images(messages) is True


class TestMultimodalExtraction:
    """Test multimodal content extraction."""

    def test_extract_multimodal_text_and_binary(self):
        """Test extracting multimodal content with text and binary image."""
        messages = [
            ModelRequest(
                parts=[
                    TextPart(content="Look at this"),
                    BinaryContent(data=b"\x89PNG", media_type="image/png")
                ],
                kind="request"
            )
        ]
        content = RequestHandler.extract_multimodal_content(messages)

        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Look at this"
        assert content[1]["type"] == "image"
        assert content[1]["source"]["type"] == "base64"

    def test_extract_multimodal_from_user_prompt_list(self):
        """Test extracting multimodal from UserPromptPart with list content."""
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
        content = RequestHandler.extract_multimodal_content(messages)

        assert len(content) >= 2
        assert any(block["type"] == "text" for block in content)
        assert any(block["type"] == "image" for block in content)

    def test_extract_multimodal_from_image_url(self):
        """Test extracting multimodal content from ImageUrl."""
        messages = [
            ModelRequest(
                parts=[ImageUrl(url="https://example.com/image.png")],
                kind="request"
            )
        ]
        content = RequestHandler.extract_multimodal_content(messages)

        assert len(content) == 1
        assert content[0]["type"] == "image"
        assert content[0]["source"]["type"] == "url"
        assert content[0]["source"]["url"] == "https://example.com/image.png"


class TestEdgeCases:
    """Test edge cases in message extraction logic."""

    def test_extract_user_message_empty_messages(self):
        """Test extracting from empty message list."""
        user_msg = RequestHandler.extract_user_message([])
        assert user_msg == ""

    def test_extract_user_message_no_user_parts(self):
        """Test extracting when no user parts exist."""
        messages = [
            ModelRequest(
                parts=[],
                kind="request"
            )
        ]
        user_msg = RequestHandler.extract_user_message(messages)
        assert user_msg == ""

    def test_extract_system_messages_empty(self):
        """Test extracting system messages when none exist."""
        system_msg = RequestHandler.extract_system_messages([])
        assert system_msg == ""

    def test_extract_system_messages_none_instructions(self):
        """Test extracting when instructions are None."""
        messages = [
            ModelRequest(
                parts=[TextPart(content="test")],
                kind="request",
                instructions=None
            )
        ]
        system_msg = RequestHandler.extract_system_messages(messages)
        assert system_msg == ""


class TestToolReturnHandling:
    """Test tool return detection and processing."""

    def test_check_for_tool_returns_detects_tool_return(self):
        """Test that check_for_tool_returns detects ToolReturnPart."""
        messages = [
            ModelRequest(
                parts=[ToolReturnPart(tool_name="test_tool", content="result")],
                kind="request"
            )
        ]

        returns = RequestHandler.check_for_tool_returns(messages)
        assert len(returns) == 1
        assert returns[0].tool_name == "test_tool"

    def test_check_for_tool_returns_no_returns(self):
        """Test that check_for_tool_returns returns empty list for no tool returns."""
        messages = [
            ModelRequest(
                parts=[TextPart(content="test")],
                kind="request"
            )
        ]

        returns = RequestHandler.check_for_tool_returns(messages)
        assert len(returns) == 0

    def test_append_tool_results_to_content(self):
        """Test appending tool results to multimodal content."""
        content = [{"type": "text", "text": "Original query"}]
        tool_returns = [
            ToolReturnPart(tool_name="tool1", content="result1"),
            ToolReturnPart(tool_name="tool2", content="result2")
        ]

        result = RequestHandler.append_tool_results_to_content(content, tool_returns)

        # All tool results are combined into a single text block
        assert len(result) == 2
        assert result[0] == {"type": "text", "text": "Original query"}
        assert result[1]["type"] == "text"
        assert "tool1" in result[1]["text"]
        assert "result1" in result[1]["text"]
        assert "tool2" in result[1]["text"]
        assert "result2" in result[1]["text"]
