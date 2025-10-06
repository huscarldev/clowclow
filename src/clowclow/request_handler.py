"""Request handler for extracting and processing Pydantic AI messages."""

from __future__ import annotations

import base64
from typing import Any
from pydantic_ai import messages as _messages


class RequestHandler:
    """Handles extraction and processing of Pydantic AI request messages."""

    @staticmethod
    def extract_user_message(messages: list[_messages.ModelMessage]) -> str:
        """Extract the most recent user message (text only).

        Args:
            messages: List of model messages

        Returns:
            Extracted user message text
        """
        for msg in reversed(messages):
            if isinstance(msg, _messages.ModelRequest):
                # Extract text from the parts
                user_parts = []
                for part in msg.parts:
                    if isinstance(part, _messages.UserPromptPart):
                        user_parts.append(part.content)
                    elif isinstance(part, _messages.TextPart):
                        user_parts.append(part.content)
                return "\n".join(user_parts)
        return ""

    @staticmethod
    def extract_multimodal_content(messages: list[_messages.ModelMessage]) -> list[dict]:
        """Extract multimodal content including text and images.

        Args:
            messages: List of model messages

        Returns:
            List of content blocks (text and image dicts)
        """
        content_blocks = []

        for msg in reversed(messages):
            if isinstance(msg, _messages.ModelRequest):
                for part in msg.parts:
                    # Handle text content
                    if isinstance(part, _messages.UserPromptPart):
                        # UserPromptPart.content can be str or list
                        if isinstance(part.content, str):
                            content_blocks.append({
                                "type": "text",
                                "text": part.content
                            })
                        elif isinstance(part.content, list):
                            # Process each item in the list
                            for item in part.content:
                                if isinstance(item, str):
                                    content_blocks.append({"type": "text", "text": item})
                                elif isinstance(item, _messages.BinaryContent):
                                    # Convert BinaryContent to dict
                                    content_blocks.append(
                                        RequestHandler._binary_content_to_dict(item)
                                    )
                                elif isinstance(item, _messages.ImageUrl):
                                    # Convert ImageUrl to dict
                                    content_blocks.append({
                                        "type": "image",
                                        "source": {
                                            "type": "url",
                                            "url": item.url
                                        }
                                    })
                                elif isinstance(item, dict):
                                    # Already a dict, use as-is
                                    content_blocks.append(item)
                    elif isinstance(part, _messages.TextPart):
                        content_blocks.append({
                            "type": "text",
                            "text": part.content
                        })
                    # Handle image content (base64)
                    elif isinstance(part, _messages.BinaryContent):
                        content_blocks.append(
                            RequestHandler._binary_content_to_dict(part)
                        )
                    # Handle image URLs
                    elif isinstance(part, _messages.ImageUrl):
                        content_blocks.append({
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": part.url
                            }
                        })

                # Return after processing the most recent user message
                if content_blocks:
                    return content_blocks

        return content_blocks

    @staticmethod
    def _binary_content_to_dict(binary_content: _messages.BinaryContent) -> dict:
        """Convert BinaryContent to image dict format.

        Args:
            binary_content: Binary content to convert

        Returns:
            Image dict with base64 data
        """
        # Encode image data to base64 if not already
        if isinstance(binary_content.data, bytes):
            image_b64 = base64.b64encode(binary_content.data).decode('utf-8')
        else:
            image_b64 = binary_content.data

        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": binary_content.media_type or "image/png",
                "data": image_b64
            }
        }

    @staticmethod
    def has_images(messages: list[_messages.ModelMessage]) -> bool:
        """Check if the messages contain any images.

        Args:
            messages: List of model messages

        Returns:
            True if images are present, False otherwise
        """
        for msg in messages:
            if isinstance(msg, _messages.ModelRequest):
                for part in msg.parts:
                    # Check direct types
                    if isinstance(part, (_messages.BinaryContent, _messages.ImageUrl)):
                        return True
                    # Check if UserPromptPart contains a list (multimodal)
                    if isinstance(part, _messages.UserPromptPart):
                        if isinstance(part.content, list):
                            return True  # List format indicates multimodal content
        return False

    @staticmethod
    def extract_system_messages(messages: list[_messages.ModelMessage]) -> str:
        """Extract and combine system messages.

        Args:
            messages: List of model messages

        Returns:
            Combined system prompt
        """
        system_parts = []
        for msg in messages:
            if isinstance(msg, _messages.ModelRequest):
                if msg.instructions:
                    system_parts.append(msg.instructions)
        return "\n".join(system_parts)

    @staticmethod
    def check_for_tool_returns(messages: list[_messages.ModelMessage]) -> list[_messages.ToolReturnPart]:
        """Check if messages contain tool return parts from previous turns.

        Args:
            messages: Message history

        Returns:
            List of ToolReturnPart objects
        """
        tool_returns = []
        for msg in messages:
            # Tool returns can be in ModelResponse OR ModelRequest messages
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if isinstance(part, _messages.ToolReturnPart):
                        tool_returns.append(part)
        return tool_returns

    @staticmethod
    def append_tool_results_to_content(
        user_content: str | list[dict],
        tool_returns: list[_messages.ToolReturnPart]
    ) -> str | list[dict]:
        """Append tool results to user content.

        Args:
            user_content: Original user content (str or list)
            tool_returns: List of tool return parts

        Returns:
            Updated user content with tool results appended
        """
        tool_results_text = "\n\n".join([
            f"Tool '{tr.tool_name}' returned: {tr.content}"
            for tr in tool_returns
        ])

        # Append tool results to user content
        if isinstance(user_content, str):
            return f"{user_content}\n\n{tool_results_text}"
        else:
            # For list content, append as text block
            updated_content = user_content.copy()
            updated_content.append({"type": "text", "text": tool_results_text})
            return updated_content
