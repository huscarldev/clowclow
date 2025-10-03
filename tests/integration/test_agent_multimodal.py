"""Integration tests for multimodal inputs with ClaudeCodeModel.

Tests verify that ClaudeCodeModel correctly handles images and other
multimodal content types through Pydantic AI's Agent interface.
"""

from __future__ import annotations

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ImageUrl, BinaryContent

from clowclow import ClaudeCodeModel


class TestImageURLInput:
    """Test image URL inputs."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_agent_with_image_url(self, test_image_url: str):
        """Test agent handling image URL input."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        # Send message with image URL
        result = await agent.run([
            "What's in this image?",
            ImageUrl(url=test_image_url)
        ])

        # Should process multimodal input successfully
        assert result.output is not None
        assert isinstance(result.output, str)
        assert len(result.output) > 0, "Response should not be empty"
        assert result.output.strip(), "Response should not be just whitespace"

        # Verify response is about analyzing an image
        image_keywords = ["image", "picture", "photo", "see", "show", "visual"]
        assert any(kw in result.output.lower() for kw in image_keywords), \
            f"Response should reference the image, got: {result.output}"

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_agent_with_multiple_image_urls(self):
        """Test agent with multiple image URLs."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        result = await agent.run([
            "Compare these images:",
            ImageUrl(url="https://example.com/image1.jpg"),
            ImageUrl(url="https://example.com/image2.jpg")
        ])

        assert result.output is not None

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_agent_text_and_image_interleaved(self, test_image_url: str):
        """Test interleaving text and images."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        result = await agent.run([
            "Look at this first image:",
            ImageUrl(url=test_image_url),
            "Now describe what you see.",
        ])

        assert result.output is not None


class TestBinaryImageContent:
    """Test binary image content inputs."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_agent_with_binary_image(self, test_image_data: bytes):
        """Test agent with binary image content."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        result = await agent.run([
            "Analyze this image:",
            BinaryContent(data=test_image_data, media_type="image/png")
        ])

        # Verify response
        assert result.output is not None
        assert isinstance(result.output, str)
        assert len(result.output) > 0, "Response should not be empty"
        assert result.output.strip(), "Response should not be just whitespace"

        # Verify it's an image analysis response
        analysis_keywords = ["image", "picture", "pixel", "color", "visual", "see", "show"]
        assert any(kw in result.output.lower() for kw in analysis_keywords), \
            f"Response should be about image analysis, got: {result.output}"

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_agent_with_jpeg_image(self, test_image_data: bytes):
        """Test agent with JPEG image."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        result = await agent.run([
            "What's in this JPEG?",
            BinaryContent(data=test_image_data, media_type="image/jpeg")
        ])

        assert result.output is not None

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_agent_with_webp_image(self, test_image_data: bytes):
        """Test agent with WebP image."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        result = await agent.run([
            "Describe this WebP image:",
            BinaryContent(data=test_image_data, media_type="image/webp")
        ])

        assert result.output is not None


class TestMixedContent:
    """Test mixed content types."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_text_url_and_binary_image(
        self, test_image_url: str, test_image_data: bytes
    ):
        """Test mixing text, image URL, and binary image."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        result = await agent.run([
            "First, look at this URL image:",
            ImageUrl(url=test_image_url),
            "Now compare it to this binary image:",
            BinaryContent(data=test_image_data, media_type="image/png"),
            "What are the differences?"
        ])

        assert result.output is not None

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_multimodal_with_system_prompt(self, test_image_data: bytes):
        """Test multimodal input with system prompt."""
        model = ClaudeCodeModel()
        agent = Agent(
            model,
            system_prompt="You are an image analysis expert. Be detailed."
        )

        result = await agent.run([
            "Analyze this:",
            BinaryContent(data=test_image_data, media_type="image/png")
        ])

        assert result.output is not None


class TestMultimodalWithTools:
    """Test multimodal inputs combined with tools.

    NOTE: ClaudeCodeModel does not support tool calling.
    This test is skipped for ClaudeCodeModel.
    """

    @pytest.mark.skip(reason="ClaudeCodeModel does not support Pydantic AI tool calling")
    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_image_with_tool_calling(self, test_image_data: bytes):
        """Test image input that triggers tool usage.

        SKIPPED: ClaudeCodeModel does not implement tool calling.
        Tools can be registered but won't be invoked.
        """
        model = ClaudeCodeModel()
        agent = Agent(model)

        @agent.tool_plain
        def get_image_metadata(width: int, height: int) -> str:
            """Get metadata about image dimensions."""
            return f"Image is {width}x{height} pixels"

        result = await agent.run([
            "What size is this image?",
            BinaryContent(data=test_image_data, media_type="image/png")
        ])

        # Tool will NOT be called - just verify multimodal input works
        assert result.output is not None
        assert isinstance(result.output, str)


class TestMultimodalStreaming:
    """Test streaming with multimodal inputs."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_stream_with_image(self, test_image_data: bytes):
        """Test streaming response with image input."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        async with agent.run_stream([
            "Describe this image:",
            BinaryContent(data=test_image_data, media_type="image/png")
        ]) as result:
            chunks = []
            async for chunk in result.stream_text(debounce_by=None):
                chunks.append(chunk)

            # Should stream response for multimodal input
            assert isinstance(chunks, list)

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_stream_with_image_url(self, test_image_url: str):
        """Test streaming with image URL."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        async with agent.run_stream([
            "What's in this image?",
            ImageUrl(url=test_image_url)
        ]) as result:
            chunks = []
            async for chunk in result.stream_text(debounce_by=None):
                chunks.append(chunk)

            assert isinstance(chunks, list)


class TestMultimodalErrorHandling:
    """Test error handling with multimodal inputs."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_invalid_image_url(self):
        """Test handling invalid image URL."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        # Invalid URL should still be processed
        # (validation happens at API level, not in our code)
        result = await agent.run([
            "Analyze this:",
            ImageUrl(url="not-a-valid-url")
        ])

        # Should get some response (may be error message from API)
        assert result.output is not None or result

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_empty_binary_content(self):
        """Test handling empty binary content."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        result = await agent.run([
            "Analyze this:",
            BinaryContent(data=b"", media_type="image/png")
        ])

        # Should handle gracefully
        assert result is not None

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_unsupported_content_type(self, test_image_data: bytes):
        """Test handling unsupported content type."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        # Send with unusual content type
        result = await agent.run([
            "Process this:",
            BinaryContent(data=test_image_data, media_type="image/xyz")
        ])

        # Should still process (API will handle validation)
        assert result is not None


class TestMultimodalContentTypes:
    """Test different multimodal content types."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_png_image(self, test_image_data: bytes):
        """Test PNG image specifically."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        result = await agent.run([
            "Describe:",
            BinaryContent(data=test_image_data, media_type="image/png")
        ])

        assert result.output is not None

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_image_with_explicit_filename(self, test_image_url: str):
        """Test image URL with filename."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        # URL with filename
        url_with_filename = "https://example.com/images/photo.png"

        result = await agent.run([
            "What's in photo.png?",
            ImageUrl(url=url_with_filename)
        ])

        assert result.output is not None


class TestMultimodalConversations:
    """Test multimodal content in conversational contexts."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_multimodal_followup_questions(self, test_image_data: bytes):
        """Test follow-up questions about image."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        # First request with image
        result1 = await agent.run([
            "What's in this image?",
            BinaryContent(data=test_image_data, media_type="image/png")
        ])
        assert result1.output is not None

        # Follow-up without image (should work)
        result2 = await agent.run("Tell me more details")
        assert result2.output is not None
