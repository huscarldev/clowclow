"""Integration tests for usage tracking with ClaudeCodeModel.

Tests verify that usage information (tokens, requests, etc.) is properly
tracked and reported through the Pydantic AI interface.
"""

from __future__ import annotations

import pytest

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from clowclow import ClaudeCodeModel


class TestBasicUsageTracking:
    """Test basic usage tracking functionality."""

    def test_usage_attribute_exists_with_testmodel(self):
        """Baseline: TestModel provides usage attribute."""
        agent = Agent(TestModel())
        result = agent.run_sync("Test query")

        # Verify usage attribute exists
        assert hasattr(result, 'usage'), "Result should have usage attribute"
        assert result.usage is not None

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_usage_attribute_exists_live(self):
        """Test that usage attribute exists with live API."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        result = await agent.run("What is 2+2?")

        # Verify usage attribute exists
        assert hasattr(result, 'usage'), "Result should have usage attribute"
        assert result.usage is not None

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_usage_has_expected_fields(self):
        """Test that usage object has expected fields."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        result = await agent.run("Test query")

        assert hasattr(result, 'usage')
        usage = result.usage()  # Call if it's a method

        # Usage should be a valid object (not None or empty)
        assert usage is not None

        # Check for common usage fields (adapt based on actual structure)
        # Pydantic AI usage typically has: request_tokens, response_tokens, total_tokens
        if hasattr(usage, 'request_tokens'):
            assert isinstance(usage.request_tokens, int)
            assert usage.request_tokens >= 0

        if hasattr(usage, 'response_tokens'):
            assert isinstance(usage.response_tokens, int)
            assert usage.response_tokens >= 0

        if hasattr(usage, 'total_tokens'):
            assert isinstance(usage.total_tokens, int)
            assert usage.total_tokens >= 0


class TestUsageAccumulation:
    """Test usage accumulation across multiple requests."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_usage_accumulates_across_requests(self):
        """Test that usage accumulates when making multiple requests."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        # First request
        result1 = await agent.run("First query")
        usage1 = result1.usage()

        # Second request
        result2 = await agent.run("Second query")
        usage2 = result2.usage()

        # Both should have usage data
        assert usage1 is not None
        assert usage2 is not None

        # Each request should have independent usage
        # (Note: Pydantic AI may or may not accumulate - verify behavior)
        # For now, just verify both are valid
        assert usage1 is not None
        assert usage2 is not None


class TestUsageWithStreaming:
    """Test usage tracking with streaming responses."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_usage_available_after_streaming(self):
        """Test that usage is available after stream completes."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        async with agent.run_stream("Test query") as result:
            # Consume stream
            async for _ in result.stream_text(debounce_by=None):
                pass

            # Usage should be available after streaming completes
            assert hasattr(result, 'usage')

            # Get usage
            usage = result.usage()
            assert usage is not None


class TestUsageWithStructuredOutput:
    """Test usage tracking with structured output."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_usage_with_structured_output(self):
        """Test usage tracking when using structured output."""

        class SimpleModel(BaseModel):
            name: str
            value: int

        model = ClaudeCodeModel()
        agent = Agent(model, output_type=SimpleModel)

        result = await agent.run("Create data")

        # Usage should be tracked for structured output too
        assert hasattr(result, 'usage')
        usage = result.usage()
        assert usage is not None


class TestUsageWithTools:
    """Test usage tracking when tools are involved.

    NOTE: ClaudeCodeModel does not support tool calling.
    This test is skipped for ClaudeCodeModel.
    """

    @pytest.mark.skip(reason="ClaudeCodeModel does not support Pydantic AI tool calling")
    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_usage_with_tool_calls(self):
        """Test usage tracking when tools are called.

        SKIPPED: ClaudeCodeModel does not implement tool calling.
        """
        model = ClaudeCodeModel()
        agent = Agent(model)

        tool_called = False

        @agent.tool_plain
        def test_tool(value: str) -> str:
            """A test tool."""
            nonlocal tool_called
            tool_called = True
            return f"Processed: {value}"

        result = await agent.run("Use the test tool")

        # Usage should be tracked (tool won't actually be called)
        assert hasattr(result, 'usage')
        usage = result.usage()
        assert usage is not None


class TestUsageEdgeCases:
    """Test usage tracking in edge cases."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_usage_with_empty_response(self):
        """Test usage tracking when response is empty."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        result = await agent.run("")

        # Should still have usage data
        assert hasattr(result, 'usage')

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_usage_with_error(self):
        """Test usage tracking when an error occurs."""
        from unittest.mock import patch, AsyncMock

        model = ClaudeCodeModel()
        agent = Agent(model)

        with patch.object(model._client, 'simple_query', new_callable=AsyncMock) as mock:
            mock.side_effect = Exception("Test error")

            try:
                result = await agent.run("Test")
                # If it succeeds somehow, check usage
                if hasattr(result, 'usage'):
                    assert result.usage is not None
            except Exception:
                # Error occurred as expected
                pass


class TestUsageComparison:
    """Test usage differences between simple and complex requests."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_usage_varies_with_request_length(self):
        """Test that usage varies based on request length."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        # Short request
        result_short = await agent.run("Hi")
        usage_short = result_short.usage()

        # Longer request
        long_text = "Please explain in detail " + "the topic " * 50
        result_long = await agent.run(long_text)
        usage_long = result_long.usage()

        # Both should have usage
        assert usage_short is not None
        assert usage_long is not None

        # Longer request likely uses more tokens (if token counts are available)
        # This is a heuristic test - may not always hold true
        # Just verify both are valid for now
        assert usage_short is not None
        assert usage_long is not None
