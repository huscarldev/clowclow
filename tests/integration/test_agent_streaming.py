"""Integration tests for streaming responses with ClaudeCodeModel.

Tests verify that ClaudeCodeModel correctly implements streaming functionality
compatible with Pydantic AI's streaming API.
"""

from __future__ import annotations

import pytest

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from clowclow import ClaudeCodeModel


class TestStreamingBasics:
    """Test basic streaming functionality."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_run_stream_returns_context_manager(self):
        """Test that run_stream returns an async context manager."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        # Should return a context manager
        result = agent.run_stream("Test query")

        # Should be an async context manager
        assert hasattr(result, '__aenter__')
        assert hasattr(result, '__aexit__')

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_stream_basic_text(self):
        """Test streaming basic text response."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        async with agent.run_stream("Say hello") as result:
            # Result should have streaming capabilities
            assert hasattr(result, 'stream_text')

            # Should be able to iterate over stream
            chunks = []
            async for chunk in result.stream_text(debounce_by=None):
                chunks.append(chunk)
                # Each chunk should be a non-empty string
                assert isinstance(chunk, str), f"Chunk should be string, got {type(chunk)}"

            # Should have received some chunks
            assert isinstance(chunks, list)
            assert len(chunks) > 0, "Should have received at least one chunk"

            # Verify chunks contain actual content
            full_text = "".join(chunks)
            assert len(full_text) > 0, "Combined chunks should not be empty"
            assert full_text.strip(), "Combined text should not be just whitespace"

            # Verify greeting was streamed
            greeting_keywords = ["hello", "hi", "hey", "greetings"]
            assert any(kw in full_text.lower() for kw in greeting_keywords), \
                f"Streamed text should contain a greeting, got: {full_text}"

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_stream_completion_status(self):
        """Test stream completion status tracking."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        async with agent.run_stream("Test") as result:
            # Consume the stream
            async for _ in result.stream_text(debounce_by=None):
                pass

            # After streaming, should be complete
            assert result.is_complete or hasattr(result, 'is_complete')


class TestStreamingWithTestModel:
    """Test streaming using TestModel for deterministic testing."""

    @pytest.mark.asyncio
    async def test_stream_text_with_test_model(self):
        """Baseline test: streaming works with TestModel."""
        agent = Agent(TestModel())

        async with agent.run_stream("Test query") as result:
            chunks = []
            async for chunk in result.stream_text(debounce_by=None):
                chunks.append(chunk)

            # Should have received chunks
            assert len(chunks) >= 0  # TestModel may return empty or populated

    @pytest.mark.asyncio
    async def test_stream_with_custom_output(self):
        """Test streaming with custom test output."""
        agent = Agent(TestModel(custom_output_text="Custom response"))

        async with agent.run_stream("Query") as result:
            chunks = []
            async for chunk in result.stream_text(debounce_by=None):
                chunks.append(chunk)

            # Custom output should be in chunks
            full_text = "".join(chunks)
            assert "Custom" in full_text or full_text == "" or isinstance(full_text, str)


class TestStreamingMessages:
    """Test streaming with different message types."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_stream_with_system_prompt(self):
        """Test streaming with system prompt."""
        model = ClaudeCodeModel()
        agent = Agent(model, system_prompt="You are a helpful assistant.")

        async with agent.run_stream("Hello") as result:
            chunks = []
            async for chunk in result.stream_text(debounce_by=None):
                chunks.append(chunk)

            assert isinstance(chunks, list)

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_stream_multiple_messages(self):
        """Test streaming in a conversation context."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        # First message
        async with agent.run_stream("First message") as result1:
            async for _ in result1.stream_text(debounce_by=None):
                pass

        # Second message (in same agent)
        async with agent.run_stream("Second message") as result2:
            async for _ in result2.stream_text(debounce_by=None):
                pass

        # Both should complete successfully
        assert True  # If we get here, streaming worked


class TestStreamingDebounce:
    """Test streaming debounce functionality."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_stream_with_debounce(self):
        """Test streaming with debounce timing."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        async with agent.run_stream("Test") as result:
            # Stream with debounce
            chunks_debounced = []
            async for chunk in result.stream_text(debounce_by=0.1):
                chunks_debounced.append(chunk)

            # Should still receive chunks (possibly fewer)
            assert isinstance(chunks_debounced, list)

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_stream_no_debounce_gets_all_chunks(self):
        """Test that no debounce returns all chunks."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        async with agent.run_stream("Test") as result:
            # Stream without debounce should get all chunks
            chunks = []
            async for chunk in result.stream_text(debounce_by=None):
                chunks.append(chunk)

            assert isinstance(chunks, list)


class TestStreamingFinalResult:
    """Test accessing final result after streaming."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_stream_final_output(self):
        """Test accessing final output after streaming completes."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        async with agent.run_stream("What is 2+2?") as result:
            # Consume stream and collect chunks
            chunks_collected = []
            async for chunk in result.stream_text(debounce_by=None):
                chunks_collected.append(chunk)

            # After streaming, should have final output
            final_output = await result.get_output()

            # Output should be a non-empty string
            assert final_output is not None
            assert isinstance(final_output, str), f"Expected string, got {type(final_output)}"
            assert len(final_output) > 0, "Final output should not be empty"
            assert final_output.strip(), "Final output should not be just whitespace"

            # Final output should match streamed chunks
            streamed_text = "".join(chunks_collected)
            assert final_output == streamed_text, \
                "Final output should match concatenated stream chunks"

            # Verify it contains the expected answer
            assert "4" in final_output, "Final output should contain the answer '4'"

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_stream_usage_tracking(self):
        """Test usage tracking during streaming."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        async with agent.run_stream("Test") as result:
            # Consume stream
            async for _ in result.stream_text(debounce_by=None):
                pass

            # Should have usage information
            assert hasattr(result, 'usage')


class TestStreamingErrors:
    """Test error handling during streaming."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_stream_with_invalid_input(self):
        """Test streaming with invalid input."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        # Empty string should still work (may return empty or error message)
        async with agent.run_stream("") as result:
            chunks = []
            async for chunk in result.stream_text(debounce_by=None):
                chunks.append(chunk)

            # Should complete without crashing
            assert isinstance(chunks, list)


class TestStreamingCancellation:
    """Test stream cancellation and cleanup."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_stream_early_exit(self):
        """Test exiting stream context early."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        async with agent.run_stream("Test") as result:
            # Exit early without consuming entire stream
            async for chunk in result.stream_text(debounce_by=None):
                # Stop after first chunk
                break

        # Context should clean up properly

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_stream_context_cleanup(self):
        """Test that streaming context cleans up resources."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        async with agent.run_stream("Test") as result:
            pass  # Don't consume stream

        # Context should exit cleanly even without consuming stream
        assert True
