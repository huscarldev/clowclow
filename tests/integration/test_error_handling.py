"""Integration tests for error handling and edge cases with ClaudeCodeModel.

Tests verify that ClaudeCodeModel handles errors gracefully and provides
meaningful error messages for various failure scenarios.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, AsyncMock

from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior

from clowclow import ClaudeCodeModel


class TestStructuredOutputErrors:
    """Test error handling with structured output."""

    @pytest.mark.asyncio
    async def test_malformed_json_in_structured_output(self):
        """Test handling of malformed JSON in structured output."""

        class SimpleModel(BaseModel):
            name: str
            value: int

        model = ClaudeCodeModel()

        # Mock structured_query to return malformed response
        with patch.object(model._client, 'structured_query', new_callable=AsyncMock) as mock:
            # This should trigger validation error or retry
            mock.side_effect = ValidationError.from_exception_data(
                "SimpleModel",
                [{"type": "missing", "loc": ("name",), "msg": "Field required"}]
            )

            agent = Agent(model, output_type=SimpleModel)

            with pytest.raises((ValidationError, RuntimeError, Exception)) as exc_info:
                await agent.run("Get data")

            # Should have meaningful error message
            assert exc_info.value is not None
            error_msg = str(exc_info.value)
            assert len(error_msg) > 0, "Error message should not be empty"

    @pytest.mark.asyncio
    async def test_missing_required_field_in_structured_output(self):
        """Test handling when required fields are missing."""

        class StrictModel(BaseModel):
            required_field: str
            required_number: int

        model = ClaudeCodeModel()

        # Mock to return partial data
        with patch.object(model._client, 'structured_query', new_callable=AsyncMock) as mock:
            # Create instance missing a field
            class PartialModel(BaseModel):
                required_field: str
                required_number: int | None = None

            mock.return_value = PartialModel(required_field="test", required_number=None)

            agent = Agent(model, output_type=StrictModel)

            # Should handle missing field
            try:
                result = await agent.run("Get data")
                # If it succeeds, verify it has valid data
                if result.output:
                    assert hasattr(result.output, 'required_field')
                    assert hasattr(result.output, 'required_number')
            except (ValidationError, Exception) as e:
                # Expected - validation should fail
                assert "required" in str(e).lower() or "field" in str(e).lower()

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_schema_validation_with_constraints(self):
        """Test that Pydantic constraints are enforced."""

        class ConstrainedModel(BaseModel):
            positive_number: int = Field(gt=0, description="Must be positive")
            limited_string: str = Field(max_length=10, description="Max 10 chars")

        model = ClaudeCodeModel()
        agent = Agent(model, output_type=ConstrainedModel)

        # Ask for data that should meet constraints
        result = await agent.run("Give me a positive number (1-100) and a short word (max 10 letters)")

        # Verify constraints are met
        assert result.output is not None
        assert result.output.positive_number > 0, "Number should be positive per constraint"
        assert len(result.output.limited_string) <= 10, \
            f"String should be max 10 chars, got {len(result.output.limited_string)}: {result.output.limited_string}"


class TestNetworkAndTimeoutErrors:
    """Test handling of network and timeout errors."""

    @pytest.mark.asyncio
    async def test_client_exception_handling(self):
        """Test that client exceptions are properly wrapped."""
        model = ClaudeCodeModel()

        with patch.object(model._client, 'simple_query', new_callable=AsyncMock) as mock:
            mock.side_effect = Exception("Connection timeout")

            agent = Agent(model)

            with pytest.raises(RuntimeError) as exc_info:
                await agent.run("Test query")

            # Should wrap in RuntimeError with context
            assert "Claude Code request failed" in str(exc_info.value)
            assert exc_info.value.__cause__ is not None

    @pytest.mark.asyncio
    async def test_empty_response_handling(self):
        """Test handling of empty/None responses from client."""
        model = ClaudeCodeModel()

        with patch.object(model._client, 'simple_query', new_callable=AsyncMock) as mock:
            mock.return_value = ""

            agent = Agent(model)
            result = await agent.run("Test")

            # Should handle empty response gracefully
            assert result.output is not None
            # May be empty string, which is valid


class TestInputValidation:
    """Test validation of various input types."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_very_long_prompt(self):
        """Test handling of very long prompts."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        # Create a long prompt
        long_prompt = "Tell me about Python. " * 1000  # ~25,000 chars

        # Should handle long input (may truncate or process in chunks)
        result = await agent.run(long_prompt)

        assert result.output is not None
        assert isinstance(result.output, str)
        assert len(result.output) > 0

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_special_characters_in_prompt(self):
        """Test handling of special characters in prompts."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        # Prompt with various special characters
        special_prompt = 'Test with "quotes", \'apostrophes\', and symbols: @#$%^&*(){}[]<>|\\/'

        result = await agent.run(special_prompt)

        assert result.output is not None
        assert isinstance(result.output, str)
        assert len(result.output) > 0

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_unicode_and_emoji_in_prompt(self):
        """Test handling of Unicode and emoji characters."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        unicode_prompt = "What does this mean: 你好 🌍 Здравствуй مرحبا"

        result = await agent.run(unicode_prompt)

        assert result.output is not None
        assert isinstance(result.output, str)
        assert len(result.output) > 0


class TestToolErrors:
    """Test error handling in tool execution."""

    def test_tool_raises_exception(self):
        """Test handling when tool raises an exception."""
        agent = Agent(TestModel())

        @agent.tool_plain
        def failing_tool(x: int) -> int:
            """Tool that always fails."""
            raise ValueError("Tool execution failed")

        # Should handle tool exception gracefully
        try:
            result = agent.run_sync("Use the tool")
            # May succeed with error message in output
            assert result.output is not None
        except Exception as e:
            # Or may propagate exception
            assert "fail" in str(e).lower() or "error" in str(e).lower()

    def test_tool_returns_invalid_type(self):
        """Test handling when tool returns unexpected type."""
        agent = Agent(TestModel())

        @agent.tool_plain
        def invalid_tool() -> str:
            """Tool with incorrect return type."""
            return {"invalid": "dict"}  # type: ignore - intentional type violation

        # Should handle type mismatch
        result = agent.run_sync("Use tool")
        assert result.output is not None


class TestConcurrentRequests:
    """Test concurrent request handling."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self):
        """Test handling multiple concurrent requests."""
        import asyncio

        model = ClaudeCodeModel()
        agent = Agent(model)

        # Make 3 concurrent requests
        tasks = [
            agent.run("What is 1+1?"),
            agent.run("What is 2+2?"),
            agent.run("What is 3+3?"),
        ]

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 3
        for result in results:
            assert result.output is not None
            assert isinstance(result.output, str)
            assert len(result.output) > 0

        # Verify different answers
        outputs = [r.output for r in results]
        assert "2" in outputs[0] or "1+1" in outputs[0].lower()
        assert "4" in outputs[1] or "2+2" in outputs[1].lower()
        assert "6" in outputs[2] or "3+3" in outputs[2].lower()


class TestEdgeCases:
    """Test various edge cases."""

    @pytest.mark.asyncio
    async def test_agent_with_none_system_prompt(self):
        """Test agent with None system prompt."""
        model = ClaudeCodeModel()
        agent = Agent(model, system_prompt=None)

        # Should work with TestModel override
        with agent.override(model=TestModel()):
            result = await agent.run("Test")
            assert result.output is not None

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_empty_string_query(self):
        """Test handling of empty string query."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        result = await agent.run("")

        # Should handle gracefully
        assert result is not None
        assert hasattr(result, 'output')

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_whitespace_only_query(self):
        """Test handling of whitespace-only query."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        result = await agent.run("   \n\t  ")

        # Should handle gracefully
        assert result is not None
        assert hasattr(result, 'output')


class TestRetryBehavior:
    """Test retry mechanisms."""

    def test_model_retry_propagates(self):
        """Test that ModelRetry exception propagates correctly."""
        agent = Agent(TestModel())

        retry_count = 0

        @agent.tool_plain
        def retry_tool() -> str:
            """Tool that retries multiple times."""
            nonlocal retry_count
            retry_count += 1
            if retry_count < 3:
                raise ModelRetry(f"Retry attempt {retry_count}")
            return "Success"

        result = agent.run_sync("Use tool")

        # Should have retried
        assert retry_count >= 2, f"Expected at least 2 retries, got {retry_count}"
        assert result.output is not None
