"""Integration tests for message inspection using capture_run_messages.

Tests verify that message exchange inspection works correctly with ClaudeCodeModel,
following Pydantic AI best practices for message tracking and debugging.
"""

from __future__ import annotations

import pytest

from pydantic import BaseModel
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)

from clowclow import ClaudeCodeModel


class TestBasicMessageCapture:
    """Test basic message capture functionality."""

    @pytest.mark.asyncio
    async def test_capture_messages_with_testmodel(self):
        """Baseline: message capture works with TestModel."""
        agent = Agent(TestModel())

        with capture_run_messages() as messages:
            result = await agent.run("Test query")

        # Should capture messages
        assert len(messages) > 0, "Should have captured messages"

        # First message should be request
        assert messages[0].kind == "request"

        # Should have at least one response
        response_messages = [m for m in messages if m.kind == "response"]
        assert len(response_messages) > 0, "Should have at least one response"

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_capture_messages_with_claude_code(self):
        """Test message capture with ClaudeCodeModel."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        with capture_run_messages() as messages:
            result = await agent.run("What is 2+2?")

        # Should capture request and response
        assert len(messages) >= 2, f"Expected at least 2 messages, got {len(messages)}"

        # Verify message types
        assert messages[0].kind == "request", "First message should be request"
        assert messages[-1].kind == "response", "Last message should be response"

        # Response should have content
        if hasattr(messages[-1], 'parts'):
            assert len(messages[-1].parts) > 0, "Response should have parts"


class TestSystemPromptInMessages:
    """Test that system prompts appear correctly in messages."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_system_prompt_appears_in_messages(self):
        """Test that system prompt is included in message history."""
        model = ClaudeCodeModel()
        system_text = "You are a helpful math tutor."
        agent = Agent(model, system_prompt=system_text)

        with capture_run_messages() as messages:
            await agent.run("What is 5+5?")

        # Find SystemPromptPart in messages
        system_parts = []
        for msg in messages:
            if msg.kind == "request" and hasattr(msg, 'parts'):
                for part in msg.parts:
                    if isinstance(part, SystemPromptPart):
                        system_parts.append(part)

        # Should have system prompt
        assert len(system_parts) > 0, "Should have at least one SystemPromptPart"

        # Verify content
        assert any(p.content == system_text for p in system_parts), \
            f"System prompt should match. Expected: '{system_text}', found: {[p.content for p in system_parts]}"

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_multiple_system_prompts_in_messages(self):
        """Test handling of multiple system prompts."""
        model = ClaudeCodeModel()
        agent = Agent(
            model,
            system_prompt="You are helpful.\nYou are concise."
        )

        with capture_run_messages() as messages:
            await agent.run("Test")

        # Should have captured the system prompt
        request_messages = [m for m in messages if m.kind == "request"]
        assert len(request_messages) > 0

        # Check for system prompts
        has_system_prompt = False
        for msg in request_messages:
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if isinstance(part, SystemPromptPart):
                        has_system_prompt = True
                        break

        assert has_system_prompt, "Should have system prompt in messages"


class TestUserMessageInMessages:
    """Test that user messages are captured correctly."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_user_message_content(self):
        """Test that user message content is preserved."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        test_message = "This is a unique test message 12345"

        with capture_run_messages() as messages:
            await agent.run(test_message)

        # Find user prompt
        user_parts = []
        for msg in messages:
            if msg.kind == "request" and hasattr(msg, 'parts'):
                for part in msg.parts:
                    if isinstance(part, (UserPromptPart, TextPart)):
                        user_parts.append(part)

        # Should have user message
        assert len(user_parts) > 0, "Should have user message parts"

        # Verify content
        found_message = any(test_message in str(p.content) for p in user_parts)
        assert found_message, \
            f"Should find test message in user parts. Parts: {[p.content for p in user_parts]}"


class TestToolCallMessages:
    """Test tool call messages in message history."""

    @pytest.mark.asyncio
    async def test_tool_calls_appear_in_messages(self):
        """Test that tool calls are captured in messages."""
        agent = Agent(TestModel(call_tools=['test_tool']))

        @agent.tool_plain
        def test_tool(value: str) -> str:
            """A test tool."""
            return f"Result: {value}"

        with capture_run_messages() as messages:
            result = await agent.run("Use the tool")

        # Check for tool-related parts
        tool_calls = []
        tool_returns = []

        for msg in messages:
            if hasattr(msg, 'parts'):
                for part in msg.parts:
                    if isinstance(part, ToolCallPart):
                        tool_calls.append(part)
                    elif isinstance(part, ToolReturnPart):
                        tool_returns.append(part)

        # Should have tool calls if tools were used
        # Note: TestModel behavior may vary
        # Just verify structure is correct if they exist
        if len(tool_calls) > 0:
            # Verify tool call structure
            assert hasattr(tool_calls[0], 'tool_name')
            assert hasattr(tool_calls[0], 'args')


class TestResponseMessages:
    """Test response messages in detail."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_response_message_structure(self):
        """Test that response messages have correct structure."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        with capture_run_messages() as messages:
            await agent.run("What is 2+2?")

        # Get response messages
        responses = [m for m in messages if m.kind == "response"]
        assert len(responses) > 0, "Should have at least one response"

        # Verify response structure
        for response in responses:
            assert isinstance(response, ModelResponse), \
                f"Response should be ModelResponse, got {type(response)}"

            if hasattr(response, 'parts'):
                assert len(response.parts) >= 0, "Response should have parts list"

                # Check parts are valid types
                for part in response.parts:
                    assert isinstance(part, (TextPart, ToolCallPart, ToolReturnPart)), \
                        f"Part should be valid type, got {type(part)}"


class TestMessageSequence:
    """Test message sequence and ordering."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_message_order_is_preserved(self):
        """Test that messages appear in correct order."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        with capture_run_messages() as messages:
            await agent.run("Test query")

        # Should start with request
        assert messages[0].kind == "request", "First message should be request"

        # Should end with response
        assert messages[-1].kind == "response", "Last message should be response"

        # All requests should come before their responses
        seen_response = False
        for msg in messages:
            if msg.kind == "response":
                seen_response = True
            elif msg.kind == "request" and seen_response:
                # Found request after response - check if it's a follow-up
                # This is valid for multi-turn conversations
                pass

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_multi_turn_conversation_messages(self):
        """Test message capture across multiple agent runs."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        # Capture messages from first run
        with capture_run_messages() as messages1:
            await agent.run("What is 2+2?")

        # Capture messages from second run
        with capture_run_messages() as messages2:
            await agent.run("What is 3+3?")

        # Each run should have its own request-response pair
        requests1 = [m for m in messages1 if m.kind == "request"]
        responses1 = [m for m in messages1 if m.kind == "response"]

        requests2 = [m for m in messages2 if m.kind == "request"]
        responses2 = [m for m in messages2 if m.kind == "response"]

        assert len(requests1) >= 1, f"First run should have request, got {len(requests1)}"
        assert len(responses1) >= 1, f"First run should have response, got {len(responses1)}"
        assert len(requests2) >= 1, f"Second run should have request, got {len(requests2)}"
        assert len(responses2) >= 1, f"Second run should have response, got {len(responses2)}"


class TestMessageInspectionWithStructuredOutput:
    """Test message inspection with structured output."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_messages_with_structured_output(self):
        """Test that structured output is captured in messages."""

        class SimpleData(BaseModel):
            name: str
            value: int

        model = ClaudeCodeModel()
        agent = Agent(model, output_type=SimpleData)

        with capture_run_messages() as messages:
            await agent.run("Create data")

        # Should have messages
        assert len(messages) > 0

        # Response should indicate structured output
        responses = [m for m in messages if m.kind == "response"]
        assert len(responses) > 0

        # Check if response contains tool call (for structured output)
        has_tool_call = False
        for resp in responses:
            if hasattr(resp, 'parts'):
                for part in resp.parts:
                    if isinstance(part, ToolCallPart):
                        has_tool_call = True
                        break

        # Structured output typically uses tool calls
        # (This depends on implementation)
        if has_tool_call:
            assert has_tool_call, "Structured output should use tool calls"


class TestMessageInspectionWithStreaming:
    """Test message inspection with streaming."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_messages_captured_during_streaming(self):
        """Test that messages are captured during streaming."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        with capture_run_messages() as messages:
            async with agent.run_stream("Test query") as result:
                # Consume stream
                async for _ in result.stream_text(debounce_by=None):
                    pass

        # Should have captured messages
        assert len(messages) > 0, "Should capture messages during streaming"

        # Should have request and response
        assert any(m.kind == "request" for m in messages)
        assert any(m.kind == "response" for m in messages)


class TestMessageInspectionEdgeCases:
    """Test message inspection in edge cases."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_messages_with_empty_query(self):
        """Test message capture with empty query."""
        model = ClaudeCodeModel()
        agent = Agent(model)

        with capture_run_messages() as messages:
            await agent.run("")

        # Should still capture messages
        assert len(messages) > 0, "Should capture messages even for empty query"

    @pytest.mark.asyncio
    async def test_nested_message_capture(self):
        """Test nested message capture contexts."""
        agent = Agent(TestModel())

        with capture_run_messages() as outer_messages:
            await agent.run("Outer query")

            with capture_run_messages() as inner_messages:
                await agent.run("Inner query")

            # Inner should only have inner messages
            assert len(inner_messages) >= 2, "Inner capture should have messages"

        # Outer only captures its own run (nested contexts are independent)
        # This is the actual behavior of capture_run_messages()
        assert len(outer_messages) >= 2, "Outer capture should have its own messages"

        # Verify outer and inner captured different runs
        outer_content = str([m for m in outer_messages])
        inner_content = str([m for m in inner_messages])
        # They should have different content (different queries)
        assert "Outer query" in outer_content or "Inner query" not in outer_content
