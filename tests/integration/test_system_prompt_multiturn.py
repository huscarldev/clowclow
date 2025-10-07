"""Test to verify system prompts are passed correctly in multi-turn conversations."""

import pytest
from pydantic_ai import Agent
from clowclow import ClaudeCodeModel


class TestSystemPromptMultiTurn:
    """Test system prompt handling in multi-turn conversations."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_system_prompt_in_first_turn(self):
        """Verify system prompt is used in first turn."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(
            model,
            system_prompt="Always start your response with 'SYSTEM_TEST:'"
        )

        result = await agent.run("What is 2+2?")
        print(f"\nFirst turn response: {result.output}")

        # Check if system prompt was applied
        assert "SYSTEM_TEST:" in result.output or result.output.startswith("SYSTEM_TEST:")

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_system_prompt_in_second_turn(self):
        """Verify system prompt is still applied in second turn."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(
            model,
            system_prompt="Always start your response with 'SYSTEM_TEST:'"
        )

        # First turn
        result1 = await agent.run("What is 2+2?")
        print(f"\nFirst turn response: {result1.output}")

        # Second turn
        result2 = await agent.run(
            "What is 3+3?",
            message_history=result1.new_messages()
        )
        print(f"\nSecond turn response: {result2.output}")

        # System prompt should still be applied in second turn
        assert "SYSTEM_TEST:" in result2.output or result2.output.startswith("SYSTEM_TEST:")

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_system_prompt_extraction_from_messages(self):
        """Test that system prompts are extracted correctly from message history."""
        from clowclow.request_handler import RequestHandler
        from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart

        # Create mock message history with system prompt
        messages = [
            ModelRequest(parts=[
                SystemPromptPart(content="You are a helpful assistant."),
                UserPromptPart(content="Hello")
            ])
        ]

        system_prompt = RequestHandler.extract_system_messages(messages)
        print(f"\nExtracted system prompt: {system_prompt}")

        assert system_prompt == "You are a helpful assistant."

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_system_prompt_with_conversation_history(self):
        """Verify system prompt and conversation history both work together."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(
            model,
            system_prompt="You are a math tutor. Always show your work."
        )

        # First turn
        result1 = await agent.run("What is 5 + 3?")
        print(f"\nFirst turn: {result1.output}")

        # Second turn - should maintain both system prompt AND previous context
        result2 = await agent.run(
            "Now add 2 to that result.",
            message_history=result1.new_messages()
        )
        print(f"\nSecond turn: {result2.output}")

        # Should understand "that result" refers to 8 from first turn
        assert "10" in result2.output
