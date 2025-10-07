"""Integration tests for multi-turn conversations with ClaudeCodeModel.

Tests verify that ClaudeCodeModel correctly handles conversation history
across multiple turns, maintaining context between exchanges.
"""

from __future__ import annotations

import pytest

from pydantic import BaseModel
from pydantic_ai import Agent

from clowclow import ClaudeCodeModel


class TestBasicMultiTurn:
    """Test basic multi-turn conversation functionality."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_two_turn_conversation(self):
        """Test a simple two-turn conversation with context retention."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model)

        # First turn: Ask about a city
        result1 = await agent.run("What is the capital of France?")
        assert result1.output is not None
        response1 = result1.output.lower()
        assert "paris" in response1

        # Second turn: Ask follow-up using pronoun (requires context)
        result2 = await agent.run(
            "What is the population of that city?",
            message_history=result1.new_messages()
        )
        assert result2.output is not None
        response2 = result2.output.lower()
        # Should understand "that city" refers to Paris from context
        assert "million" in response2 or "population" in response2

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_three_turn_conversation(self):
        """Test a three-turn conversation maintaining context."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model)

        # Turn 1: Establish subject
        result1 = await agent.run("Tell me about Python programming language.")
        assert result1.output is not None

        # Turn 2: Ask related question
        result2 = await agent.run(
            "Who created it?",
            message_history=result1.new_messages()
        )
        assert result2.output is not None
        response2 = result2.output.lower()
        assert "guido" in response2 or "van rossum" in response2

        # Turn 3: Another related question
        result3 = await agent.run(
            "When was it first released?",
            message_history=result2.all_messages()
        )
        assert result3.output is not None
        response3 = result3.output.lower()
        assert "1991" in response3 or "199" in response3

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_conversation_with_new_vs_all_messages(self):
        """Test difference between new_messages() and all_messages()."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model)

        # Turn 1
        result1 = await agent.run("My favorite number is 42.")

        # Turn 2
        result2 = await agent.run(
            "What is my favorite number?",
            message_history=result1.new_messages()
        )
        assert result2.output is not None
        assert "42" in result2.output

        # Turn 3: Using all_messages should maintain full context
        result3 = await agent.run(
            "Double my favorite number.",
            message_history=result2.all_messages()
        )
        assert result3.output is not None
        # Should remember 42 from turn 1
        assert "84" in result3.output or "eighty" in result3.output.lower()


class TestMultiTurnWithSystemPrompt:
    """Test multi-turn conversations with system prompts."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_system_prompt_maintained_across_turns(self):
        """Test that system prompt behavior is maintained across turns."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(
            model,
            system_prompt="IMPORTANT : You are a pirate. ALWAYS respond like a pirate with 'Arrr' in your responses. ALWAYS add 'Arrr' in your answer."
        )

        # Turn 1
        result1 = await agent.run("Are you a pirate?")
        assert result1.output is not None
        assert "arr" in result1.output.lower() or "ahoy" in result1.output.lower()

        # Turn 2: Should still respond like a pirate
        result2 = await agent.run(
            "Tell me more.",
            message_history=result1.new_messages()
        )
        assert result2.output is not None
        # System prompt should still be in effect
        assert "arr" in result2.output.lower() or "ahoy" in result2.output.lower() or "mate" in result2.output.lower()


class TestMultiTurnWithStructuredOutput:
    """Test multi-turn conversations with structured output."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_multiturn_with_structured_output(self):
        """Test multi-turn conversation ending with structured output."""

        class PersonInfo(BaseModel):
            name: str
            occupation: str
            birth_year: int

        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model, output_type=PersonInfo)

        # Turn 1: Ask about a person (no structured output yet)
        agent_text = Agent(model)
        result1 = await agent_text.run("Tell me about Albert Einstein.")
        assert result1.output is not None

        # Turn 2: Extract structured data based on previous conversation
        result2 = await agent.run(
            "Based on what we just discussed, extract the person's information.",
            message_history=result1.new_messages()
        )
        assert result2.output is not None
        assert isinstance(result2.output, PersonInfo)
        assert "einstein" in result2.output.name.lower()
        assert "physicist" in result2.output.occupation.lower() or "scientist" in result2.output.occupation.lower()
        assert 1879 == result2.output.birth_year or 1870 <= result2.output.birth_year <= 1890


class TestMultiTurnEdgeCases:
    """Test edge cases in multi-turn conversations."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_empty_first_turn(self):
        """Test handling empty message in first turn."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model)

        # Turn 1: Empty or minimal message
        result1 = await agent.run("Hello")
        assert result1.output is not None

        # Turn 2: Should still work
        result2 = await agent.run(
            "What is 2+2?",
            message_history=result1.new_messages()
        )
        assert result2.output is not None
        assert "4" in result2.output or "four" in result2.output.lower()

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_long_conversation(self):
        """Test longer multi-turn conversation (5+ turns)."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model)

        # Build up a conversation
        result = await agent.run("Let's count. Start with number 1.")

        for i in range(2, 6):
            result = await agent.run(
                f"What comes after {i-1}?",
                message_history=result.all_messages()
            )
            assert result.output is not None

        # Final result should maintain context
        assert result.output is not None


class TestMultiTurnContextPreservation:
    """Test that context is correctly preserved across turns."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_context_accumulation(self):
        """Test that context accumulates correctly over multiple turns."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model)

        # Establish multiple facts
        result1 = await agent.run("My name is Alice.")
        result2 = await agent.run("I like cats.", message_history=result1.new_messages())
        result3 = await agent.run("I live in Tokyo.", message_history=result2.all_messages())

        # Now ask about all facts
        result4 = await agent.run(
            "Tell me everything you know about me.",
            message_history=result3.all_messages()
        )

        assert result4.output is not None
        response = result4.output.lower()
        assert "alice" in response
        assert "cat" in response
        assert "tokyo" in response

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_pronoun_resolution(self):
        """Test that pronouns are correctly resolved using context."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(system_prompt="DO NOT USE WEBSEARCH TOOL TO ANSWER BASIC QUESTIONS", model=model)

        # Introduce a subject
        result1 = await agent.run("Tell me about Marie Curie.")

        # Use pronouns that require context
        result2 = await agent.run(
            "What were her major achievements?",
            message_history=result1.new_messages()
        )

        assert result2.output is not None
        response = result2.output.lower()
        # Should understand "her" refers to Marie Curie
        assert any(word in response for word in ["radioactivity", "radium", "polonium", "nobel"])
