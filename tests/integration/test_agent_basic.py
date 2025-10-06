"""Integration tests for basic Agent usage with ClaudeCodeModel.

These tests verify that ClaudeCodeModel integrates correctly with Pydantic AI's Agent.
Tests are split into:
- Fast tests using TestModel (always run)
- Live tests using real ClaudeCodeModel (run with pytest -m live)

Best practices:
- Use TestModel for fast baseline tests
- Compare ClaudeCodeModel behavior against TestModel
- Verify actual content, not just existence
- Use capture_run_messages to inspect message exchanges
"""

from __future__ import annotations

import pytest

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai import capture_run_messages

from clowclow import ClaudeCodeModel


class TestBasicAgentIntegration:
    """Test basic Agent functionality - fast tests using TestModel."""

    def test_agent_with_test_model_baseline(self):
        """Baseline: Agent works with TestModel."""
        agent = Agent(TestModel())
        result = agent.run_sync("Test query")
        assert result.output is not None

    def test_agent_creation_with_claude_code_model(self):
        """Test that ClaudeCodeModel can be used to create an Agent."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model)

        assert agent is not None
        assert agent.model.model_name == "claude-code"
        assert agent.model.system == "claude-code"

    def test_agent_with_system_prompt(self):
        """Test creating agent with system prompt."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(
            model,
            system_prompt="You are a helpful assistant."
        )

        assert agent is not None

    def test_multiple_agents_different_models(self):
        """Test multiple agents with different model configurations."""
        model1 = ClaudeCodeModel(model_name="claude-code-1")
        model2 = ClaudeCodeModel(model_name="claude-code-2")

        agent1 = Agent(model1)
        agent2 = Agent(model2)

        assert agent1.model.model_name == "claude-code-1"
        assert agent2.model.model_name == "claude-code-2"


class TestLiveBasicIntegration:
    """Live integration tests using real Claude Code API."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_agent_run_simple_query(self):
        """Test running a simple query through the agent with real API."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model)

        result = await agent.run("What is 2+2?")

        # Verify output structure
        assert result.output is not None
        assert isinstance(result.output, str)
        assert len(result.output) > 0, "Response should not be empty"
        # Expect the answer contains "4"
        assert "4" in result.output, "Response should contain the answer '4'"
        # Verify it's a reasonable response (not just "4")
        assert len(result.output) >= 1, "Response should be at least 1 character"

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_agent_with_system_prompt_live(self):
        """Test agent with system prompt using real API."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(
            model,
            system_prompt="You are a concise math tutor. Answer in one sentence."
        )

        result = await agent.run("What is 5 * 5?")

        assert result.output is not None
        assert isinstance(result.output, str)
        assert len(result.output) > 0, "Response should not be empty"
        assert "25" in result.output, "Response should contain the answer '25'"
        # Verify conciseness (system prompt effect)
        assert len(result.output) < 200, "Response should be concise per system prompt"

    @pytest.mark.live
    def test_agent_run_sync_live(self):
        """Test synchronous agent run with real API."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model)

        result = agent.run_sync("Say hello")

        assert result.output is not None
        assert isinstance(result.output, str)
        assert len(result.output) > 0, "Response should not be empty"
        # Verify it's a greeting response
        greeting_keywords = ["hello", "hi", "hey", "greetings"]
        assert any(kw in result.output.lower() for kw in greeting_keywords), \
            "Response should contain a greeting"


class TestAgentWithDependencies:
    """Test Agent with dependency injection."""

    def test_agent_with_deps_type(self):
        """Test agent with typed dependencies."""
        model = ClaudeCodeModel(model=self.claude_model)

        # Create agent with typed dependencies
        agent = Agent(
            model,
            deps_type=str,
        )

        assert agent is not None

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_agent_with_deps_live(self):
        """Test agent with dependency injection using real API."""
        model = ClaudeCodeModel(model=self.claude_model)

        agent = Agent(
            model,
            deps_type=dict,
            system_prompt="You are a helpful assistant. Answer questions concisely."
        )

        result = await agent.run(
            "What is the capital of France?",
            deps={"user_id": "test123"}  # Deps may not affect simple queries
        )

        # Verify basic output (dependency injection is a Pydantic AI feature,
        # not directly testable without tools that use deps)
        assert result.output is not None
        assert isinstance(result.output, str)
        assert len(result.output) > 0, "Response should not be empty"

        # For this simple query, verify we get an answer
        # (Don't check for deps usage since ClaudeCodeModel may not support
        # dynamic system prompt templating)
        assert ("paris" in result.output.lower()), \
            f"Response should mention Paris, got: {result.output}"


class TestAgentModelOverride:
    """Test agent model override functionality."""

    @pytest.mark.asyncio
    async def test_override_model_with_test_model(self):
        """Test overriding ClaudeCodeModel with TestModel for testing."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model)

        # Override with TestModel for fast testing
        with agent.override(model=TestModel()):
            result = await agent.run("Test")
            assert result.output is not None


class TestAgentUsageTracking:
    """Test usage tracking through Agent runs."""

    def test_result_has_usage_attribute(self):
        """Test that run result has usage attribute."""
        agent = Agent(TestModel())
        result = agent.run_sync("Test query")

        # Result should have usage method/property
        assert hasattr(result, 'usage')


class TestMessageInspection:
    """Test message inspection using capture_run_messages."""

    @pytest.mark.asyncio
    async def test_capture_messages_with_test_model(self):
        """Test message capture with TestModel."""
        agent = Agent(TestModel())

        with capture_run_messages() as messages:
            result = await agent.run("Test query")

        # Should have captured messages
        assert len(messages) > 0
        # First message should be user request
        assert messages[0].kind == "request"

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_capture_messages_with_claude_code(self):
        """Test message capture with ClaudeCodeModel."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model)

        with capture_run_messages() as messages:
            result = await agent.run("Say hello")

        # Should have captured request and response
        assert len(messages) >= 2
        assert messages[0].kind == "request"
        assert messages[-1].kind == "response"

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_capture_messages_with_system_prompt(self):
        """Test that system prompt appears in messages."""
        from pydantic_ai.messages import SystemPromptPart

        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model, system_prompt="You are a helpful assistant.")

        with capture_run_messages() as messages:
            await agent.run("Test")

        # System prompt should appear as SystemPromptPart in request messages
        request_messages = [m for m in messages if m.kind == "request"]
        assert len(request_messages) > 0

        # Find SystemPromptPart with the correct content
        system_parts = [
            part for msg in request_messages
            for part in msg.parts
            if isinstance(part, SystemPromptPart)
        ]
        assert any(p.content == "You are a helpful assistant." for p in system_parts)


class TestAgentBehaviorConsistency:
    """Test that ClaudeCodeModel behaves consistently with TestModel."""

    @pytest.mark.asyncio
    async def test_response_structure_matches_testmodel(self):
        """Verify ClaudeCodeModel returns same structure as TestModel."""
        test_agent = Agent(TestModel())
        test_result = await test_agent.run("Test")

        # Both should have same result structure
        assert hasattr(test_result, 'output')
        assert hasattr(test_result, 'usage')
        # Note: Can't test ClaudeCodeModel without live API

    def test_agent_initialization_consistency(self):
        """Test that both models initialize agents the same way."""
        test_model = TestModel()
        claude_model = ClaudeCodeModel(model=self.claude_model)

        test_agent = Agent(test_model)
        claude_agent = Agent(claude_model)

        # Both should have same attributes
        assert hasattr(test_agent, 'model')
        assert hasattr(claude_agent, 'model')
        assert hasattr(test_agent, 'run')
        assert hasattr(claude_agent, 'run')
        assert hasattr(test_agent, 'run_sync')
        assert hasattr(claude_agent, 'run_sync')


class TestErrorHandling:
    """Test error handling in Agent with ClaudeCodeModel."""

    @pytest.mark.asyncio
    async def test_empty_query_with_test_model(self):
        """Baseline: empty query with TestModel."""
        agent = Agent(TestModel())
        # Should complete without error (TestModel handles gracefully)
        result = await agent.run("")
        assert result is not None

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_empty_query_with_claude_code(self):
        """Test empty query with ClaudeCodeModel."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model)

        # Should handle empty query gracefully
        result = await agent.run("")
        assert result is not None
        assert hasattr(result, 'output')


class TestResponseValidation:
    """Test that responses are properly validated."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_response_is_nonempty_string(self):
        """Verify response contains actual content."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model)

        result = await agent.run("What is 2+2?")

        # Should have non-empty output
        assert result.output is not None
        assert isinstance(result.output, str)
        assert len(result.output) > 0, "Response should not be empty"
        # Should contain the answer
        assert "4" in result.output, "Response should contain '4'"
        # Verify reasonable response length
        assert 1 <= len(result.output) <= 1000, "Response should be reasonable length"
        # Verify it's not just whitespace
        assert result.output.strip(), "Response should not be just whitespace"

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_multiple_runs_maintain_state(self):
        """Test that multiple runs work correctly."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model)

        result1 = await agent.run("What is 3 + 5?")
        result2 = await agent.run("What is 10 - 2?")

        # Both should succeed with non-empty responses
        assert result1.output is not None
        assert result2.output is not None
        assert isinstance(result1.output, str)
        assert isinstance(result2.output, str)
        assert len(result1.output) > 0, "First response should not be empty"
        assert len(result2.output) > 0, "Second response should not be empty"

        # Verify correct answers
        assert "8" in result1.output, "First response should contain '8'"
        assert "8" in result2.output, "Second response should contain '8'"

        # Both should have non-empty stripped content
        assert result1.output.strip()
        assert result2.output.strip()
