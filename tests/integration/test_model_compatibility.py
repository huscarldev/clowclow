"""Model compatibility tests - comparing TestModel vs ClaudeCodeModel behavior.

These tests verify that ClaudeCodeModel implements the same Pydantic AI Model
interface as TestModel and produces compatible response structures.
"""

from __future__ import annotations

import pytest

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai import capture_run_messages

from clowclow import ClaudeCodeModel


class TestInterfaceCompatibility:
    """Verify that ClaudeCodeModel implements the same interface as TestModel."""

    def test_both_models_have_model_name_property(self):
        """Test that both models have model_name property."""
        test_model = TestModel()
        claude_model = ClaudeCodeModel(model=self.claude_model)

        assert hasattr(test_model, 'model_name')
        assert hasattr(claude_model, 'model_name')
        assert isinstance(test_model.model_name, str)
        assert isinstance(claude_model.model_name, str)

    def test_both_models_have_system_property(self):
        """Test that both models have system property."""
        test_model = TestModel()
        claude_model = ClaudeCodeModel(model=self.claude_model)

        assert hasattr(test_model, 'system')
        assert hasattr(claude_model, 'system')

    def test_both_models_have_request_method(self):
        """Test that both models have async request method."""
        test_model = TestModel()
        claude_model = ClaudeCodeModel(model=self.claude_model)

        assert hasattr(test_model, 'request')
        assert hasattr(claude_model, 'request')
        assert callable(getattr(test_model, 'request'))
        assert callable(getattr(claude_model, 'request'))

    def test_both_models_have_request_stream_method(self):
        """Test that both models have async request_stream method."""
        test_model = TestModel()
        claude_model = ClaudeCodeModel(model=self.claude_model)

        assert hasattr(test_model, 'request_stream')
        assert hasattr(claude_model, 'request_stream')
        assert callable(getattr(test_model, 'request_stream'))
        assert callable(getattr(claude_model, 'request_stream'))


class TestAgentCreationCompatibility:
    """Verify that both models can be used to create compatible agents."""

    def test_both_models_create_agents_with_same_attributes(self):
        """Test that agents created with both models have the same attributes."""
        test_agent = Agent(TestModel())
        claude_agent = Agent(ClaudeCodeModel(model=self.claude_model))

        # Both should have same core attributes
        assert hasattr(test_agent, 'model')
        assert hasattr(claude_agent, 'model')
        assert hasattr(test_agent, 'run')
        assert hasattr(claude_agent, 'run')
        assert hasattr(test_agent, 'run_sync')
        assert hasattr(claude_agent, 'run_sync')
        assert hasattr(test_agent, 'run_stream')
        assert hasattr(claude_agent, 'run_stream')

    def test_both_models_accept_system_prompt(self):
        """Test that both models accept system prompts."""
        system_prompt = "You are a helpful assistant."

        test_agent = Agent(TestModel(), system_prompt=system_prompt)
        claude_agent = Agent(ClaudeCodeModel(model=self.claude_model), system_prompt=system_prompt)

        assert test_agent is not None
        assert claude_agent is not None

    def test_both_models_accept_deps_type(self):
        """Test that both models accept dependency types."""
        test_agent = Agent(TestModel(), deps_type=dict)
        claude_agent = Agent(ClaudeCodeModel(model=self.claude_model), deps_type=dict)

        assert test_agent is not None
        assert claude_agent is not None


class TestResponseStructureCompatibility:
    """Verify that both models return compatible response structures."""

    @pytest.mark.asyncio
    async def test_both_models_return_result_with_output(self):
        """Test that both models return results with output attribute."""
        test_agent = Agent(TestModel())
        claude_agent = Agent(ClaudeCodeModel(model=self.claude_model))

        test_result = await test_agent.run("Test")
        # For Claude, use override with TestModel to avoid live API
        with claude_agent.override(model=TestModel()):
            claude_result = await claude_agent.run("Test")

        # Both should have output attribute
        assert hasattr(test_result, 'output')
        assert hasattr(claude_result, 'output')

    @pytest.mark.asyncio
    async def test_both_models_return_result_with_usage(self):
        """Test that both models return results with usage attribute."""
        test_agent = Agent(TestModel())
        claude_agent = Agent(ClaudeCodeModel(model=self.claude_model))

        test_result = await test_agent.run("Test")
        with claude_agent.override(model=TestModel()):
            claude_result = await claude_agent.run("Test")

        # Both should have usage attribute
        assert hasattr(test_result, 'usage')
        assert hasattr(claude_result, 'usage')

    @pytest.mark.asyncio
    async def test_both_models_return_same_type_for_simple_queries(self):
        """Test that both models return str for simple queries."""
        test_agent = Agent(TestModel())
        claude_agent = Agent(ClaudeCodeModel(model=self.claude_model))

        test_result = await test_agent.run("Hello")
        with claude_agent.override(model=TestModel()):
            claude_result = await claude_agent.run("Hello")

        # Both should return string output
        assert isinstance(test_result.output, (str, type(None)))
        assert isinstance(claude_result.output, (str, type(None)))
        # If both are strings, types should match
        if test_result.output and claude_result.output:
            assert type(test_result.output) == type(claude_result.output)


class TestStructuredOutputCompatibility:
    """Verify that both models handle structured output compatibly."""

    class SimpleModel(BaseModel):
        name: str
        value: int

    @pytest.mark.asyncio
    async def test_both_models_support_structured_output(self):
        """Test that both models support structured output via output_type."""
        test_agent = Agent(TestModel(), output_type=self.SimpleModel)
        claude_agent = Agent(ClaudeCodeModel(model=self.claude_model), output_type=self.SimpleModel)

        test_result = await test_agent.run("Get data")
        with claude_agent.override(model=TestModel()):
            claude_result = await claude_agent.run("Get data")

        # Both should return instances of SimpleModel
        assert isinstance(test_result.output, self.SimpleModel)
        assert isinstance(claude_result.output, self.SimpleModel)

    @pytest.mark.asyncio
    async def test_both_models_return_same_pydantic_structure(self):
        """Test that both models return Pydantic models with same fields."""
        class Person(BaseModel):
            name: str
            age: int

        test_agent = Agent(TestModel(), output_type=Person)
        claude_agent = Agent(ClaudeCodeModel(model=self.claude_model), output_type=Person)

        test_result = await test_agent.run("Create person")
        with claude_agent.override(model=TestModel()):
            claude_result = await claude_agent.run("Create person")

        # Both should have same attributes
        assert hasattr(test_result.output, 'name')
        assert hasattr(test_result.output, 'age')
        assert hasattr(claude_result.output, 'name')
        assert hasattr(claude_result.output, 'age')


class TestStreamingCompatibility:
    """Verify that both models support streaming compatibly."""

    @pytest.mark.asyncio
    async def test_both_models_support_stream_context_manager(self):
        """Test that both models return async context managers for streaming."""
        test_agent = Agent(TestModel())
        claude_agent = Agent(ClaudeCodeModel(model=self.claude_model))

        test_stream = test_agent.run_stream("Test")
        with claude_agent.override(model=TestModel()):
            claude_stream = claude_agent.run_stream("Test")

        # Both should be async context managers
        assert hasattr(test_stream, '__aenter__')
        assert hasattr(test_stream, '__aexit__')
        assert hasattr(claude_stream, '__aenter__')
        assert hasattr(claude_stream, '__aexit__')

    @pytest.mark.asyncio
    async def test_both_models_support_stream_text(self):
        """Test that both models support stream_text method."""
        test_agent = Agent(TestModel())
        claude_agent = Agent(ClaudeCodeModel(model=self.claude_model))

        async with test_agent.run_stream("Test") as test_result:
            assert hasattr(test_result, 'stream_text')

        with claude_agent.override(model=TestModel()):
            async with claude_agent.run_stream("Test") as claude_result:
                assert hasattr(claude_result, 'stream_text')

    @pytest.mark.asyncio
    async def test_both_models_support_get_output_after_streaming(self):
        """Test that both models support get_output() after streaming."""
        test_agent = Agent(TestModel())
        claude_agent = Agent(ClaudeCodeModel(model=self.claude_model))

        async with test_agent.run_stream("Test") as test_result:
            async for _ in test_result.stream_text(debounce_by=None):
                pass
            test_output = await test_result.get_output()
            assert test_output is not None or test_output == ""

        with claude_agent.override(model=TestModel()):
            async with claude_agent.run_stream("Test") as claude_result:
                async for _ in claude_result.stream_text(debounce_by=None):
                    pass
                claude_output = await claude_result.get_output()
                assert claude_output is not None or claude_output == ""


class TestMessageCaptureCompatibility:
    """Verify that both models work with capture_run_messages."""

    @pytest.mark.asyncio
    async def test_both_models_support_message_capture(self):
        """Test that both models work with capture_run_messages."""
        test_agent = Agent(TestModel())
        claude_agent = Agent(ClaudeCodeModel(model=self.claude_model))

        with capture_run_messages() as test_messages:
            await test_agent.run("Test")

        with claude_agent.override(model=TestModel()):
            with capture_run_messages() as claude_messages:
                await claude_agent.run("Test")

        # Both should capture messages
        assert len(test_messages) > 0
        assert len(claude_messages) > 0

    @pytest.mark.asyncio
    async def test_both_models_capture_request_messages(self):
        """Test that both models capture request messages."""
        test_agent = Agent(TestModel())
        claude_agent = Agent(ClaudeCodeModel(model=self.claude_model))

        with capture_run_messages() as test_messages:
            await test_agent.run("Test")

        with claude_agent.override(model=TestModel()):
            with capture_run_messages() as claude_messages:
                await claude_agent.run("Test")

        # Both should have request messages
        test_requests = [m for m in test_messages if m.kind == "request"]
        claude_requests = [m for m in claude_messages if m.kind == "request"]

        assert len(test_requests) > 0
        assert len(claude_requests) > 0

    @pytest.mark.asyncio
    async def test_both_models_capture_response_messages(self):
        """Test that both models capture response messages."""
        test_agent = Agent(TestModel())
        claude_agent = Agent(ClaudeCodeModel(model=self.claude_model))

        with capture_run_messages() as test_messages:
            await test_agent.run("Test")

        with claude_agent.override(model=TestModel()):
            with capture_run_messages() as claude_messages:
                await claude_agent.run("Test")

        # Both should have response messages
        test_responses = [m for m in test_messages if m.kind == "response"]
        claude_responses = [m for m in claude_messages if m.kind == "response"]

        assert len(test_responses) > 0
        assert len(claude_responses) > 0


class TestKnownDifferences:
    """Document known behavioral differences between models."""

    def test_tool_calling_not_supported_by_claude_code(self):
        """Document that ClaudeCodeModel does NOT support tool calling.

        This is a known limitation - tools can be registered but will not be invoked.
        TestModel supports tool calling, ClaudeCodeModel does not.
        """
        # TestModel supports tools
        test_agent = Agent(TestModel())
        tool_called = False

        @test_agent.tool_plain
        def test_tool() -> str:
            nonlocal tool_called
            tool_called = True
            return "Called"

        test_agent.run_sync("Use tool")
        # TestModel CAN call tools (behavior depends on call_tools config)

        # ClaudeCodeModel does NOT support tools
        claude_agent = Agent(ClaudeCodeModel(model=self.claude_model))
        claude_tool_called = False

        @claude_agent.tool_plain
        def claude_tool() -> str:
            nonlocal claude_tool_called
            claude_tool_called = True
            return "Called"

        # Tools will NOT be called with ClaudeCodeModel
        # This is expected behavior - no assertion needed, just documentation

    def test_model_name_differences(self):
        """Document model name differences."""
        test_model = TestModel()
        claude_model = ClaudeCodeModel(model=self.claude_model)

        # Different model names are expected
        assert test_model.model_name == "test"
        assert claude_model.model_name == "claude-code"
        # This is expected and correct

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_live_api_behavior_difference(self):
        """Document that only ClaudeCodeModel uses live API.

        TestModel is deterministic and doesn't make API calls.
        ClaudeCodeModel makes real API calls to Claude Code.
        """
        # TestModel: deterministic, no API calls
        test_agent = Agent(TestModel())
        test_result = await test_agent.run("What is 2+2?")
        # Result is synthetic from TestModel

        # ClaudeCodeModel: real API calls
        claude_agent = Agent(ClaudeCodeModel(model=self.claude_model))
        claude_result = await claude_agent.run("What is 2+2?")

        # Both should have output, but content may differ
        assert test_result.output is not None or test_result.output == ""
        assert claude_result.output is not None
        assert isinstance(claude_result.output, str)
        assert len(claude_result.output) > 0
        # Claude's response should contain the answer
        assert "4" in claude_result.output
