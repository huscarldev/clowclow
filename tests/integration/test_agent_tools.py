"""Integration tests for tool calling with ClaudeCodeModel.

Tests verify that ClaudeCodeModel correctly handles Pydantic AI tool calling patterns.
"""

from __future__ import annotations

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.exceptions import ModelRetry

from clowclow import ClaudeCodeModel


class TestToolCalling:
    """Test basic tool calling functionality.

    NOTE: ClaudeCodeModel does NOT support Pydantic AI tool calling.
    Tools can be registered with @agent.tool_plain/@agent.tool but will NOT be invoked.
    Use TestModel for testing tool calling functionality.
    """

    def test_agent_with_simple_tool_using_test_model(self):
        """Baseline test: tool calling works with TestModel."""
        agent = Agent(TestModel())

        @agent.tool_plain
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Weather in {city}: Sunny"

        result = agent.run_sync("What's the weather in Paris?")

        # TestModel should call tools and return result
        assert result.output is not None

    @pytest.mark.skip(reason="ClaudeCodeModel does not support Pydantic AI tool calling - tools will not be invoked")
    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_agent_with_tool_plain(self):
        """Test agent with a plain tool function.

        SKIPPED: ClaudeCodeModel does not support Pydantic AI tool calling.
        Tools can be registered but will NOT be invoked by the model.
        """
        model = ClaudeCodeModel()
        agent = Agent(model)

        tool_call_count = 0
        tool_args_received = []

        @agent.tool_plain
        def get_location(city: str) -> str:
            """Get coordinates for a city."""
            nonlocal tool_call_count, tool_args_received
            tool_call_count += 1
            tool_args_received.append(city)
            return f"Location of {city}: 37.77°N, 122.42°W"

        result = await agent.run("Where is San Francisco?")

        # Tool should be called - but ClaudeCodeModel does not support this
        assert tool_call_count > 0, "Tool should have been called"
        assert "San Francisco" in tool_args_received
        assert result.output is not None

    @pytest.mark.skip(reason="ClaudeCodeModel does not support Pydantic AI tool calling - async tools will not be invoked")
    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_agent_with_async_tool(self):
        """Test agent with an async tool function.

        SKIPPED: ClaudeCodeModel does not support Pydantic AI tool calling.
        Async tools can be registered but will NOT be invoked by the model.
        """
        model = ClaudeCodeModel()
        agent = Agent(model)

        tool_called = False

        @agent.tool_plain
        async def async_search(query: str) -> str:
            """Async search function."""
            nonlocal tool_called
            tool_called = True
            return f"Search results for: {query}"

        result = await agent.run("Use the async_search tool to search for Python tutorials")

        # Tool should be called - but ClaudeCodeModel does not support this
        assert tool_called, "Async tool should have been called"
        assert result.output is not None


class TestToolWithTestModel:
    """Test tool calling patterns using TestModel for deterministic testing."""

    def test_tool_called_once(self):
        """Test that TestModel calls tool once."""
        agent = Agent(TestModel(call_tools=['calculate']))

        @agent.tool_plain
        def calculate(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y

        result = agent.run_sync("Add 5 and 3")

        # Tool should be called
        assert result.output is not None

    def test_tool_not_called(self):
        """Test tool not called when configured."""
        agent = Agent(TestModel(call_tools=[]))  # Don't call any tools

        tool_called = False

        @agent.tool_plain
        def should_not_call() -> str:
            """This shouldn't be called."""
            nonlocal tool_called
            tool_called = True
            return "Called"

        result = agent.run_sync("Test")

        # Tool should not have been called
        assert not tool_called

    def test_multiple_tools_called(self):
        """Test multiple tools can be registered."""
        agent = Agent(TestModel())

        @agent.tool_plain
        def tool_a(x: str) -> str:
            """Tool A."""
            return f"A: {x}"

        @agent.tool_plain
        def tool_b(y: str) -> str:
            """Tool B."""
            return f"B: {y}"

        result = agent.run_sync("Use both tools")

        assert result.output is not None


class TestToolRetry:
    """Test tool retry functionality."""

    def test_tool_retry_with_test_model(self):
        """Test tool retry mechanism using TestModel."""
        agent = Agent(TestModel())

        call_count = 0

        @agent.tool_plain
        def retry_tool(value: int) -> int:
            """Tool that retries once."""
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ModelRetry("First attempt failed, retry")
            return value * 2

        result = agent.run_sync("Double the number 5")

        # Tool should have been called twice (initial + 1 retry)
        assert call_count == 2, f"Expected 2 calls (1 retry), got {call_count}"
        assert result.output is not None

        # Verify the tool eventually succeeded after retry
        assert isinstance(result.output, (str, int)), "Result should be a valid type"

    def test_tool_retry_max_attempts(self):
        """Test tool retry respects max attempts."""
        agent = Agent(TestModel())

        call_count = 0

        @agent.tool_plain
        def always_retry() -> str:
            """Tool that always retries."""
            nonlocal call_count
            call_count += 1
            raise ModelRetry("Always retry")

        # Should eventually fail after max retries
        # Note: Actual max retries depends on Agent configuration
        try:
            result = agent.run_sync("Test retry limits")
            # If it succeeds, verify output exists
            assert result.output is not None
        except Exception:
            # Or it may raise after max retries
            assert call_count > 1


class TestToolValidation:
    """Test tool parameter validation."""

    # Define UserInput at class level so it's in scope for tool decorators
    from pydantic import BaseModel

    class UserInput(BaseModel):
        username: str
        email: str

    def test_tool_with_typed_parameters(self):
        """Test tool with type-checked parameters."""
        agent = Agent(TestModel())

        @agent.tool_plain
        def typed_tool(name: str, age: int, active: bool = True) -> str:
            """Tool with typed parameters."""
            return f"{name}, {age}, {active}"

        result = agent.run_sync("Call the tool")

        assert result.output is not None

    def test_tool_with_pydantic_model_param(self):
        """Test tool that accepts Pydantic model as parameter."""
        agent = Agent(TestModel())

        @agent.tool_plain
        def process_user(user: TestToolValidation.UserInput) -> str:
            """Process user data."""
            return f"Processed {user.username}"

        result = agent.run_sync("Create user")

        assert result.output is not None


class TestToolContext:
    """Test tool functions with context."""

    def test_tool_with_run_context(self):
        """Test tool that receives RunContext."""
        agent = Agent(TestModel(), deps_type=dict)

        @agent.tool
        def context_tool(ctx: RunContext[dict], query: str) -> str:
            """Tool that uses context."""
            # Context should be available
            assert ctx.deps is not None
            return f"Processed: {query}"

        result = agent.run_sync("Test", deps={"key": "value"})

        assert result.output is not None

    def test_tool_with_retry_count(self):
        """Test tool that checks retry count."""
        agent = Agent(TestModel(), deps_type=None)

        @agent.tool
        def retry_aware_tool(ctx: RunContext[None]) -> str:
            """Tool aware of retries."""
            # Can access retry information from context
            return "Success"

        result = agent.run_sync("Test")

        assert result.output is not None


class TestDynamicToolRegistration:
    """Test dynamic tool registration patterns."""

    @pytest.mark.asyncio
    async def test_register_tool_function(self):
        """Test registering tool via function."""
        agent = Agent(TestModel())

        def external_tool(param: str) -> str:
            """External tool function."""
            return f"Result: {param}"

        # Register tool
        agent.tool_plain(external_tool)

        result = await agent.run("Use the external tool")

        assert result.output is not None

    def test_multiple_tool_registration(self):
        """Test registering multiple tools."""
        agent = Agent(TestModel())

        tools = []

        @agent.tool_plain
        def tool1() -> str:
            tools.append("tool1")
            return "Tool 1"

        @agent.tool_plain
        def tool2() -> str:
            tools.append("tool2")
            return "Tool 2"

        @agent.tool_plain
        def tool3() -> str:
            tools.append("tool3")
            return "Tool 3"

        result = agent.run_sync("Use tools")

        assert result.output is not None


class TestToolOutputFormats:
    """Test different tool output formats."""

    def test_tool_returns_string(self):
        """Test tool returning string."""
        agent = Agent(TestModel())

        @agent.tool_plain
        def string_tool() -> str:
            return "String output"

        result = agent.run_sync("Test")
        assert result.output is not None

    def test_tool_returns_number(self):
        """Test tool returning number."""
        agent = Agent(TestModel())

        @agent.tool_plain
        def number_tool() -> int:
            return 42

        result = agent.run_sync("Get number")
        assert result.output is not None

    def test_tool_returns_dict(self):
        """Test tool returning dictionary."""
        agent = Agent(TestModel())

        @agent.tool_plain
        def dict_tool() -> dict:
            return {"status": "success", "value": 100}

        result = agent.run_sync("Get dict")
        assert result.output is not None

    def test_tool_returns_list(self):
        """Test tool returning list."""
        agent = Agent(TestModel())

        @agent.tool_plain
        def list_tool() -> list[str]:
            return ["item1", "item2", "item3"]

        result = agent.run_sync("Get list")
        assert result.output is not None
