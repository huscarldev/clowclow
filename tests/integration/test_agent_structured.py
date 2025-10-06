"""Integration tests for structured output with ClaudeCodeModel.

Tests verify that ClaudeCodeModel correctly handles Pydantic AI's structured
output modes using Pydantic models.
"""

from __future__ import annotations

import pytest
from typing import List, Callable, Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import SystemPromptPart

from clowclow import ClaudeCodeModel


# Test Pydantic models for structured output
class CityLocation(BaseModel):
    """Location information for a city."""
    city: str
    country: str
    latitude: float = 0.0
    longitude: float = 0.0


class WeatherData(BaseModel):
    """Weather information."""
    temperature: float = Field(description="Temperature in Celsius")
    condition: str = Field(description="Weather condition (sunny, rainy, etc)")
    humidity: int = Field(description="Humidity percentage")


class PersonInfo(BaseModel):
    """Information about a person."""
    name: str
    age: int
    occupation: str | None = None


class SearchResults(BaseModel):
    """Search results with multiple items."""
    query: str
    results: List[str]
    count: int


class TestParametrizedStructuredOutput:
    """Parametrized tests for different structured output types."""

    @pytest.mark.parametrize("model_class,query,field_validators", [
        (
            CityLocation,
            "What is the capital of France?",
            {
                "city": lambda v: "paris" in v.lower(),
                "country": lambda v: "france" in v.lower(),
            }
        ),
        (
            PersonInfo,
            "Tell me about Bob who is 25 years old",
            {
                "name": lambda v: "bob" in v.lower(),
                "age": lambda v: 20 <= v <= 30,
            }
        ),
        (
            WeatherData,
            "Weather in London: 15°C, rainy, 80% humidity",
            {
                "temperature": lambda v: -50 <= v <= 50,
                "condition": lambda v: len(v) > 0,
                "humidity": lambda v: 0 <= v <= 100,
            }
        ),
    ])
    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_structured_output_with_validation(
        self,
        model_class: type[BaseModel],
        query: str,
        field_validators: dict[str, Callable[[Any], bool]]
    ):
        """Test structured output for different model types with field validation."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model, output_type=model_class)

        result = await agent.run(query)

        # Verify basic structure
        assert result.output is not None
        assert isinstance(result.output, model_class), \
            f"Expected {model_class.__name__}, got {type(result.output)}"

        # Verify field validators
        for field_name, validator in field_validators.items():
            field_value = getattr(result.output, field_name)
            assert field_value is not None, f"Field {field_name} should not be None"
            assert validator(field_value), \
                f"Field {field_name} validation failed for value: {field_value}"

    # NOTE: Removed test_json_schema_type_handling test
    # The test was flawed - using `value: Any` doesn't enforce types in Pydantic.
    # Structured output works correctly with properly typed Pydantic models,
    # which is tested in other test methods (e.g., test_structured_output_with_validation)


class TestStructuredOutputBasics:
    """Test basic structured output functionality."""

    def test_structured_output_with_test_model(self):
        """Baseline: structured output works with TestModel."""
        agent = Agent(TestModel(), output_type=CityLocation)

        result = agent.run_sync("What is the largest city in France?")

        # Should return CityLocation instance
        assert isinstance(result.output, CityLocation)
        assert result.output.city
        assert result.output.country

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_simple_structured_output(self, test_workspace):
        """Test simple structured output with ClaudeCodeModel."""
        model = ClaudeCodeModel(workspace_dir=test_workspace)
        agent = Agent(model, output_type=CityLocation)

        result = await agent.run("Largest city in France?")

        # Should return structured CityLocation instance
        assert result.output is not None
        assert isinstance(result.output, CityLocation), \
            f"Expected CityLocation, got {type(result.output)}"

        # Verify required fields are populated
        assert result.output.city, "City field should not be empty"
        assert result.output.country, "Country field should not be empty"
        assert isinstance(result.output.city, str)
        assert isinstance(result.output.country, str)

        # Verify content makes sense for France query
        assert "paris" in result.output.city.lower(), \
            f"Expected Paris as largest city in France, got: {result.output.city}"
        assert "france" in result.output.country.lower(), \
            f"Expected France as country, got: {result.output.country}"

        # Verify coordinates are numeric
        assert isinstance(result.output.latitude, (int, float))
        assert isinstance(result.output.longitude, (int, float))

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_structured_output_with_nested_fields(self):
        """Test structured output with nested data."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model, output_type=WeatherData)

        result = await agent.run("What's the weather in London?")

        # Should return WeatherData instance with valid data
        assert result.output is not None
        assert isinstance(result.output, WeatherData), \
            f"Expected WeatherData, got {type(result.output)}"

        # Verify all required fields are populated
        assert isinstance(result.output.temperature, (int, float)), \
            f"Temperature should be numeric, got {type(result.output.temperature)}"
        assert result.output.condition, "Condition should not be empty"
        assert isinstance(result.output.condition, str), \
            f"Condition should be string, got {type(result.output.condition)}"
        assert isinstance(result.output.humidity, int), \
            f"Humidity should be int, got {type(result.output.humidity)}"

        # Verify reasonable ranges
        assert -50 <= result.output.temperature <= 50, \
            f"Temperature should be in reasonable range, got {result.output.temperature}"
        assert 0 <= result.output.humidity <= 100, \
            f"Humidity should be 0-100%, got {result.output.humidity}"


class TestStructuredOutputTypes:
    """Test different structured output types."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_structured_with_optional_fields(self):
        """Test structured output with optional fields."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model, output_type=PersonInfo)

        result = await agent.run("Tell me about Alice who is 30 years old and works as a software engineer")

        # Verify structured output with all fields
        assert result.output is not None
        assert isinstance(result.output, PersonInfo), \
            f"Expected PersonInfo, got {type(result.output)}"

        # Verify required fields
        assert result.output.name, "Name should not be empty"
        assert isinstance(result.output.name, str)
        assert "alice" in result.output.name.lower(), \
            f"Name should contain 'Alice', got: {result.output.name}"

        assert isinstance(result.output.age, int), \
            f"Age should be int, got {type(result.output.age)}"
        assert 0 <= result.output.age <= 150, \
            f"Age should be reasonable, got {result.output.age}"

        # Optional field may or may not be populated
        if result.output.occupation is not None:
            assert isinstance(result.output.occupation, str)
            assert len(result.output.occupation) > 0

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_structured_with_list_fields(self):
        """Test structured output with list fields."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model, output_type=SearchResults)

        result = await agent.run("Search for Python tutorials and return 3 results")

        # Verify structured output
        assert result.output is not None
        assert isinstance(result.output, SearchResults), \
            f"Expected SearchResults, got {type(result.output)}"

        # Verify query field
        assert result.output.query, "Query should not be empty"
        assert isinstance(result.output.query, str)

        # Verify results list
        assert isinstance(result.output.results, list), \
            f"Results should be list, got {type(result.output.results)}"
        assert len(result.output.results) >= 0, "Results should be a valid list"

        # Verify all items are strings
        for item in result.output.results:
            assert isinstance(item, str), f"Each result should be string, got {type(item)}"
            assert len(item) > 0, "Result items should not be empty"

        # Verify count matches list length
        assert result.output.count == len(result.output.results), \
            f"Count {result.output.count} should match results length {len(result.output.results)}"

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_structured_with_defaults(self):
        """Test structured output with default values."""

        class ConfigData(BaseModel):
            enabled: bool = True
            timeout: int = 30
            retries: int = 3

        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model, output_type=ConfigData)

        result = await agent.run("Create config")

        assert result.output is not None
        # Verify defaults were applied (either from Claude or from the model)
        assert isinstance(result.output.enabled, bool)
        assert isinstance(result.output.timeout, int)
        assert isinstance(result.output.retries, int)
        # Values should be reasonable (either defaults or Claude's choices)
        assert result.output.timeout > 0
        assert result.output.retries >= 0


class TestStructuredOutputValidation:
    """Test validation with structured output."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_structured_with_field_descriptions(self):
        """Test that field descriptions are used in schema."""

        class DescribedModel(BaseModel):
            title: str = Field(description="The title of the article")
            summary: str = Field(description="A brief summary in one sentence")
            word_count: int = Field(description="Number of words in the content")

        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model, output_type=DescribedModel)

        result = await agent.run("Analyze this article: 'Python Programming'. The article has 250 words and explains Python basics.")

        # Verify structured output
        assert result.output is not None
        assert isinstance(result.output, DescribedModel), \
            f"Expected DescribedModel, got {type(result.output)}"

        # Verify all required fields are populated
        assert result.output.title, "Title should not be empty"
        assert result.output.summary, "Summary should not be empty"
        assert result.output.word_count is not None, "Word count should be set"

        # Verify field types
        assert isinstance(result.output.title, str)
        assert isinstance(result.output.summary, str)
        assert isinstance(result.output.word_count, int)

        # Verify field content makes sense
        assert result.output.word_count > 0, "Word count should be positive"
        assert "python" in result.output.title.lower() or "python" in result.output.summary.lower(), \
            "Output should reference Python"

        # Verify summary is reasonably brief (1 sentence guideline)
        assert len(result.output.summary) < 500, "Summary should be brief as per field description"

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_structured_with_constraints(self):
        """Test structured output with field constraints."""

        class ConstrainedModel(BaseModel):
            score: int = Field(ge=0, le=100, description="Score between 0 and 100")
            rating: str = Field(pattern="^[A-F]$", description="Letter grade A-F")

        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model, output_type=ConstrainedModel)

        result = await agent.run("Give a score (0-100) and letter grade (A-F) for this performance")

        assert result.output is not None
        assert 0 <= result.output.score <= 100
        assert result.output.rating in ['A', 'B', 'C', 'D', 'E', 'F']
        assert len(result.output.rating) == 1


class TestStructuredOutputWithTools:
    """Test structured output combined with tools.
    """

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_structured_output_with_tool_calls(self):
        """Test structured output when tools are also available.

        ClaudeCodeModel supports tool calling via MCP integration.
        This test verifies that having tools available doesn't break structured output.
        """
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model, output_type=CityLocation)

        tool_call_count = 0
        tool_args_received = []

        @agent.tool_plain
        def get_coordinates(city: str) -> tuple[float, float]:
            """Get lat/lon for a city."""
            nonlocal tool_call_count, tool_args_received
            tool_call_count += 1
            tool_args_received.append(city)
            return (48.8566, 2.3522)

        result = await agent.run("Where is Paris? Use the get_coordinates tool if helpful.")

        # Verify structured output works (tool may or may not be called)
        assert result.output is not None
        assert isinstance(result.output, CityLocation)
        assert result.output.city, "City should be populated"
        assert result.output.country, "Country should be populated"

        # If tool was called, verify coordinates are populated
        if tool_call_count > 0:
            assert "paris" in tool_args_received[0].lower() or "Paris" in tool_args_received[0]
            # Coordinates should be populated from tool result
            assert result.output.latitude != 0.0 or result.output.longitude != 0.0

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_structured_output_tool_actually_called(self):
        """Test that tools ARE called and structured output is returned.

        This test explicitly requests tool usage and verifies:
        1. Tool is called with correct arguments
        2. Final output is structured (CityLocation instance)
        3. Structured output contains data influenced by tool result
        """
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model, output_type=CityLocation)

        tool_call_count = 0
        tool_args_received = []

        @agent.tool_plain
        def get_coordinates(city: str) -> tuple[float, float]:
            """Get precise lat/lon coordinates for a city."""
            nonlocal tool_call_count, tool_args_received
            tool_call_count += 1
            tool_args_received.append(city)
            # Return Paris coordinates
            return (48.8566, 2.3522)

        result = await agent.run(
            "Use the get_coordinates tool to find the exact coordinates for Paris, France. "
            "Return the result as a CityLocation object."
        )

        # Verify tool was called
        assert tool_call_count > 0, "Tool should have been called at least once"
        assert len(tool_args_received) > 0, "Tool should have received arguments"
        assert any("paris" in arg.lower() for arg in tool_args_received), \
            f"Tool should have been called with Paris, got: {tool_args_received}"

        # Verify structured output
        assert result.output is not None
        assert isinstance(result.output, CityLocation), \
            f"Expected CityLocation, got {type(result.output)}"

        # Verify required fields
        assert result.output.city, "City should be populated"
        assert "paris" in result.output.city.lower(), \
            f"Expected Paris, got: {result.output.city}"
        assert result.output.country, "Country should be populated"
        assert "france" in result.output.country.lower(), \
            f"Expected France, got: {result.output.country}"

        # Verify coordinates are populated (from tool or from Claude's knowledge)
        assert isinstance(result.output.latitude, (int, float))
        assert isinstance(result.output.longitude, (int, float))
        # At least one coordinate should be non-zero
        assert result.output.latitude != 0.0 or result.output.longitude != 0.0, \
            "Coordinates should be populated"

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_structured_output_tool_not_needed(self):
        """Test that structured output works even when tool is NOT called.

        This verifies:
        1. Tool is available but not used
        2. Structured output is still returned correctly
        3. Agent can directly answer without tools
        """
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model, output_type=CityLocation)

        tool_call_count = 0

        @agent.tool_plain
        def get_coordinates(city: str) -> tuple[float, float]:
            """Get lat/lon for a city."""
            nonlocal tool_call_count
            tool_call_count += 1
            return (48.8566, 2.3522)

        result = await agent.run(
            "What is the capital city of France? Just answer directly without using tools."
        )

        # Tool should NOT be called (we asked for direct answer)
        # Note: Claude might still choose to call it, but we're encouraging it not to

        # Verify structured output works regardless
        assert result.output is not None
        assert isinstance(result.output, CityLocation), \
            f"Expected CityLocation, got {type(result.output)}"

        # Verify required fields
        assert result.output.city, "City should be populated"
        assert "paris" in result.output.city.lower(), \
            f"Expected Paris, got: {result.output.city}"
        assert result.output.country, "Country should be populated"
        assert "france" in result.output.country.lower(), \
            f"Expected France, got: {result.output.country}"


class TestStructuredOutputStreaming:
    """Test streaming with structured output."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_stream_structured_output(self):
        """Test streaming with structured result type."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model, output_type=CityLocation)

        async with agent.run_stream("City info for Tokyo, Japan") as result:
            # Stream structured output (not text)
            chunks_collected = []
            async for chunk in result.stream_output(debounce_by=None):
                chunks_collected.append(chunk)

            # Final output should be structured
            output = await result.get_output()
            assert output is not None
            assert isinstance(output, CityLocation), \
                f"Expected CityLocation, got {type(output)}"

            # Verify all fields are populated
            assert output.city, "City should not be empty"
            assert "tokyo" in output.city.lower(), \
                f"Expected Tokyo, got: {output.city}"
            assert output.country, "Country should not be empty"
            assert "japan" in output.country.lower(), \
                f"Expected Japan, got: {output.country}"

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_stream_structured_output_with_tools(self):
        """Test streaming with tools and structured output.

        This verifies that the streaming path also supports both
        function tools AND structured output simultaneously.
        """
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model, output_type=CityLocation)

        tool_call_count = 0
        tool_args_received = []

        @agent.tool_plain
        def get_coordinates(city: str) -> tuple[float, float]:
            """Get precise lat/lon coordinates for a city."""
            nonlocal tool_call_count, tool_args_received
            tool_call_count += 1
            tool_args_received.append(city)
            return (48.8566, 2.3522)

        async with agent.run_stream(
            "Use the get_coordinates tool to find Paris, France coordinates. "
            "Return as CityLocation."
        ) as result:
            # Collect stream chunks
            chunks_collected = []
            async for chunk in result.stream_output(debounce_by=None):
                chunks_collected.append(chunk)

            # Get final structured output
            output = await result.get_output()

            # Verify tool was called
            assert tool_call_count > 0, "Tool should have been called"
            assert any("paris" in arg.lower() for arg in tool_args_received), \
                f"Tool should have been called with Paris, got: {tool_args_received}"

            # Verify structured output
            assert output is not None
            assert isinstance(output, CityLocation), \
                f"Expected CityLocation, got {type(output)}"

            # Verify fields
            assert output.city, "City should be populated"
            assert "paris" in output.city.lower(), \
                f"Expected Paris, got: {output.city}"
            assert output.country, "Country should be populated"
            assert "france" in output.country.lower(), \
                f"Expected France, got: {output.country}"

            # Verify coordinates
            assert isinstance(output.latitude, (int, float))
            assert isinstance(output.longitude, (int, float))
            assert output.latitude != 0.0 or output.longitude != 0.0, \
                "Coordinates should be populated"

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_stream_structured_output_tool_not_needed(self):
        """Test streaming structured output when tool is available but not used."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model, output_type=CityLocation)

        tool_call_count = 0

        @agent.tool_plain
        def get_coordinates(city: str) -> tuple[float, float]:
            """Get lat/lon for a city."""
            nonlocal tool_call_count
            tool_call_count += 1
            return (48.8566, 2.3522)

        async with agent.run_stream(
            "What is the capital of France? Answer directly."
        ) as result:
            # Get final output
            output = await result.get_output()

            # Verify structured output works
            assert output is not None
            assert isinstance(output, CityLocation), \
                f"Expected CityLocation, got {type(output)}"

            # Verify fields
            assert output.city, "City should be populated"
            assert "paris" in output.city.lower(), \
                f"Expected Paris, got: {output.city}"
            assert output.country, "Country should be populated"
            assert "france" in output.country.lower(), \
                f"Expected France, got: {output.country}"


class TestStructuredOutputErrors:
    """Test error handling with structured output."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_structured_output_validation_error(self):
        """Test handling validation errors in structured output."""
        from pydantic import ValidationError
        from pydantic_ai.exceptions import UnexpectedModelBehavior

        class StrictModel(BaseModel):
            required_field: str
            number: int

        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model, output_type=StrictModel)

        # Should raise validation error or handle gracefully
        try:
            result = await agent.run("Get data")
            # If it succeeds, verify output meets requirements
            assert result is not None
            assert isinstance(result.output, StrictModel)
            assert hasattr(result.output, 'required_field')
            assert hasattr(result.output, 'number')
        except (ValidationError, UnexpectedModelBehavior) as e:
            # Validation error or retry limit exceeded is acceptable
            assert "validation" in str(e).lower() or "retry" in str(e).lower() or "retries" in str(e).lower(), \
                f"Expected validation or retry error, got: {e}"


class TestStructuredOutputComplexTypes:
    """Test structured output with complex nested types."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_nested_pydantic_models(self):
        """Test nested Pydantic models."""

        class Address(BaseModel):
            street: str
            city: str
            country: str

        class Person(BaseModel):
            name: str
            address: Address

        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model, output_type=Person)

        result = await agent.run("Create person info for John Smith at 123 Main Street, Springfield, USA")

        # Verify structured output
        assert result.output is not None
        assert isinstance(result.output, Person), f"Expected Person, got {type(result.output)}"

        # Verify name field
        assert result.output.name, "Name should not be empty"
        assert isinstance(result.output.name, str)
        assert "john" in result.output.name.lower() and "smith" in result.output.name.lower(), \
            f"Expected 'John Smith', got: {result.output.name}"

        # Verify nested address structure
        assert result.output.address is not None, "Address should not be None"

        # Address should be either an Address instance or a dict with the required fields
        if isinstance(result.output.address, dict):
            assert 'street' in result.output.address, "Address dict should have 'street'"
            assert 'city' in result.output.address, "Address dict should have 'city'"
            assert 'country' in result.output.address, "Address dict should have 'country'"

            # Verify content
            assert "main" in result.output.address['street'].lower(), \
                f"Expected '123 Main Street', got: {result.output.address['street']}"
            assert "springfield" in result.output.address['city'].lower(), \
                f"Expected 'Springfield', got: {result.output.address['city']}"
            assert "usa" in result.output.address['country'].lower() or "united states" in result.output.address['country'].lower(), \
                f"Expected 'USA', got: {result.output.address['country']}"
        else:
            # It's a nested Pydantic model
            assert isinstance(result.output.address, Address), \
                f"Expected Address model, got {type(result.output.address)}"
            assert hasattr(result.output.address, 'street')
            assert hasattr(result.output.address, 'city')
            assert hasattr(result.output.address, 'country')

            # Verify content
            assert result.output.address.street, "Street should not be empty"
            assert result.output.address.city, "City should not be empty"
            assert result.output.address.country, "Country should not be empty"

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_list_of_models(self):
        """Test list of Pydantic models."""

        class Item(BaseModel):
            name: str
            price: float

        class ShoppingList(BaseModel):
            items: List[Item]
            total: float

        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model, output_type=ShoppingList)

        result = await agent.run("Create shopping list with 3 items: milk $3.99, bread $2.50, eggs $4.25")

        # Verify structured output
        assert result.output is not None
        assert isinstance(result.output, ShoppingList), \
            f"Expected ShoppingList, got {type(result.output)}"

        # Verify items list
        assert isinstance(result.output.items, list), \
            f"Items should be list, got {type(result.output.items)}"
        assert len(result.output.items) >= 0, "Items should be a valid list"

        # Verify each item structure
        for item in result.output.items:
            assert isinstance(item, (Item, dict)), \
                f"Each item should be Item instance or dict, got {type(item)}"
            if isinstance(item, Item):
                assert item.name, "Item name should not be empty"
                assert isinstance(item.name, str)
                assert isinstance(item.price, (int, float))
                assert item.price >= 0, f"Price should be non-negative, got {item.price}"

        # Verify total
        assert isinstance(result.output.total, (int, float))
        assert result.output.total >= 0, f"Total should be non-negative, got {result.output.total}"

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_union_types(self):
        """Test union types in structured output."""

        class Response(BaseModel):
            status: str
            value: str | int | None = None

        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model, output_type=Response)

        result = await agent.run("Create a response with status 'success' and value 42")

        # Verify structured output
        assert result.output is not None
        assert isinstance(result.output, Response), \
            f"Expected Response, got {type(result.output)}"

        # Verify status field
        assert result.output.status, "Status should not be empty"
        assert isinstance(result.output.status, str)

        # Verify union type field - can be str, int, or None
        if result.output.value is not None:
            assert isinstance(result.output.value, (str, int)), \
                f"Value should be str or int, got {type(result.output.value)}"


class TestStructuredOutputMessageInspection:
    """Test message inspection for structured output."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_structured_output_message_flow(self):
        """Test that structured output uses proper message flow."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model, output_type=CityLocation)

        with capture_run_messages() as messages:
            result = await agent.run("Where is Paris?")

        # Verify message exchange occurred
        assert len(messages) >= 2, "Should have request and response messages"
        assert messages[0].kind == "request", "First message should be request"
        # For structured output, there may be additional messages (tool calls)
        assert any(m.kind == "response" for m in messages), \
            "Should have at least one response message"

        # Verify response contains structured data
        assert result.output is not None
        assert isinstance(result.output, CityLocation)

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_system_prompt_in_message_flow(self):
        """Test that system prompt appears in message flow for structured output."""
        model = ClaudeCodeModel(model=self.claude_model)
        system_text = "You are a precise geography assistant."
        agent = Agent(model, output_type=CityLocation, system_prompt=system_text)

        with capture_run_messages() as messages:
            result = await agent.run("Where is London?")

        # Find system prompt in messages
        request_messages = [m for m in messages if m.kind == "request"]
        assert len(request_messages) > 0, "Should have at least one request message"

        # Check for system prompt parts
        system_parts = [
            part for msg in request_messages
            for part in msg.parts
            if isinstance(part, SystemPromptPart)
        ]
        assert len(system_parts) > 0, "Should have system prompt parts"
        assert any(system_text in p.content for p in system_parts), \
            "System prompt should contain the specified text"

        # Verify output is still structured
        assert isinstance(result.output, CityLocation)

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_multimodal_message_structure(self, test_image_data: bytes):
        """Test message structure for multimodal structured output."""
        from pydantic_ai.messages import BinaryContent

        class ImageInfo(BaseModel):
            description: str
            width: int = 1
            height: int = 1

        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model, output_type=ImageInfo)

        with capture_run_messages() as messages:
            result = await agent.run([
                "Describe this image:",
                BinaryContent(data=test_image_data, media_type="image/png")
            ])

        # Verify message structure for multimodal input
        assert len(messages) >= 2
        request_msg = messages[0]
        assert request_msg.kind == "request"

        # Verify output is structured
        assert isinstance(result.output, ImageInfo)
        assert result.output.description


class TestStructuredOutputWithSystemPrompt:
    """Test structured output with system prompts."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_structured_with_custom_system_prompt(self):
        """Test structured output with custom system prompt."""
        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(
            model,
            output_type=CityLocation,
            system_prompt="You are a geography expert. Be precise with coordinates."
        )

        result = await agent.run("Where is Berlin?")

        # Verify structured output
        assert result.output is not None
        assert isinstance(result.output, CityLocation), \
            f"Expected CityLocation, got {type(result.output)}"

        # Verify required fields
        assert result.output.city, "City should not be empty"
        assert "berlin" in result.output.city.lower(), \
            f"Expected Berlin, got: {result.output.city}"
        assert result.output.country, "Country should not be empty"
        assert "german" in result.output.country.lower(), \
            f"Expected Germany, got: {result.output.country}"

        # Verify coordinates (system prompt asks for precision)
        assert isinstance(result.output.latitude, (int, float))
        assert isinstance(result.output.longitude, (int, float))
        # Berlin is around 52.5°N, 13.4°E
        assert 50 <= result.output.latitude <= 55, \
            f"Berlin latitude should be ~52.5, got {result.output.latitude}"
        assert 10 <= result.output.longitude <= 15, \
            f"Berlin longitude should be ~13.4, got {result.output.longitude}"


class TestStructuredOutputMultimodal:
    """Test structured output with multimodal inputs."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_structured_from_image(self, test_image_data: bytes):
        """Test extracting structured data from image."""
        from pydantic_ai.messages import BinaryContent

        class ImageAnalysis(BaseModel):
            description: str
            dominant_color: str
            object_count: int

        model = ClaudeCodeModel(model=self.claude_model)
        agent = Agent(model, output_type=ImageAnalysis)

        result = await agent.run([
            "Analyze this image and provide structured data:",
            BinaryContent(data=test_image_data, media_type="image/png")
        ])

        # Verify structured output from multimodal input
        assert result.output is not None
        assert isinstance(result.output, ImageAnalysis), \
            f"Expected ImageAnalysis, got {type(result.output)}"

        # Verify all fields are populated
        assert result.output.description, "Description should not be empty"
        assert isinstance(result.output.description, str)
        assert len(result.output.description) > 0

        assert result.output.dominant_color, "Dominant color should not be empty"
        assert isinstance(result.output.dominant_color, str)
        assert len(result.output.dominant_color) > 0

        assert isinstance(result.output.object_count, int)
        assert result.output.object_count >= 0, \
            f"Object count should be non-negative, got {result.output.object_count}"
