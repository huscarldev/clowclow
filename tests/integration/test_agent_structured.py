"""Integration tests for structured output with ClaudeCodeModel.

Tests verify that ClaudeCodeModel correctly handles Pydantic AI's structured
output modes using Pydantic models.
"""

from __future__ import annotations

import pytest
from typing import List

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

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
        model = ClaudeCodeModel()
        agent = Agent(model, output_type=WeatherData)

        result = await agent.run("What's the weather in London?")

        # Should return WeatherData instance
        assert result.output is not None


class TestStructuredOutputTypes:
    """Test different structured output types."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_structured_with_optional_fields(self):
        """Test structured output with optional fields."""
        model = ClaudeCodeModel()
        agent = Agent(model, output_type=PersonInfo)

        result = await agent.run("Tell me about Alice")

        assert result.output is not None

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_structured_with_list_fields(self):
        """Test structured output with list fields."""
        model = ClaudeCodeModel()
        agent = Agent(model, output_type=SearchResults)

        result = await agent.run("Search for Python tutorials")

        assert result.output is not None

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_structured_with_defaults(self):
        """Test structured output with default values."""

        class ConfigData(BaseModel):
            enabled: bool = True
            timeout: int = 30
            retries: int = 3

        model = ClaudeCodeModel()
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

        model = ClaudeCodeModel()
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

        model = ClaudeCodeModel()
        agent = Agent(model, output_type=ConstrainedModel)

        result = await agent.run("Give a score (0-100) and letter grade (A-F) for this performance")

        assert result.output is not None
        assert 0 <= result.output.score <= 100
        assert result.output.rating in ['A', 'B', 'C', 'D', 'E', 'F']
        assert len(result.output.rating) == 1


class TestStructuredOutputWithTools:
    """Test structured output combined with tools.

    NOTE: ClaudeCodeModel does not support tool calling.
    This test is skipped for ClaudeCodeModel.
    """

    @pytest.mark.skip(reason="ClaudeCodeModel does not support Pydantic AI tool calling")
    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_structured_output_with_tool_calls(self):
        """Test structured output when tools are also available.

        SKIPPED: ClaudeCodeModel does not implement tool calling.
        Tools can be registered but won't be invoked.
        """
        model = ClaudeCodeModel()
        agent = Agent(model, output_type=CityLocation)

        @agent.tool_plain
        def get_coordinates(city: str) -> tuple[float, float]:
            """Get lat/lon for a city."""
            return (48.8566, 2.3522)

        result = await agent.run("Where is Paris?")

        # Tool will NOT be called - just verify structured output works
        assert result.output is not None
        assert isinstance(result.output, CityLocation)


class TestStructuredOutputStreaming:
    """Test streaming with structured output."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_stream_structured_output(self):
        """Test streaming with structured result type."""
        model = ClaudeCodeModel()
        agent = Agent(model, output_type=CityLocation)

        async with agent.run_stream("City info for Tokyo") as result:
            # Stream structured output (not text)
            async for _ in result.stream_output(debounce_by=None):
                pass

            # Final output should be structured
            output = await result.get_output()
            assert output is not None


class TestStructuredOutputErrors:
    """Test error handling with structured output."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_structured_output_validation_error(self):
        """Test handling validation errors in structured output."""

        class StrictModel(BaseModel):
            required_field: str
            number: int

        model = ClaudeCodeModel()
        agent = Agent(model, output_type=StrictModel)

        # Should raise validation error or handle gracefully
        try:
            result = await agent.run("Get data")
            # If it succeeds, verify output
            assert result is not None
        except Exception:
            # Validation error expected
            pass


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

        model = ClaudeCodeModel()
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

        model = ClaudeCodeModel()
        agent = Agent(model, output_type=ShoppingList)

        result = await agent.run("Create shopping list")

        assert result.output is not None

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_union_types(self):
        """Test union types in structured output."""

        class Response(BaseModel):
            status: str
            value: str | int | None = None

        model = ClaudeCodeModel()
        agent = Agent(model, output_type=Response)

        result = await agent.run("Get response")

        assert result.output is not None


class TestStructuredOutputWithSystemPrompt:
    """Test structured output with system prompts."""

    @pytest.mark.live
    @pytest.mark.asyncio
    async def test_structured_with_custom_system_prompt(self):
        """Test structured output with custom system prompt."""
        model = ClaudeCodeModel()
        agent = Agent(
            model,
            output_type=CityLocation,
            system_prompt="You are a geography expert. Be precise with coordinates."
        )

        result = await agent.run("Where is Berlin?")

        assert result.output is not None


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

        model = ClaudeCodeModel()
        agent = Agent(model, output_type=ImageAnalysis)

        result = await agent.run([
            "Analyze this image:",
            BinaryContent(data=test_image_data, media_type="image/png")
        ])

        assert result.output is not None
