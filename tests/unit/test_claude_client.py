"""Unit tests for CustomClaudeCodeClient - testing client delegation to strategies."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path
from pydantic import BaseModel

from clowclow.claude_client import CustomClaudeCodeClient


class TestClientInitialization:
    """Test CustomClaudeCodeClient initialization."""

    def test_initialization_default_workspace(self):
        """Test client initializes with default workspace directory."""
        client = CustomClaudeCodeClient()

        assert client.config.workspace_dir is not None
        assert client.config.workspace_dir.exists()

    def test_initialization_custom_workspace(self, test_workspace):
        """Test client initializes with custom workspace directory."""
        client = CustomClaudeCodeClient(workspace_dir=test_workspace)

        assert client.config.workspace_dir == test_workspace

    def test_initialization_custom_model(self):
        """Test client initializes with custom model name."""
        client = CustomClaudeCodeClient(model="custom-model")

        assert client.config.model == "custom-model"

    def test_strategies_created(self):
        """Test that all strategies are created during initialization."""
        client = CustomClaudeCodeClient()

        assert client.simple_strategy is not None
        assert client.structured_strategy is not None
        assert client.tools_strategy is not None

    def test_multimodal_handler_created(self):
        """Test that multimodal handler is created."""
        client = CustomClaudeCodeClient()

        assert client.multimodal_handler is not None


class TestSimpleQuery:
    """Test simple_query method."""

    @pytest.mark.asyncio
    async def test_simple_query_delegates_to_strategy(self):
        """Test that simple_query delegates to SimpleQueryStrategy."""
        client = CustomClaudeCodeClient()

        with patch.object(client.simple_strategy, 'execute', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = "Strategy response"

            result = await client.simple_query("Test message")

            assert result == "Strategy response"
            mock_execute.assert_called_once_with(
                message="Test message",
                system_prompt=None,
                max_turns=1
            )

    @pytest.mark.asyncio
    async def test_simple_query_with_system_prompt(self):
        """Test simple_query passes system prompt to strategy."""
        client = CustomClaudeCodeClient()

        with patch.object(client.simple_strategy, 'execute', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = "Response"

            await client.simple_query("Test", system_prompt="Custom system")

            call_args = mock_execute.call_args
            assert call_args.kwargs['system_prompt'] == "Custom system"

    @pytest.mark.asyncio
    async def test_simple_query_with_custom_max_turns(self):
        """Test simple_query passes custom max_turns to strategy."""
        client = CustomClaudeCodeClient()

        with patch.object(client.simple_strategy, 'execute', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = "Response"

            await client.simple_query("Test", max_turns=5)

            call_args = mock_execute.call_args
            assert call_args.kwargs['max_turns'] == 5

    @pytest.mark.asyncio
    async def test_simple_query_with_multimodal_content(self):
        """Test simple_query handles multimodal content."""
        client = CustomClaudeCodeClient()

        multimodal_content = [
            {"type": "text", "text": "Analyze this"},
            {"type": "image", "source": {"type": "url", "url": "https://example.com/img.png"}}
        ]

        with patch.object(client.simple_strategy, 'execute', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = "Image analyzed"

            result = await client.simple_query(multimodal_content)

            assert result == "Image analyzed"
            call_args = mock_execute.call_args
            assert call_args.kwargs['message'] == multimodal_content


class TestStructuredQuery:
    """Test structured_query method."""

    @pytest.mark.asyncio
    async def test_structured_query_delegates_to_strategy(self):
        """Test that structured_query delegates to StructuredQueryStrategy."""
        client = CustomClaudeCodeClient()

        class TestModel(BaseModel):
            value: str

        schema = {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"]
        }

        with patch.object(client.structured_strategy, 'execute', new_callable=AsyncMock) as mock_execute:
            mock_response = TestModel(value="test")
            mock_execute.return_value = mock_response

            result = await client.structured_query("Test", schema, TestModel)

            assert result == mock_response
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_structured_query_passes_all_parameters(self):
        """Test structured_query passes all parameters to strategy."""
        client = CustomClaudeCodeClient()

        class TestModel(BaseModel):
            value: str

        with patch.object(client.structured_strategy, 'execute', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = TestModel(value="test")

            await client.structured_query(
                message="Test message",
                pydantic_class=TestModel,
                system_prompt="Custom system",
                max_turns=3
            )

            # Verify execute was called
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_structured_query_with_multimodal(self):
        """Test structured_query with multimodal content."""
        client = CustomClaudeCodeClient()

        class TestModel(BaseModel):
            description: str

        multimodal_content = [
            {"type": "text", "text": "Describe"},
            {"type": "image", "source": {"type": "url", "url": "https://example.com/img.png"}}
        ]

        with patch.object(client.structured_strategy, 'execute', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = TestModel(description="An image")

            result = await client.structured_query(multimodal_content, TestModel)

            # Verify execute was called
            mock_execute.assert_called_once()


class TestToolsQuery:
    """Test tools_query method."""

    @pytest.mark.asyncio
    async def test_tools_query_delegates_to_strategy(self):
        """Test that tools_query delegates to ToolsQueryStrategy."""
        client = CustomClaudeCodeClient()

        tools = [{"name": "test_tool", "description": "Test", "parameters_json_schema": {}}]

        with patch.object(client.tools_strategy, 'execute', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {"tool_name": "test_tool", "tool_input": {}}

            result = await client.tools_query("Test", tools)

            assert result == {"tool_name": "test_tool", "tool_input": {}}
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_tools_query_passes_all_parameters(self):
        """Test tools_query passes all parameters to strategy."""
        client = CustomClaudeCodeClient()

        tools = [{"name": "tool1", "description": "Tool 1", "parameters_json_schema": {}}]

        with patch.object(client.tools_strategy, 'execute', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = "Response"

            await client.tools_query(
                message="Test message",
                tools=tools,
                system_prompt="Custom system",
                max_turns=5
            )

            # Verify execute was called
            mock_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_tools_query_default_max_turns(self):
        """Test tools_query uses default max_turns when not specified."""
        client = CustomClaudeCodeClient()

        tools = [{"name": "test", "description": "Test", "parameters_json_schema": {}}]

        with patch.object(client.tools_strategy, 'execute', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = "Response"

            await client.tools_query("Test", tools)

            call_args = mock_execute.call_args
            # Should use default from config (5)
            assert call_args.kwargs['max_turns'] == 5


class TestConfigurationSharing:
    """Test that configuration is shared correctly across strategies."""

    def test_config_shared_with_strategies(self):
        """Test that the same config instance is used by all strategies."""
        client = CustomClaudeCodeClient()

        # All strategies should use the same config
        assert client.simple_strategy.config is client.config
        assert client.structured_strategy.config is client.config
        assert client.tools_strategy.config is client.config

    def test_multimodal_handler_shared_with_strategies(self):
        """Test that the same multimodal handler is used by all strategies."""
        client = CustomClaudeCodeClient()

        assert client.simple_strategy.multimodal_handler is client.multimodal_handler
        assert client.structured_strategy.multimodal_handler is client.multimodal_handler
        assert client.tools_strategy.multimodal_handler is client.multimodal_handler

    def test_workspace_dir_propagated(self, test_workspace):
        """Test that workspace_dir is propagated to all components."""
        client = CustomClaudeCodeClient(workspace_dir=test_workspace)

        assert client.config.workspace_dir == test_workspace
        assert client.multimodal_handler.workspace_dir == test_workspace
