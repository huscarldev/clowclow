"""Custom Claude Code SDK client wrapper for pydantic-ai integration."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Type, TypeVar
from pydantic import BaseModel

from .constants import ClaudeCodeConfig
from .multimodal_handler import MultimodalContentHandler
from .query_strategies import SimpleQueryStrategy, StructuredQueryStrategy, ToolsQueryStrategy

T = TypeVar('T', bound=BaseModel)


class BasicResponse(BaseModel):
    """Basic response model for simple text outputs."""
    answer: str


class CustomClaudeCodeClient:
    """Custom wrapper around Claude Code SDK for pydantic-ai integration."""

    def __init__(
        self,
        api_key: str | None = None,
        workspace_dir: Path | None = None,
        model: str | None = None
    ):
        """Initialize the custom Claude Code client.

        Args:
            api_key: Anthropic API key. If not provided, will use ANTHROPIC_API_KEY env var.
            workspace_dir: Working directory for temporary files. If not provided, uses temp directory.
            model: Model to use (e.g., "claude-3-5-sonnet-20241022"). If not provided, uses SDK default.
        """
        # API key validation is handled by ClaudeSDKClient internally
        _ = api_key  # Unused but kept for interface compatibility

        # Initialize configuration
        self.config = ClaudeCodeConfig(
            workspace_dir=workspace_dir or Path(tempfile.gettempdir()),
            model=model
        )

        # Initialize handlers
        self.multimodal_handler = MultimodalContentHandler(self.config.workspace_dir)

        # Initialize strategies
        self.simple_strategy = SimpleQueryStrategy(self.config, self.multimodal_handler)
        self.structured_strategy = StructuredQueryStrategy(self.config, self.multimodal_handler)
        self.tools_strategy = ToolsQueryStrategy(self.config, self.multimodal_handler)

    async def simple_query(
        self,
        message: str | list[dict],
        system_prompt: str | None = None,
        max_turns: int = 1
    ) -> str:
        """Execute a simple query and return the text response.

        Args:
            message: The user message/query (str or list of content blocks for multimodal)
            system_prompt: Optional system prompt
            max_turns: Maximum number of conversation turns

        Returns:
            The response text
        """
        return await self.simple_strategy.execute(
            message=message,
            system_prompt=system_prompt,
            max_turns=max_turns
        )

    async def structured_query(
        self,
        message: str | list[dict],
        pydantic_class: Type[T],
        system_prompt: str | None = None,
        custom_instructions: str | None = None,
        max_turns: int | None = None
    ) -> T:
        """Execute a structured query with schema tag method.

        Args:
            message: The user message/query (str or list of content blocks for multimodal)
            pydantic_class: The Pydantic model class for the expected response
            system_prompt: Optional system prompt
            custom_instructions: Additional instructions for the model
            max_turns: Maximum number of conversation turns (None for default behavior)

        Returns:
            An instance of the provided Pydantic class
        """
        return await self.structured_strategy.execute(
            message=message,
            pydantic_class=pydantic_class,
            system_prompt=system_prompt,
            custom_instructions=custom_instructions,
            max_turns=max_turns
        )

    async def tools_query(
        self,
        message: str | list[dict],
        tools: list[dict],
        system_prompt: str | None = None,
        max_turns: int = 5
    ) -> dict:
        """Execute a query with function tools available.

        Args:
            message: The user message/query (str or list of content blocks for multimodal)
            tools: List of tool definitions in Pydantic AI format
            system_prompt: Optional system prompt
            max_turns: Maximum number of conversation turns

        Returns:
            Dict with 'tool_calls' (list of tool call dicts) and 'text' (final response text)
        """
        return await self.tools_strategy.execute(
            message=message,
            tools=tools,
            system_prompt=system_prompt,
            max_turns=max_turns
        )
