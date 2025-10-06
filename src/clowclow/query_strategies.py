"""Query strategy pattern for different Claude Code query types."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Type, TypeVar, Any

from pydantic import BaseModel
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    tool,
    create_sdk_mcp_server,
    HookMatcher
)

from .constants import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_STRUCTURED_SYSTEM_SUFFIX,
    STRUCTURED_OUTPUT_INSTRUCTIONS,
    STRUCTURED_QUERY_CUSTOM_INSTRUCTIONS,
    MCP_TOOL_PREFIX,
    MCP_SERVER_NAME,
    MCP_SERVER_VERSION,
    ClaudeCodeConfig
)
from .multimodal_handler import MultimodalContentHandler
from .dynamic_model_builder import DynamicModelBuilder


T = TypeVar('T', bound=BaseModel)


class QueryStrategy(ABC):
    """Abstract base class for query strategies."""

    def __init__(self, config: ClaudeCodeConfig, multimodal_handler: MultimodalContentHandler):
        """Initialize the query strategy.

        Args:
            config: Claude Code configuration
            multimodal_handler: Handler for multimodal content
        """
        self.config = config
        self.multimodal_handler = multimodal_handler

    @abstractmethod
    async def execute(self, message: str | list[dict], **kwargs) -> Any:
        """Execute the query strategy.

        Args:
            message: The user message/query
            **kwargs: Additional strategy-specific arguments

        Returns:
            Strategy-specific result
        """
        pass


class SimpleQueryStrategy(QueryStrategy):
    """Strategy for simple text queries."""

    async def execute(
        self,
        message: str | list[dict],
        system_prompt: str | None = None,
        max_turns: int | None = None
    ) -> str:
        """Execute a simple text query.

        Args:
            message: The user message/query
            system_prompt: Optional system prompt
            max_turns: Maximum number of conversation turns

        Returns:
            The response text
        """
        with self.multimodal_handler.managed_content(message) as (final_prompt, _):
            # Add instruction to answer directly without using tools
            enhanced_system = (system_prompt or self.config.default_system_prompt) + "\n\nIMPORTANT: Answer directly based on your knowledge. Do not use any tools or search capabilities."

            options = ClaudeAgentOptions(
                system_prompt=enhanced_system,
                max_turns=max_turns if max_turns is not None else self.config.max_turns_simple,
                cwd=str(self.config.workspace_dir),
                permission_mode=self.config.permission_mode_simple,
                allowed_tools=[],  # Disable tools for simple text queries
                model=self.config.model
            )

            response_parts = []

            async with ClaudeSDKClient(options=options) as client:
                await client.query(final_prompt)

                # Collect the streaming response
                async for response_message in client.receive_response():
                    if hasattr(response_message, 'content'):
                        for block in response_message.content:
                            if hasattr(block, 'text'):
                                response_parts.append(block.text)

            return ''.join(response_parts)


class StructuredQueryStrategy(QueryStrategy):
    """Strategy for structured output queries using <schema> tag method."""

    async def execute(
        self,
        message: str | list[dict],
        pydantic_class: Type[T],
        system_prompt: str | None = None,
        custom_instructions: str | None = None,
        max_turns: int | None = None
    ) -> T:
        """Execute a structured output query.

        Args:
            message: The user message/query
            pydantic_class: The Pydantic model class for the expected response
            system_prompt: Optional system prompt
            custom_instructions: Additional instructions for the model
            max_turns: Maximum number of conversation turns

        Returns:
            An instance of the provided Pydantic class
        """
        # Generate JSON schema and resolve $ref references
        schema = pydantic_class.model_json_schema()
        resolved_schema = DynamicModelBuilder.resolve_schema_refs(schema)
        schema_json = json.dumps(resolved_schema, indent=2)

        with self.multimodal_handler.managed_content(message) as (user_prompt, _):
            # Create final prompt with schema
            final_prompt = f"""<schema>
{schema_json}
</schema>

{user_prompt}

{custom_instructions if custom_instructions else ''}

{STRUCTURED_OUTPUT_INSTRUCTIONS}"""

            # Create enhanced system prompt
            enhanced_system = f"""{system_prompt or self.config.default_system_prompt}

{DEFAULT_STRUCTURED_SYSTEM_SUFFIX}

IMPORTANT: Answer directly based on your knowledge without using any tools or search capabilities. Provide only the JSON response."""

            options = ClaudeAgentOptions(
                system_prompt=enhanced_system,
                max_turns=max_turns if max_turns is not None else self.config.max_turns_structured,
                cwd=str(self.config.workspace_dir),
                permission_mode=self.config.permission_mode_structured,
                allowed_tools=[],  # Disable tools for structured output
                model=self.config.model
            )

            response_parts = []

            async with ClaudeSDKClient(options=options) as client:
                await client.query(final_prompt)

                # Collect the streaming response
                async for response_message in client.receive_response():
                    if hasattr(response_message, 'content'):
                        for block in response_message.content:
                            if hasattr(block, 'text'):
                                response_parts.append(block.text)

            response_text = ''.join(response_parts).strip()

            # Try to extract JSON from the response
            json_text = self._extract_json_from_response(response_text)

            # Parse and validate the JSON
            json_data = json.loads(json_text)

            # Validate and return the Pydantic model
            return pydantic_class.model_validate(json_data)

    @staticmethod
    def _extract_json_from_response(response_text: str) -> str:
        """Extract JSON from response text that might contain extra content."""
        # First try to find JSON between triple backticks
        json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        match = re.search(json_pattern, response_text, re.DOTALL)
        if match:
            return match.group(1)

        # Try to find JSON object boundaries
        start = response_text.find('{')
        if start == -1:
            raise ValueError("No JSON object found in response")

        # Find matching closing brace
        brace_count = 0
        end = start
        for i, char in enumerate(response_text[start:], start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break

        if brace_count != 0:
            raise ValueError("Unmatched braces in JSON response")

        return response_text[start:end]


class ToolsQueryStrategy(QueryStrategy):
    """Strategy for queries with function tools."""

    async def execute(
        self,
        message: str | list[dict],
        tools: list[dict],
        system_prompt: str | None = None,
        max_turns: int | None = None
    ) -> dict:
        """Execute a query with function tools available.

        Args:
            message: The user message/query
            tools: List of tool definitions in Pydantic AI format
            system_prompt: Optional system prompt
            max_turns: Maximum number of conversation turns

        Returns:
            Dict with 'tool_calls' (list of tool call dicts) and 'text' (final response text)
        """
        tool_calls_captured = []

        # Create hook to capture tool calls BEFORE execution
        async def capture_tool_calls_hook(input_data, tool_use_id, context):
            """Capture tool calls for Pydantic AI to execute."""
            tool_name = input_data.get('tool_name', '')

            # Check if this is one of our Pydantic AI tools
            if tool_name.startswith(MCP_TOOL_PREFIX):
                actual_tool_name = tool_name.replace(MCP_TOOL_PREFIX, '')
                tool_input = input_data.get('tool_input', {})

                # Store the tool call
                tool_calls_captured.append({
                    'tool_name': actual_tool_name,
                    'args': tool_input,
                    'tool_call_id': tool_use_id
                })

                # Allow the tool to execute (it will return placeholder text)
                return {}

            # For other tools (Read, Write), let them execute normally
            return {}

        # Create MCP tools from Pydantic AI tool definitions
        mcp_tools = []

        for tool_def in tools:
            tool_name = tool_def['name']
            tool_description = tool_def.get('description', '')
            tool_schema = tool_def.get('parameters_json_schema', {})

            # Create async tool function that returns placeholder
            async def tool_func(args, *, _tool_name=tool_name):
                # Return placeholder - actual execution happens in Pydantic AI
                return {
                    "content": [{
                        "type": "text",
                        "text": f"[Tool {_tool_name} will be executed by Pydantic AI]"
                    }]
                }

            # Register as MCP tool
            mcp_tool = tool(tool_name, tool_description, tool_schema)(tool_func)
            mcp_tools.append(mcp_tool)

        # Create SDK MCP server
        server = create_sdk_mcp_server(
            name=MCP_SERVER_NAME,
            version=MCP_SERVER_VERSION,
            tools=mcp_tools
        )

        # Prepare allowed tools list
        allowed_tools = [f"{MCP_TOOL_PREFIX}{t['name']}" for t in tools]

        with self.multimodal_handler.managed_content(message) as (final_prompt, _):
            # Configure options with MCP server and hook
            options = ClaudeAgentOptions(
                system_prompt=system_prompt or self.config.default_system_prompt,
                max_turns=max_turns if max_turns is not None else self.config.max_turns_tools,
                cwd=str(self.config.workspace_dir),
                permission_mode=self.config.permission_mode_tools,  # Auto-accept so tools execute and we capture them
                allowed_tools=["Read", "Write"] + allowed_tools,
                mcp_servers={MCP_SERVER_NAME: server},
                hooks={
                    "PreToolUse": [HookMatcher(
                        hooks=[capture_tool_calls_hook]  # List of hook functions
                    )]
                },
                model=self.config.model
            )

            response_parts = []

            async with ClaudeSDKClient(options=options) as client:
                await client.query(final_prompt)

                # Collect the streaming response
                async for response_message in client.receive_response():
                    if hasattr(response_message, 'content'):
                        for block in response_message.content:
                            if hasattr(block, 'text'):
                                response_parts.append(block.text)

            return {
                'tool_calls': tool_calls_captured,
                'text': ''.join(response_parts)
            }
