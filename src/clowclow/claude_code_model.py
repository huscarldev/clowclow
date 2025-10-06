"""Claude Code model adapter for pydantic-ai."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Literal

from pydantic_ai.models import Model, ModelResponse, ModelSettings, StreamedResponse, ModelRequestParameters, RunContext, RequestUsage
from pydantic_ai import messages as _messages
from pydantic_ai._parts_manager import ModelResponsePartsManager

from .claude_client import CustomClaudeCodeClient
from .request_handler import RequestHandler
from .dynamic_model_builder import DynamicModelBuilder
from .constants import (
    DEFAULT_TOOL_CALL_ID,
    STRUCTURED_QUERY_CUSTOM_INSTRUCTIONS,
    STRUCTURED_QUERY_WITH_TOOLS_INSTRUCTIONS,
    STRUCTURED_QUERY_FROM_RESPONSE_INSTRUCTIONS
)


class ClaudeCodeModel(Model):
    """A pydantic-ai Model implementation that uses Claude Code SDK."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "claude-code",
        workspace_dir: Path | None = None,
        model: str | None = None
    ) -> None:
        """Initialize the Claude Code model.

        Args:
            api_key: Anthropic API key. If not provided, will use ANTHROPIC_API_KEY env var.
            model_name: Model identifier, defaults to "claude-code"
            workspace_dir: Working directory for temporary files
            model: Anthropic model to use (e.g., "claude-3-5-sonnet-20241022"). If not provided, uses SDK default.
        """
        self._model_name = model_name
        self._client = CustomClaudeCodeClient(api_key=api_key, workspace_dir=workspace_dir, model=model)

    @property
    def model_name(self) -> str:
        """The name of the model."""
        return self._model_name

    @property
    def system(self) -> Literal["claude-code"]:
        """The system/provider name."""
        return "claude-code"

    def _convert_tools_to_client_format(self, tools: list) -> list[dict]:
        """Convert Pydantic AI tool definitions to client format.

        Args:
            tools: List of ToolDefinition objects from Pydantic AI

        Returns:
            List of tool dicts for CustomClaudeCodeClient
        """
        result = []
        for tool in tools:
            result.append({
                'name': tool.name,
                'description': tool.description,
                'parameters_json_schema': tool.parameters_json_schema
            })
        return result

    async def request(
        self,
        messages: list[_messages.ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters
    ) -> ModelResponse:
        """Make a request to Claude Code.

        Args:
            messages: The message history to send
            model_settings: Optional model settings
            model_request_parameters: Optional request parameters

        Returns:
            The model response
        """
        try:
            # Extract messages using RequestHandler
            has_images = RequestHandler.has_images(messages)
            user_content = (
                RequestHandler.extract_multimodal_content(messages)
                if has_images
                else RequestHandler.extract_user_message(messages)
            )
            system_prompt = RequestHandler.extract_system_messages(messages)

            # Check for function tools (user-defined tools that Claude can call)
            text_from_tools = None
            if (model_request_parameters and
                hasattr(model_request_parameters, 'function_tools') and
                model_request_parameters.function_tools):

                # Check if this is a continuation with tool results
                tool_returns = RequestHandler.check_for_tool_returns(messages)
                if tool_returns:
                    user_content = RequestHandler.append_tool_results_to_content(
                        user_content, tool_returns
                    )

                # Convert tools to client format and execute
                tools = self._convert_tools_to_client_format(model_request_parameters.function_tools)
                result = await self._client.tools_query(
                    message=user_content,
                    tools=tools,
                    system_prompt=system_prompt
                )

                # Check if Claude called any tools
                if result['tool_calls']:
                    parts = [
                        _messages.ToolCallPart(
                            tool_name=tc['tool_name'],
                            args=tc['args'],
                            tool_call_id=tc['tool_call_id']
                        )
                        for tc in result['tool_calls']
                    ]
                    return ModelResponse(parts=parts, timestamp=datetime.now())
                else:
                    text_from_tools = result['text']

            # Check if this is a structured output request (tool mode)
            if (model_request_parameters and
                hasattr(model_request_parameters, 'output_mode') and
                model_request_parameters.output_mode == 'tool' and
                model_request_parameters.output_tools):

                # Extract the JSON schema from the output tool
                output_tool = model_request_parameters.output_tools[0]
                schema_dict = output_tool.parameters_json_schema
                tool_name = output_tool.name

                # Create dynamic model from schema
                DynamicModel = DynamicModelBuilder.create_model_from_schema(schema_dict)

                # Determine message and instructions based on context
                if text_from_tools:
                    # We have text from tools query, use it to extract structured data
                    structured_message = self._create_structured_message_from_tools(
                        user_content, text_from_tools
                    )
                    custom_instructions = STRUCTURED_QUERY_WITH_TOOLS_INSTRUCTIONS
                else:
                    structured_message = user_content
                    custom_instructions = STRUCTURED_QUERY_CUSTOM_INSTRUCTIONS

                # Execute structured query
                structured_response = await self._client.structured_query(
                    message=structured_message,
                    pydantic_class=DynamicModel,
                    system_prompt=system_prompt,
                    custom_instructions=custom_instructions
                )

                # Post-process and convert to tool call
                args = DynamicModelBuilder.post_process_model_data(
                    structured_response.model_dump(), schema_dict
                )

                tool_call = _messages.ToolCallPart(
                    tool_name=tool_name,
                    args=args,
                    tool_call_id=DEFAULT_TOOL_CALL_ID
                )

                return ModelResponse(parts=[tool_call], timestamp=datetime.now())
            else:
                # Simple text query
                response = await self._client.simple_query(
                    message=user_content,
                    system_prompt=system_prompt
                )
                return self._convert_response(response)

        except Exception as e:
            raise RuntimeError(f"Claude Code request failed: {e}") from e

    @staticmethod
    def _create_structured_message_from_tools(
        user_content: str | list[dict],
        text_from_tools: str
    ) -> str | list[dict]:
        """Create a structured message that combines user content with tool response.

        Args:
            user_content: Original user content
            text_from_tools: Text response from tools

        Returns:
            Combined message for structured extraction
        """
        addition = f"Original request: {user_content}\n\nResponse from tools: {text_from_tools}\n\nBased on the above information, provide the data in the required structured format."

        if isinstance(user_content, str):
            return addition
        else:
            # For list content, append as text block
            return user_content + [{"type": "text", "text": f"\n\nResponse from tools: {text_from_tools}\n\nBased on the above information, provide the data in the required structured format."}]

    def _convert_response(self, response: str) -> ModelResponse:
        """Convert Claude Code response to pydantic-ai ModelResponse."""
        return ModelResponse(
            parts=[_messages.TextPart(content=response)],
            timestamp=datetime.now(),
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[_messages.ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext | None = None
    ) -> AsyncIterator[StreamedResponse]:
        """Make a streaming request to Claude Code.

        Args:
            messages: The message history to send
            model_settings: Optional model settings
            model_request_parameters: Request parameters
            run_context: Optional run context

        Returns:
            A streaming response
        """
        response = ClaudeCodeStreamedResponse(
            model=self,
            messages=messages,
            model_settings=model_settings,
            model_request_parameters=model_request_parameters,
            run_context=run_context
        )
        yield response


class ClaudeCodeStreamedResponse(StreamedResponse):
    """Streaming response implementation for Claude Code."""

    def __init__(
        self,
        model: ClaudeCodeModel,
        messages: list[_messages.ModelMessage],
        model_settings: ModelSettings | None = None,
        model_request_parameters: ModelRequestParameters | None = None,
        run_context: RunContext | None = None,
    ):
        self._model = model
        self._messages = messages
        self._model_settings = model_settings
        self._model_request_parameters = model_request_parameters
        self._run_context = run_context
        self._timestamp = datetime.now()
        self._parts_manager = ModelResponsePartsManager()
        self._usage = RequestUsage()

    @property
    def model_name(self) -> str:
        """The name of the model."""
        return self._model.model_name

    @property
    def provider_name(self) -> str:
        """The provider name."""
        return self._model.system

    @property
    def timestamp(self) -> datetime:
        """The response timestamp."""
        return self._timestamp

    @property
    def model_request_parameters(self) -> ModelRequestParameters | None:
        """The model request parameters."""
        return self._model_request_parameters

    async def _get_event_iterator(self) -> AsyncIterator[_messages.ModelResponseStreamEvent]:
        """Get an async iterator of stream events."""
        # Note: This is a simple non-streaming fallback
        # TODO: Implement actual streaming if claude-code-sdk supports it

        try:
            # Extract messages using RequestHandler
            has_images = RequestHandler.has_images(self._messages)
            user_content = (
                RequestHandler.extract_multimodal_content(self._messages)
                if has_images
                else RequestHandler.extract_user_message(self._messages)
            )
            system_message = RequestHandler.extract_system_messages(self._messages)

            # Check for function tools
            text_from_tools = None
            if (self._model_request_parameters and
                hasattr(self._model_request_parameters, 'function_tools') and
                self._model_request_parameters.function_tools):

                # Check for tool results
                tool_returns = RequestHandler.check_for_tool_returns(self._messages)
                if tool_returns:
                    user_content = RequestHandler.append_tool_results_to_content(
                        user_content, tool_returns
                    )

                # Convert tools and execute
                tools = self._model._convert_tools_to_client_format(
                    self._model_request_parameters.function_tools
                )
                result = await self._model._client.tools_query(
                    message=user_content,
                    tools=tools,
                    system_prompt=system_message
                )

                # Check if Claude called any tools
                if result['tool_calls']:
                    for tc in result['tool_calls']:
                        yield self._parts_manager.handle_tool_call_part(
                            vendor_part_id=0,
                            tool_name=tc['tool_name'],
                            args=tc['args'],
                            tool_call_id=tc['tool_call_id']
                        )
                    return  # Early return after yielding tool calls
                else:
                    text_from_tools = result['text']

            # Check if this is a structured output request
            if (self._model_request_parameters and
                hasattr(self._model_request_parameters, 'output_mode') and
                self._model_request_parameters.output_mode == 'tool' and
                self._model_request_parameters.output_tools):

                # Extract schema and create dynamic model
                output_tool = self._model_request_parameters.output_tools[0]
                schema_dict = output_tool.parameters_json_schema
                tool_name = output_tool.name

                DynamicModel = DynamicModelBuilder.create_model_from_schema(schema_dict)

                # Determine message and instructions
                if text_from_tools:
                    structured_message = self._model._create_structured_message_from_tools(
                        user_content, text_from_tools
                    )
                    custom_instructions = STRUCTURED_QUERY_FROM_RESPONSE_INSTRUCTIONS
                else:
                    structured_message = user_content
                    custom_instructions = STRUCTURED_QUERY_CUSTOM_INSTRUCTIONS

                # Execute structured query
                structured_response = await self._model._client.structured_query(
                    message=structured_message,
                    pydantic_class=DynamicModel,
                    system_prompt=system_message,
                    custom_instructions=custom_instructions
                )

                # Post-process and yield tool call
                args = DynamicModelBuilder.post_process_model_data(
                    structured_response.model_dump(), schema_dict
                )

                yield self._parts_manager.handle_tool_call_part(
                    vendor_part_id=0,
                    tool_name=tool_name,
                    args=args,
                    tool_call_id=DEFAULT_TOOL_CALL_ID
                )
            else:
                # Simple text request
                response = await self._model._client.simple_query(
                    message=user_content,
                    system_prompt=system_message
                )

                # Yield the response
                maybe_event = self._parts_manager.handle_text_delta(
                    vendor_part_id=0,
                    content=response
                )
                if maybe_event is not None:
                    yield maybe_event

        except Exception as e:
            # Yield an error event
            maybe_event = self._parts_manager.handle_text_delta(
                vendor_part_id=0,
                content=f"Error: {e}"
            )
            if maybe_event is not None:
                yield maybe_event
