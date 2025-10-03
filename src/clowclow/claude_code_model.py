"""Claude Code model adapter for pydantic-ai."""

from __future__ import annotations

import os
import base64
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Literal

from pydantic_ai.models import Model, ModelResponse, ModelRequest, ModelSettings, StreamedResponse, ModelRequestParameters, RunContext, RequestUsage
from pydantic_ai import messages as _messages
from pydantic_ai._parts_manager import ModelResponsePartsManager

from .claude_client import CustomClaudeCodeClient, BasicResponse


class ClaudeCodeModel(Model):
    """A pydantic-ai Model implementation that uses Claude Code SDK."""
    
    def __init__(
        self, 
        api_key: str | None = None, 
        model_name: str = "claude-code",
        workspace_dir: Path | None = None
    ) -> None:
        """Initialize the Claude Code model.
        
        Args:
            api_key: Anthropic API key. If not provided, will use ANTHROPIC_API_KEY env var.
            model_name: Model identifier, defaults to "claude-code"
            workspace_dir: Working directory for temporary files
        """
        self._model_name = model_name
        self._client = CustomClaudeCodeClient(api_key=api_key, workspace_dir=workspace_dir)

    @property
    def model_name(self) -> str:
        """The name of the model."""
        return self._model_name

    @property
    def system(self) -> Literal["claude-code"]:
        """The system/provider name."""
        return "claude-code"

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
            # Check if messages contain images
            has_images = self._has_images(messages)

            # Extract messages
            if has_images:
                # Use multimodal content extraction
                user_content = self._extract_multimodal_content(messages)
            else:
                # Use text-only extraction
                user_content = self._extract_user_message(messages)

            system_prompt = self._extract_system_messages(messages)

            # Check if this is a structured output request (tool mode)
            if (model_request_parameters and
                hasattr(model_request_parameters, 'output_mode') and
                model_request_parameters.output_mode == 'tool' and
                model_request_parameters.output_tools):

                # Extract the JSON schema from the output tool
                output_tool = model_request_parameters.output_tools[0]
                schema_dict = output_tool.parameters_json_schema
                tool_name = output_tool.name

                # Create a dynamic Pydantic model from the schema
                from pydantic import create_model

                # Build field definitions from schema
                fields = {}
                properties = schema_dict.get('properties', {})
                required = schema_dict.get('required', [])

                for field_name, field_schema in properties.items():
                    field_type = self._get_type_from_schema(field_schema)

                    # Make field optional if not required
                    if field_name not in required:
                        # Check if the schema specifies a default value
                        if 'default' in field_schema:
                            # Use the default value from the schema
                            fields[field_name] = (field_type, field_schema['default'])
                        # Provide default values for optional fields without explicit defaults
                        elif field_schema.get('type') == 'array':
                            fields[field_name] = (field_type, [])
                        elif field_schema.get('type') == 'object':
                            fields[field_name] = (field_type, {})
                        else:
                            # Only add | None if the type doesn't already include None
                            # (anyOf with null already returns type | None)
                            if not ('anyOf' in field_schema):
                                field_type = field_type | None
                            fields[field_name] = (field_type, None)
                    else:
                        # Required field - check if it has a default (shouldn't normally, but handle it)
                        if 'default' in field_schema:
                            fields[field_name] = (field_type, field_schema['default'])
                        else:
                            fields[field_name] = (field_type, ...)

                # Create the dynamic model
                DynamicModel = create_model(schema_dict.get('title', 'OutputModel'), **fields)

                # Use structured query for JSON output
                structured_response = await self._client.structured_query(
                    message=user_content,  # Can be str or list[dict]
                    pydantic_class=DynamicModel,
                    system_prompt=system_prompt,
                    custom_instructions="Generate JSON that exactly matches the required schema. For list/array fields, use empty array [] instead of null if there are no items."
                )

                # Post-process: Replace None with [] for list fields and {} for dict fields
                args = structured_response.model_dump()
                for field_name, field_schema in properties.items():
                    if field_schema.get('type') == 'array' and args.get(field_name) is None:
                        args[field_name] = []
                    elif field_schema.get('type') == 'object' and args.get(field_name) is None:
                        args[field_name] = {}

                # Convert to a tool call response (this is what pydantic-ai expects for tool mode)
                tool_call = _messages.ToolCallPart(
                    tool_name=tool_name,
                    args=args,
                    tool_call_id="tool_call_1"
                )

                return ModelResponse(
                    parts=[tool_call],
                    timestamp=datetime.now(),
                )
            else:
                # Use simple query for text output
                response = await self._client.simple_query(
                    message=user_content,  # Can be str or list[dict]
                    system_prompt=system_prompt
                )
                return self._convert_response(response)

        except Exception as e:
            raise RuntimeError(f"Claude Code request failed: {e}") from e

    def _extract_user_message(self, messages: list[_messages.ModelMessage]) -> str:
        """Extract the most recent user message (text only)."""
        for msg in reversed(messages):
            if isinstance(msg, _messages.ModelRequest):
                # Extract text from the parts
                user_parts = []
                for part in msg.parts:
                    if isinstance(part, _messages.UserPromptPart):
                        user_parts.append(part.content)
                    elif isinstance(part, _messages.TextPart):
                        user_parts.append(part.content)
                return "\n".join(user_parts)
        return ""

    def _extract_multimodal_content(self, messages: list[_messages.ModelMessage]) -> list[dict]:
        """Extract multimodal content including text and images."""
        content_blocks = []

        for msg in reversed(messages):
            if isinstance(msg, _messages.ModelRequest):
                for part in msg.parts:
                    # Handle text content
                    if isinstance(part, _messages.UserPromptPart):
                        # UserPromptPart.content can be str or list
                        if isinstance(part.content, str):
                            content_blocks.append({
                                "type": "text",
                                "text": part.content
                            })
                        elif isinstance(part.content, list):
                            # Process each item in the list
                            for item in part.content:
                                if isinstance(item, str):
                                    content_blocks.append({"type": "text", "text": item})
                                elif isinstance(item, _messages.BinaryContent):
                                    # Convert BinaryContent to dict
                                    if isinstance(item.data, bytes):
                                        image_b64 = base64.b64encode(item.data).decode('utf-8')
                                    else:
                                        image_b64 = item.data
                                    content_blocks.append({
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": item.media_type or "image/png",
                                            "data": image_b64
                                        }
                                    })
                                elif isinstance(item, _messages.ImageUrl):
                                    # Convert ImageUrl to dict
                                    content_blocks.append({
                                        "type": "image",
                                        "source": {
                                            "type": "url",
                                            "url": item.url
                                        }
                                    })
                                elif isinstance(item, dict):
                                    # Already a dict, use as-is
                                    content_blocks.append(item)
                    elif isinstance(part, _messages.TextPart):
                        content_blocks.append({
                            "type": "text",
                            "text": part.content
                        })
                    # Handle image content (base64)
                    elif isinstance(part, _messages.BinaryContent):
                        # Encode image data to base64 if not already
                        if isinstance(part.data, bytes):
                            image_b64 = base64.b64encode(part.data).decode('utf-8')
                        else:
                            image_b64 = part.data

                        content_blocks.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": part.media_type or "image/png",
                                "data": image_b64
                            }
                        })
                    # Handle image URLs
                    elif isinstance(part, _messages.ImageUrl):
                        content_blocks.append({
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": part.url
                            }
                        })

                # Return after processing the most recent user message
                if content_blocks:
                    return content_blocks

        return content_blocks

    def _has_images(self, messages: list[_messages.ModelMessage]) -> bool:
        """Check if the messages contain any images."""
        for msg in messages:
            if isinstance(msg, _messages.ModelRequest):
                for part in msg.parts:
                    # Check direct types
                    if isinstance(part, (_messages.BinaryContent, _messages.ImageUrl)):
                        return True
                    # Check if UserPromptPart contains a list (multimodal)
                    if isinstance(part, _messages.UserPromptPart):
                        if isinstance(part.content, list):
                            return True  # List format indicates multimodal content
        return False


    def _extract_system_messages(self, messages: list[_messages.ModelMessage]) -> str:
        """Extract and combine system messages."""
        system_parts = []
        for msg in messages:
            if isinstance(msg, _messages.ModelRequest):
                if msg.instructions:
                    system_parts.append(msg.instructions)
        return "\n".join(system_parts)

    def _get_type_from_schema(self, field_schema: dict) -> type:
        """Convert JSON schema type to Python type."""
        # Handle $ref references (nested models)
        if '$ref' in field_schema:
            # For $ref, treat as dict since we'll handle nested validation separately
            # The actual structure will be validated by the nested Pydantic model
            return dict

        # Handle anyOf (union types, including optional fields with None)
        if 'anyOf' in field_schema:
            any_of = field_schema['anyOf']
            # Check if this is an optional field (union with null)
            has_null = any(item.get('type') == 'null' for item in any_of)
            # Get the non-null type
            non_null_types = [item for item in any_of if item.get('type') != 'null']
            if non_null_types:
                base_type = self._get_type_from_schema(non_null_types[0])
                if has_null:
                    return base_type | None
                return base_type
            return str  # Fallback

        schema_type = field_schema.get('type', 'string')

        if schema_type == 'string':
            return str
        elif schema_type == 'integer':
            return int
        elif schema_type == 'number':
            return float
        elif schema_type == 'boolean':
            return bool
        elif schema_type == 'array':
            # Handle arrays like list[str], list[int], etc.
            items_schema = field_schema.get('items', {})
            item_type = self._get_type_from_schema(items_schema)
            return list[item_type]
        elif schema_type == 'object':
            # Handle objects like dict[str, int], dict[str, str], etc.
            additional_properties = field_schema.get('additionalProperties', {})
            if additional_properties:
                value_type = self._get_type_from_schema(additional_properties)
                return dict[str, value_type]
            else:
                # If no additionalProperties, treat as generic dict
                return dict[str, str]
        else:
            # Default fallback
            return str

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
        # For now, we'll implement a simple non-streaming fallback
        # TODO: Implement actual streaming if claude-code-sdk supports it

        try:
            # Check if messages contain images
            has_images = self._model._has_images(self._messages)

            # Extract messages
            if has_images:
                user_content = self._model._extract_multimodal_content(self._messages)
            else:
                user_content = self._model._extract_user_message(self._messages)

            system_message = self._model._extract_system_messages(self._messages)

            # Check if this is a structured output request (tool mode)
            if (self._model_request_parameters and
                self._model_request_parameters.output_mode == 'tool' and
                self._model_request_parameters.output_tools):

                # Extract the JSON schema from the output tool
                output_tool = self._model_request_parameters.output_tools[0]
                schema_dict = output_tool.parameters_json_schema
                tool_name = output_tool.name

                # Create a dynamic Pydantic model from the schema
                from pydantic import create_model

                # Build field definitions from schema
                fields = {}
                properties = schema_dict.get('properties', {})
                required = schema_dict.get('required', [])

                for field_name, field_schema in properties.items():
                    field_type = self._model._get_type_from_schema(field_schema)

                    # Make field optional if not required
                    if field_name not in required:
                        # Check if the schema specifies a default value
                        if 'default' in field_schema:
                            # Use the default value from the schema
                            fields[field_name] = (field_type, field_schema['default'])
                        # Provide default values for optional fields without explicit defaults
                        elif field_schema.get('type') == 'array':
                            fields[field_name] = (field_type, [])
                        elif field_schema.get('type') == 'object':
                            fields[field_name] = (field_type, {})
                        else:
                            # Only add | None if the type doesn't already include None
                            # (anyOf with null already returns type | None)
                            if not ('anyOf' in field_schema):
                                field_type = field_type | None
                            fields[field_name] = (field_type, None)
                    else:
                        # Required field - check if it has a default (shouldn't normally, but handle it)
                        if 'default' in field_schema:
                            fields[field_name] = (field_type, field_schema['default'])
                        else:
                            fields[field_name] = (field_type, ...)

                # Create the dynamic model
                DynamicModel = create_model(schema_dict.get('title', 'OutputModel'), **fields)

                # Use structured query for JSON output
                structured_response = await self._model._client.structured_query(
                    message=user_content,  # Can be str or list[dict]
                    pydantic_class=DynamicModel,
                    system_prompt=system_message,
                    custom_instructions="Generate JSON that exactly matches the required schema. For list/array fields, use empty array [] instead of null if there are no items."
                )

                # Post-process: Replace None with [] for list fields and {} for dict fields
                args = structured_response.model_dump()
                for field_name, field_schema in properties.items():
                    if field_schema.get('type') == 'array' and args.get(field_name) is None:
                        args[field_name] = []
                    elif field_schema.get('type') == 'object' and args.get(field_name) is None:
                        args[field_name] = {}

                # Convert to a tool call response
                tool_call = _messages.ToolCallPart(
                    tool_name=tool_name,
                    args=args,
                    tool_call_id="tool_call_1"
                )

                # Yield a tool call event using parts manager
                yield self._parts_manager.handle_tool_call_part(
                    vendor_part_id=0,
                    tool_name=tool_call.tool_name,
                    args=tool_call.args,
                    tool_call_id=tool_call.tool_call_id
                )
            else:
                # Make simple text request to Claude Code
                response = await self._model._client.simple_query(
                    message=user_content,
                    system_prompt=system_message
                )

                # Yield the response using parts manager
                maybe_event = self._parts_manager.handle_text_delta(
                    vendor_part_id=0,
                    content=response
                )
                if maybe_event is not None:
                    yield maybe_event
            
        except Exception as e:
            # Yield an error event using parts manager
            maybe_event = self._parts_manager.handle_text_delta(
                vendor_part_id=0,
                content=f"Error: {e}"
            )
            if maybe_event is not None:
                yield maybe_event