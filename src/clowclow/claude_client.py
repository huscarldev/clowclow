"""Custom Claude Code SDK client wrapper for pydantic-ai integration."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Type, TypeVar
from pydantic import BaseModel

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions


T = TypeVar('T', bound=BaseModel)


class BasicResponse(BaseModel):
    """Basic response model for simple text outputs."""
    answer: str


def _resolve_schema_refs(schema: dict) -> dict:
    """Resolve $ref references in a JSON schema by inlining definitions.

    Converts schemas with $defs and $ref into flat schemas with nested objects inline.

    Args:
        schema: JSON schema with potential $ref references

    Returns:
        Schema with all $ref references resolved and inlined
    """
    def resolve_ref(ref_path: str, root_schema: dict) -> dict:
        """Resolve a $ref path like '#/$defs/Address' to its definition."""
        if not ref_path.startswith('#/'):
            raise ValueError(f"Only local refs supported, got: {ref_path}")

        parts = ref_path[2:].split('/')  # Remove '#/' and split
        current = root_schema
        for part in parts:
            current = current[part]
        return current

    def resolve_object(obj: dict, root_schema: dict) -> dict:
        """Recursively resolve all $ref in an object."""
        if isinstance(obj, dict):
            if '$ref' in obj:
                # Replace the $ref with the actual definition
                ref_def = resolve_ref(obj['$ref'], root_schema)
                # Recursively resolve the referenced definition
                return resolve_object(ref_def, root_schema)
            else:
                # Recursively resolve nested objects
                return {k: resolve_object(v, root_schema) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_object(item, root_schema) for item in obj]
        else:
            return obj

    # Create a copy and resolve all refs
    resolved = resolve_object(schema, schema)

    # Remove $defs from the final schema since everything is inlined
    if isinstance(resolved, dict) and '$defs' in resolved:
        resolved = {k: v for k, v in resolved.items() if k != '$defs'}

    return resolved


class CustomClaudeCodeClient:
    """Custom wrapper around Claude Code SDK for pydantic-ai integration."""
    
    def __init__(self, api_key: str | None = None, workspace_dir: Path | None = None):
        """Initialize the custom Claude Code client.
        
        Args:
            api_key: Anthropic API key. If not provided, will use ANTHROPIC_API_KEY env var.
            workspace_dir: Working directory for temporary files. If not provided, uses temp directory.
        """
        # API key validation is handled by ClaudeSDKClient internally
        _ = api_key  # Unused but kept for interface compatibility
        
        self.workspace_dir = workspace_dir or Path(tempfile.gettempdir())
        self.workspace_dir.mkdir(exist_ok=True)
    
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
        options = ClaudeAgentOptions(
            system_prompt=system_prompt or "You are a helpful assistant.",
            max_turns=max_turns,
            cwd=str(self.workspace_dir),
            permission_mode="acceptEdits",
            allowed_tools=["Read", "Write"]
        )

        response_parts = []
        temp_image_files = []

        try:
            # Handle multimodal content by saving images to files
            if isinstance(message, list):
                prompt_parts = []

                for block in message:
                    if block.get("type") == "text":
                        prompt_parts.append(block["text"])
                    elif block.get("type") == "image":
                        # Save image to temp file and ask Claude to read it
                        import base64
                        import tempfile

                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            # Decode base64 image
                            image_data = base64.b64decode(source["data"])
                            media_type = source.get("media_type", "image/png")
                            ext = media_type.split("/")[-1]

                            # Create temp file with descriptive name
                            import time
                            timestamp = int(time.time() * 1000)
                            temp_filename = f"vision_input_{timestamp}.{ext}"
                            temp_filepath = Path(self.workspace_dir) / temp_filename

                            # Write image data with explicit flush and sync
                            with open(temp_filepath, 'wb') as f:
                                f.write(image_data)
                                f.flush()
                                import os
                                os.fsync(f.fileno())

                            # Verify file was written correctly
                            if not temp_filepath.exists():
                                raise FileNotFoundError(f"Failed to create temp image file: {temp_filepath}")

                            file_size = temp_filepath.stat().st_size
                            if file_size != len(image_data):
                                raise IOError(f"Temp file size mismatch: {file_size} != {len(image_data)}")

                            temp_image_files.append(str(temp_filepath))

                            # Add instruction to read the image with absolute path
                            prompt_parts.append(f"Please read and analyze the image file at this exact path: {temp_filepath.absolute()}")

                # Combine all prompt parts
                final_prompt = "\n\n".join(prompt_parts)
            else:
                # Simple string message
                final_prompt = message

            async with ClaudeSDKClient(options=options) as client:
                await client.query(final_prompt)

                # Collect the streaming response
                async for response_message in client.receive_response():
                    if hasattr(response_message, 'content'):
                        for block in response_message.content:
                            if hasattr(block, 'text'):
                                response_parts.append(block.text)

            return ''.join(response_parts)

        finally:
            # Clean up temporary image files
            for temp_file in temp_image_files:
                try:
                    Path(temp_file).unlink()
                except Exception:
                    pass

    async def _structured_query_schema_tag(
        self,
        message: str | list[dict],
        pydantic_class: Type[T],
        system_prompt: str | None = None,
        custom_instructions: str | None = None,
        max_turns: int = None
    ) -> T:
        """Execute a query expecting structured output using <schema> tag method.

        Args:
            message: The user message/query (str or list of content blocks for multimodal)
            pydantic_class: The Pydantic model class for the expected response
            system_prompt: Optional system prompt
            custom_instructions: Additional instructions for the model
            max_turns: Maximum number of conversation turns

        Returns:
            An instance of the provided Pydantic class
        """
        temp_image_files = []

        try:
            # Generate JSON schema and resolve $ref references
            schema = pydantic_class.model_json_schema()
            resolved_schema = _resolve_schema_refs(schema)
            schema_json = json.dumps(resolved_schema, indent=2)

            # Handle multimodal content by saving images to files
            if isinstance(message, list):
                prompt_parts = []

                for block in message:
                    if block.get("type") == "text":
                        prompt_parts.append(block["text"])
                    elif block.get("type") == "image":
                        # Save image to temp file
                        import base64
                        import tempfile

                        source = block.get("source", {})
                        if source.get("type") == "base64":
                            # Decode base64 image
                            image_data = base64.b64decode(source["data"])
                            media_type = source.get("media_type", "image/png")
                            ext = media_type.split("/")[-1]

                            # Create temp file with descriptive name
                            import time
                            timestamp = int(time.time() * 1000)
                            temp_filename = f"vision_input_{timestamp}.{ext}"
                            temp_filepath = Path(self.workspace_dir) / temp_filename

                            # Write image data with explicit flush and sync
                            with open(temp_filepath, 'wb') as f:
                                f.write(image_data)
                                f.flush()
                                import os
                                os.fsync(f.fileno())

                            # Verify file was written correctly
                            if not temp_filepath.exists():
                                raise FileNotFoundError(f"Failed to create temp image file: {temp_filepath}")

                            file_size = temp_filepath.stat().st_size
                            if file_size != len(image_data):
                                raise IOError(f"Temp file size mismatch: {file_size} != {len(image_data)}")

                            temp_image_files.append(str(temp_filepath))

                            # Use absolute path for Claude Code to read
                            prompt_parts.append(f"Please read and analyze the image file at this exact path: {temp_filepath.absolute()}")

                user_prompt = "\n\n".join(prompt_parts)
            else:
                user_prompt = message

            # Create final prompt with schema
            final_prompt = f"""<schema>
{schema_json}
</schema>

{user_prompt}

{custom_instructions if custom_instructions else ''}

IMPORTANT: Respond with ONLY valid JSON that matches the schema above. No additional text or explanation.
- Follow the EXACT structure defined in the schema - include ONLY the fields listed in "properties"
- Do NOT add any extra fields not defined in the schema
- For nested objects (type: "object"), create a proper nested JSON object with ONLY the fields from that nested schema's properties
- Include ALL required fields (check the "required" array for each object and nested object)
- For optional fields with "default" values: you can either include them with appropriate values OR omit them (defaults will be used)
- Use the exact data types specified (string, number, object, array, etc.)
- RESPECT ALL CONSTRAINTS: "minimum", "maximum", "pattern", "minLength", "maxLength", etc.
- For fields with "pattern", ensure the value EXACTLY matches the regex pattern (e.g., "^[A-F]$" means a single letter A-F)
- For fields with "minimum"/"maximum", ensure the value is within the specified range"""

            # Create enhanced system prompt
            enhanced_system = f"""{system_prompt or 'You are a helpful assistant.'}

You must respond with valid JSON that exactly matches the provided schema.
Follow all constraints, required fields, and data types specified.
For nested objects (type: "object" with properties), create proper nested JSON objects with all their required fields.
Do not include any text outside of the JSON response."""

            options = ClaudeAgentOptions(
                system_prompt=enhanced_system,
                max_turns=max_turns,  # Allow multiple turns for tool usage (Read tool)
                cwd=str(self.workspace_dir),
                permission_mode="acceptEdits",
                allowed_tools=["Read", "Write"]
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

        except Exception as e:
            raise RuntimeError(f"Schema tag structured query failed: {e}") from e

        finally:
            # Clean up temporary image files
            for temp_file in temp_image_files:
                try:
                    Path(temp_file).unlink()
                except Exception:
                    pass

    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from response text that might contain extra content."""
        # First try to find JSON between triple backticks
        import re
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
        return await self._structured_query_schema_tag(
            message, pydantic_class, system_prompt, custom_instructions, max_turns
        )