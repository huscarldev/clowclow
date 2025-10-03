# Architecture Patterns and Design Decisions

## Core Architecture

### 1. Model Adapter Pattern
`ClaudeCodeModel` acts as an adapter between Pydantic AI's `Model` interface and Claude Code SDK:
- Implements required methods: `request()`, `request_stream()`, `model_name()`, `system()`
- Translates Pydantic AI message format → Claude SDK format
- Translates Claude SDK responses → Pydantic AI response format

### 2. Structured Output Strategy
Uses **`<schema>` tag method** for structured output:
- Embeds JSON schema in prompt within `<schema>` tags
- Claude returns JSON matching the schema
- JSON is extracted and validated against dynamically created Pydantic model
- Post-processing: `None` → `[]` for arrays, `None` → `{}` for objects

### 3. Multimodal Content Handling
Images are converted from base64 to temporary files:
1. Detect image content in messages (base64 or ImageUrl)
2. Save image to temp file in workspace directory
3. Instruct Claude to read the file with absolute path
4. Clean up temp files after query completion
5. Uses increased `max_turns=5` for multimodal queries

### 4. Message Extraction Logic
- **User Message**: Extracts most recent user message from history
- **System Messages**: Combines all system messages with newlines
- **Multimodal**: Detects images in messages and creates file instructions
- Handles `UserPromptPart`, `TextPart`, `BinaryContent`, `ImageUrl` types

## Key Design Decisions

### Dynamic Pydantic Model Creation
```python
# JSON schema → Pydantic model at runtime
model_class = create_model(
    schema["title"] or "GeneratedModel",
    **fields
)
```
- Allows flexible structured output without pre-defined models
- Used in `_get_type_from_schema()` method

### Tool Mode vs Simple Mode
- **Tool Mode**: When Pydantic AI passes `tool_definitions` → structured output
- **Simple Mode**: No tools → text response
- Detected in `request()` method

### ClaudeCodeOptions Configuration
```python
ClaudeCodeOptions(
    permission_mode="acceptEdits",  # Auto-accept file edits
    allowed_tools=["Read", "Write"], # Limited for safety
    cwd=workspace_dir,              # For temp file ops
    max_turns=1 or 5                # Based on query type
)
```

### Response Streaming
- `request_stream()` wraps `simple_query()` in async iterator
- Returns `ClaudeCodeStreamedResponse` implementing `StreamedResponse`
- Yields complete response as single `TextPart` (no actual streaming to Claude SDK)

## Code Flow Examples

### Structured Output Flow
1. Pydantic AI calls `request()` with `tool_definitions`
2. Extract JSON schema from tool definition
3. Create dynamic Pydantic model via `create_model()`
4. Call `structured_query()` with schema embedded in `<schema>` tags
5. Extract JSON from response text
6. Validate against Pydantic model
7. Convert to `ToolCallPart` for Pydantic AI

### Multimodal Flow
1. Detect images in message parts
2. Save base64/URL images to temp files
3. Add file reading instructions to user message
4. Use `max_turns=5` for extended conversation
5. Clean up temp files after completion

## File Organization Philosophy
- **Separation of Concerns**: Model logic separate from client logic
- **Single Responsibility**: Each class has clear purpose
- **Encapsulation**: Private methods prefixed with `_`
- **Type Safety**: Extensive type hints throughout
