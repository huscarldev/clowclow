# Pydantic AI Message Structure (Important Implementation Details)

## Critical Finding: System Prompts in Messages

### How System Prompts Are Stored

In Pydantic AI's `capture_run_messages()`, system prompts appear as **`SystemPromptPart`** in message parts, **NOT** in the `instructions` field.

### Incorrect Approach (Common Mistake)
```python
# ❌ WRONG - instructions is always None
with capture_run_messages() as messages:
    await agent.run("Test")

request_messages = [m for m in messages if m.kind == "request"]
# This will FAIL - instructions is None
assert any(m.instructions == "You are a helpful assistant." for m in request_messages)
```

### Correct Approach
```python
# ✅ CORRECT - Check SystemPromptPart in message parts
from pydantic_ai.messages import SystemPromptPart

with capture_run_messages() as messages:
    await agent.run("Test")

# Find SystemPromptPart in request messages
request_messages = [m for m in messages if m.kind == "request"]
system_parts = [
    part for msg in request_messages
    for part in msg.parts
    if isinstance(part, SystemPromptPart)
]
assert any(p.content == "You are a helpful assistant." for p in system_parts)
```

## Message Structure Details

### Request Message with System Prompt
```
Message (kind="request"):
  instructions: None  ← Not used for system prompts!
  parts: [
    SystemPromptPart(content="You are a helpful assistant."),  ← System prompt here!
    UserPromptPart(content="User's query")
  ]
```

### Request Message without System Prompt
```
Message (kind="request"):
  instructions: None
  parts: [
    UserPromptPart(content="User's query")
  ]
```

### Response Message
```
Message (kind="response"):
  parts: [
    TextPart(content="Model's response")
  ]
```

## Available Message Part Types

From `pydantic_ai.messages`:

1. **SystemPromptPart** - System prompt content
   - Attribute: `content` (str)
   
2. **UserPromptPart** - User message content
   - Attribute: `content` (str or list for multimodal)
   
3. **TextPart** - Text response from model
   - Attribute: `content` (str)
   
4. **ToolCallPart** - Tool/function call
   - Attributes: `tool_name`, `args`, `tool_call_id`
   
5. **ToolReturnPart** - Tool execution result
   - Attributes: `tool_name`, `content`, `tool_call_id`
   
6. **BinaryContent** - Binary data (images, etc.)
   - Attributes: `data`, `media_type`
   
7. **ImageUrl** - Image URL
   - Attribute: `url`

## Testing Message Inspection

### Pattern for Testing System Prompts
```python
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import SystemPromptPart

@pytest.mark.asyncio
async def test_system_prompt_in_messages():
    agent = Agent(model, system_prompt="You are helpful.")
    
    with capture_run_messages() as messages:
        await agent.run("Test")
    
    # Find all SystemPromptParts
    system_parts = [
        part for msg in messages if msg.kind == "request"
        for part in msg.parts
        if isinstance(part, SystemPromptPart)
    ]
    
    # Verify system prompt exists and has correct content
    assert len(system_parts) > 0
    assert any(p.content == "You are helpful." for p in system_parts)
```

### Pattern for Testing Multimodal Content
```python
from pydantic_ai.messages import BinaryContent, UserPromptPart

@pytest.mark.asyncio
async def test_multimodal_messages():
    with capture_run_messages() as messages:
        await agent.run([
            "Analyze this:",
            BinaryContent(data=image_bytes, media_type="image/png")
        ])
    
    # Check for image content in messages
    request_msg = messages[0]
    has_image = any(
        isinstance(part, BinaryContent) 
        for part in request_msg.parts
    )
    assert has_image
```

### Pattern for Testing Tool Calls
```python
from pydantic_ai.messages import ToolCallPart, ToolReturnPart

@pytest.mark.asyncio
async def test_tool_calling_in_messages():
    agent = Agent(model)
    
    @agent.tool_plain
    def my_tool(param: str) -> str:
        return f"Result: {param}"
    
    with capture_run_messages() as messages:
        await agent.run("Use the tool")
    
    # Find tool calls
    tool_calls = [
        part for msg in messages
        for part in msg.parts
        if isinstance(part, ToolCallPart)
    ]
    
    # Find tool returns
    tool_returns = [
        part for msg in messages
        for part in msg.parts
        if isinstance(part, ToolReturnPart)
    ]
    
    assert len(tool_calls) > 0
    assert len(tool_returns) > 0
```

## Message Kinds

Messages have a `kind` attribute with these values:

- `"request"` - User/system request to model
- `"response"` - Model's response
- `"retry-prompt"` - Retry after error (with ModelRetry)

## Common Pitfalls

### ❌ Pitfall 1: Checking instructions field
```python
# This will always be None for system prompts
assert msg.instructions == "System prompt"
```

### ❌ Pitfall 2: Not handling multimodal content
```python
# This fails for multimodal messages where content is a list
user_msg = msg.parts[0].content  # May be a list, not a string!
```

### ❌ Pitfall 3: Assuming message order
```python
# Messages may have retries or multiple rounds
first_msg = messages[0]  # May not always be the user's original request
```

### ✅ Best Practice: Iterate and Filter
```python
# Filter by kind and part type
user_prompts = [
    part.content for msg in messages if msg.kind == "request"
    for part in msg.parts
    if isinstance(part, UserPromptPart)
]

system_prompts = [
    part.content for msg in messages if msg.kind == "request"
    for part in msg.parts
    if isinstance(part, SystemPromptPart)
]
```

## Why This Matters for Testing

When testing custom Model implementations (like ClaudeCodeModel):

1. **Message inspection validates correct behavior** - Ensures your model properly constructs and sends messages
2. **System prompt handling must be tested correctly** - Use SystemPromptPart, not instructions
3. **Multimodal content must be validated** - Check for BinaryContent or ImageUrl parts
4. **Tool calling verification** - Inspect ToolCallPart and ToolReturnPart

## Reference

- [Pydantic AI Messages API](https://ai.pydantic.dev/api/messages/)
- [Testing Documentation](https://ai.pydantic.dev/testing/)
- Test example: `tests/integration/test_agent_basic.py::TestMessageInspection`
