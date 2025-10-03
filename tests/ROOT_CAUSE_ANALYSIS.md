# Root Cause Analysis: Why ClaudeCodeModel Cannot Use Tools

## Executive Summary

**ClaudeCodeModel's `request()` method ignores the `function_tools` parameter that Pydantic AI passes to it.**

This is **not** a Pydantic AI issue - it's a **missing implementation** in ClaudeCodeModel.

## Technical Root Cause

### What Pydantic AI Does (Correctly)

When you register tools with an agent:

```python
agent = Agent(ClaudeCodeModel())

@agent.tool_plain
def test_tool(x: str) -> str:
    return f"Result: {x}"
```

Pydantic AI:
1. ✅ Stores the tool in the agent's toolset
2. ✅ Creates a `ToolDefinition` for the tool
3. ✅ Passes it to the model via `ModelRequestParameters.function_tools`

### What ClaudeCodeModel Does (Incorrectly)

**Evidence from intercepted request**:

```
ClaudeCodeModel.request() called with:
  - output_mode: text
  - function_tools: 1 tools
      - test_tool: A test tool that processes input.
  - output_tools: 0 tools
```

**What happens next**:
1. ❌ `request()` method checks `output_mode == 'tool'` → FALSE (it's 'text')
2. ❌ `request()` method checks `output_tools` → EMPTY (tools are in `function_tools`!)
3. ❌ Falls through to simple text query
4. ❌ Never looks at `model_request_parameters.function_tools`
5. ❌ Never formats tools for the API
6. ❌ Never handles tool call responses
7. ❌ Never invokes the user's tool function

## Code Analysis

### Current Implementation

```python
# src/clowclow/claude_code_model.py, line 78-82
if (model_request_parameters and
    hasattr(model_request_parameters, 'output_mode') and
    model_request_parameters.output_mode == 'tool' and
    model_request_parameters.output_tools):  # ← This is for structured OUTPUT

    # Handle structured output...
else:
    # ❌ Falls through to simple text query
    # ❌ NEVER checks model_request_parameters.function_tools
    response = await self._client.simple_query(...)
```

### What's Missing

The `request()` method needs to:

1. **Check for function_tools**:
   ```python
   if model_request_parameters.function_tools:
       # Handle tool calling mode
   ```

2. **Format tools for Claude API**:
   ```python
   tools = [
       {
           "name": tool.name,
           "description": tool.description,
           "input_schema": tool.parameters_json_schema
       }
       for tool in model_request_parameters.function_tools
   ]
   ```

3. **Send tools to Claude**:
   ```python
   response = await claude_client.message_with_tools(
       message=user_content,
       tools=tools,
       system_prompt=system_prompt
   )
   ```

4. **Handle tool call responses**:
   ```python
   if response contains tool_calls:
       # Extract tool calls from response
       # Return as ToolCallPart for Pydantic AI to invoke
       return ModelResponse(parts=[ToolCallPart(...)])
   ```

5. **Handle tool results**:
   ```python
   if messages contain ToolReturnPart:
       # Format tool results for Claude
       # Continue conversation with tool results
   ```

## Comparison: output_tools vs function_tools

### `output_tools` (Implemented ✅)
- **Purpose**: Structured output (Pydantic model as response)
- **Usage**: `agent = Agent(model, output_type=MyModel)`
- **Mode**: `output_mode = 'tool'`
- **What it does**: Asks Claude to respond in a specific JSON schema format
- **Implementation**: Uses `<schema>` tag method

### `function_tools` (NOT Implemented ❌)
- **Purpose**: User-defined tools that Claude can call
- **Usage**: `@agent.tool_plain def my_tool(...)`
- **Mode**: `output_mode = 'text'` (with tools available)
- **What it should do**: Let Claude decide when to call tools, invoke them, and continue
- **Implementation**: **MISSING**

## Why Tests Exposed This

### Weak Test (Didn't Catch It)
```python
@agent.tool_plain
def test_tool(x: str) -> str:
    return f"Result: {x}"

result = await agent.run("Use the tool")
assert result.output is not None  # ✅ PASSES - gets text response
```

**Weakness**: Only checked that _some_ response came back, not that the tool was used.

### Strong Test (Caught It)
```python
tool_call_count = 0

@agent.tool_plain
def test_tool(x: str) -> str:
    nonlocal tool_call_count
    tool_call_count += 1
    return f"Result: {x}"

result = await agent.run("Use the tool")
assert tool_call_count > 0  # ❌ FAILS - tool never called!
```

**Strength**: Verified that the tool was actually invoked.

## Evidence Summary

| Check | Result | Evidence |
|-------|--------|----------|
| Pydantic AI passes tools? | ✅ YES | `function_tools: 1 tools` in request |
| Tool definition correct? | ✅ YES | `test_tool: A test tool that processes input.` |
| ClaudeCodeModel checks function_tools? | ❌ NO | No code path checks this field |
| Tool gets invoked? | ❌ NO | `tool_called: False` |
| Claude aware of tool? | ❌ NO | Response: "I don't see a tool called 'test'" |

## Fix Complexity

**Complexity**: HIGH - Requires significant implementation

**What needs to be added**:
1. Tool calling request/response cycle
2. Tool call extraction from Claude responses
3. ToolCallPart creation for Pydantic AI
4. Tool result handling from ToolReturnPart
5. Multi-turn conversation with tool calls
6. Error handling for tool execution
7. Retry logic for tool call errors

**Estimated effort**:
- Code: ~200-300 lines
- Testing: ~50-100 lines
- Documentation: Significant
- **Total**: 2-3 days for experienced developer

## Recommendation

### Short Term ✅
- **Document** this limitation clearly
- **Skip** tool-related tests with clear explanations
- **Guide users** to use other Pydantic AI models for tool calling

### Long Term ⏭️
- **Implement** tool calling support in ClaudeCodeModel
- **Follow** Claude's function calling API patterns
- **Test** thoroughly with the improved test suite
- **Update** documentation when complete

## Conclusion

The root cause is clear: **ClaudeCodeModel.request() never checks `model_request_parameters.function_tools`**

This is:
- ✅ Not a Pydantic AI bug
- ✅ Not a test issue
- ✅ A missing feature in ClaudeCodeModel
- ✅ Properly exposed by improved testing

The good news: The test improvements **worked exactly as intended** - they exposed a real limitation that weak assertions were hiding.

---

**Analysis Date**: 2025-10-03
**Method**: Request interception and code inspection
**Confidence**: 100% - Root cause confirmed
