"""Claude Code integration for pydantic-ai."""

from .claude_code_model import ClaudeCodeModel

__all__ = ["ClaudeCodeModel", "main"]


def main() -> None:
    print("Hello from clowclow!")
    print("Use: from clowclow import ClaudeCodeModel")
    print("Then: agent = Agent(ClaudeCodeModel())")
