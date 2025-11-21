"""
Agents Package

AI agents with tool calling and orchestration capabilities.
"""

from .tool_calling import (
    ToolRegistry,
    ToolCallingAgent,
    Tool,
    ToolCall,
    ToolResult,
    ToolParameter,
    ToolType,
    create_tool_registry,
    create_tool_calling_agent,
    integrate_tool_calling
)

__all__ = [
    'ToolRegistry',
    'ToolCallingAgent',
    'Tool',
    'ToolCall',
    'ToolResult',
    'ToolParameter',
    'ToolType',
    'create_tool_registry',
    'create_tool_calling_agent',
    'integrate_tool_calling'
]
