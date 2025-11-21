"""
Tool-Calling Execution Engine

Implements function calling capabilities for AI agents with safety controls
and integration with existing model infrastructure.
"""

import json
import logging
import inspect
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor
import traceback

logger = logging.getLogger(__name__)


class ToolType(Enum):
    """Types of tools available"""
    SEARCH = "search"
    CALCULATION = "calculation"
    FILE_OPERATION = "file_operation"
    API_CALL = "api_call"
    DATABASE_QUERY = "database_query"
    DOCUMENT_ANALYSIS = "document_analysis"
    LEGAL_LOOKUP = "legal_lookup"


@dataclass
class ToolParameter:
    """Tool parameter definition"""
    name: str
    param_type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    enum_values: Optional[List[str]] = None


@dataclass
class Tool:
    """Tool definition"""
    name: str
    description: str
    tool_type: ToolType
    function: Callable
    parameters: List[ToolParameter]
    is_async: bool = False
    timeout_seconds: int = 30
    requires_confirmation: bool = False
    safety_level: str = "safe"  # "safe", "moderate", "dangerous"
    usage_count: int = 0
    avg_execution_time: float = 0.0


@dataclass
class ToolCall:
    """Tool call request"""
    tool_name: str
    parameters: Dict[str, Any]
    call_id: str = field(default_factory=lambda: str(int(time.time() * 1000)))
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolResult:
    """Tool execution result"""
    call_id: str
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)


class ToolRegistry:
    """
    Registry for managing available tools
    """
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Safety settings
        self.enable_dangerous_tools = False
        self.require_confirmation_for_moderate = True
        
        # Statistics
        self.stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "avg_execution_time": 0.0
        }
        
        # Initialize built-in tools
        self._register_builtin_tools()
        
        logger.info("🔧 Tool Registry initialized")
        logger.info(f"   Available tools: {len(self.tools)}")
    
    def register_tool(
        self,
        name: str,
        function: Callable,
        description: str,
        tool_type: ToolType,
        parameters: List[ToolParameter],
        **kwargs
    ) -> bool:
        """Register a new tool"""
        try:
            # Validate function
            if not callable(function):
                raise ValueError("Function must be callable")
            
            # Check if async
            is_async = asyncio.iscoroutinefunction(function)
            
            tool = Tool(
                name=name,
                description=description,
                tool_type=tool_type,
                function=function,
                parameters=parameters,
                is_async=is_async,
                **kwargs
            )
            
            self.tools[name] = tool
            logger.info(f"✅ Registered tool: {name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to register tool '{name}': {e}")
            return False
    
    def _register_builtin_tools(self):
        """Register built-in tools"""
        
        # Search tool
        self.register_tool(
            name="search_documents",
            function=self._search_documents,
            description="Search through legal documents",
            tool_type=ToolType.SEARCH,
            parameters=[
                ToolParameter("query", "string", "Search query", required=True),
                ToolParameter("limit", "number", "Maximum results", required=False, default=5)
            ]
        )
        
        # Calculator tool
        self.register_tool(
            name="calculate",
            function=self._calculate,
            description="Perform mathematical calculations",
            tool_type=ToolType.CALCULATION,
            parameters=[
                ToolParameter("expression", "string", "Mathematical expression", required=True)
            ]
        )
        
        # Legal lookup tool
        self.register_tool(
            name="legal_lookup",
            function=self._legal_lookup,
            description="Look up legal terms and definitions",
            tool_type=ToolType.LEGAL_LOOKUP,
            parameters=[
                ToolParameter("term", "string", "Legal term to look up", required=True),
                ToolParameter("jurisdiction", "string", "Legal jurisdiction", required=False, default="general")
            ]
        )
        
        # Document analysis tool
        self.register_tool(
            name="analyze_document",
            function=self._analyze_document,
            description="Analyze document structure and content",
            tool_type=ToolType.DOCUMENT_ANALYSIS,
            parameters=[
                ToolParameter("document_id", "string", "Document identifier", required=True),
                ToolParameter("analysis_type", "string", "Type of analysis", required=False, 
                            enum_values=["structure", "content", "summary"], default="content")
            ]
        )
        
        logger.info(f"✅ Registered {len(self.tools)} built-in tools")
    
    async def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call"""
        start_time = time.time()
        
        if tool_call.tool_name not in self.tools:
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                success=False,
                error=f"Tool '{tool_call.tool_name}' not found"
            )
        
        tool = self.tools[tool_call.tool_name]
        
        # Safety check
        if not self._is_tool_allowed(tool):
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                success=False,
                error=f"Tool '{tool_call.tool_name}' not allowed (safety level: {tool.safety_level})"
            )
        
        try:
            # Validate parameters
            validated_params = self._validate_parameters(tool, tool_call.parameters)
            
            # Execute tool
            if tool.is_async:
                result = await asyncio.wait_for(
                    tool.function(**validated_params),
                    timeout=tool.timeout_seconds
                )
            else:
                # Run sync function in thread pool
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda: tool.function(**validated_params)
                    ),
                    timeout=tool.timeout_seconds
                )
            
            execution_time = time.time() - start_time
            
            # Update statistics
            tool.usage_count += 1
            tool.avg_execution_time = tool.avg_execution_time * 0.9 + execution_time * 0.1
            self.stats["total_calls"] += 1
            self.stats["successful_calls"] += 1
            self.stats["avg_execution_time"] = (
                self.stats["avg_execution_time"] * 0.9 + execution_time * 0.1
            )
            
            logger.info(f"✅ Tool executed: {tool_call.tool_name} ({execution_time:.2f}s)")
            
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                success=True,
                result=result,
                execution_time=execution_time
            )
            
        except asyncio.TimeoutError:
            error_msg = f"Tool execution timed out after {tool.timeout_seconds}s"
            logger.error(f"⏰ {error_msg}")
            self.stats["failed_calls"] += 1
            
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                success=False,
                error=error_msg,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            logger.debug(traceback.format_exc())
            self.stats["failed_calls"] += 1
            
            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                success=False,
                error=error_msg,
                execution_time=time.time() - start_time
            )
    
    def _is_tool_allowed(self, tool: Tool) -> bool:
        """Check if tool is allowed based on safety settings"""
        if tool.safety_level == "dangerous" and not self.enable_dangerous_tools:
            return False
        
        if tool.safety_level == "moderate" and self.require_confirmation_for_moderate:
            # In a real implementation, this would prompt for confirmation
            logger.warning(f"⚠️  Tool '{tool.name}' requires confirmation (moderate safety)")
        
        return True
    
    def _validate_parameters(self, tool: Tool, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool parameters"""
        validated = {}
        
        for param in tool.parameters:
            if param.name in parameters:
                value = parameters[param.name]
                
                # Type validation (basic)
                if param.param_type == "number" and not isinstance(value, (int, float)):
                    try:
                        value = float(value)
                    except ValueError:
                        raise ValueError(f"Parameter '{param.name}' must be a number")
                
                elif param.param_type == "boolean" and not isinstance(value, bool):
                    if isinstance(value, str):
                        value = value.lower() in ("true", "1", "yes")
                    else:
                        value = bool(value)
                
                # Enum validation
                if param.enum_values and value not in param.enum_values:
                    raise ValueError(f"Parameter '{param.name}' must be one of: {param.enum_values}")
                
                validated[param.name] = value
                
            elif param.required:
                raise ValueError(f"Required parameter '{param.name}' missing")
            
            elif param.default is not None:
                validated[param.name] = param.default
        
        return validated
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get OpenAI-compatible tool schema"""
        if tool_name not in self.tools:
            return None
        
        tool = self.tools[tool_name]
        
        properties = {}
        required = []
        
        for param in tool.parameters:
            prop = {
                "type": param.param_type,
                "description": param.description
            }
            
            if param.enum_values:
                prop["enum"] = param.enum_values
            
            if param.default is not None:
                prop["default"] = param.default
            
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
    
    def get_all_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all available tools"""
        schemas = []
        for tool_name in self.tools:
            schema = self.get_tool_schema(tool_name)
            if schema:
                schemas.append(schema)
        return schemas
    
    # Built-in tool implementations
    def _search_documents(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Search documents (placeholder implementation)"""
        try:
            # This would integrate with the existing RAG system
            from src.rag.retriever import LegalRetriever
            
            # Placeholder results
            results = [
                {
                    "title": f"Document {i+1}",
                    "content": f"Content related to '{query}'",
                    "score": 0.9 - i * 0.1
                }
                for i in range(min(limit, 3))
            ]
            
            return {
                "query": query,
                "results": results,
                "total_found": len(results)
            }
            
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return {"error": str(e)}
    
    def _calculate(self, expression: str) -> Dict[str, Any]:
        """Safe calculator (basic math only)"""
        try:
            # Whitelist of safe operations
            allowed_chars = set("0123456789+-*/()., ")
            allowed_names = {"abs", "round", "min", "max", "pow"}
            
            # Basic safety check
            if not all(c in allowed_chars for c in expression if c.isalnum() or c in "+-*/()."):
                raise ValueError("Expression contains unsafe characters")
            
            # Evaluate safely (very basic - in production use a proper math parser)
            result = eval(expression, {"__builtins__": {}}, {
                "abs": abs, "round": round, "min": min, "max": max, "pow": pow
            })
            
            return {
                "expression": expression,
                "result": result,
                "type": type(result).__name__
            }
            
        except Exception as e:
            return {"error": f"Calculation failed: {str(e)}"}
    
    def _legal_lookup(self, term: str, jurisdiction: str = "general") -> Dict[str, Any]:
        """Legal term lookup (placeholder)"""
        # This would integrate with a legal database
        legal_definitions = {
            "contract": "A legally binding agreement between two or more parties",
            "tort": "A civil wrong that causes harm to another person",
            "jurisdiction": "The authority of a court to hear and decide cases",
            "precedent": "A legal principle established in a previous case"
        }
        
        definition = legal_definitions.get(term.lower())
        
        if definition:
            return {
                "term": term,
                "definition": definition,
                "jurisdiction": jurisdiction,
                "source": "Legal Dictionary"
            }
        else:
            return {
                "term": term,
                "error": "Term not found in legal dictionary"
            }
    
    def _analyze_document(self, document_id: str, analysis_type: str = "content") -> Dict[str, Any]:
        """Document analysis (placeholder)"""
        # This would integrate with document processing pipeline
        return {
            "document_id": document_id,
            "analysis_type": analysis_type,
            "summary": f"Analysis of document {document_id}",
            "key_points": [
                "Key point 1",
                "Key point 2",
                "Key point 3"
            ],
            "metadata": {
                "pages": 10,
                "word_count": 2500,
                "language": "English"
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics"""
        tool_stats = {}
        for name, tool in self.tools.items():
            tool_stats[name] = {
                "usage_count": tool.usage_count,
                "avg_execution_time": tool.avg_execution_time,
                "tool_type": tool.tool_type.value,
                "safety_level": tool.safety_level
            }
        
        return {
            "global_stats": self.stats,
            "tool_stats": tool_stats,
            "total_tools": len(self.tools)
        }


class ToolCallingAgent:
    """
    Agent that can call tools based on model responses
    """
    
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self.conversation_history = []
        
        logger.info("🤖 Tool Calling Agent initialized")
    
    async def process_with_tools(
        self,
        query: str,
        context: str = "",
        model_key: str = "mamba",
        max_tool_calls: int = 3
    ) -> Dict[str, Any]:
        """
        Process query with tool calling capabilities
        
        Args:
            query: User query
            context: Additional context
            model_key: Model to use for generation
            max_tool_calls: Maximum number of tool calls allowed
            
        Returns:
            Response with tool call results
        """
        try:
            # Get available tools
            available_tools = self.tool_registry.get_all_tool_schemas()
            
            # Generate initial response with tool awareness
            system_prompt = self._create_system_prompt(available_tools)
            full_prompt = f"{system_prompt}\n\nUser: {query}\nAssistant:"
            
            # Generate response (this would integrate with existing generation)
            from src.core.generator import generate_answer
            
            response = generate_answer(
                model_key=model_key,
                prompt=full_prompt,
                context=context,
                max_new_tokens=512
            )
            
            initial_answer = response.get("answer", "")
            
            # Parse for tool calls
            tool_calls = self._parse_tool_calls(initial_answer)
            
            tool_results = []
            final_answer = initial_answer
            
            # Execute tool calls
            if tool_calls and len(tool_calls) <= max_tool_calls:
                logger.info(f"🔧 Executing {len(tool_calls)} tool calls")
                
                for tool_call in tool_calls:
                    result = await self.tool_registry.execute_tool(tool_call)
                    tool_results.append(result)
                    
                    if result.success:
                        logger.info(f"✅ Tool '{tool_call.tool_name}' succeeded")
                    else:
                        logger.warning(f"❌ Tool '{tool_call.tool_name}' failed: {result.error}")
                
                # Generate final response with tool results
                if tool_results:
                    tool_context = self._format_tool_results(tool_results)
                    final_prompt = f"{full_prompt}\n\nTool Results:\n{tool_context}\n\nFinal Answer:"
                    
                    final_response = generate_answer(
                        model_key=model_key,
                        prompt=final_prompt,
                        context=context,
                        max_new_tokens=512
                    )
                    
                    final_answer = final_response.get("answer", initial_answer)
            
            # Store conversation
            self.conversation_history.append({
                "query": query,
                "initial_answer": initial_answer,
                "tool_calls": [{"name": tc.tool_name, "params": tc.parameters} for tc in tool_calls],
                "tool_results": [{"success": tr.success, "result": tr.result} for tr in tool_results],
                "final_answer": final_answer,
                "timestamp": time.time()
            })
            
            return {
                "answer": final_answer,
                "tool_calls_made": len(tool_calls),
                "tool_results": tool_results,
                "model_used": model_key,
                "has_tool_calls": len(tool_calls) > 0
            }
            
        except Exception as e:
            logger.error(f"Tool calling process failed: {e}")
            return {
                "answer": f"I encountered an error while processing your request: {str(e)}",
                "error": str(e),
                "tool_calls_made": 0,
                "tool_results": []
            }
    
    def _create_system_prompt(self, available_tools: List[Dict[str, Any]]) -> str:
        """Create system prompt with tool information"""
        tool_descriptions = []
        for tool_schema in available_tools:
            func = tool_schema["function"]
            tool_descriptions.append(f"- {func['name']}: {func['description']}")
        
        tools_text = "\n".join(tool_descriptions)
        
        return f"""You are a helpful AI assistant with access to tools. You can call tools to help answer questions.

Available tools:
{tools_text}

To call a tool, use this format:
TOOL_CALL: tool_name(parameter1="value1", parameter2="value2")

You can call multiple tools if needed. Always provide a helpful response based on the tool results."""
    
    def _parse_tool_calls(self, text: str) -> List[ToolCall]:
        """Parse tool calls from model response"""
        tool_calls = []
        
        # Simple regex-based parsing (in production, use a proper parser)
        import re
        
        pattern = r'TOOL_CALL:\s*(\w+)\((.*?)\)'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for tool_name, params_str in matches:
            try:
                # Parse parameters (very basic - in production use proper parsing)
                parameters = {}
                if params_str.strip():
                    # Simple key=value parsing
                    param_pairs = params_str.split(',')
                    for pair in param_pairs:
                        if '=' in pair:
                            key, value = pair.split('=', 1)
                            key = key.strip().strip('"\'')
                            value = value.strip().strip('"\'')
                            parameters[key] = value
                
                tool_call = ToolCall(
                    tool_name=tool_name,
                    parameters=parameters
                )
                tool_calls.append(tool_call)
                
            except Exception as e:
                logger.warning(f"Failed to parse tool call: {e}")
        
        return tool_calls
    
    def _format_tool_results(self, tool_results: List[ToolResult]) -> str:
        """Format tool results for model context"""
        formatted_results = []
        
        for result in tool_results:
            if result.success:
                formatted_results.append(f"{result.tool_name}: {json.dumps(result.result, indent=2)}")
            else:
                formatted_results.append(f"{result.tool_name}: ERROR - {result.error}")
        
        return "\n".join(formatted_results)


# Factory functions
def create_tool_registry() -> ToolRegistry:
    """Create tool registry with built-in tools"""
    return ToolRegistry()


def create_tool_calling_agent(tool_registry: ToolRegistry) -> ToolCallingAgent:
    """Create tool calling agent"""
    return ToolCallingAgent(tool_registry)


# Integration with existing system
def integrate_tool_calling():
    """
    Integration function for existing MARK AI system
    """
    try:
        logger.info("🔗 Integrating tool calling with existing system")
        
        registry = create_tool_registry()
        agent = create_tool_calling_agent(registry)
        
        async def enhanced_generate_with_tools(
            query: str,
            context: str = "",
            model_key: str = "mamba",
            enable_tools: bool = True,
            **kwargs
        ):
            """Enhanced generation with tool calling"""
            
            if enable_tools:
                return await agent.process_with_tools(
                    query=query,
                    context=context,
                    model_key=model_key
                )
            else:
                # Regular generation without tools
                from src.core.generator import generate_answer
                return generate_answer(
                    model_key=model_key,
                    prompt=query,
                    context=context,
                    **kwargs
                )
        
        return enhanced_generate_with_tools, registry, agent
        
    except Exception as e:
        logger.error(f"⚠️  Could not integrate tool calling: {e}")
        return None, None, None
