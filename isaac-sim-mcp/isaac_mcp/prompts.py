#!/usr/bin/env python3
"""
Prompts module for Isaac Sim MCP Interface
Centralized location for all LLM prompts and instructions
"""

class SystemPrompts:
    """System-level prompts for LLM orchestration"""
    
    @staticmethod
    def get_tool_calling_prompt(tools_list: str, backend: str = "claude", tool_schemas: list = None) -> str:
        """
        Get system prompt for tool calling with MCP tools
        
        Args:
            tools_list: Comma-separated list of available tool names
            backend: LLM backend name (claude, chatgpt, ollama)
            tool_schemas: Full tool schemas with parameters (optional)
            
        Returns:
            Formatted system prompt string
        """
        base_prompt = f"""You are a helper agent for assembly processes in Isaac Sim. Use these MCP tools: {tools_list}

When responding with tool calls, use this format:
TOOLS: [{{"tool": "tool_name", "params": {{"key": "value"}}}}]

Example:
I'll check the scene for you.
TOOLS: [{{"tool": "get_scene_info", "params": {{}}}}]

If no tools are needed, respond normally without TOOLS: line."""
        
        # Add detailed tool information for non-Claude models
        if backend != "claude" and tool_schemas:
            base_prompt += "\n\nIMPORTANT TOOL PARAMETERS:\n"
            for tool in tool_schemas:
                if isinstance(tool, dict) and "name" in tool and "parameters" in tool:
                    tool_name = tool["name"]
                    params = tool.get("parameters", {}).get("properties", {})
                    required = tool.get("parameters", {}).get("required", [])
                    
                    if params:
                        base_prompt += f"\n{tool_name}:\n"
                        for param_name, param_info in params.items():
                            param_type = param_info.get("type", "unknown")
                            is_required = param_name in required
                            base_prompt += f"  - {param_name} ({param_type}){' [REQUIRED]' if is_required else ' [optional]'}\n"
        
        # Backend-specific customizations
        if backend == "claude":
            # Claude-specific instructions
            base_prompt += "\n\nYou are an expert at Isaac Sim assembly tasks. Use tools efficiently and provide clear explanations."
        elif backend == "chatgpt":
            # ChatGPT needs explicit parameter guidance
            base_prompt += "\n\nCRITICAL: Use EXACT parameter names from the tool definitions above. Do not guess parameter names."
        elif backend == "ollama":
            # Ollama needs very explicit instructions
            base_prompt += "\n\nCRITICAL: Use EXACT parameter names from the tool definitions above. Do not guess parameter names. Follow the required parameters exactly."
            
        return base_prompt
    
    @staticmethod
    def get_conversation_context() -> str:
        """Get instructions for maintaining conversation context"""
        return """
You are an AI assistant helping users with assembly processes in Isaac Sim.
Maintain context from previous messages to provide coherent responses.
If a user's request is unclear, ask clarifying questions before executing tools.
"""
    
    @staticmethod
    def get_error_handling() -> str:
        """Get instructions for handling errors gracefully"""
        return """
If a tool fails or returns an error:
1. Explain what went wrong in simple terms
2. Suggest alternative approaches if available
3. Ask the user for clarification if needed
"""
