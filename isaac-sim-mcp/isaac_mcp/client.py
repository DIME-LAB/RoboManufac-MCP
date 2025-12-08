#!/usr/bin/env python3
"""
MCP client
"""

import asyncio
import json
import os
from typing import List, Dict, Any

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    print("Install MCP: pip install mcp")
    exit(1)

class MCPClient:
    """MCP client that loads servers from configuration file"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Default to mcp_config.json in the same directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, "mcp_config.json")
        
        self.config_path = config_path
        self.config = self._load_config()
        self.available_tools = []
        self.servers = {}
        self.tool_to_server = {}  # Map tool names to server names
    
    def _load_config(self) -> Dict[str, Any]:
        """Load MCP configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file not found: {self.config_path}")
            return {"mcpServers": {}}
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in config file: {e}")
            return {"mcpServers": {}}
    
    def get_server_config(self, server_name: str) -> Dict[str, Any]:
        """Get configuration for a specific server"""
        return self.config.get("mcpServers", {}).get(server_name, {})
    
    def list_servers(self) -> List[str]:
        """List all available servers from config"""
        return list(self.config.get("mcpServers", {}).keys())
    
    async def run_with_server(self, server_name: str, tool_calls: List[Dict[str, Any]]):
        """Connect to a specific server and execute tools"""
        server_config = self.get_server_config(server_name)
        
        if not server_config:
            print(f"Server '{server_name}' not found in configuration")
            return
        
        if server_config.get("disabled", False):
            print(f"Server '{server_name}' is disabled")
            return
        
        # Create server parameters from config
        server_params = StdioServerParameters(
            command=server_config.get("command", "python3"),
            args=server_config.get("args", [])
        )
        
        timeout = server_config.get("timeout", 60)
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize
                await session.initialize()
                
                # Get available tools
                tools_result = await session.list_tools()
                self.available_tools = [tool.name for tool in tools_result.tools]
                
                print(f"Connected to {server_name}! Available tools: {self.available_tools}")
                
                # Execute tool calls
                if tool_calls:
                    print(f" Executing {len(tool_calls)} tool(s)...")
                    
                    for tool_call in tool_calls:
                        tool_name = tool_call.get("tool", tool_call.get("name"))
                        params = tool_call.get("params", tool_call.get("arguments", {}))
                        
                        if tool_name in self.available_tools:
                            print(f"     Calling {tool_name} with {params}")
                            
                            try:
                                result = await session.call_tool(tool_name, params)
                                if hasattr(result, 'content') and result.content:
                                    print(f"   Result: {result.content[0].text}")
                                else:
                                    print(f"   Tool executed successfully")
                            except Exception as e:
                                print(f"   Error: {e}")
                        else:
                            print(f"   Tool '{tool_name}' not available")
                else:
                    print("   No tools to execute")
    
    async def get_available_tools(self, server_name: str = None):
        """Get available tools from a specific server or all servers"""
        if server_name:
            servers_to_check = [server_name]
        else:
            servers_to_check = self.list_servers()
        
        all_tools = []
        self.tool_to_server = {}  # Reset tool mapping
        
        for server in servers_to_check:
            server_config = self.get_server_config(server)
            if not server_config or server_config.get("disabled", False):
                continue
            
            try:
                server_params = StdioServerParameters(
                    command=server_config.get("command", "python3"),
                    args=server_config.get("args", [])
                )
                
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools_result = await session.list_tools()
                        # Send full tool schemas
                        server_tools = []
                        for tool in tools_result.tools:
                            tool_info = {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.inputSchema if hasattr(tool, 'inputSchema') else {}
                            }
                            server_tools.append(tool_info)
                            # Map tool name to server for routing
                            self.tool_to_server[tool.name] = server
                        
                        all_tools.extend(server_tools)
                        
                        print(f"Server '{server}' tools: {server_tools}")
                        
            except Exception as e:
                print(f"Could not get tools from {server}: {e}")
        
        self.available_tools = all_tools
        return all_tools
    
    async def run_with_isaac(self, tool_calls: List[Dict[str, Any]]):
        """Convenience method to run with Isaac Sim server"""
        return await self.run_with_server("isaac-sim", tool_calls)
    
    async def run_tools(self, tool_calls: List[Dict[str, Any]]):
        """Run tools by routing them to the appropriate server"""
        if not tool_calls:
            return
        
        # Group tools by server
        tools_by_server = {}
        for tool_call in tool_calls:
            tool_name = tool_call.get("tool", tool_call.get("name"))
            server = self.tool_to_server.get(tool_name)
            
            if server:
                if server not in tools_by_server:
                    tools_by_server[server] = []
                tools_by_server[server].append(tool_call)
            else:
                print(f"Warning: Tool '{tool_name}' not found in any server")
        
        # Execute tools on each server
        for server, server_tools in tools_by_server.items():
            print(f"Executing {len(server_tools)} tool(s) on {server} server...")
            await self.run_with_server(server, server_tools)