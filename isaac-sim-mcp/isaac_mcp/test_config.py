#!/usr/bin/env python3
"""
Test script for MCP configuration
"""

import asyncio
import sys
import os
import argparse

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from client import MCPClient

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    print("Install MCP: pip install mcp")
    exit(1)

async def test_config(server_name=None, test_tools=False, test_prompts=False):
    """Test the MCP configuration"""
    print("Testing MCP Configuration...")
    
    # Create config client
    client = MCPClient()
    
    # List available servers
    servers = client.list_servers()
    print(f"Available servers: {servers}")
    
    # Test getting server config
    for server in servers:
        config = client.get_server_config(server)
        print(f"Server '{server}' config: {config}")
    
    # Test specific server or all servers
    if server_name:
        if server_name not in servers:
            print(f"Error: Server '{server_name}' not found in configuration")
            print(f"Available servers: {servers}")
            return
        
        print(f"\nTesting server: {server_name}")
        try:
            tools = await client.get_available_tools(server_name)
            print(f"Available tools from {server_name}: {tools}")
            
            if test_tools and tools:
                print(f"\nTesting tools from {server_name}...")
                # Test a simple tool call if available
                for tool in tools[:1]:  # Test first tool only
                    tool_name = tool.get("name", tool.get("tool"))
                    print(f"Testing tool: {tool_name}")
                    # You can add specific tool tests here
            elif test_tools and not tools:
                print(f"\nNo tools available from {server_name}")
                print("Note: This server might provide prompts instead of tools")
                print("Prompts are different from tools in MCP - they provide text templates rather than executable functions")
            
            # Test prompts if requested
            if test_prompts:
                print(f"\nTesting prompts from {server_name}...")
                try:
                    # Try to get prompts from the server
                    server_config = client.get_server_config(server_name)
                    server_params = StdioServerParameters(
                        command=server_config.get("command", "python3"),
                        args=server_config.get("args", [])
                    )
                    
                    async with stdio_client(server_params) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            
                            # Try to list prompts
                            try:
                                prompts_result = await session.list_prompts()
                                if hasattr(prompts_result, 'prompts') and prompts_result.prompts:
                                    print(f"Available prompts from {server_name}:")
                                    for prompt in prompts_result.prompts:
                                        print(f"  - {prompt.name}: {prompt.description}")
                                else:
                                    print(f"No prompts found from {server_name}")
                            except Exception as e:
                                print(f"Could not get prompts from {server_name}: {e}")
                                
                except Exception as e:
                    print(f"Error testing prompts from {server_name}: {e}")
                    
        except Exception as e:
            print(f"Error testing server {server_name}: {e}")
    else:
        # Test all servers
        print("\nTesting tool discovery for all servers...")
        try:
            tools = await client.get_available_tools()
            print(f"Available tools: {tools}")
        except Exception as e:
            print(f"Error getting tools: {e}")

def interactive_server_selection():
    """Interactive server selection"""
    client = MCPClient()
    servers = client.list_servers()
    
    if not servers:
        print("No servers found in configuration")
        return None
    
    print("\nAvailable servers:")
    for i, server in enumerate(servers, 1):
        print(f"{i}. {server}")
    
    while True:
        try:
            choice = input(f"\nSelect server (1-{len(servers)}) or 'all' for all servers: ").strip()
            
            if choice.lower() == 'all':
                return None
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(servers):
                return servers[choice_num - 1]
            else:
                print(f"Please enter a number between 1 and {len(servers)}")
        except ValueError:
            print("Please enter a valid number or 'all'")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None

def main():
    parser = argparse.ArgumentParser(description="Test MCP configuration")
    parser.add_argument("--server", "-s", help="Server name to test")
    parser.add_argument("--test-tools", "-t", action="store_true", help="Test available tools")
    parser.add_argument("--test-prompts", "-p", action="store_true", help="Test available prompts")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive server selection")
    
    args = parser.parse_args()
    
    if args.interactive:
        server_name = interactive_server_selection()
    else:
        server_name = args.server
    
    asyncio.run(test_config(server_name, args.test_tools, args.test_prompts))

if __name__ == "__main__":
    main()
