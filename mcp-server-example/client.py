import streamlit as st
import asyncio
import json
import traceback
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
import os
from dotenv import load_dotenv
import sys
from typing import Dict, List, Tuple, Any

# Load MCP configuration from config file
def load_mcp_config(config_path="mcp_config.json"):
    """Load MCP server configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Config file {config_path} not found")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}")
        return None

# Load environment variables
load_dotenv()

# Load configuration
config = load_mcp_config()
if not config:
    print("Failed to load MCP configuration")
    exit(1)

# Find all enabled servers
enabled_servers = []
for server_name, server_config in config["mcpServers"].items():
    if not server_config.get("disabled", False):
        enabled_servers.append((server_name, server_config))

if not enabled_servers:
    print("No enabled servers found in configuration")
    exit(1)

# Store server configurations for UI selection
SERVER_CONFIGS = {name: config for name, config in enabled_servers}

def get_openai_tools_from_servers(tools_by_server: Dict[str, Any], enabled_tool_names: Dict[str, List[str]] = None):
    """Convert MCP tools from multiple servers to OpenAI format"""
    all_tools = []
    
    for server_name, tools_result in tools_by_server.items():
        server_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": f"[{server_name}] {tool.description}",
                    "parameters": tool.inputSchema,
                },
            }
            for tool in tools_result.tools
        ]
        
        # Filter tools based on enabled list for this server
        if enabled_tool_names and server_name in enabled_tool_names:
            server_tools = [
                tool for tool in server_tools 
                if tool["function"]["name"] in enabled_tool_names[server_name]
            ]
        
        all_tools.extend(server_tools)
    
    return all_tools

def fetch_available_tools(server_name):
    """Fetch available tools from a server synchronously"""
    async def _fetch():
        server_config = SERVER_CONFIGS[server_name]
        server_params = StdioServerParameters(
            command=server_config["command"],
            args=server_config["args"]
        )
        
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
                    return [(tool.name, tool.description) for tool in tools_result.tools]
        except Exception as e:
            st.error(f"Error fetching tools from {server_name}: {str(e)}")
            return []
    
    return asyncio.run(_fetch())

async def chat_response_multi_server(messages, sessions_dict, tool_to_server_map, openai_tools, client):
    """Handle chat response with multiple server sessions"""
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=messages,
        tools=openai_tools,
        tool_choice="auto",
    )
    messages.append(response.choices[0].message)
    tool_calls = response.choices[0].message.tool_calls
    
    # Handle any tool calls
    if response.choices[0].message.tool_calls:
        for tool_execution in tool_calls:
            tool_name = tool_execution.function.name
            
            # Find which server has this tool
            server_name = tool_to_server_map.get(tool_name)
            if not server_name or server_name not in sessions_dict:
                # Add error response if tool not found
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_execution.id,
                    "content": f"Error: Tool {tool_name} not found in any connected server",
                })
                continue
            
            # Execute tool call on the correct server
            session = sessions_dict[server_name]
            try:
                result = await session.call_tool(
                    tool_name,
                    arguments=json.loads(tool_execution.function.arguments),
                )
                # Add tool response to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_execution.id,
                    "content": result.content[0].text,
                })
            except Exception as e:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_execution.id,
                    "content": f"Error executing tool {tool_name} on {server_name}: {str(e)}",
                })
        
        # Get response from LLM after tool execution
        response = client.chat.completions.create(
            model='gpt-4o',
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
        )
        
        if response.choices[0].finish_reason == "tool_calls":
            # Recursively handle more tool calls
            return await chat_response_multi_server(messages, sessions_dict, tool_to_server_map, openai_tools, client)
        elif response.choices[0].finish_reason == "stop":
            messages.append(response.choices[0].message)
            return response.choices[0].message.content
    else:
        messages.append(response.choices[0].message)
        return response.choices[0].message.content

def sync_chat_response_multi_server(messages, user_input, selected_servers, enabled_tools_dict=None):
    """Handle chat with multiple servers simultaneously"""
    async def _chat():
        from contextlib import AsyncExitStack
        
        # Create sessions for all selected servers
        sessions_dict = {}
        tools_by_server = {}
        tool_to_server_map = {}
        
        async with AsyncExitStack() as stack:
            # Connect to all selected servers
            for server_name in selected_servers:
                server_config = SERVER_CONFIGS[server_name]
                server_params = StdioServerParameters(
                    command=server_config["command"],
                    args=server_config["args"]
                )
                
                # Create server context with proper async context management
                read, write = await stack.enter_async_context(stdio_client(server_params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                
                # Store session
                sessions_dict[server_name] = session
                
                # Get tools from this server
                tools_result = await session.list_tools()
                tools_by_server[server_name] = tools_result
                
                # Map tools to servers
                for tool in tools_result.tools:
                    tool_to_server_map[tool.name] = server_name
            
            # Aggregate tools from all servers
            openai_tools = get_openai_tools_from_servers(tools_by_server, enabled_tools_dict)
            
            # Add user message
            messages.append({"role": "user", "content": user_input})
            
            # Get chat response using all servers
            client = OpenAI()
            response = await chat_response_multi_server(
                messages, sessions_dict, tool_to_server_map, openai_tools, client
            )
            
            return response
    
    return asyncio.run(_chat())

def main():
    st.title("🤖 MCP Chatbot (Multi-Server Support)")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "selected_servers" not in st.session_state:
        st.session_state["selected_servers"] = []
    if "available_tools" not in st.session_state:
        st.session_state["available_tools"] = {}
    if "enabled_tools" not in st.session_state:
        st.session_state["enabled_tools"] = {}
    
    # Server selection in sidebar
    with st.sidebar:
        st.header("🖥️ Server Selection")
        server_names = list(SERVER_CONFIGS.keys())
        
        # Multi-select servers
        st.markdown("**Select MCP Servers:**")
        selected_servers = []
        for server_name in server_names:
            is_selected = server_name in st.session_state["selected_servers"]
            selected = st.checkbox(
                f"**{server_name}**",
                value=is_selected,
                key=f"server_select_{server_name}"
            )
            if selected:
                selected_servers.append(server_name)
        
        # Update selected servers in session state
        if set(selected_servers) != set(st.session_state["selected_servers"]):
            st.session_state["selected_servers"] = selected_servers
            # Reset chat when servers change
            st.session_state["messages"] = []
            st.session_state["history"] = []
            st.rerun()
        
        if not selected_servers:
            st.warning("⚠️ No servers selected. Please select at least one server.")
        else:
            st.success(f"✓ {len(selected_servers)} server(s) selected")
        
        st.markdown("---")
        
        # Tool selection for each selected server
        st.header("🛠️ Tool Settings")
        
        for server_name in selected_servers:
            # Fetch available tools for this server (cached per server)
            if server_name not in st.session_state["available_tools"]:
                with st.spinner(f"Fetching tools from {server_name}..."):
                    tools = fetch_available_tools(server_name)
                    st.session_state["available_tools"][server_name] = tools
                    # By default, enable all tools
                    st.session_state["enabled_tools"][server_name] = [tool[0] for tool in tools]
            
            available_tools = st.session_state["available_tools"].get(server_name, [])
            
            # Expandable section for each server's tools
            with st.expander(f"📦 {server_name} ({len(available_tools)} tools)", expanded=False):
                if available_tools:
                    # Add "Select All" / "Deselect All" buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✓ All", key=f"select_all_{server_name}"):
                            st.session_state["enabled_tools"][server_name] = [tool[0] for tool in available_tools]
                            st.rerun()
                    with col2:
                        if st.button("✗ None", key=f"deselect_all_{server_name}"):
                            st.session_state["enabled_tools"][server_name] = []
                            st.rerun()
                    
                    # Show enabled tool count
                    enabled_count = len(st.session_state["enabled_tools"].get(server_name, []))
                    st.caption(f"✓ {enabled_count} / {len(available_tools)} enabled")
                    
                    st.markdown("---")
                    
                    # Display each tool with checkbox
                    for tool_name, tool_description in available_tools:
                        is_enabled = tool_name in st.session_state["enabled_tools"].get(server_name, [])
                        
                        # Create checkbox for each tool
                        enabled = st.checkbox(
                            f"{tool_name}",
                            value=is_enabled,
                            key=f"tool_{server_name}_{tool_name}",
                            help=tool_description[:200] + "..." if len(tool_description) > 200 else tool_description
                        )
                        
                        # Update enabled tools list
                        if enabled and tool_name not in st.session_state["enabled_tools"][server_name]:
                            st.session_state["enabled_tools"][server_name].append(tool_name)
                        elif not enabled and tool_name in st.session_state["enabled_tools"][server_name]:
                            st.session_state["enabled_tools"][server_name].remove(tool_name)
                else:
                    st.warning("No tools available for this server")
    
    # Main chat interface
    col1, col2 = st.columns([6, 1])
    with col2:
        reset = st.button("🔄 Reset")
    
    if reset:
        st.session_state["messages"] = []
        st.session_state["history"] = []
        st.rerun()
    
    # Display info about active servers
    if selected_servers:
        total_tools = sum(
            len(st.session_state["enabled_tools"].get(server, []))
            for server in selected_servers
        )
        st.info(f"🎯 Active: **{', '.join(selected_servers)}** | 🛠️ {total_tools} tools enabled")
    
    user_input = st.text_input("You:", key="user_input", placeholder="Ask me anything...")
    
    # Warn if no servers selected
    if not selected_servers:
        st.warning("⚠️ Please select at least one server from the sidebar to start chatting.")
    
    # Warn if no tools are enabled across all servers
    total_enabled = sum(
        len(st.session_state["enabled_tools"].get(server, []))
        for server in selected_servers
    )
    if selected_servers and total_enabled == 0:
        st.warning("⚠️ No tools are enabled across any server. The AI will not be able to use any server functions.")
    
    if st.button("Send", type="primary") and user_input:
        if not selected_servers:
            st.error("Please select at least one server first!")
        else:
            with st.spinner(f"Thinking with {len(selected_servers)} server(s)..."):
                try:
                    response = sync_chat_response_multi_server(
                        st.session_state["messages"], 
                        user_input, 
                        selected_servers,
                        st.session_state["enabled_tools"]
                    )
                    st.session_state["history"].append((user_input, response))
                    st.rerun()
                except Exception as e:
                    st.error("An error occurred:")
                    st.error(traceback.format_exc())
    
    # Display chat history
    st.markdown("---")
    for user, bot in st.session_state["history"]:
        st.markdown(f"**👤 You:** {user}")
        st.markdown(f"**🤖 AI:** {bot}")
        st.markdown("---")

def run_streamlit():
    main()

if __name__ == "__main__":
    run_streamlit()
