#!/usr/bin/env python3
"""
Isaac Sim MCP Interface - Terminal Application
Simplified main entry point with agent selection at startup
"""
import os
import sys
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

try:
    from host import MCPHost, check_backend_availability
    HOST_AVAILABLE = True
except ImportError as e:
    print(f" Host module not available: {e}")
    print("Make sure host.py is in the same directory")
    sys.exit(1)

class IsaacSimTerminal:
    """Terminal interface for Isaac Sim using MCP Host"""
    
    def __init__(self):
        self.host = None
        self.current_backend = None
        self.conversation_history = []
        self.available_backends = []
        
        # Check which backends are available
        self.available_backends = check_backend_availability()
        
        if not self.available_backends:
            print(" No API keys found!")
            print("Please set one or both of these in your .env file:")
            print("  • ANTHROPIC_API_KEY=your_claude_key_here")
            print("  • OPENAI_API_KEY=your_openai_key_here")
            sys.exit(1)
    
    def display_header(self):
        """Display the application header"""
        print(" Isaac Sim MCP Interface")
        print("=" * 50)
        print("Control Isaac Sim using natural language with AI agents")
        print("")
        print("Available commands:")
        print("  • Type your request in natural language")
        print("  • 'switch' - Change between Claude/ChatGPT")
        print("  • 'backends' - Show available backends")
        print("  • 'history' - Show conversation history")
        print("  • 'clear' - Clear conversation history")
        print("  • 'test' - Test connection to Isaac Sim")
        print("  • 'quit' or 'exit' - Exit the application")
        print("=" * 50)
    
    def select_backend(self):
        """Let user select which backend to use at startup"""
        print(f"\n Available AI backends: {', '.join(self.available_backends)}")
        
        if len(self.available_backends) == 1:
            selected = self.available_backends[0]
            print(f" Only one backend available: {selected.title()}")
            return selected
        
        print(f"\nSelect your AI backend:")
        for i, backend in enumerate(self.available_backends, 1):
            emoji = "" if backend == "claude" else ""
            print(f"  {i}. {emoji} {backend.title()}")
        
        while True:
            try:
                choice = input(f"\nEnter your choice (1-{len(self.available_backends)}) or backend name: ").strip().lower()
                
                # Check if it's a number
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(self.available_backends):
                        return self.available_backends[idx]
                    else:
                        print(f" Invalid choice. Please enter 1-{len(self.available_backends)}")
                        continue
                
                # Check if it's a backend name
                if choice in self.available_backends:
                    return choice
                
                print(f" Invalid choice. Available options: {', '.join(self.available_backends)}")
                
            except KeyboardInterrupt:
                print("\n Goodbye!")
                sys.exit(0)
            except Exception as e:
                print(f" Error: {e}")
    
    async def initialize_host(self, backend=None):
        """Initialize the MCP host with specified backend"""
        try:
            if backend is None:
                # Default to Claude if available, otherwise first available
                backend = "claude" if "claude" in self.available_backends else self.available_backends[0]
            
            if backend not in self.available_backends:
                print(f" Backend '{backend}' not available. Available: {', '.join(self.available_backends)}")
                return False
            
            print(f" Initializing {backend.title()} backend...")
            
            self.host = MCPHost(backend)
            self.current_backend = backend
            print(f" {backend.title()} backend initialized successfully!")
            return True
            
        except Exception as e:
            print(f" Failed to initialize {backend} backend: {e}")
            if "API_KEY" in str(e):
                print(" Make sure you have the required API key set in your .env file")
            return False
    
    async def switch_backend(self):
        """Switch between available backends"""
        if len(self.available_backends) < 2:
            print(f" Only one backend available: {self.current_backend}")
            return
        
        # Find next backend
        current_index = self.available_backends.index(self.current_backend)
        next_index = (current_index + 1) % len(self.available_backends)
        next_backend = self.available_backends[next_index]
        
        print(f" Switching from {self.current_backend.title()} to {next_backend.title()}...")
        
        if await self.initialize_host(next_backend):
            print(f" Successfully switched to {next_backend.title()}")
            # Clear history when switching to avoid context confusion
            if self.conversation_history:
                print(" Conversation history cleared due to backend switch")
                self.conversation_history.clear()
        else:
            print(f" Failed to switch to {next_backend}")
    
    def show_backends(self):
        """Show available backends and current selection"""
        print(f"\n Available backends: {', '.join([b.title() for b in self.available_backends])}")
        print(f" Current backend: {self.current_backend.title()}")
        
        # Show API key status
        claude_key = "" if os.getenv("ANTHROPIC_API_KEY") else ""
        openai_key = "" if os.getenv("OPENAI_API_KEY") else ""
        print(f" API Keys - Claude: {claude_key}, OpenAI: {openai_key}")
    
    def show_history(self):
        """Show conversation history"""
        if not self.conversation_history:
            print(" No conversation history yet")
            return
        
        print(f"\n Conversation History ({len(self.conversation_history)} messages):")
        print("-" * 40)
        
        for i, msg in enumerate(self.conversation_history[-10:], 1):  # Show last 10
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            backend = msg.get("backend", "unknown")
            
            # Truncate long messages
            if len(content) > 100:
                content = content[:97] + "..."
            
            role_emoji = "" if role == "user" else ""
            backend_emoji = "" if backend == "claude" else ""
            print(f"{i}. {role_emoji} [{role.upper()} - {backend_emoji} {backend.title()}] {content}")
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        print(" Conversation history cleared")
    
    def show_help(self):
        """Show available Isaac Sim tools and examples"""
        print("\n  Available Isaac Sim Tools:")
        print("-" * 40)
        
        if self.host and hasattr(self.host.mcp_client, 'available_tools'):
            if self.host.mcp_client.available_tools:
                for tool in self.host.mcp_client.available_tools:
                    print(f"• {tool}")
            else:
                print("• No tools available (connect to Isaac Sim first)")
        else:
            print("• Tools will be discovered when you connect to Isaac Sim")
        
    
    async def test_connection(self):
        """Test connection to Isaac Sim"""
        if not self.host:
            print(" No host initialized")
            return
        
        print("🔧 Testing connection to Isaac Sim...")
        try:
            # Test with a simple scene info request
            response = await self.host.process_request("Check the scene status", self.conversation_history)
            print(f" Connection test successful!")
            print(f"Response: {response}")
        except Exception as e:
            print(f" Connection test failed: {e}")
            print(" Make sure Isaac Sim is running and the server.py is accessible")
    
    async def process_user_input(self, user_input: str):
        """Process user input and get response from host"""
        if not self.host:
            print(" No host initialized")
            return
        
        try:
            # Add user message to history
            user_msg = {
                "role": "user", 
                "content": user_input,
                "backend": self.current_backend
            }
            self.conversation_history.append(user_msg)
            
            # Get response from host
            print(f" {self.current_backend.title()} is thinking...")
            response = await self.host.process_request(user_input, self.conversation_history)
            
            # Add assistant response to history
            assistant_msg = {
                "role": "assistant",
                "content": response,
                "backend": self.current_backend
            }
            self.conversation_history.append(assistant_msg)
            
            # Display response
            backend_emoji = "" if self.current_backend == "claude" else ""
            print(f"\n{backend_emoji} {self.current_backend.title()}: {response}")
            
        except Exception as e:
            print(f" Error processing request: {e}")
    
    async def run(self):
        """Main application loop"""
        self.display_header()
        
        # Let user select backend at startup
        selected_backend = self.select_backend()
        
        # Initialize selected backend
        if not await self.initialize_host(selected_backend):
            print(" Failed to initialize selected backend. Exiting.")
            return
        
        backend_emoji = "" if self.current_backend == "claude" else ""
        print(f"\n Ready! Using {backend_emoji} {self.current_backend.title()} backend")
        print(" What would you like to do in Isaac Sim?")
        print(" Type 'help' to see available commands")
        
        while True:
            try:
                # Get user input
                user_input = input(f"\n You ({self.current_backend}): ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'q', 'exit']:
                    break
                elif user_input.lower() == 'switch':
                    await self.switch_backend()
                elif user_input.lower() == 'backends':
                    self.show_backends()
                elif user_input.lower() == 'history':
                    self.show_history()
                elif user_input.lower() == 'clear':
                    self.clear_history()
                elif user_input.lower() == 'help':
                    self.show_help()
                elif user_input.lower() == 'test':
                    await self.test_connection()
                else:
                    # Process as Isaac Sim request
                    await self.process_user_input(user_input)
                
            except KeyboardInterrupt:
                print("\n\n Interrupted by user")
                break
            except Exception as e:
                print(f" Unexpected error: {e}")
        
        print("\n Thanks for using Isaac Sim MCP Interface!")

async def main():
    """Entry point"""
    terminal = IsaacSimTerminal()
    await terminal.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n Goodbye!")
    except Exception as e:
        print(f" Fatal error: {e}")
        sys.exit(1)