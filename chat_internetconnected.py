#!/usr/bin/env python3
"""Text-based internet-connected chatbot for testing Ollama tool support with Firecrawl."""

import os
import sys
from datetime import datetime

import requests
from dotenv import load_dotenv

# Import all tools from the centralized tools module
from tools import TOOLS, execute_tool_call


# Load environment variables from .env file
load_dotenv()

# Configuration from environment variables
FIRECRAWL_URL = os.getenv("FIRECRAWL_URL", "http://localhost:3002")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")  # Tool-capable model required


# ============================================================================
# Enhanced Ollama Client with Tool Support
# ============================================================================

class OllamaClientWithTools:
    """Enhanced Ollama client that supports tool calling."""
    
    def __init__(
        self,
        url: str = "http://localhost:11434/api/chat",
        model: str = "llama3.2:1b",
        system_prompt: str = None,
    ) -> None:
        self.url = url
        self.model = model
        self.conversation_history: list[dict] = []
        
        if system_prompt:
            self.conversation_history.append({
                "role": "system",
                "content": system_prompt
            })
    
    def query_with_tools(self, user_text: str, tools: list[dict], verbose: bool = True) -> str:
        """Query Ollama with tool support and handle tool calls."""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_text
        })
        
        max_iterations = 5  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            if verbose and iteration > 1:
                print(f"\n🔄 [ITERATION {iteration}] Processing...")
            
            # Make request to Ollama with tools
            payload = {
                "model": self.model,
                "messages": self.conversation_history,
                "stream": False,
                "tools": tools
            }
            
            try:
                response = requests.post(self.url, json=payload, timeout=120)
                
                if response.status_code != 200:
                    error_msg = f"Ollama error: {response.status_code} - {response.text}"
                    print(f"\n✗ {error_msg}")
                    return "Sorry, I encountered an error processing your request."
                
                data = response.json()
                message = data.get("message", {})
                
                # Check if model wants to use tools
                tool_calls = message.get("tool_calls", [])
                
                if tool_calls:
                    if verbose:
                        print(f"\n🔧 [TOOL EXECUTION] Model requested {len(tool_calls)} tool call(s)")
                    
                    # Add assistant's tool call message to history
                    self.conversation_history.append(message)
                    
                    # Execute each tool call
                    for tool_call in tool_calls:
                        function = tool_call.get("function", {})
                        tool_name = function.get("name", "")
                        arguments = function.get("arguments", {})
                        
                        if verbose:
                            print(f"   → {tool_name}({arguments})")
                        
                        # Execute the tool
                        tool_result = execute_tool_call(tool_name, arguments)
                        
                        # Add tool result to conversation history
                        self.conversation_history.append({
                            "role": "tool",
                            "content": tool_result
                        })
                    
                    # Continue loop to let model process tool results
                    continue
                
                else:
                    # No tool calls, model provided final answer
                    content = message.get("content", "")
                    
                    # Add assistant's response to history
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": content
                    })
                    
                    return content.strip()
                    
            except Exception as e:
                error_msg = f"Error querying Ollama: {e}"
                print(f"\n✗ {error_msg}")
                return "Sorry, I encountered an error processing your request."
        
        return "I apologize, but I couldn't complete your request after multiple attempts."
    
    def reset_conversation(self):
        """Clear conversation history (except system prompt)."""
        system_prompt = None
        if self.conversation_history and self.conversation_history[0].get("role") == "system":
            system_prompt = self.conversation_history[0]
        
        self.conversation_history = []
        if system_prompt:
            self.conversation_history.append(system_prompt)


# ============================================================================
# Main Chat Interface
# ============================================================================

def print_banner():
    """Print welcome banner."""
    print("=" * 70)
    print("🌐 Internet-Connected Chatbot (Test Mode)")
    print("=" * 70)
    print()
    print("This chatbot can search the web, scrape webpages, and check weather!")
    print()
    print("Commands:")
    print("  - Type your question and press Enter")
    print("  - 'quit' or 'exit' to quit")
    print("  - 'clear' to clear conversation history")
    print("  - 'help' for example queries")
    print()
    print(f"Ollama URL: {OLLAMA_URL}")
    print(f"Ollama Model: {OLLAMA_MODEL}")
    print(f"Firecrawl URL: {FIRECRAWL_URL}")
    print(f"Weather API: {'✓ Configured' if OPENWEATHERMAP_API_KEY else '✗ Not configured'}")
    print()


def print_help():
    """Print example queries."""
    print()
    print("📚 Example Queries:")
    print()
    print("  Weather Examples:")
    print("    • What's the weather in Tokyo?")
    print("    • How's the weather in New York?")
    print("    • Get weather for London, UK")
    print()
    print("  Web Search Examples:")
    print("    • What's the latest news about AI?")
    print("    • Find Python tutorials")
    print("    • Search for SpaceX launches")
    print()
    print("  Webpage Scraping Examples:")
    print("    • Read the article at example.com")
    print("    • Scrape https://example.com")
    print("    • What does the Wikipedia page about Python say?")
    print()
    print("  Direct Questions (no tools):")
    print("    • Tell me a joke")
    print("    • What is 2+2?")
    print("    • Explain quantum computing")
    print()


def check_services():
    """Check if required services are running."""
    print("Checking services...")
    
    # Check Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("  ✓ Ollama is running")
        else:
            print(f"  ✗ Ollama returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Cannot connect to Ollama: {e}")
        return False
    
    # Check Firecrawl (using root endpoint since /health doesn't exist)
    try:
        response = requests.get(f"{FIRECRAWL_URL}/", timeout=5)
        if response.status_code == 200:
            print(f"  ✓ Firecrawl is accessible at {FIRECRAWL_URL}")
        else:
            print(f"  ⚠️  Firecrawl returned status {response.status_code}")
    except Exception as e:
        print(f"  ⚠️  Cannot reach Firecrawl: {e}")
        print(f"     (You can still use the chatbot, but internet features won't work)")
    
    print()
    return True


def main():
    """Main chat loop."""
    print_banner()
    
    # Check services
    if not check_services():
        print("✗ Ollama is not running. Please start it first.")
        print("  Run: ollama serve")
        return 1
    
    # Initialize chatbot with environment variables
    # Include current date/time so the model knows what "now" is
    current_datetime = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    system_prompt = f"""You are a helpful AI assistant with access to web search and webpage scraping tools. 

IMPORTANT: Today's date is {current_datetime}. When searching for current or recent information, use the current year (2025) in your search queries, not outdated years like 2023 or 2024.

Use these tools when you need current information or to look up specific details online. Be concise but informative in your responses."""
    
    chatbot = OllamaClientWithTools(
        url=OLLAMA_URL,
        model=OLLAMA_MODEL,
        system_prompt=system_prompt
    )
    
    print("=" * 70)
    print("Ready! Type your question or 'help' for examples.")
    print("=" * 70)
    print()
    
    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                chatbot.reset_conversation()
                print("\n✓ Conversation history cleared.\n")
                continue
            
            if user_input.lower() == 'help':
                print_help()
                continue
            
            # Process query
            print()  # Blank line before processing
            
            response = chatbot.query_with_tools(user_input, TOOLS, verbose=True)
            
            print()
            print(f"Bot: {response}")
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}\n")
            continue
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

