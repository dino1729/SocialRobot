#!/usr/bin/env python3
"""Text-based internet-connected chatbot for testing Ollama tool support with Firecrawl."""

import os
import sys
from typing import Any

import requests
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

# Configuration from environment variables
FIRECRAWL_URL = os.getenv("FIRECRAWL_URL", "http://localhost:3002")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")  # Tool-capable model required


# ============================================================================
# Firecrawl Tool Functions
# ============================================================================

def search_web(query: str) -> str:
    """Search the web using Firecrawl's search API.
    
    Args:
        query: The search query string
        
    Returns:
        A summary of search results as a string
    """
    try:
        print(f"\n🔍 [TOOL CALL] Searching web for: '{query}'")
        
        headers = {"Content-Type": "application/json"}
        if FIRECRAWL_API_KEY:
            headers["Authorization"] = f"Bearer {FIRECRAWL_API_KEY}"
        
        response = requests.post(
            f"{FIRECRAWL_URL}/v2/search",
            json={"query": query},
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Firecrawl v2 returns: {"data": {"web": [...]}}
            if "data" in data and isinstance(data["data"], dict) and "web" in data["data"]:
                results = data["data"]["web"]
            elif "data" in data and isinstance(data["data"], list):
                results = data["data"]
            elif "results" in data:
                results = data["results"]
            else:
                results = data if isinstance(data, list) else []
            
            # Ensure results is a list
            if not isinstance(results, list):
                return f"Unexpected response format. Got {type(results).__name__}, expected list"
            
            if not results:
                return "No search results found."
            
            # Format results
            summary = f"Found {len(results)} results:\n\n"
            try:
                # Safely iterate with limit
                for i, result in enumerate(results[:5] if len(results) > 5 else results, 1):
                    if isinstance(result, dict):
                        title = result.get("title", result.get("name", "Untitled"))
                        url = result.get("url", result.get("link", ""))
                        snippet = result.get("description", result.get("snippet", result.get("content", "")))
                        
                        # Safely truncate snippet
                        if snippet and len(snippet) > 200:
                            snippet = snippet[:200] + "..."
                        
                        summary += f"{i}. {title}\n   URL: {url}\n   {snippet}\n\n"
                    else:
                        summary += f"{i}. {str(result)}\n\n"
            except Exception as e:
                return f"Error formatting results: {str(e)}"
            
            print(f"✓ Search completed successfully")
            return summary
        else:
            error_msg = f"Search failed with status code: {response.status_code}"
            print(f"✗ {error_msg}")
            return error_msg
            
    except Exception as e:
        error_msg = f"Search error: {str(e)}"
        print(f"✗ {error_msg}")
        return error_msg


def scrape_webpage(url: str) -> str:
    """Scrape a webpage using Firecrawl's scrape API.
    
    Args:
        url: The URL to scrape
        
    Returns:
        The scraped content as a string
    """
    try:
        print(f"\n🌐 [TOOL CALL] Scraping webpage: {url}")
        
        headers = {"Content-Type": "application/json"}
        if FIRECRAWL_API_KEY:
            headers["Authorization"] = f"Bearer {FIRECRAWL_API_KEY}"
        
        response = requests.post(
            f"{FIRECRAWL_URL}/v2/scrape",
            json={
                "url": url,
                "formats": ["markdown"]
            },
            headers=headers,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            content = data.get("data", {}).get("markdown", "")
            
            if not content:
                return "No content found on the page."
            
            # Limit content length
            max_length = 3000  # Larger for text interface
            if len(content) > max_length:
                content = content[:max_length] + "...\n[Content truncated]"
            
            print(f"✓ Scraped {len(content)} characters")
            return f"Content from {url}:\n\n{content}"
        else:
            error_msg = f"Scraping failed with status code: {response.status_code}"
            print(f"✗ {error_msg}")
            return error_msg
            
    except Exception as e:
        error_msg = f"Scraping error: {str(e)}"
        print(f"✗ {error_msg}")
        return error_msg


# ============================================================================
# Tool Definitions for Ollama
# ============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information on a topic. Use this when you need current information, facts, news, or anything you don't have in your training data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (e.g., 'latest AI news', 'weather in Tokyo', 'Python tutorials')"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_webpage",
            "description": "Scrape and read the content of a specific webpage. Use this when you need to read detailed information from a specific URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full URL of the webpage to scrape (e.g., 'https://example.com/article')"
                    }
                },
                "required": ["url"]
            }
        }
    }
]


# Map function names to actual functions
TOOL_FUNCTIONS = {
    "search_web": search_web,
    "scrape_webpage": scrape_webpage
}


def execute_tool_call(tool_name: str, arguments: dict[str, Any]) -> str:
    """Execute a tool function and return its result."""
    if tool_name in TOOL_FUNCTIONS:
        try:
            result = TOOL_FUNCTIONS[tool_name](**arguments)
            return result
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
    else:
        return f"Unknown tool: {tool_name}"


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
    print("This chatbot can search the web and scrape webpages!")
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
    print()


def print_help():
    """Print example queries."""
    print()
    print("📚 Example Queries:")
    print()
    print("  Web Search Examples:")
    print("    • What's the latest news about AI?")
    print("    • What's the weather in Tokyo?")
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
    chatbot = OllamaClientWithTools(
        url=OLLAMA_URL,
        model=OLLAMA_MODEL,
        system_prompt="You are a helpful AI assistant with access to web search and webpage scraping tools. Use these tools when you need current information or to look up specific details online. Be concise but informative in your responses."
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

