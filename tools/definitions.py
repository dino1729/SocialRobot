"""Tool definitions and registry for Ollama function calling."""

from typing import Any
from .web_tools import search_web, scrape_webpage
from .weather_tool import get_weather


# Tool definitions for Ollama
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather information for a specific location. Use this when asked about weather, temperature, or climate conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or 'city, country code' (e.g., 'London', 'Tokyo, JP', 'New York, US')"
                    }
                },
                "required": ["location"]
            }
        }
    }
]


# Map function names to actual functions
TOOL_FUNCTIONS = {
    "search_web": search_web,
    "scrape_webpage": scrape_webpage,
    "get_weather": get_weather
}


def execute_tool_call(tool_name: str, arguments: dict[str, Any]) -> str:
    """Execute a tool function and return its result.
    
    Args:
        tool_name: Name of the tool to execute
        arguments: Dictionary of arguments to pass to the tool
        
    Returns:
        Result from the tool as a string
    """
    if tool_name in TOOL_FUNCTIONS:
        try:
            result = TOOL_FUNCTIONS[tool_name](**arguments)
            return result
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
    else:
        return f"Unknown tool: {tool_name}"

