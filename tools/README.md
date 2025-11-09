# Tools Module

This module contains all the internet-connected tools for the Social Robot assistant.

## Structure

```
tools/
├── __init__.py          # Package exports: TOOLS, execute_tool_call, and all functions
├── definitions.py       # Tool definitions (TOOLS list) and registry
├── web_tools.py         # Web search and scraping functions (Firecrawl)
└── weather_tool.py      # Weather information function (OpenWeatherMap)
```

## Available Tools

### 1. **search_web** (web_tools.py)
Search the web using Firecrawl's search API.

**Usage:**
```python
from tools import search_web
result = search_web("latest AI news")
```

### 2. **scrape_webpage** (web_tools.py)
Scrape webpage content using Firecrawl's scrape API.

**Usage:**
```python
from tools import scrape_webpage
content = scrape_webpage("https://example.com")
```

### 3. **get_weather** (weather_tool.py)
Get current weather for a location using OpenWeatherMap API.

**Usage:**
```python
from tools import get_weather
weather = get_weather("Tokyo, JP")
```

## Using in Your Scripts

### Import Everything
```python
from tools import TOOLS, execute_tool_call
```

### Import Individual Functions
```python
from tools import search_web, scrape_webpage, get_weather
```

### Execute Tool Calls
```python
from tools import execute_tool_call

result = execute_tool_call("search_web", {"query": "Python tutorials"})
result = execute_tool_call("get_weather", {"location": "London, UK"})
```

## Configuration

All tools use environment variables from `.env`:
- `FIRECRAWL_URL` - Firecrawl API endpoint
- `FIRECRAWL_API_KEY` - Firecrawl API key (optional for self-hosted)
- `OPENWEATHERMAP_API_KEY` - OpenWeatherMap API key

## Adding New Tools

To add a new tool:

1. Create a new file in `tools/` (e.g., `my_tool.py`)
2. Implement your function with proper docstrings
3. Add the tool definition to `definitions.py` in the `TOOLS` list
4. Add the function mapping to `TOOL_FUNCTIONS` in `definitions.py`
5. Export the function in `__init__.py`

Example:
```python
# tools/my_tool.py
def my_function(param: str) -> str:
    """My custom tool function."""
    return f"Processed: {param}"

# tools/definitions.py - add to TOOLS list
{
    "type": "function",
    "function": {
        "name": "my_function",
        "description": "What my function does",
        "parameters": {
            "type": "object",
            "properties": {
                "param": {
                    "type": "string",
                    "description": "Parameter description"
                }
            },
            "required": ["param"]
        }
    }
}

# tools/definitions.py - add to TOOL_FUNCTIONS
from .my_tool import my_function

TOOL_FUNCTIONS = {
    # ... existing tools ...
    "my_function": my_function
}

# tools/__init__.py - add to exports
from .my_tool import my_function

__all__ = [
    # ... existing exports ...
    "my_function",
]
```

## Benefits of This Structure

✅ **No Code Duplication** - Single source of truth for all tools  
✅ **Easy Maintenance** - Update tools in one place  
✅ **Modular** - Each tool in its own file  
✅ **Testable** - Easy to unit test individual tools  
✅ **Scalable** - Simple to add new tools  
✅ **Clean Imports** - Import what you need

