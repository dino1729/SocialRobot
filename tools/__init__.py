"""Internet-connected tools for the Social Robot assistant.

This module provides:
- Web search via Firecrawl
- Webpage scraping via Firecrawl
- Weather information via OpenWeatherMap

All tools are ready to use with Ollama function calling.
"""

from .web_tools import search_web, scrape_webpage
from .weather_tool import get_weather
from .definitions import TOOLS, TOOL_FUNCTIONS, execute_tool_call

__all__ = [
    "search_web",
    "scrape_webpage",
    "get_weather",
    "TOOLS",
    "TOOL_FUNCTIONS",
    "execute_tool_call",
]

