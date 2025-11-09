# test_tools.py - Tool Testing Guide

## Overview

Comprehensive test script for all internet-connected tools in the Social Robot project.

## What It Tests

✅ **Tool Registry** - Verifies all tools are properly registered  
✅ **Tool Definitions** - Validates tool definition format  
✅ **Tool Execution** - Tests the execute_tool_call() function  
✅ **Web Search** - Tests search_web() with Firecrawl  
✅ **Webpage Scraping** - Tests scrape_webpage() with Firecrawl  
✅ **Weather Lookup** - Tests get_weather() with OpenWeatherMap  

## Requirements

Before running tests, make sure:
1. Virtual environment is activated
2. All dependencies are installed (`pip install -r requirements.txt`)
3. Services are configured in `.env`:
   - `FIRECRAWL_URL` (default: http://localhost:3002)
   - `OPENWEATHERMAP_API_KEY` (optional, required for weather tests)
4. Firecrawl service is running (if testing web/scraping tools)

## Usage

### Quick Test (All Tests)
```bash
python test_tools.py
```

### With Virtual Environment
```bash
source ~/robot/.venv/bin/activate
python test_tools.py
```

### Make Executable and Run
```bash
chmod +x test_tools.py
./test_tools.py
```

## Example Output

```
======================================================================
  Tool Testing Suite
======================================================================
Date: 2025-11-09 14:23:45
Total tools registered: 3

📋 Configuration Status:
  Firecrawl URL: http://localhost:3002
  Firecrawl API Key: ✗ Not set
  OpenWeatherMap API Key: ✓ Set

======================================================================
  Running Tests
======================================================================

🧪 TEST: Tool Registry
----------------------------------------------------------------------
Expected tools: ['search_web', 'scrape_webpage', 'get_weather']
Found tools: ['search_web', 'scrape_webpage', 'get_weather']
✅ All 3 tools are registered

🧪 TEST: Tool Definition Format
----------------------------------------------------------------------
✅ All 3 tool definitions are properly formatted

🧪 TEST: Tool Execution via execute_tool_call()
----------------------------------------------------------------------
✅ Valid tool call handled correctly
✅ Invalid tool name handled correctly
✅ Missing arguments handled correctly

Passed 3/3 execution tests

🧪 TEST: Web Search Tool
----------------------------------------------------------------------
✅ Firecrawl is accessible at http://localhost:3002
Searching for: 'Python programming language'
✅ Search returned 456 characters

First 200 chars of result:
Found 5 results:

1. Python (programming language) - Wikipedia
   URL: https://en.wikipedia.org/wiki/Python_(programming_language)
   Python is a high-level, general-purpose programming language...

🧪 TEST: Webpage Scraping Tool
----------------------------------------------------------------------
✅ Firecrawl is accessible at http://localhost:3002
Scraping URL: https://example.com
✅ Scrape returned 342 characters

First 200 chars of result:
Content from https://example.com:

# Example Domain

This domain is for use in illustrative examples in documents...

🧪 TEST: Weather Tool
----------------------------------------------------------------------
✅ OpenWeatherMap API key is configured
Getting weather for: London, UK
✅ Weather lookup successful

Weather result:
Weather in London, UK:
Status: Clear Sky
Temperature: 15.3°C (59.5°F)
Feels like: 14.8°C (58.6°F)
Min/Max: 14.0°C / 16.5°C
Humidity: 72%
Wind Speed: 3.1 m/s
Clouds: 20%

======================================================================
  Test Summary
======================================================================

Results: 6/6 tests passed

  ✅ Tool Registry
  ✅ Tool Definitions
  ✅ Tool Execution
  ✅ Web Search
  ✅ Webpage Scraping
  ✅ Weather Lookup

🎉 All tests passed!
```

## Exit Codes

- `0` - All tests passed
- `1` - Some tests failed (≥70% passed)
- `2` - Many tests failed (<70% passed)
- `130` - Tests interrupted by user (Ctrl+C)

## Troubleshooting

### "Cannot reach Firecrawl"
- Make sure Firecrawl is running
- Check `FIRECRAWL_URL` in `.env`
- Test with: `curl http://localhost:3002`

### "OpenWeatherMap API key not configured"
- Add `OPENWEATHERMAP_API_KEY` to your `.env` file
- Get a free API key at https://openweathermap.org/api
- See `WEATHER_SETUP.md` for details

### "No module named 'requests'"
- Activate your virtual environment
- Install dependencies: `pip install -r requirements.txt`

### "ModuleNotFoundError: No module named 'tools'"
- Make sure you're running from the project root directory
- The `tools/` directory should be in the same location

## Testing Individual Tools

You can import and test individual tools in Python:

```python
from tools import search_web, scrape_webpage, get_weather

# Test search
result = search_web("Python tutorials")
print(result)

# Test scraping
content = scrape_webpage("https://example.com")
print(content)

# Test weather
weather = get_weather("Tokyo, JP")
print(weather)
```

## Continuous Integration

This script is designed to work in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Test tools
  run: |
    source ~/robot/.venv/bin/activate
    python test_tools.py
```

## Adding New Tests

To add tests for new tools:

1. Add the tool to `tools/` module
2. Create a test function in `test_tools.py`:
   ```python
   def test_my_new_tool():
       """Test my new tool."""
       print_test("My New Tool")
       try:
           result = my_new_tool(args)
           if result:
               print_result(True, "Tool works!")
               return True
       except Exception as e:
           print_result(False, f"Failed: {e}")
           return False
   ```
3. Add it to `run_all_tests()`:
   ```python
   results["My New Tool"] = test_my_new_tool()
   ```

---

Happy testing! 🧪🤖

