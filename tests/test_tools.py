#!/usr/bin/env python3
"""Test script for all internet-connected tools."""

import os
import sys
from datetime import datetime

import requests
from dotenv import load_dotenv

# Import all tools
from tools import TOOLS, execute_tool_call, search_web, scrape_webpage, get_weather

# Load environment variables
load_dotenv()

# Configuration
FIRECRAWL_URL = os.getenv("FIRECRAWL_URL", "http://localhost:3002")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_test(test_name: str):
    """Print a test name."""
    print(f"\n🧪 TEST: {test_name}")
    print("-" * 70)


def print_result(success: bool, message: str):
    """Print a test result."""
    icon = "✅" if success else "❌"
    print(f"{icon} {message}")


def check_firecrawl():
    """Check if Firecrawl service is accessible."""
    try:
        response = requests.get(f"{FIRECRAWL_URL}/", timeout=5)
        if response.status_code == 200:
            print_result(True, f"Firecrawl is accessible at {FIRECRAWL_URL}")
            return True
        else:
            print_result(False, f"Firecrawl returned status {response.status_code}")
            return False
    except Exception as e:
        print_result(False, f"Cannot reach Firecrawl: {e}")
        return False


def check_openweathermap():
    """Check if OpenWeatherMap API key is configured."""
    if OPENWEATHERMAP_API_KEY:
        print_result(True, "OpenWeatherMap API key is configured")
        return True
    else:
        print_result(False, "OpenWeatherMap API key not configured")
        return False


def test_tool_registry():
    """Test that all tools are properly registered."""
    print_test("Tool Registry")
    
    # Check TOOLS list
    expected_tools = ["search_web", "scrape_webpage", "get_weather"]
    found_tools = [tool["function"]["name"] for tool in TOOLS]
    
    print(f"Expected tools: {expected_tools}")
    print(f"Found tools: {found_tools}")
    
    all_found = all(tool in found_tools for tool in expected_tools)
    if all_found:
        print_result(True, f"All {len(expected_tools)} tools are registered")
    else:
        missing = [t for t in expected_tools if t not in found_tools]
        print_result(False, f"Missing tools: {missing}")
    
    return all_found


def test_search_web():
    """Test the web search tool."""
    print_test("Web Search Tool")
    
    # Check if Firecrawl is available
    if not check_firecrawl():
        print_result(False, "Skipping search test - Firecrawl not available")
        return False
    
    try:
        # Test search
        query = "Python programming language"
        print(f"Searching for: '{query}'")
        result = search_web(query)
        
        # Check result
        if result and "error" not in result.lower() and len(result) > 50:
            print_result(True, f"Search returned {len(result)} characters")
            print(f"\nFirst 200 chars of result:\n{result[:200]}...")
            return True
        else:
            print_result(False, f"Search returned unexpected result: {result[:200]}")
            return False
            
    except Exception as e:
        print_result(False, f"Search test failed: {e}")
        return False


def test_scrape_webpage():
    """Test the webpage scraping tool."""
    print_test("Webpage Scraping Tool")
    
    # Check if Firecrawl is available
    if not check_firecrawl():
        print_result(False, "Skipping scrape test - Firecrawl not available")
        return False
    
    try:
        # Test scraping a simple page
        url = "https://example.com"
        print(f"Scraping URL: {url}")
        result = scrape_webpage(url)
        
        # Check result
        if result and "error" not in result.lower() and len(result) > 50:
            print_result(True, f"Scrape returned {len(result)} characters")
            print(f"\nFirst 200 chars of result:\n{result[:200]}...")
            return True
        else:
            print_result(False, f"Scrape returned unexpected result: {result[:200]}")
            return False
            
    except Exception as e:
        print_result(False, f"Scrape test failed: {e}")
        return False


def test_get_weather():
    """Test the weather tool."""
    print_test("Weather Tool")
    
    # Check if API key is configured
    if not check_openweathermap():
        print_result(False, "Skipping weather test - API key not configured")
        return False
    
    try:
        # Test weather lookup
        location = "London, UK"
        print(f"Getting weather for: {location}")
        result = get_weather(location)
        
        # Check result
        if result and "error" not in result.lower() and "Temperature" in result:
            print_result(True, f"Weather lookup successful")
            print(f"\nWeather result:\n{result}")
            return True
        else:
            print_result(False, f"Weather returned unexpected result: {result}")
            return False
            
    except Exception as e:
        print_result(False, f"Weather test failed: {e}")
        return False


def test_execute_tool_call():
    """Test the execute_tool_call function."""
    print_test("Tool Execution via execute_tool_call()")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Valid tool call
    tests_total += 1
    try:
        result = execute_tool_call("get_weather", {"location": "Tokyo, JP"})
        if "error" not in result.lower() or "not configured" in result.lower():
            print_result(True, "Valid tool call handled correctly")
            tests_passed += 1
        else:
            print_result(False, f"Unexpected result: {result[:100]}")
    except Exception as e:
        print_result(False, f"Valid tool call failed: {e}")
    
    # Test 2: Invalid tool name
    tests_total += 1
    try:
        result = execute_tool_call("invalid_tool", {})
        if "unknown tool" in result.lower():
            print_result(True, "Invalid tool name handled correctly")
            tests_passed += 1
        else:
            print_result(False, f"Unexpected result for invalid tool: {result}")
    except Exception as e:
        print_result(False, f"Invalid tool test failed: {e}")
    
    # Test 3: Missing arguments
    tests_total += 1
    try:
        result = execute_tool_call("search_web", {})
        if "error" in result.lower():
            print_result(True, "Missing arguments handled correctly")
            tests_passed += 1
        else:
            print_result(False, f"Unexpected result for missing args: {result[:100]}")
    except Exception as e:
        print_result(True, "Missing arguments raised exception (expected)")
        tests_passed += 1
    
    success = tests_passed == tests_total
    print(f"\nPassed {tests_passed}/{tests_total} execution tests")
    return success


def test_tool_definitions():
    """Test that tool definitions are properly formatted."""
    print_test("Tool Definition Format")
    
    issues = []
    
    for tool in TOOLS:
        tool_name = tool.get("function", {}).get("name", "UNKNOWN")
        
        # Check required fields
        if "type" not in tool:
            issues.append(f"{tool_name}: Missing 'type' field")
        elif tool["type"] != "function":
            issues.append(f"{tool_name}: Invalid type '{tool['type']}'")
        
        if "function" not in tool:
            issues.append(f"{tool_name}: Missing 'function' field")
            continue
        
        func = tool["function"]
        
        # Check function fields
        required_fields = ["name", "description", "parameters"]
        for field in required_fields:
            if field not in func:
                issues.append(f"{tool_name}: Missing '{field}' field")
        
        # Check parameters structure
        if "parameters" in func:
            params = func["parameters"]
            if "type" not in params or params["type"] != "object":
                issues.append(f"{tool_name}: Invalid parameters type")
            if "properties" not in params:
                issues.append(f"{tool_name}: Missing 'properties' in parameters")
            if "required" not in params:
                issues.append(f"{tool_name}: Missing 'required' field in parameters")
    
    if issues:
        print_result(False, f"Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print_result(True, f"All {len(TOOLS)} tool definitions are properly formatted")
        return True


def run_all_tests():
    """Run all tests and provide a summary."""
    print_header("Tool Testing Suite")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total tools registered: {len(TOOLS)}")
    
    # Configuration status
    print("\n📋 Configuration Status:")
    print(f"  Firecrawl URL: {FIRECRAWL_URL}")
    print(f"  Firecrawl API Key: {'✓ Set' if FIRECRAWL_API_KEY else '✗ Not set'}")
    print(f"  OpenWeatherMap API Key: {'✓ Set' if OPENWEATHERMAP_API_KEY else '✗ Not set'}")
    
    # Run tests
    results = {}
    
    print_header("Running Tests")
    
    results["Tool Registry"] = test_tool_registry()
    results["Tool Definitions"] = test_tool_definitions()
    results["Tool Execution"] = test_execute_tool_call()
    results["Web Search"] = test_search_web()
    results["Webpage Scraping"] = test_scrape_webpage()
    results["Weather Lookup"] = test_get_weather()
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nResults: {passed}/{total} tests passed\n")
    
    for test_name, result in results.items():
        icon = "✅" if result else "❌"
        print(f"  {icon} {test_name}")
    
    # Overall result
    print()
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    elif passed >= total * 0.7:
        print("⚠️  Most tests passed, but some failures detected.")
        return 1
    else:
        print("❌ Many tests failed. Check configuration and services.")
        return 2


def main():
    """Main entry point."""
    try:
        exit_code = run_all_tests()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n👋 Tests interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

