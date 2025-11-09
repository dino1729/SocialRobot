#!/usr/bin/env python3
"""Test script to debug Firecrawl search API response."""

import json
import os
import requests
from dotenv import load_dotenv

load_dotenv()

FIRECRAWL_URL = os.getenv("FIRECRAWL_URL", "http://localhost:3002")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")

print("=" * 70)
print("Firecrawl Search API Test")
print("=" * 70)
print(f"Firecrawl URL: {FIRECRAWL_URL}")
print()

# Test query
test_query = "test search"

print(f"Testing search with query: '{test_query}'")
print()

headers = {"Content-Type": "application/json"}
if FIRECRAWL_API_KEY:
    headers["Authorization"] = f"Bearer {FIRECRAWL_API_KEY}"

try:
    response = requests.post(
        f"{FIRECRAWL_URL}/v2/search",
        json={"query": test_query},
        headers=headers,
        timeout=30
    )
    
    print(f"Status Code: {response.status_code}")
    print()
    
    if response.status_code == 200:
        data = response.json()
        print("Raw Response:")
        print(json.dumps(data, indent=2))
        print()
        
        print("Response Keys:")
        if isinstance(data, dict):
            for key in data.keys():
                value = data[key]
                print(f"  - {key}: {type(value).__name__}")
                if isinstance(value, list) and len(value) > 0:
                    print(f"    └─ First item type: {type(value[0]).__name__}")
                    if isinstance(value[0], dict):
                        print(f"       Keys: {list(value[0].keys())}")
        
        print()
        print("✓ Test completed")
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"✗ Exception: {e}")
    import traceback
    traceback.print_exc()

