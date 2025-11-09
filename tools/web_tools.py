"""Web search and scraping tools using Firecrawl."""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

FIRECRAWL_URL = os.getenv("FIRECRAWL_URL", "http://localhost:3002")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")


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
            max_length = 3000
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

