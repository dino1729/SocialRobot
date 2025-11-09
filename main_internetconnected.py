"""Entrypoint for the internet-connected voice assistant with tool support."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Optional

import requests
from dotenv import load_dotenv

from audio.stt import FasterWhisperSTT
from audio.tts import KokoroTTS
from audio.vad import VADConfig, VADListener
from llm.ollama import OllamaClient


# Load environment variables from .env file
load_dotenv()

# Configuration from environment variables
FIRECRAWL_URL = os.getenv("FIRECRAWL_URL", "http://localhost:3002")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")  # Optional for self-hosted
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")  # Tool-capable model required


def _get_memory_stats() -> dict[str, float]:
    """Get system memory statistics (lightweight, no external dependencies)."""
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        
        mem_total = mem_available = mem_free = 0
        for line in meminfo.split('\n'):
            if line.startswith('MemTotal:'):
                mem_total = int(line.split()[1]) / 1024  # Convert to MB
            elif line.startswith('MemAvailable:'):
                mem_available = int(line.split()[1]) / 1024
            elif line.startswith('MemFree:'):
                mem_free = int(line.split()[1]) / 1024
        
        mem_used = mem_total - mem_available
        mem_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0
        
        return {
            'total': mem_total,
            'used': mem_used,
            'available': mem_available,
            'percent': mem_percent
        }
    except Exception:
        return {'total': 0, 'used': 0, 'available': 0, 'percent': 0}


def _get_process_memory() -> float:
    """Get current process memory usage in MB."""
    try:
        pid = os.getpid()
        with open(f'/proc/{pid}/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024  # Convert to MB
    except Exception:
        pass
    return 0.0


def _format_memory_stats(stats: dict[str, float], process_mem: float) -> str:
    """Format memory statistics for display."""
    return (
        f"💾 RAM: {stats['used']:.0f}/{stats['total']:.0f}MB "
        f"({stats['percent']:.1f}%) | "
        f"Process: {process_mem:.0f}MB"
    )


def _memory_monitor(stop_event: threading.Event, interval: int = 60) -> None:
    """Background thread to monitor and display memory usage."""
    while not stop_event.is_set():
        stats = _get_memory_stats()
        process_mem = _get_process_memory()
        print(f"\n{_format_memory_stats(stats, process_mem)}")
        stop_event.wait(interval)  # Sleep but can be interrupted


def _detect_whisper_device() -> str:
    """Detects the best available device for ctranslate2 (CUDA or CPU)."""
    try:
        import ctranslate2  # type: ignore

        if ctranslate2.get_cuda_device_count() > 0:  # type: ignore[attr-defined]
            return "cuda"
    except Exception:
        pass
    return "cpu"


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
        print(f"🔍 Searching web for: {query}")
        
        headers = {}
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
            
            return summary
        else:
            return f"Search failed with status code: {response.status_code}"
            
    except Exception as e:
        print(f"❌ Search error: {e}")
        return f"Search error: {str(e)}"


def scrape_webpage(url: str) -> str:
    """Scrape a webpage using Firecrawl's scrape API.
    
    Args:
        url: The URL to scrape
        
    Returns:
        The scraped content as a string
    """
    try:
        print(f"🌐 Scraping webpage: {url}")
        
        headers = {"Content-Type": "application/json"}
        if FIRECRAWL_API_KEY:
            headers["Authorization"] = f"Bearer {FIRECRAWL_API_KEY}"
        
        response = requests.post(
            f"{FIRECRAWL_URL}/v2/scrape",
            json={
                "url": url,
                "formats": ["markdown"]  # Get clean markdown content
            },
            headers=headers,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            content = data.get("data", {}).get("markdown", "")
            
            if not content:
                return "No content found on the page."
            
            # Limit content length for LLM context
            max_length = 2000
            if len(content) > max_length:
                content = content[:max_length] + "...\n[Content truncated]"
            
            return f"Content from {url}:\n\n{content}"
        else:
            return f"Scraping failed with status code: {response.status_code}"
            
    except Exception as e:
        print(f"❌ Scraping error: {e}")
        return f"Scraping error: {str(e)}"


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
        stream: bool = False,  # Disable streaming for tool support
        system_prompt: Optional[str] = None,
    ) -> None:
        self.url = url
        self.model = model
        self.stream = stream
        self.system_prompt = system_prompt
        self.conversation_history: list[dict] = []
        
        if system_prompt:
            self.conversation_history.append({
                "role": "system",
                "content": system_prompt
            })
    
    def query_with_tools(self, user_text: str, tools: list[dict]) -> str:
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
            
            # Make request to Ollama with tools
            payload = {
                "model": self.model,
                "messages": self.conversation_history,
                "stream": self.stream,
                "tools": tools
            }
            
            try:
                response = requests.post(self.url, json=payload, timeout=120)
                
                if response.status_code != 200:
                    print(f"Ollama error: {response.status_code}")
                    return "Sorry, I encountered an error processing your request."
                
                data = response.json()
                message = data.get("message", {})
                
                # Check if model wants to use tools
                tool_calls = message.get("tool_calls", [])
                
                if tool_calls:
                    print(f"🔧 Model requested {len(tool_calls)} tool call(s)")
                    
                    # Add assistant's tool call message to history
                    self.conversation_history.append(message)
                    
                    # Execute each tool call
                    for tool_call in tool_calls:
                        function = tool_call.get("function", {})
                        tool_name = function.get("name", "")
                        arguments = function.get("arguments", {})
                        
                        print(f"   Calling: {tool_name}({arguments})")
                        
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
                print(f"Error querying Ollama: {e}")
                return "Sorry, I encountered an error processing your request."
        
        return "I apologize, but I couldn't complete your request after multiple attempts."


# ============================================================================
# Main Function
# ============================================================================

def main(enable_memory_monitor: bool = True, monitor_interval: int = 60) -> None:
    """Initializes all components and starts the main interaction loop.
    
    Args:
        enable_memory_monitor: Whether to show periodic memory usage stats (default: True)
        monitor_interval: Seconds between memory stat updates (default: 60)
    """
    print("-> Initializing internet-connected voice assistant...")
    print(f"-> Ollama URL: {OLLAMA_URL}")
    print(f"-> Ollama Model: {OLLAMA_MODEL}")
    print(f"-> Firecrawl URL: {FIRECRAWL_URL}")
    
    # Check Firecrawl connectivity (using root endpoint since /health doesn't exist)
    try:
        response = requests.get(f"{FIRECRAWL_URL}/", timeout=5)
        if response.status_code == 200:
            print("-> ✓ Firecrawl is accessible")
        else:
            print(f"-> ⚠️  Firecrawl returned status {response.status_code}")
    except Exception as e:
        print(f"-> ⚠️  Warning: Cannot reach Firecrawl at {FIRECRAWL_URL}: {e}")
        print("   Make sure Firecrawl is running (see QUICKSTART.md)")
    
    # Display initial memory stats
    if enable_memory_monitor:
        stats = _get_memory_stats()
        process_mem = _get_process_memory()
        print(f"-> Initial {_format_memory_stats(stats, process_mem)}")
    
    # Start background memory monitor if enabled
    memory_stop_event = threading.Event()
    memory_thread = None
    if enable_memory_monitor:
        memory_thread = threading.Thread(
            target=_memory_monitor,
            args=(memory_stop_event, monitor_interval),
            daemon=True
        )
        memory_thread.start()
    
    vad_config = VADConfig(sample_rate=16000, frame_duration_ms=30, padding_duration_ms=360, aggressiveness=2)

    stt_device = _detect_whisper_device()
    stt_model = FasterWhisperSTT(model_size_or_path="tiny.en", device=stt_device, compute_type="int8")

    # Use a tool-capable model (configured via environment variable)
    ollama_client = OllamaClientWithTools(
        url=OLLAMA_URL,
        model=OLLAMA_MODEL,
        stream=False,  # Disable streaming for tool support
        system_prompt="You are a helpful AI assistant with access to web search and webpage scraping tools. Use these tools when you need current information or to look up specific details online. Always be concise in your responses.",
    )

    tts_model = KokoroTTS(voice="af_bella", speed=1.0)

    vad_listener: Optional[VADListener] = None
    last_bot_response: str = ""

    def on_speech_detected(raw_bytes: bytes) -> None:
        """Callback function triggered when VAD detects speech."""
        nonlocal vad_listener, last_bot_response
        if vad_listener is None:
            return

        vad_listener.disable_vad()

        try:
            recognized_text = stt_model.run_stt(raw_bytes, sample_rate=vad_listener.sample_rate)
        except Exception as exc:  # pragma: no cover - defensive logging only
            print("STT error:", exc)
            recognized_text = ""

        print("-> User said:", recognized_text)

        if not recognized_text.strip():
            vad_listener.enable_vad()
            return

        # Simple echo cancellation: ignore if user input matches the last bot response
        normalized_user = recognized_text.strip().lower()
        normalized_bot = last_bot_response.strip().lower()
        if normalized_user and normalized_bot:
            if (
                normalized_user == normalized_bot
                or normalized_user in normalized_bot
                or normalized_bot in normalized_user
            ):
                print("-> Ignoring self-echo from recent response.")
                vad_listener.enable_vad()
                return

        # Query with tool support
        llm_response = ollama_client.query_with_tools(recognized_text, TOOLS)
        print("-> Bot response:", llm_response)
        
        if not llm_response.strip():
            vad_listener.enable_vad()
            return

        try:
            audio_data = tts_model.synthesize(llm_response)
        except Exception as exc:  # pragma: no cover - defensive logging only
            print("TTS error:", exc)
            vad_listener.enable_vad()
            return

        last_bot_response = llm_response

        # Play audio without amplitude callback (no face animation)
        tts_model.play_audio_with_amplitude(audio_data, amplitude_callback=None)
        
        # Display memory stats after interaction
        if enable_memory_monitor:
            stats = _get_memory_stats()
            process_mem = _get_process_memory()
            print(f"-> {_format_memory_stats(stats, process_mem)}")
        
        vad_listener.enable_vad()

    vad_listener = VADListener(config=vad_config, device_index=None, on_speech_callback=on_speech_detected)

    print("-> Voice assistant ready! Listening for speech...")
    print("-> 🌐 Internet access enabled via Firecrawl tools")
    print("-> Ask me to search for information or scrape webpages!")
    if enable_memory_monitor:
        print(f"-> Memory monitoring enabled (updates every {monitor_interval}s)")
    print("-> Press Ctrl+C to stop.")
    
    try:
        vad_listener.start()
    except KeyboardInterrupt:
        print("\n-> Shutting down...")
    finally:
        vad_listener.stop()
        
        # Stop memory monitor thread
        if enable_memory_monitor:
            memory_stop_event.set()
            if memory_thread and memory_thread.is_alive():
                memory_thread.join(timeout=1)
        
        # Display final memory stats
        if enable_memory_monitor:
            stats = _get_memory_stats()
            process_mem = _get_process_memory()
            print(f"-> Final {_format_memory_stats(stats, process_mem)}")
        
        print("-> Goodbye!")


if __name__ == "__main__":
    # Default: memory monitoring enabled with 60s interval
    main()
    
    # To disable memory monitoring, use:
    # main(enable_memory_monitor=False)
    
    # To change update interval (e.g., every 30 seconds):
    # main(enable_memory_monitor=True, monitor_interval=30)

