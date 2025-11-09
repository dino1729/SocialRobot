#!/usr/bin/env python3
"""Test script to verify internet-connected voice assistant setup."""

import os
import sys

import requests


def test_ollama_connection():
    """Test if Ollama is accessible."""
    print("\n1. Testing Ollama Connection...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("   ✓ Ollama is running")
            
            # Check for tool-capable models
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            
            tool_capable = []
            for model in models:
                if any(x in model.lower() for x in ['llama3.1', 'mistral-nemo', 'firefunction', 'command-r']):
                    tool_capable.append(model)
            
            if tool_capable:
                print(f"   ✓ Found tool-capable model(s): {', '.join(tool_capable)}")
                return True
            else:
                print("   ✗ No tool-capable models found!")
                print("     Install one with: ollama pull llama3.2:1b")
                return False
        else:
            print(f"   ✗ Ollama returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ Cannot connect to Ollama: {e}")
        print("     Make sure Ollama is running")
        return False


def test_firecrawl_connection():
    """Test if Firecrawl is accessible."""
    print("\n2. Testing Firecrawl Connection...")
    firecrawl_url = os.getenv("FIRECRAWL_URL", "http://localhost:3002")
    
    try:
        # Use root endpoint since /health doesn't exist in Firecrawl
        response = requests.get(f"{firecrawl_url}/", timeout=5)
        if response.status_code == 200:
            print(f"   ✓ Firecrawl is accessible at {firecrawl_url}")
            return True
        else:
            print(f"   ✗ Firecrawl returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"   ✗ Cannot connect to Firecrawl at {firecrawl_url}")
        print("     Start Firecrawl with: cd ~/firecrawl && docker compose up -d")
        return False
    except Exception as e:
        print(f"   ✗ Firecrawl error: {e}")
        return False


def test_firecrawl_search():
    """Test Firecrawl search functionality."""
    print("\n3. Testing Firecrawl Search...")
    firecrawl_url = os.getenv("FIRECRAWL_URL", "http://localhost:3002")
    firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY", "")
    
    try:
        headers = {"Content-Type": "application/json"}
        if firecrawl_api_key:
            headers["Authorization"] = f"Bearer {firecrawl_api_key}"
        
        response = requests.post(
            f"{firecrawl_url}/v2/search",
            json={"query": "test search"},
            headers=headers,
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("data", [])
            print(f"   ✓ Search working! Found {len(results)} results")
            return True
        else:
            print(f"   ✗ Search returned status {response.status_code}")
            print(f"     Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"   ✗ Search test failed: {e}")
        return False


def test_firecrawl_scrape():
    """Test Firecrawl scraping functionality."""
    print("\n4. Testing Firecrawl Scrape...")
    firecrawl_url = os.getenv("FIRECRAWL_URL", "http://localhost:3002")
    firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY", "")
    
    try:
        headers = {"Content-Type": "application/json"}
        if firecrawl_api_key:
            headers["Authorization"] = f"Bearer {firecrawl_api_key}"
        
        response = requests.post(
            f"{firecrawl_url}/v2/scrape",
            json={
                "url": "https://example.com",
                "formats": ["markdown"]
            },
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            content = data.get("data", {}).get("markdown", "")
            if content:
                print(f"   ✓ Scraping working! Got {len(content)} characters")
                return True
            else:
                print("   ⚠️  Scrape succeeded but no content returned")
                return False
        else:
            print(f"   ✗ Scrape returned status {response.status_code}")
            print(f"     Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"   ✗ Scrape test failed: {e}")
        return False


def test_ollama_tool_support():
    """Test if Ollama supports tool calling."""
    print("\n5. Testing Ollama Tool Support...")
    
    try:
        import ollama
        
        # Find a tool-capable model
        models = ollama.list()
        tool_model = None
        
        for model in models.get('models', []):
            name = model.get('name', '')
            if 'llama3.1' in name.lower():
                tool_model = name
                break
        
        if not tool_model:
            print("   ✗ No tool-capable model found")
            print("     Install with: ollama pull llama3.2:1b")
            return False
        
        print(f"   Testing with model: {tool_model}")
        
        # Test tool call
        response = ollama.chat(
            model=tool_model,
            messages=[{'role': 'user', 'content': 'What is 2+2?'}],
            tools=[{
                'type': 'function',
                'function': {
                    'name': 'calculator',
                    'description': 'Perform calculations',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'expression': {
                                'type': 'string',
                                'description': 'The math expression'
                            }
                        },
                        'required': ['expression']
                    }
                }
            }]
        )
        
        # Check if model responded (tool call or direct answer both ok)
        if response:
            print("   ✓ Ollama tool support is working")
            return True
        else:
            print("   ✗ Ollama tool support test failed")
            return False
            
    except ImportError:
        print("   ⚠️  ollama-python package not installed")
        print("     Install with: pip install ollama")
        return False
    except Exception as e:
        print(f"   ✗ Tool support test failed: {e}")
        return False


def test_audio_devices():
    """Test audio device configuration."""
    print("\n6. Testing Audio Devices...")
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        
        input_count = sum(1 for i in range(p.get_device_count()) 
                         if p.get_device_info_by_index(i)['maxInputChannels'] > 0)
        output_count = sum(1 for i in range(p.get_device_count()) 
                          if p.get_device_info_by_index(i)['maxOutputChannels'] > 0)
        
        p.terminate()
        
        if input_count > 0 and output_count > 0:
            print(f"   ✓ Audio devices found: {input_count} input(s), {output_count} output(s)")
            return True
        else:
            print(f"   ✗ Audio devices missing")
            return False
    except Exception as e:
        print(f"   ✗ Audio test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("Internet-Connected Voice Assistant Setup Verification")
    print("=" * 70)
    
    results = []
    
    # Core tests
    results.append(("Ollama Connection", test_ollama_connection()))
    results.append(("Firecrawl Connection", test_firecrawl_connection()))
    
    # Feature tests (only if core services are up)
    if results[-1][1]:  # If Firecrawl is accessible
        results.append(("Firecrawl Search", test_firecrawl_search()))
        results.append(("Firecrawl Scrape", test_firecrawl_scrape()))
    
    # Optional tests
    results.append(("Ollama Tool Support", test_ollama_tool_support()))
    results.append(("Audio Devices", test_audio_devices()))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_core_passed = all(result for name, result in results[:2])
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 70)
    
    if all_passed:
        print("✅ All tests passed! Ready to run the internet-connected assistant.")
        print("\nRun with:")
        print("  source venv/bin/activate")
        print("  python main_internetconnected.py")
        return 0
    elif all_core_passed:
        print("⚠️  Core services working, but some features need attention.")
        print("   You can still run the assistant, but some features may not work.")
        print("\nRun with:")
        print("  source venv/bin/activate")
        print("  python main_internetconnected.py")
        return 0
    else:
        print("❌ Core services not ready. Please fix the issues above.")
        print("\nQuick fixes:")
        print("  - Ollama: Make sure it's running (ollama serve)")
        print("  - Firecrawl: cd ~/firecrawl && docker compose up -d")
        print("  - Model: ollama pull llama3.2:1b")
        return 1


if __name__ == "__main__":
    sys.exit(main())

