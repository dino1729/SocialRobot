#!/usr/bin/env python3
"""Quick setup verification script for the voice assistant."""

import sys

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing module imports...")
    
    try:
        import pyaudio
        print("  ✓ pyaudio")
    except ImportError as e:
        print(f"  ✗ pyaudio: {e}")
        return False
    
    try:
        import webrtcvad
        print("  ✓ webrtcvad")
    except ImportError as e:
        print(f"  ✗ webrtcvad: {e}")
        return False
    
    try:
        from faster_whisper import WhisperModel
        print("  ✓ faster-whisper")
    except ImportError as e:
        print(f"  ✗ faster-whisper: {e}")
        return False
    
    try:
        from kokoro_onnx import Kokoro
        print("  ✓ kokoro-onnx")
    except ImportError as e:
        print(f"  ✗ kokoro-onnx: {e}")
        return False
    
    try:
        import requests
        print("  ✓ requests")
    except ImportError as e:
        print(f"  ✗ requests: {e}")
        return False
    
    try:
        import numpy
        print("  ✓ numpy")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False
    
    return True

def test_cuda():
    """Check CUDA availability for faster-whisper."""
    print("\nChecking CUDA availability...")
    try:
        import ctranslate2
        cuda_devices = ctranslate2.get_cuda_device_count()
        if cuda_devices > 0:
            print(f"  ✓ CUDA available: {cuda_devices} device(s)")
            return True
        else:
            print("  ⚠ CUDA not available (will use CPU)")
            return True
    except Exception as e:
        print(f"  ⚠ CUDA check failed: {e} (will use CPU)")
        return True

def test_audio_devices():
    """Check available audio devices."""
    print("\nChecking audio devices...")
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        
        print("\n  Audio Input Devices:")
        input_count = 0
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"    [{i}] {info['name']} ({info['maxInputChannels']} channels)")
                input_count += 1
        
        if input_count == 0:
            print("    ✗ No input devices found!")
        else:
            print(f"    ✓ Found {input_count} input device(s)")
        
        print("\n  Audio Output Devices:")
        output_count = 0
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info['maxOutputChannels'] > 0:
                print(f"    [{i}] {info['name']} ({info['maxOutputChannels']} channels)")
                output_count += 1
        
        if output_count == 0:
            print("    ✗ No output devices found!")
        else:
            print(f"    ✓ Found {output_count} output device(s)")
        
        p.terminate()
        return input_count > 0 and output_count > 0
        
    except Exception as e:
        print(f"  ✗ Audio device check failed: {e}")
        return False

def test_ollama():
    """Check if Ollama is running and has the required model."""
    print("\nChecking Ollama setup...")
    try:
        import requests
        
        # Check if Ollama is running
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print("  ✓ Ollama server is running")
                
                # Check for gemma3:270m model
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                
                if any('gemma3:270m' in model for model in models):
                    print("  ✓ gemma3:270m model is available")
                    return True
                else:
                    print("  ✗ gemma3:270m model not found!")
                    print("    Available models:", models)
                    print("    Run: ollama run gemma3:270m")
                    return False
            else:
                print(f"  ✗ Ollama returned status code: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            print("  ✗ Cannot connect to Ollama server")
            print("    Make sure Ollama is running: ollama serve")
            return False
        except requests.exceptions.Timeout:
            print("  ✗ Connection to Ollama timed out")
            return False
            
    except Exception as e:
        print(f"  ✗ Ollama check failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Voice Assistant Setup Verification")
    print("=" * 60)
    
    results = []
    
    # Test imports
    results.append(("Module Imports", test_imports()))
    
    # Test CUDA
    results.append(("CUDA Support", test_cuda()))
    
    # Test audio devices
    results.append(("Audio Devices", test_audio_devices()))
    
    # Test Ollama
    results.append(("Ollama Setup", test_ollama()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All checks passed! Ready to run the voice assistant.")
        print("\nRun: python main.py")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

