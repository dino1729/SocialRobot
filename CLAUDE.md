# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SocialRobot is a headless voice assistant that runs locally on NVIDIA Jetson or desktop Linux. It combines wake word detection, speech recognition (STT), LLM inference, and text-to-speech (TTS) into a conversational AI pipeline.

## Common Commands

### Running the Assistant

```bash
# Activate virtual environment first
source .venv/bin/activate

# Basic continuous listening (local Ollama)
python main.py

# Wake word activated
python main.py --wakeword

# With internet tools (web search, weather)
python main.py --tools

# Full featured: wake word + online LLM + tools
python main.py --wakeword --llm litellm --tools

# Custom TTS engine
python main.py --tts-engine chatterbox --voice morgan_freeman --tts-gpu
python main.py --tts-engine vibevoice --vibevoice-speaker Carter --tts-gpu
```

### Testing

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run integration tests (requires audio hardware)
pytest tests/integration/

# Run a specific test file
pytest tests/unit/test_vad.py

# Run with coverage
pytest --cov=audio --cov=llm --cov=tools
```

### Testing Individual Components

```bash
# Test TTS engines
python test_tts.py --engine kokoro --voice af_bella "Hello world"
python test_tts.py --engine chatterbox --voice c3po "Hello world"
python test_tts.py --engine vibevoice --speaker Carter "Hello world"
python test_tts.py --list-voices

# Test STT engines
python test_stt.py
python test_stt.py --engine openai-whisper --gpu
```

### Bot Management (Background Operation)

```bash
./start_bot.sh                    # Start with defaults
./start_bot.sh wakeword           # Start wake word mode
./stop_bot.sh                     # Stop the bot
./status_bot.sh                   # Check status and logs
tail -f bot.log                   # Watch logs
```

### Audio Configuration

```bash
./check_audio_defaults.sh         # Show current audio devices
./set_audio_defaults.sh           # Configure default mic/speaker
```

## Architecture

### Core Pipeline Flow

```
[Microphone] -> VAD -> STT -> LLM -> TTS -> [Speaker]
                 |
            [Wake Word] (optional)
```

### Module Structure

**`audio/`** - Audio processing components:
- `vad.py` - Voice Activity Detection using WebRTC-VAD
- `stt.py` - Faster-Whisper STT (ctranslate2-based)
- `stt_openai.py` - OpenAI Whisper STT (PyTorch-based, better RTX 50xx support)
- `tts.py` - Kokoro TTS (ONNX, CPU-optimized)
- `tts_piper.py` - Piper TTS (lightweight local)
- `tts_chatterbox.py` - Chatterbox TTS (zero-shot voice cloning, GPU)
- `tts_vibevoice.py` - Microsoft VibeVoice TTS (neural, GPU)
- `engine_config.py` - Factory functions for STT/TTS engine creation
- `suppress_warnings.py` - Utilities to suppress ALSA/JACK/ML library warnings

**`llm/`** - LLM backends:
- `ollama.py` - Local Ollama inference
- `litellm.py` - Online APIs (OpenAI, Anthropic, etc.) with tool calling

**`tools/`** - Internet tools for LLM function calling:
- `web_tools.py` - Web search and URL scraping via Firecrawl
- `weather_tool.py` - Weather via OpenWeatherMap API
- `definitions.py` - Tool schemas for LLM function calling

**`voices/`** - WAV files for Chatterbox voice cloning (e.g., `morgan_freeman.wav`)

**`personas/`** - TXT files with personality prompts (e.g., `morgan_freeman.txt`)

### Engine Selection

The codebase supports swappable engines via `audio/engine_config.py`:

| Component | Options | Default |
|-----------|---------|---------|
| STT | `faster-whisper`, `openai-whisper` | `faster-whisper` |
| TTS | `kokoro`, `piper`, `chatterbox`, `vibevoice` | `kokoro` |
| LLM | `ollama`, `litellm` | `ollama` |

### Configuration

All settings are in `.env` (copy from `env.example`). Key sections:
- Feature flags: `USE_WAKEWORD`, `USE_TOOLS`, `LLM_BACKEND`
- LLM: `OLLAMA_URL`, `OLLAMA_MODEL`, `LITELLM_*`
- VAD: `VAD_AGGRESSIVENESS`, `VAD_PADDING_DURATION_MS`
- STT/TTS: `STT_ENGINE`, `TTS_ENGINE`, `TTS_GPU`, `TTS_VOICE`

### Voice Cloning (Chatterbox)

To add a new voice character:
1. Add `voices/{name}.wav` - 5-10 second voice sample
2. Add `personas/{name}.txt` - personality system prompt
3. Use with `--voice {name}` flag

## Key Patterns

### Warning Suppression

Use `audio/suppress_warnings.py` for clean output:
```python
from audio.suppress_warnings import suppress_ml_warnings, get_pyaudio
suppress_ml_warnings()  # Call early, before ML imports
pyaudio = get_pyaudio()  # Silent PyAudio initialization
```

### Engine Factory Pattern

Create engines via factory functions, not direct imports:
```python
from audio.engine_config import create_stt_engine, create_tts_engine
stt = create_stt_engine(engine="openai-whisper", device="cuda")
tts = create_tts_engine(engine="chatterbox", use_gpu=True, voice_path="voices/c3po.wav")
```

## Dependencies

- **Local venv**: `.venv/` with `transformers==4.51.3` (required for VibeVoice compatibility)
- **External**: VibeVoice cloned to `/home/dino/myprojects/VibeVoice` (editable install)
- **Services**: Ollama for local LLM, Firecrawl for web tools (optional)
