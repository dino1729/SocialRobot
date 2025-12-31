# Social Robot 🤖

A lightweight, headless voice assistant that listens, thinks, and responds. Designed to run locally (e.g., NVIDIA Jetson Orin Nano), with optional “online LLM” and optional internet tools.

## What It Does

- **VAD**: WebRTC VAD detects when you start/stop speaking.
- **STT**: Faster-Whisper (default) or OpenAI Whisper (PyTorch).
- **LLM**: Local Ollama (default) or online via LiteLLM-compatible endpoints.
- **TTS**: Kokoro-ONNX (default) plus optional Piper / Chatterbox / VibeVoice.

Everything is driven by a single entrypoint: `main.py` (feature flags are CLI options and/or `.env` defaults).

## Modes & Examples

Run continuous listening (local Ollama):
```bash
python main.py
```

Wake word (“Hey Jarvis” by default):
```bash
python main.py --wakeword
```

Enable internet tools (web search/scrape + weather):
```bash
python main.py --tools
```

Full featured (wake word + online LLM + tools):
```bash
python main.py --wakeword --llm litellm --tools
```

TTS/STT examples:
```bash
python main.py --tts-engine piper --tts-gpu
python main.py --tts-engine chatterbox --voice rick_sanchez --tts-gpu
python main.py --stt-engine openai-whisper
python main.py --device cpu --compute-type int8
```

## Setup

### 1) System dependencies (Linux)
```bash
sudo apt update
sudo apt install -y git curl python3-venv python3-dev build-essential \
  libportaudio2 portaudio19-dev libasound-dev
```

### 2) Python environment + install
Python version is pinned in `.python-version`.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional GPU / extra engines:
```bash
pip install -r requirements_gpu.txt
```

### 3) Configure `.env`
```bash
cp env.example .env
```
Edit keys as needed:
- Local LLM: `OLLAMA_URL`, `OLLAMA_MODEL`
- Online LLM: `LITELLM_URL`, `LITELLM_MODEL`, `LITELLM_API_KEY`
- Tools: `FIRECRAWL_URL`, `FIRECRAWL_API_KEY`, `OPENWEATHERMAP_API_KEY`
- Wake word: `WAKEWORD_MODEL`, `WAKEWORD_THRESHOLD`

### 4) Ollama (local LLM)
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run gemma3:270m
```

## Audio Setup (PulseAudio / PipeWire)

Quick path:
```bash
./set_audio_defaults.sh --auto
./check_audio_defaults.sh
```

Manual path:
```bash
pactl list short sinks
pactl set-default-sink <sink_name>
pactl list short sources
pactl set-default-source <source_name>
```

## Run in the Background

Use the bot manager scripts (logs to `bot.log`, PID in `.bot.pid`):
```bash
./start_bot.sh basic
./start_bot.sh wakeword_tools_online --tts-engine piper --tts-gpu
./status_bot.sh
./stop_bot.sh
```
See `BOT_SCRIPTS_README.md` for more examples.

## Wake Word Setup

Install and validate OpenWakeWord + models:
```bash
./setup_wakeword.sh
```
Then run: `python main.py --wakeword` (tune `WAKEWORD_THRESHOLD` in `.env` if needed).

## Tests

Run all tests:
```bash
python -m pytest -q
```
Unit tests only:
```bash
python -m pytest -q tests/unit
```
Integration tests may require audio devices, GPU, or API keys (see `tests/integration/`).

Manual smoke tests:
- STT: `python test_stt.py`
- TTS: `python test_tts.py`

Benchmarks (optional):
- `python benchmark_engines.py`
- `python benchmark_tts_engines.py`

## Troubleshooting

- **GPU STT failures (cuDNN / CUDA issues)**: try `python main.py --device cpu` or switch engines with `--stt-engine openai-whisper`.
- **No audio devices / wrong defaults**: run `./set_audio_defaults.sh --auto` and re-check with `./check_audio_defaults.sh`.
- **Ollama not responding**: start it with `ollama serve`.
