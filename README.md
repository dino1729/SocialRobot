# Social Robot 🤖

A lightweight, headless voice assistant powered by AI that listens, thinks, and responds—all running locally on an NVIDIA Jetson Orin Nano.

## What is it?

Talk to your robot and it talks back! This conversational AI assistant uses:
- **Voice Activity Detection** (WebRTC-VAD) to hear when you're speaking
- **Speech Recognition** (Faster-Whisper) to understand what you're saying
- **Language Model** (Ollama with gemma3:270m) to generate intelligent responses
- **Text-to-Speech** (Kokoro-ONNX) to speak back naturally

No cloud services required—everything runs locally in headless mode, perfect for SSH access and minimal setups.

## Available Modes

| Mode | Script | Features |
|------|--------|----------|
| **Basic** | `main.py` | Continuous listening, offline, no internet |
| **Internet** | `main_internetconnected.py` | Continuous listening, web search, weather tools |
| **Wake Word** | `main_wakeword.py` | Wake word activation ("Hey Jarvis"), offline |
| **Wake Word + Internet** | `main_wakeword_internetconnected.py` | Wake word activation with internet tools |

See [WAKEWORD_GUIDE.md](WAKEWORD_GUIDE.md) for wake word setup and configuration.

## Quick Start

### 1. Install System Dependencies
```bash
sudo apt update
sudo apt install -y git curl python3-venv python3-dev build-essential \
    libportaudio2 portaudio19-dev libasound-dev
```

### 2. Clone & Setup
```bash
git clone https://github.com/OminousIndustries/SocialRobot.git
cd SocialRobot
python3 -m venv ~/robot/.venv
source ~/robot/.venv/bin/activate
pip install -r requirements.txt
```

### 3. Install Ollama LLM
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run gemma3:270m
```

### 4. Configure Audio
Connect a USB microphone and speakers, then set your default audio devices:
```bash
pactl list short sinks    # find your output
pactl set-default-sink <sink_name>

pactl list short sources  # find your input
pactl set-default-source <source_name>
```

### 5. Run!

**Option A: Continuous Listening (default)**
```bash
python main.py
```

**Option B: Wake Word Activation (hands-free)**
```bash
python main_wakeword.py
```

Start talking—your robot is listening! 🎤

> **💡 Wake Word Mode**: Say "Hey Jarvis" to activate the assistant, then speak your command. See [WAKEWORD_GUIDE.md](WAKEWORD_GUIDE.md) for details.

## Troubleshooting

**First run taking forever?**  
Models are downloading (Faster-Whisper, Kokoro, Ollama). Grab a coffee ☕

**No audio devices found?**  
Check that PulseAudio can see your hardware, and run as your desktop user (not root).

**Want a different voice?**  
Modify the `KokoroTTS` parameters in your code. List available voices with `KokoroTTS().available_voices()`.

**Ollama not responding?**  
Make sure the service is running: `ollama serve`

---

Made with ❤️ for tinkerers, makers, and AI enthusiasts.
