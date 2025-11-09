# Social Robot - Basic Voice Assistant (Headless Mode)

![Hero shot placeholder](images/hero-shot-placeholder.jpg)

> **⚠️ IMPORTANT NOTE:** This is a **simplified, headless version** of the Social Robot that removes all face animation and visual display components. Perfect for running on a Jetson Orin Nano (8GB) in headless mode with minimal memory footprint.

## Overview
This basic voice assistant version provides a conversational AI experience without the visual face animation system. The software stack includes:

- **Compute & audio**: An NVIDIA Jetson Orin Nano drives the voice interaction pipeline.
- **Software**: WebRTC-based voice activity detection, Faster-Whisper speech recognition, Kokoro-ONNX text-to-speech, and an Ollama-hosted LLM (gemma3:270m - the smallest model available).
- **No visual display required**: Operates entirely in headless mode, perfect for SSH access or minimal setups.

### What's been removed from the original project:
- **Face animation system** (pygame, FaceAnimator, visual assets)
- **Display requirements** (no need for HDMI output, projector, mirrors, or screen)
- **Threading for face rendering** (simplified single-thread operation with VAD callbacks)

> [!IMPORTANT]
> **Availability & Kits** > Social Robot is free to make and print for personal use. For convenience, a limited number of pre-printed kits (including all parts **except** the Jetson) are available here: [ominousindustries.com](https://ominousindustries.com/products/ai-social-robot-kit-compatible-with-nvidia-jetson-orin-nano-super-jetson-not-included).

### Quick Links
- <h3>📺 <a href="https://vimeo.com/1120539378/5c16415a2a">Full robot build instruction video</a></h3>
- <h3>📺 <a href="https://youtu.be/0hvnBBC9HRI">Bijan Bowen YouTube Video On Robot & Software setup tutorial</a></h3>
- <h3>🎛️ <a href="https://vimeo.com/1120544089">Optional Arduino head-movement video</a></h3>

![Exploded view placeholder](images/exploded-view-placeholder.jpg)

## Software Setup

### 1. OS Prerequisites (Jetson)
Install system packages once per Jetson Orin Nano to support audio and build tooling:
```bash
sudo apt update
sudo apt install -y git curl python3-venv python3-dev build-essential \
    libportaudio2 portaudio19-dev libasound-dev
```
- `pyaudio` depends on PortAudio/ALSA headers and libraries.
- `webrtcvad` requires development headers on aarch64.

### 2. Clone the Repository
```bash
git clone https://github.com/OminousIndustries/SocialRobot.git
cd SocialRobot
```

### 3. Create the Python Environment
```bash
python3 -m venv ~/robot/.venv
source ~/robot/.venv/bin/activate
python -m pip install -r requirements.txt
```
The first run downloads the Faster-Whisper `tiny.en` model and Kokoro ONNX/voice files into `~/.cache/kokoro_onnx`.

### 4. ~~Provide Face Assets~~ (Not needed for basic voice assistant)
**Note:** The face animation system has been removed in this basic voice assistant version. The robot now operates in headless mode without visual display. 

### 5. Install & Prepare Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run gemma3:270m   # downloads the gemma3 270m model
```
The application reaches `http://localhost:11434/api/chat` and expects the `gemma3:270m` model in streaming mode. Keep the Ollama service running in the background (`ollama serve` if systemd is unavailable).

### 6. Configure Audio Devices
1. Connect a USB microphone and speakers/headphones.
2. Set default input/output devices via the desktop sound settings.
3. When operating headless, adjust PulseAudio defaults over SSH:
```bash
pactl list short sinks       # identify HDMI or other outputs
pactl set-default-sink <sink_name>

pactl list short sources     # identify microphone input
pactl set-default-source <source_name>
```
PulseAudio settings reset on reboot unless persisted via configuration files.

### 7. Run the Application
```bash
# inside the venv and repository root
python main.py
```
The voice assistant runs in headless mode. The VAD loop listens for speech, processes it through Whisper STT, generates responses using Ollama LLM (gemma3:270m - the lowest memory model), and speaks back using Kokoro TTS. Whisper falls back to CPU if CUDA is unavailable; otherwise, ctranslate2 leverages GPU acceleration.

## Bill of Materials

### Electronics
- NVIDIA Jetson Orin Nano Super Developer Kit (available from Micro Center and other retailers)
- Yoton Y3 Mini Projector — <https://a.co/d/gWdswff>
- Generic USB Microphone — <https://a.co/d/i5nXfO6>
- 1 ft HDMI Cable — <https://a.co/d/eZBGOwM>
- Right-angle Male-to-Female HDMI Adapter (90° or 270°) — <https://a.co/d/8lm3JhN>
- DisplayPort-to-HDMI Adapter — <https://a.co/d/3IJHrtP>

### Hardware & Materials
- 3 mm thick 6"×4" acrylic mirrors (two required) — <https://marketingholders.com/acrylic-fashion-displays-and-organizers/acrylic-mirror-stands-and-sheets/4-x-6-acrylic-mirror-sheet-for-replacement-or-diy-crafts>
- 10 mil milky translucent PET (Mylar) sheet, 12"×12" — <https://a.co/d/gCJCugM>
- Hex socket head M4 screw and nut set — <https://a.co/d/7dKUKiq>
- Hex socket head M2/M3/M4 assorted screw and nut set — <https://a.co/d/aQrlfdq>
- 4 mm nylon spacer for sound level sensor (optional) — <https://a.co/d/1VccBdR>
- Two 1 kg rolls of 3D-printer filament in your preferred colors/materials

### Optional Head Movement System (Arduino)
- Arduino Uno R3 — <https://a.co/d/bvDaFiA>
- 28BYJ-48 stepper motor & ULN2003 driver board — <https://a.co/d/c2HEamP>
- Mini 170-point breadboard — <https://a.co/d/5Bglmuq>
- Two LM393 3-pin sound level sensors — <https://a.co/d/bQ231V5>
- Arduino jumper wires — <https://a.co/d/es0TScP>

![Electronics layout placeholder](images/electronics-layout-placeholder.jpg)

## 3D Printed Parts

All print files are available here:  
**Printables:** <https://www.printables.com/model/1420907-ominous-industries-social-robot>

A **limited number of pre‑printed kits** (Not Including the Jetson) are available here [ominousindustries.com](https://ominousindustries.com/products/ai-social-robot-kit-compatible-with-nvidia-jetson-orin-nano-super-jetson-not-included).

---

## Mechanical Assembly Overview
Follow along with the **[Build instruction video](https://vimeo.com/1120539378/5c16415a2a)** for step-by-step visuals. 

### Optional Head Movement System
If you plan to add head motion, watch the **[Arduino instruction video](https://vimeo.com/1120544089)**

## Runtime Expectations & Troubleshooting
- First launch may take several minutes while models download (Faster-Whisper, Kokoro assets, Ollama LLM).
- If `pyaudio` reports missing devices, confirm PulseAudio/ALSA can access hardware, or run as the desktop user instead of root.
- Adjust `FaceSettings` in `main.py` to tweak display size, rotation, and offsets.
- Change voices or playback rate by passing different parameters when constructing `KokoroTTS`. List options with `KokoroTTS().available_voices()`.
- If an Ubuntu software update window blocks the projection, locate and terminate the process:
  ```bash
  pgrep -af update
  kill <PID>
  ```
- If you are unable to get Chromium to launch on the Jetson/Jetpack 6.2, see these instructions: [Chromium Jetson Fix](https://forums.developer.nvidia.com/t/chromium-other-browsers-not-working-after-flashing-or-updating-heres-why-and-quick-fix/338891)

![Completed robot placeholder](images/completed-robot-placeholder.jpg)
