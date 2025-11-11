# Bot Management Scripts

These scripts allow you to easily start, stop, and check the status of your SocialRobot voice assistant in the background.

## Scripts Overview

- **`start_bot.sh`** - Start the bot in the background
- **`stop_bot.sh`** - Stop the running bot
- **`status_bot.sh`** - Check if the bot is running and view recent logs

## Usage

### Starting the Bot

**Default (internet-connected with LiteLLM):**
```bash
./start_bot.sh
```

**Choose a specific variant:**
```bash
./start_bot.sh wakeword                        # Basic (Ollama)
./start_bot.sh wakeword_online                 # Basic (LiteLLM/OpenAI)
./start_bot.sh wakeword_internetconnected      # With internet tools (Ollama)
./start_bot.sh wakeword_internetconnected_online # With internet tools (LiteLLM/OpenAI)
```

**Pass additional arguments:**
```bash
# Use Piper TTS instead of Kokoro
./start_bot.sh wakeword_online --tts-engine piper

# Use Piper with GPU and custom wakeword threshold
./start_bot.sh wakeword_internetconnected --tts-engine piper --tts-gpu --wakeword-threshold 0.6

# Use specific compute type
./start_bot.sh wakeword --compute-type int8

# Use Kokoro with different voice and speed
./start_bot.sh wakeword_internetconnected_online --tts-voice af_sarah --tts-speed 1.2

# Disable memory monitoring
./start_bot.sh wakeword_online --no-memory-monitor
```

### Stopping the Bot

```bash
./stop_bot.sh
```

This will gracefully stop the running bot. If it doesn't stop within 10 seconds, it will be forcefully terminated.

### Checking Status

```bash
./status_bot.sh
```

This shows:
- Whether the bot is running
- Process ID (PID)
- How long it's been running
- Which script/command is running
- Recent log entries

### Viewing Logs

**View logs in real-time:**
```bash
tail -f bot.log
```

**View all logs:**
```bash
cat bot.log
```

**View last 50 lines:**
```bash
tail -n 50 bot.log
```

## Files Created

- **`.bot.pid`** - Contains the process ID of the running bot (automatically managed)
- **`bot.log`** - Contains all console output from the bot (stdout and stderr)

## Examples

### Example 1: Start with default settings
```bash
./start_bot.sh
# Bot starts with wakeword_internetconnected_online variant
```

### Example 2: Start with Piper TTS and custom settings
```bash
./start_bot.sh wakeword_internetconnected_online --tts-engine piper --tts-gpu --wakeword-threshold 0.55
```

### Example 3: Check if bot is running
```bash
./status_bot.sh
```

### Example 4: Stop the bot
```bash
./stop_bot.sh
```

### Example 5: Restart the bot with different settings
```bash
./stop_bot.sh
./start_bot.sh wakeword --tts-engine kokoro --tts-voice af_bella --tts-speed 1.1
```

## Troubleshooting

### Bot won't start
1. Check the logs: `cat bot.log`
2. Make sure your `.env` file is configured
3. Check if required services are running (Ollama, Firecrawl, etc.)
4. Try running the script directly to see error messages: `python3 main_wakeword_internetconnected_online.py`

### Bot is running but not responding
1. Check the logs: `tail -f bot.log`
2. Check the status: `./status_bot.sh`
3. Try restarting: `./stop_bot.sh && ./start_bot.sh`

### Multiple bots running
Only one bot can run at a time (per this script). If you try to start another, it will warn you that one is already running.

### Clean up stale PID file
If the bot crashed and left a stale PID file:
```bash
rm .bot.pid
```

## Background Operation

The bot runs completely in the background using `nohup`. This means:
- ✓ The bot continues running even if you close your terminal
- ✓ Output is redirected to `bot.log`
- ✓ You can safely log out and the bot keeps running
- ✓ The bot will stop if the system reboots (unless you set up autostart)

## Setting Up Autostart (Optional)

To automatically start the bot when your system boots, you can:

1. **Using systemd (recommended for Linux):** Create a systemd service file
2. **Using crontab:** Add `@reboot` entry
3. **Using rc.local:** Add the start command to `/etc/rc.local`

Let me know if you need help setting up autostart!

