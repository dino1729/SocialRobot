# Weather Feature Setup Guide 🌤️

The SocialRobot now supports real-time weather information using the OpenWeatherMap API!

## Quick Setup

### 1. Get a Free API Key

1. Go to [OpenWeatherMap](https://openweathermap.org/api)
2. Click "Sign Up" (top right)
3. Create a free account
4. Go to your API keys page: https://home.openweathermap.org/api_keys
5. Copy your API key (it may take a few minutes to activate)

**Note:** The free tier includes:
- 1,000 API calls per day
- Current weather data
- 3-hour forecast
- More than enough for personal use!

### 2. Install PyOWM Library

Make sure you're in your virtual environment, then install:

```bash
source ~/robot/.venv/bin/activate
pip install pyowm
```

Or update all requirements:

```bash
pip install -r requirements.txt
```

### 3. Add API Key to Your .env File

Edit your `.env` file and add your API key:

```bash
nano .env
```

Add this line (replace with your actual key):

```
OPENWEATHERMAP_API_KEY=your_api_key_here
```

Example `.env` file:

```bash
# Ollama Configuration
OLLAMA_URL=http://localhost:11434/api/chat
OLLAMA_MODEL=qwen3:14b

# Firecrawl Configuration
FIRECRAWL_URL=http://10.0.0.107:3002
FIRECRAWL_API_KEY=

# OpenWeatherMap Configuration
OPENWEATHERMAP_API_KEY=abc123def456ghi789jkl012mno345pq
```

Save and exit (Ctrl+X, then Y, then Enter).

## Usage

### Text Chat Mode

Run the text-based chatbot:

```bash
python chat_internetconnected.py
```

Ask weather questions:
- "What's the weather in Tokyo?"
- "How's the weather in New York?"
- "Get weather for London, UK"
- "What's the temperature in Paris?"

### Voice Assistant Mode

Run the voice assistant:

```bash
python main_internetconnected.py
```

Just ask naturally:
- "Hey robot, what's the weather like in Tokyo?"
- "How hot is it in Dubai right now?"
- "Is it raining in London?"

## Weather Information Provided

The weather tool returns:
- **Current Status** (clear, cloudy, rainy, etc.)
- **Temperature** (in both Celsius and Fahrenheit)
- **Feels Like** temperature
- **Min/Max** temperatures
- **Humidity** percentage
- **Wind Speed**
- **Cloud Coverage**
- **Rain Information** (if applicable)

## Example Output

```
Weather in Tokyo:
Status: Clear Sky
Temperature: 18.5°C (65.3°F)
Feels like: 17.2°C (63.0°F)
Min/Max: 16.0°C / 20.0°C
Humidity: 65%
Wind Speed: 3.5 m/s
Clouds: 10%
```

## Troubleshooting

**API Key Not Working?**
- Wait 10-15 minutes after creating your account (keys need activation)
- Check you copied the entire key without extra spaces
- Verify the key is correctly set in your `.env` file

**"Weather service not configured" error?**
- Make sure `OPENWEATHERMAP_API_KEY` is in your `.env` file
- Restart your script after adding the API key

**"pyowm library not found" error?**
- Install it: `pip install pyowm`
- Or: `pip install -r requirements.txt`

**Location Not Found?**
- Try adding country code: "London, UK" instead of just "London"
- Use major city names for best results
- Check spelling of city names

## API Rate Limits

Free tier limits:
- **1,000 calls/day** = ~41 calls/hour
- More than enough for personal use
- If exceeded, wait until next day or upgrade account

## Advanced: Location Formats

The weather tool accepts various location formats:

```python
"Tokyo"              # City name only
"Tokyo, JP"          # City with country code (recommended)
"New York, US"       # Works with spaces
"London, UK"         # ISO country codes
"Paris, FR"          # Two-letter codes
```

For best results, use city name + country code format!

## Security Note

⚠️ **Keep your API key private!**
- Never commit your `.env` file to git
- Don't share your API key publicly
- The `.env` file is already in `.gitignore`

---

Enjoy weather-aware conversations with your robot! 🌤️🤖

