# Weather Tool Quick Reference 🌤️

## Setup (One-Time)

```bash
# 1. Get API key from https://openweathermap.org/api
# 2. Install library
pip install pyowm

# 3. Add to .env file
echo "OPENWEATHERMAP_API_KEY=your_key_here" >> .env
```

## Usage Examples

### Text Chat
```bash
python chat_internetconnected.py
```
- "What's the weather in Tokyo?"
- "How's the weather in New York, US?"
- "Get weather for London, UK"

### Voice Assistant
```bash
python main_internetconnected.py
```
- "What's the weather like today in Paris?"
- "Is it raining in Seattle?"
- "How hot is it in Dubai?"

## Location Formats
- `Tokyo` - City only
- `Tokyo, JP` - City + country (recommended)
- `New York, US` - Works with spaces
- Use ISO country codes (US, UK, JP, FR, etc.)

## What You Get
✓ Current conditions
✓ Temperature (°C and °F)
✓ Feels like temperature
✓ Min/Max temps
✓ Humidity
✓ Wind speed
✓ Cloud coverage
✓ Rain info

## Free Tier Limits
- 1,000 calls/day
- No credit card needed
- Perfect for personal use

## Troubleshooting
| Problem | Solution |
|---------|----------|
| API key not working | Wait 10-15 min after signup |
| "Not configured" error | Add key to `.env` and restart |
| Location not found | Use "City, CC" format |
| Import error | Run `pip install pyowm` |

See [WEATHER_SETUP.md](WEATHER_SETUP.md) for detailed instructions.

