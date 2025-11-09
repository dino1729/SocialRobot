"""Weather information tool using OpenWeatherMap API."""

import os
from dotenv import load_dotenv

load_dotenv()

OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")


def get_weather(location: str) -> str:
    """Get current weather for a location using OpenWeatherMap API.
    
    Args:
        location: City name or "city, country code" (e.g., "London", "Tokyo, JP")
        
    Returns:
        Weather information as a string
    """
    try:
        if not OPENWEATHERMAP_API_KEY:
            return "Weather service is not configured. Please set OPENWEATHERMAP_API_KEY in your .env file."
        
        print(f"\n🌤️  [TOOL CALL] Getting weather for: {location}")
        
        # Try to import pyowm
        try:
            import pyowm
        except ImportError:
            return "Weather service requires 'pyowm' library. Install it with: pip install pyowm"
        
        # Initialize PyOWM
        owm = pyowm.OWM(OPENWEATHERMAP_API_KEY)
        mgr = owm.weather_manager()
        
        # Get weather for location
        observation = mgr.weather_at_place(location)
        weather = observation.weather
        
        # Get temperature in Celsius and Fahrenheit
        temp = weather.temperature('celsius')
        temp_f = weather.temperature('fahrenheit')
        
        # Format weather information
        result = f"Weather in {location}:\n"
        result += f"Status: {weather.detailed_status.title()}\n"
        result += f"Temperature: {temp['temp']:.1f}°C ({temp_f['temp']:.1f}°F)\n"
        result += f"Feels like: {temp['feels_like']:.1f}°C ({temp_f['feels_like']:.1f}°F)\n"
        result += f"Min/Max: {temp['temp_min']:.1f}°C / {temp['temp_max']:.1f}°C\n"
        result += f"Humidity: {weather.humidity}%\n"
        result += f"Wind Speed: {weather.wind()['speed']} m/s\n"
        
        # Add clouds and rain if available
        if weather.clouds:
            result += f"Clouds: {weather.clouds}%\n"
        
        rain = weather.rain
        if rain:
            result += f"Rain: {rain}\n"
        
        print(f"✓ Weather retrieved successfully")
        return result
        
    except Exception as e:
        error_msg = f"Weather error: {str(e)}"
        print(f"✗ {error_msg}")
        return error_msg

