"""Weather information tool using OpenWeatherMap API."""

import os
from dotenv import load_dotenv

load_dotenv()

OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")
DEFAULT_WEATHER_LOCATION = os.getenv("DEFAULT_WEATHER_LOCATION", "")


def get_weather(location: str = "") -> str:
    """Get current weather for a location using OpenWeatherMap API.
    
    Args:
        location: City name or "city, country code" (e.g., "London", "Tokyo, JP").
                  If empty/not provided, uses DEFAULT_WEATHER_LOCATION from .env
        
    Returns:
        Weather information as a string
    """
    try:
        if not OPENWEATHERMAP_API_KEY:
            return "Weather service is not configured. Please set OPENWEATHERMAP_API_KEY in your .env file."
        
        # Use default location if none provided
        if not location or not location.strip():
            if DEFAULT_WEATHER_LOCATION:
                location = DEFAULT_WEATHER_LOCATION
                print(f"\n🌤️  [TOOL CALL] Using default location: {location}")
            else:
                return "No location specified and no DEFAULT_WEATHER_LOCATION set in .env file. Please provide a location."
        else:
            print(f"\n🌤️  [TOOL CALL] Getting weather for: {location}")
        
        # Try to import pyowm
        try:
            import pyowm
        except ImportError:
            return "Weather service requires 'pyowm' library. Install it with: pip install pyowm"
        
        # Normalize location format for OpenWeatherMap
        # Convert "City, ST" or "City, State" to "City,US" format
        normalized_location = location.strip()
        
        # US state abbreviations to detect US locations
        us_states = {
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
        }
        
        if ',' in normalized_location:
            parts = [p.strip() for p in normalized_location.split(',')]
            if len(parts) == 2:
                city, state_or_country = parts
                # Check if it looks like a US state abbreviation
                if state_or_country.upper() in us_states:
                    normalized_location = f"{city},US"
                    print(f"   Normalized to: {normalized_location}")
        
        # Initialize PyOWM
        owm = pyowm.OWM(OPENWEATHERMAP_API_KEY)
        mgr = owm.weather_manager()
        
        # Get weather for location
        observation = mgr.weather_at_place(normalized_location)
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

