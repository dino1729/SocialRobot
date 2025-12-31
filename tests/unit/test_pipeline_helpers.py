"""Unit tests for pipeline decision helpers in main.py."""

import unittest


class TestIsAudioTooShort(unittest.TestCase):
    """Tests for is_audio_too_short() helper."""

    def test_empty_audio_is_too_short(self):
        """Verify empty audio is rejected."""
        from main import is_audio_too_short
        
        self.assertTrue(is_audio_too_short(b''))

    def test_short_audio_is_rejected(self):
        """Verify audio shorter than 1 second is rejected."""
        from main import is_audio_too_short, MIN_AUDIO_BYTES
        
        # 0.5 seconds worth at 16kHz, 16-bit = 16000 bytes
        short_audio = b'\x00' * 16000
        self.assertTrue(is_audio_too_short(short_audio))

    def test_long_audio_is_accepted(self):
        """Verify audio longer than 1 second is accepted."""
        from main import is_audio_too_short, MIN_AUDIO_BYTES
        
        # 1.5 seconds worth at 16kHz, 16-bit = 48000 bytes
        long_audio = b'\x00' * 48000
        self.assertFalse(is_audio_too_short(long_audio))

    def test_exactly_threshold_is_accepted(self):
        """Verify audio exactly at threshold is accepted."""
        from main import is_audio_too_short, MIN_AUDIO_BYTES
        
        # Exactly 1 second
        audio = b'\x00' * MIN_AUDIO_BYTES
        self.assertFalse(is_audio_too_short(audio))

    def test_custom_threshold(self):
        """Verify custom threshold works."""
        from main import is_audio_too_short
        
        audio = b'\x00' * 1000
        
        # Too short for default threshold
        self.assertTrue(is_audio_too_short(audio))
        
        # But acceptable for custom threshold
        self.assertFalse(is_audio_too_short(audio, min_bytes=500))


class TestIsWakewordEcho(unittest.TestCase):
    """Tests for is_wakeword_echo() helper."""

    def test_exact_wakeword_is_echo(self):
        """Verify exact wake word phrase is detected as echo."""
        from main import is_wakeword_echo
        
        self.assertTrue(is_wakeword_echo("hey jarvis"))
        self.assertTrue(is_wakeword_echo("jarvis"))
        self.assertTrue(is_wakeword_echo("hey jarvis."))

    def test_case_insensitive(self):
        """Verify detection is case-insensitive."""
        from main import is_wakeword_echo
        
        self.assertTrue(is_wakeword_echo("HEY JARVIS"))
        self.assertTrue(is_wakeword_echo("Hey Jarvis"))
        self.assertTrue(is_wakeword_echo("JARVIS"))

    def test_with_whitespace(self):
        """Verify leading/trailing whitespace is handled."""
        from main import is_wakeword_echo
        
        self.assertTrue(is_wakeword_echo("  hey jarvis  "))
        self.assertTrue(is_wakeword_echo("\tjarvis\n"))

    def test_real_command_not_echo(self):
        """Verify real commands are not treated as echoes."""
        from main import is_wakeword_echo
        
        self.assertFalse(is_wakeword_echo("hey jarvis what is the weather"))
        self.assertFalse(is_wakeword_echo("tell me a joke"))
        self.assertFalse(is_wakeword_echo("what time is it"))

    def test_custom_phrases(self):
        """Verify custom wake word phrases work."""
        from main import is_wakeword_echo
        
        custom_phrases = ["alexa", "hey alexa"]
        
        self.assertTrue(is_wakeword_echo("alexa", wakeword_phrases=custom_phrases))
        self.assertTrue(is_wakeword_echo("hey alexa", wakeword_phrases=custom_phrases))
        self.assertFalse(is_wakeword_echo("hey jarvis", wakeword_phrases=custom_phrases))


class TestIsSelfEcho(unittest.TestCase):
    """Tests for is_self_echo() helper."""

    def test_exact_match_is_echo(self):
        """Verify exact match between user and bot is detected as echo."""
        from main import is_self_echo
        
        bot_response = "The weather is sunny today."
        user_text = "The weather is sunny today."
        
        self.assertTrue(is_self_echo(user_text, bot_response))

    def test_case_insensitive(self):
        """Verify detection is case-insensitive."""
        from main import is_self_echo
        
        bot_response = "Hello there!"
        user_text = "HELLO THERE!"
        
        self.assertTrue(is_self_echo(user_text, bot_response))

    def test_partial_match_user_in_bot(self):
        """Verify partial match is detected (user text in bot response)."""
        from main import is_self_echo
        
        bot_response = "The temperature is 72 degrees Fahrenheit and it's sunny."
        user_text = "72 degrees"
        
        self.assertTrue(is_self_echo(user_text, bot_response))

    def test_partial_match_bot_in_user(self):
        """Verify partial match is detected (bot response in user text)."""
        from main import is_self_echo
        
        bot_response = "sunny"
        user_text = "It's sunny today"
        
        self.assertTrue(is_self_echo(user_text, bot_response))

    def test_different_text_not_echo(self):
        """Verify different text is not treated as echo."""
        from main import is_self_echo
        
        bot_response = "The weather is sunny."
        user_text = "What time is it?"
        
        self.assertFalse(is_self_echo(user_text, bot_response))

    def test_empty_user_text_not_echo(self):
        """Verify empty user text is not an echo."""
        from main import is_self_echo
        
        self.assertFalse(is_self_echo("", "some response"))
        self.assertFalse(is_self_echo("   ", "some response"))

    def test_empty_bot_response_not_echo(self):
        """Verify empty bot response doesn't cause false echo."""
        from main import is_self_echo
        
        self.assertFalse(is_self_echo("user text", ""))
        self.assertFalse(is_self_echo("user text", "   "))

    def test_both_empty_not_echo(self):
        """Verify both empty is not an echo."""
        from main import is_self_echo
        
        self.assertFalse(is_self_echo("", ""))

    def test_with_whitespace(self):
        """Verify whitespace is properly stripped."""
        from main import is_self_echo
        
        bot_response = "  hello world  "
        user_text = "\thello world\n"
        
        self.assertTrue(is_self_echo(user_text, bot_response))


if __name__ == "__main__":
    unittest.main()

