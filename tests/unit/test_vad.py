"""Unit tests for VAD (Voice Activity Detection) module."""

import threading
import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock


class TestVADConfig(unittest.TestCase):
    """Tests for VADConfig dataclass."""

    def test_default_values(self):
        """Verify VADConfig has sensible defaults."""
        from audio.vad import VADConfig
        
        config = VADConfig()
        
        self.assertEqual(config.sample_rate, 16000)
        self.assertEqual(config.frame_duration_ms, 30)
        self.assertEqual(config.padding_duration_ms, 300)
        self.assertAlmostEqual(config.activation_ratio, 0.6)
        self.assertAlmostEqual(config.deactivation_ratio, 0.85)
        self.assertEqual(config.aggressiveness, 2)

    def test_custom_values(self):
        """Verify VADConfig accepts custom values."""
        from audio.vad import VADConfig
        
        config = VADConfig(
            sample_rate=8000,
            frame_duration_ms=20,
            padding_duration_ms=200,
            activation_ratio=0.5,
            deactivation_ratio=0.9,
            aggressiveness=3,
        )
        
        self.assertEqual(config.sample_rate, 8000)
        self.assertEqual(config.frame_duration_ms, 20)
        self.assertEqual(config.padding_duration_ms, 200)
        self.assertAlmostEqual(config.activation_ratio, 0.5)
        self.assertAlmostEqual(config.deactivation_ratio, 0.9)
        self.assertEqual(config.aggressiveness, 3)


class TestVADListenerInit(unittest.TestCase):
    """Tests for VADListener initialization."""

    @patch('audio.vad.pyaudio.PyAudio')
    @patch('audio.vad.webrtcvad.Vad')
    def test_initialization_computes_frame_params(self, mock_vad_class, mock_pyaudio_class):
        """Verify VADListener computes frame parameters correctly."""
        from audio.vad import VADConfig, VADListener
        
        config = VADConfig(
            sample_rate=16000,
            frame_duration_ms=30,
            padding_duration_ms=300,
        )
        
        listener = VADListener(config=config)
        
        # frame_size = sample_rate * frame_duration_ms / 1000
        # = 16000 * 30 / 1000 = 480
        self.assertEqual(listener.frame_size, 480)
        
        # padding_frames = padding_duration_ms / frame_duration_ms
        # = 300 / 30 = 10
        self.assertEqual(listener.padding_frames, 10)
        
        # activation_count = padding_frames * activation_ratio
        # = 10 * 0.6 = 6
        self.assertEqual(listener.activation_count, 6)
        
        # deactivation_count = padding_frames * deactivation_ratio
        # = 10 * 0.85 = 8 (rounded down)
        self.assertEqual(listener.deactivation_count, 8)

    @patch('audio.vad.pyaudio.PyAudio')
    @patch('audio.vad.webrtcvad.Vad')
    def test_vad_initialized_with_aggressiveness(self, mock_vad_class, mock_pyaudio_class):
        """Verify webrtcvad.Vad is initialized with configured aggressiveness."""
        from audio.vad import VADConfig, VADListener
        
        config = VADConfig(aggressiveness=3)
        listener = VADListener(config=config)
        
        mock_vad_class.assert_called_once_with(3)


class TestVADListenerEnableDisable(unittest.TestCase):
    """Tests for VADListener enable/disable behavior."""

    @patch('audio.vad.pyaudio.PyAudio')
    @patch('audio.vad.webrtcvad.Vad')
    def test_vad_enabled_by_default(self, mock_vad_class, mock_pyaudio_class):
        """Verify VAD is enabled by default after initialization."""
        from audio.vad import VADConfig, VADListener
        
        listener = VADListener(config=VADConfig())
        
        self.assertTrue(listener._vad_enabled.is_set())

    @patch('audio.vad.pyaudio.PyAudio')
    @patch('audio.vad.webrtcvad.Vad')
    def test_disable_vad_clears_flag(self, mock_vad_class, mock_pyaudio_class):
        """Verify disable_vad() clears the enabled flag."""
        from audio.vad import VADConfig, VADListener
        
        listener = VADListener(config=VADConfig())
        listener._stream = MagicMock()
        listener._stream.is_active.return_value = True
        
        listener.disable_vad()
        
        self.assertFalse(listener._vad_enabled.is_set())
        self.assertTrue(listener._reset_buffers)
        self.assertEqual(listener._frames_to_skip, 0)

    @patch('audio.vad.pyaudio.PyAudio')
    @patch('audio.vad.webrtcvad.Vad')
    def test_enable_vad_sets_flag(self, mock_vad_class, mock_pyaudio_class):
        """Verify enable_vad() sets the enabled flag and configures skip frames."""
        from audio.vad import VADConfig, VADListener
        
        listener = VADListener(config=VADConfig())
        listener._stream = MagicMock()
        listener._stream.is_active.return_value = False
        listener._vad_enabled.clear()
        
        listener.enable_vad()
        
        self.assertTrue(listener._vad_enabled.is_set())
        self.assertTrue(listener._reset_buffers)
        # Should skip some frames to avoid processing stale audio
        self.assertGreater(listener._frames_to_skip, 0)

    @patch('audio.vad.pyaudio.PyAudio')
    @patch('audio.vad.webrtcvad.Vad')
    def test_disable_stops_stream(self, mock_vad_class, mock_pyaudio_class):
        """Verify disable_vad() stops the audio stream if active."""
        from audio.vad import VADConfig, VADListener
        
        mock_stream = MagicMock()
        mock_stream.is_active.return_value = True
        
        listener = VADListener(config=VADConfig())
        listener._stream = mock_stream
        
        listener.disable_vad()
        
        mock_stream.stop_stream.assert_called_once()

    @patch('audio.vad.pyaudio.PyAudio')
    @patch('audio.vad.webrtcvad.Vad')
    def test_enable_starts_stream(self, mock_vad_class, mock_pyaudio_class):
        """Verify enable_vad() starts the audio stream if not active."""
        from audio.vad import VADConfig, VADListener
        
        mock_stream = MagicMock()
        mock_stream.is_active.return_value = False
        
        listener = VADListener(config=VADConfig())
        listener._stream = mock_stream
        listener._vad_enabled.clear()
        
        listener.enable_vad()
        
        mock_stream.start_stream.assert_called_once()


class TestVADListenerSpeechDetection(unittest.TestCase):
    """Tests for VADListener speech segment detection."""

    @patch('audio.vad.pyaudio.PyAudio')
    @patch('audio.vad.webrtcvad.Vad')
    def test_speech_callback_invoked_on_segment(self, mock_vad_class, mock_pyaudio_class):
        """Verify callback is invoked when a speech segment is detected."""
        from audio.vad import VADConfig, VADListener
        
        # Create config with short padding for faster test
        config = VADConfig(
            sample_rate=16000,
            frame_duration_ms=30,
            padding_duration_ms=90,  # 3 frames
            activation_ratio=0.5,    # need 1-2 voiced frames to activate
            deactivation_ratio=0.5,  # need 1-2 unvoiced to deactivate
            aggressiveness=2,
        )
        
        # Track callback invocations
        callback_data = []
        def on_speech(audio_bytes):
            callback_data.append(audio_bytes)
        
        # Create mock VAD that returns True for first 5 frames, then False
        mock_vad_instance = MagicMock()
        frame_count = [0]
        def is_speech_pattern(frame, sample_rate):
            frame_count[0] += 1
            # Frames 1-5: speech, frames 6+: silence
            return frame_count[0] <= 5
        mock_vad_instance.is_speech.side_effect = is_speech_pattern
        mock_vad_class.return_value = mock_vad_instance
        
        # Create mock stream that returns dummy frames then stops
        mock_stream = MagicMock()
        frame_size = int(16000 * 30 / 1000)  # 480 samples
        dummy_frame = b'\x00' * (frame_size * 2)  # 16-bit audio = 2 bytes per sample
        
        read_count = [0]
        def stream_read(num_samples, exception_on_overflow=False):
            read_count[0] += 1
            if read_count[0] > 15:
                # Stop after enough frames
                listener.stop()
            return dummy_frame
        mock_stream.read.side_effect = stream_read
        mock_stream.is_active.return_value = True
        
        mock_pa_instance = MagicMock()
        mock_pa_instance.open.return_value = mock_stream
        mock_pyaudio_class.return_value = mock_pa_instance
        
        # Create listener and run briefly
        listener = VADListener(config=config, on_speech_callback=on_speech)
        
        # Run in thread so we can stop it
        def run_listener():
            listener.start()
        
        thread = threading.Thread(target=run_listener, daemon=True)
        thread.start()
        thread.join(timeout=2.0)
        
        # Should have detected at least one speech segment
        self.assertGreater(len(callback_data), 0, "Expected at least one speech callback")
        # The detected segment should contain audio bytes
        self.assertIsInstance(callback_data[0], bytes)
        self.assertGreater(len(callback_data[0]), 0)

    @patch('audio.vad.pyaudio.PyAudio')
    @patch('audio.vad.webrtcvad.Vad')
    def test_stop_terminates_listener(self, mock_vad_class, mock_pyaudio_class):
        """Verify stop() terminates the listening loop."""
        from audio.vad import VADConfig, VADListener
        
        mock_stream = MagicMock()
        mock_stream.read.return_value = b'\x00' * 960
        mock_stream.is_active.return_value = True
        
        mock_pa_instance = MagicMock()
        mock_pa_instance.open.return_value = mock_stream
        mock_pyaudio_class.return_value = mock_pa_instance
        
        mock_vad_instance = MagicMock()
        mock_vad_instance.is_speech.return_value = False
        mock_vad_class.return_value = mock_vad_instance
        
        listener = VADListener(config=VADConfig())
        
        # Start in background thread
        thread = threading.Thread(target=listener.start, daemon=True)
        thread.start()
        
        # Give it a moment to start
        time.sleep(0.1)
        
        # Stop should terminate the loop
        listener.stop()
        thread.join(timeout=1.0)
        
        self.assertFalse(thread.is_alive(), "Listener thread should have terminated")


class TestVADListenerBufferReset(unittest.TestCase):
    """Tests for VADListener buffer reset behavior."""

    @patch('audio.vad.pyaudio.PyAudio')
    @patch('audio.vad.webrtcvad.Vad')
    def test_reset_buffers_flag_behavior(self, mock_vad_class, mock_pyaudio_class):
        """Verify _reset_buffers flag is set correctly by enable/disable."""
        from audio.vad import VADConfig, VADListener
        
        mock_stream = MagicMock()
        mock_stream.is_active.return_value = True
        
        listener = VADListener(config=VADConfig())
        listener._stream = mock_stream
        
        # Initially, reset_buffers should be False
        self.assertFalse(listener._reset_buffers)
        
        # After disable, reset_buffers should be True
        listener.disable_vad()
        self.assertTrue(listener._reset_buffers)
        
        # After enable, reset_buffers should still be True (to clear on next frame)
        listener._reset_buffers = False
        mock_stream.is_active.return_value = False
        listener.enable_vad()
        self.assertTrue(listener._reset_buffers)


if __name__ == "__main__":
    unittest.main()

