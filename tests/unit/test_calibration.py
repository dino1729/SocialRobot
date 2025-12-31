"""Unit tests for startup noise calibration helpers.

These tests define the expected behavior of calibration functions before 
implementation (TDD approach).
"""

import math
import unittest
from unittest.mock import MagicMock, patch


class TestComputeNoiseRMS(unittest.TestCase):
    """Tests for compute_noise_rms() helper."""

    def test_silence_returns_zero(self):
        """Verify silence (all zeros) returns 0 RMS."""
        from main import compute_noise_rms
        
        silent_samples = bytes(1024)  # 512 samples of silence (int16)
        result = compute_noise_rms(silent_samples)
        
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_known_signal_rms(self):
        """Verify RMS calculation with known signal."""
        from main import compute_noise_rms
        import struct
        
        # Create a constant amplitude signal (all samples = 1000)
        amplitude = 1000
        samples = struct.pack('<' + 'h' * 100, *([amplitude] * 100))
        
        result = compute_noise_rms(samples)
        
        # RMS of constant signal = abs(amplitude)
        self.assertAlmostEqual(result, float(amplitude), places=1)

    def test_empty_input_returns_zero(self):
        """Verify empty input returns 0."""
        from main import compute_noise_rms
        
        result = compute_noise_rms(b'')
        self.assertAlmostEqual(result, 0.0, places=5)


class TestComputeNoiseDdBFS(unittest.TestCase):
    """Tests for compute_noise_dbfs() helper."""

    def test_silence_returns_negative_infinity(self):
        """Verify silence returns very low dBFS."""
        from main import compute_noise_dbfs
        
        silent_samples = bytes(1024)
        result = compute_noise_dbfs(silent_samples)
        
        # Should be negative infinity or very large negative number
        self.assertLess(result, -60)

    def test_full_scale_returns_zero(self):
        """Verify full-scale signal returns 0 dBFS."""
        from main import compute_noise_dbfs
        import struct
        
        # Full scale for int16 is 32767
        full_scale = 32767
        samples = struct.pack('<' + 'h' * 100, *([full_scale] * 100))
        
        result = compute_noise_dbfs(samples)
        
        # Should be close to 0 dBFS
        self.assertAlmostEqual(result, 0.0, delta=0.1)


class TestRecommendedVADAggressiveness(unittest.TestCase):
    """Tests for compute_recommended_vad_aggressiveness() helper."""

    def test_quiet_environment_low_aggressiveness(self):
        """Verify quiet environments use lower aggressiveness (more sensitive)."""
        from main import compute_recommended_vad_aggressiveness
        
        # -50 dBFS is very quiet
        result = compute_recommended_vad_aggressiveness(noise_dbfs=-50)
        
        # Should be 1 or 2 (more sensitive)
        self.assertIn(result, [1, 2])

    def test_moderate_noise_medium_aggressiveness(self):
        """Verify moderate noise uses medium aggressiveness."""
        from main import compute_recommended_vad_aggressiveness
        
        # -30 dBFS is moderate ambient noise (office, AC)
        result = compute_recommended_vad_aggressiveness(noise_dbfs=-30)
        
        # Should be 2
        self.assertEqual(result, 2)

    def test_loud_environment_high_aggressiveness(self):
        """Verify loud environments use higher aggressiveness (stricter)."""
        from main import compute_recommended_vad_aggressiveness
        
        # -20 dBFS is fairly loud (near a fan, busy room)
        result = compute_recommended_vad_aggressiveness(noise_dbfs=-20)
        
        # Should be 3 (most strict, filters out more noise)
        self.assertEqual(result, 3)

    def test_aggressiveness_range(self):
        """Verify result is always in valid range 0-3."""
        from main import compute_recommended_vad_aggressiveness
        
        for dbfs in range(-80, 0, 5):
            result = compute_recommended_vad_aggressiveness(noise_dbfs=dbfs)
            self.assertIn(result, [0, 1, 2, 3])


class TestWakewordThresholdBoost(unittest.TestCase):
    """Tests for compute_wakeword_threshold_boost() helper."""

    def test_quiet_environment_no_boost(self):
        """Verify quiet environments don't boost threshold."""
        from main import compute_wakeword_threshold_boost
        
        # Very quiet environment
        result = compute_wakeword_threshold_boost(noise_dbfs=-50)
        
        # No boost needed
        self.assertAlmostEqual(result, 0.0, delta=0.05)

    def test_moderate_noise_small_boost(self):
        """Verify moderate noise gives small threshold boost."""
        from main import compute_wakeword_threshold_boost
        
        # Moderate noise
        result = compute_wakeword_threshold_boost(noise_dbfs=-30)
        
        # Should give a small boost (0.05-0.15)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 0.2)

    def test_loud_environment_larger_boost(self):
        """Verify loud environments give larger threshold boost."""
        from main import compute_wakeword_threshold_boost
        
        # Loud environment
        result = compute_wakeword_threshold_boost(noise_dbfs=-15)
        
        # Should give a larger boost to reduce false positives
        self.assertGreater(result, 0.1)
        self.assertLessEqual(result, 0.3)

    def test_boost_never_negative(self):
        """Verify boost is never negative (never makes threshold more sensitive)."""
        from main import compute_wakeword_threshold_boost
        
        for dbfs in range(-80, 0, 5):
            result = compute_wakeword_threshold_boost(noise_dbfs=dbfs)
            self.assertGreaterEqual(result, 0.0)

    def test_boost_capped(self):
        """Verify boost doesn't exceed reasonable maximum."""
        from main import compute_wakeword_threshold_boost
        
        # Even very loud environment
        result = compute_wakeword_threshold_boost(noise_dbfs=-5)
        
        # Should be capped at reasonable value (0.3 max)
        self.assertLessEqual(result, 0.3)


class TestCalibrateAudioParameters(unittest.TestCase):
    """Tests for calibrate_audio_parameters() main function."""

    @patch('pyaudio.PyAudio')
    def test_returns_calibration_result_dict(self, mock_pyaudio):
        """Verify function returns expected result structure."""
        from main import calibrate_audio_parameters
        
        # Setup mock audio stream that returns silence
        mock_stream = MagicMock()
        mock_stream.read.return_value = bytes(3200)  # 100ms of silence at 16kHz
        
        mock_pa = MagicMock()
        mock_pa.open.return_value = mock_stream
        mock_pyaudio.return_value = mock_pa
        
        result = calibrate_audio_parameters(
            sample_seconds=0.5,
            base_wakeword_threshold=0.5
        )
        
        # Should return dict with expected keys
        self.assertIsInstance(result, dict)
        self.assertIn('vad_aggressiveness', result)
        self.assertIn('wakeword_threshold', result)
        self.assertIn('noise_dbfs', result)
        self.assertIn('success', result)

    @patch('pyaudio.PyAudio')
    def test_returns_defaults_on_failure(self, mock_pyaudio):
        """Verify function returns defaults when audio capture fails."""
        from main import calibrate_audio_parameters
        
        # Simulate audio capture failure
        mock_pyaudio.side_effect = Exception("No audio device")
        
        result = calibrate_audio_parameters(
            sample_seconds=0.5,
            base_wakeword_threshold=0.5
        )
        
        # Should return defaults
        self.assertFalse(result['success'])
        # Should have valid default values
        self.assertIn(result['vad_aggressiveness'], [0, 1, 2, 3])
        self.assertGreater(result['wakeword_threshold'], 0)

    @patch('pyaudio.PyAudio')
    def test_wakeword_threshold_never_lower_than_base(self, mock_pyaudio):
        """Verify wakeword threshold is never lower than configured base."""
        from main import calibrate_audio_parameters
        
        mock_stream = MagicMock()
        mock_stream.read.return_value = bytes(3200)
        
        mock_pa = MagicMock()
        mock_pa.open.return_value = mock_stream
        mock_pyaudio.return_value = mock_pa
        
        base_threshold = 0.6
        result = calibrate_audio_parameters(
            sample_seconds=0.5,
            base_wakeword_threshold=base_threshold
        )
        
        # Threshold should be >= base (boost is only ever positive)
        self.assertGreaterEqual(result['wakeword_threshold'], base_threshold)


class TestCalibrationIntegration(unittest.TestCase):
    """Integration tests for calibration with realistic scenarios."""

    def test_calibration_helpers_compatible(self):
        """Verify all calibration helpers work together."""
        from main import (
            compute_noise_rms,
            compute_noise_dbfs,
            compute_recommended_vad_aggressiveness,
            compute_wakeword_threshold_boost,
        )
        import struct
        
        # Create test signal with moderate amplitude
        amplitude = 500  # Low-moderate signal
        samples = struct.pack('<' + 'h' * 100, *([amplitude] * 100))
        
        # Compute RMS
        rms = compute_noise_rms(samples)
        self.assertGreater(rms, 0)
        
        # Compute dBFS
        dbfs = compute_noise_dbfs(samples)
        self.assertLess(dbfs, 0)  # Should be negative
        
        # Get VAD recommendation
        vad_agg = compute_recommended_vad_aggressiveness(dbfs)
        self.assertIn(vad_agg, [0, 1, 2, 3])
        
        # Get wakeword boost
        ww_boost = compute_wakeword_threshold_boost(dbfs)
        self.assertGreaterEqual(ww_boost, 0)


if __name__ == "__main__":
    unittest.main()

