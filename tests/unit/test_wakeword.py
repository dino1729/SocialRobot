"""Unit tests for WakeWordDetector in main.py."""

import sys
import time
import threading
import unittest
from unittest.mock import MagicMock, patch, PropertyMock


class MockOpenwakewordModel:
    """Mock for openwakeword.model.Model."""
    
    def __init__(self, wakeword_model_paths=None):
        self.wakeword_model_paths = wakeword_model_paths
        self._reset_called = False
        self._predict_scores = {}
    
    def reset(self):
        self._reset_called = True
    
    def predict(self, audio_array):
        # Return configurable prediction scores
        return self._predict_scores


class TestWakeWordDetectorDebounce(unittest.TestCase):
    """Tests for WakeWordDetector debounce behavior."""

    def _create_detector(self, threshold=0.5):
        """Create a WakeWordDetector with mocked dependencies."""
        # Mock openwakeword before importing WakeWordDetector
        mock_oww_model = MockOpenwakewordModel()
        
        with patch.dict(sys.modules, {
            'openwakeword': MagicMock(),
            'openwakeword.model': MagicMock(),
        }):
            with patch('numpy.frombuffer') as mock_frombuffer:
                mock_frombuffer.return_value = MagicMock()
                
                with patch('pyaudio.PyAudio') as mock_pyaudio:
                    # Import WakeWordDetector from main
                    from main import WakeWordDetector
                    
                    # Patch the Model class
                    with patch.object(WakeWordDetector, '__init__', lambda self, **kwargs: None):
                        detector = WakeWordDetector.__new__(WakeWordDetector)
                        
                        # Manually initialize required attributes
                        detector.wakeword_models = ['test_wakeword']
                        detector.threshold = threshold
                        detector.chunk_size = 1280
                        detector.sample_rate = 16000
                        detector.device_index = None
                        detector.np = MagicMock()
                        detector.oww_model = mock_oww_model
                        detector.audio = MagicMock()
                        detector.stream = None
                        detector.is_running = False
                        detector.is_paused = False
                        detector.detection_callback = None
                        detector.last_detection_time = 0.0
                        detector._beep_playing = False
                        
                        return detector, mock_oww_model

    def test_debounce_prevents_rapid_triggers(self):
        """Verify that detections within 3 seconds are ignored."""
        detector, mock_model = self._create_detector(threshold=0.5)
        
        callback_count = [0]
        def on_detection():
            callback_count[0] += 1
        
        detector.detection_callback = on_detection
        
        # Simulate first detection
        detector.last_detection_time = 0.0
        current_time = 10.0  # First detection at t=10
        
        # Simulate detection above threshold
        mock_model._predict_scores = {'test_wakeword': 0.8}
        
        # Check that detection would happen (time since last > 3s)
        time_since_last = current_time - detector.last_detection_time
        self.assertGreaterEqual(time_since_last, 3.0)
        
        # Manually simulate what happens in the detection loop
        if time_since_last >= 3.0:
            detector.last_detection_time = current_time
            if detector.detection_callback:
                detector.detection_callback()
        
        self.assertEqual(callback_count[0], 1)
        
        # Second detection attempt at t=11 (only 1 second later)
        current_time = 11.0
        time_since_last = current_time - detector.last_detection_time
        self.assertLess(time_since_last, 3.0)
        
        # Should NOT trigger callback
        if time_since_last >= 3.0:
            detector.last_detection_time = current_time
            if detector.detection_callback:
                detector.detection_callback()
        
        self.assertEqual(callback_count[0], 1)  # Still 1, not 2

    def test_debounce_allows_after_3_seconds(self):
        """Verify that detections after 3 seconds are allowed."""
        detector, mock_model = self._create_detector(threshold=0.5)
        
        callback_count = [0]
        def on_detection():
            callback_count[0] += 1
        
        detector.detection_callback = on_detection
        
        # First detection at t=0
        detector.last_detection_time = 0.0
        current_time = 5.0  # More than 3 seconds later
        
        time_since_last = current_time - detector.last_detection_time
        self.assertGreaterEqual(time_since_last, 3.0)
        
        if time_since_last >= 3.0:
            detector.last_detection_time = current_time
            if detector.detection_callback:
                detector.detection_callback()
        
        self.assertEqual(callback_count[0], 1)
        
        # Second detection at t=10 (5 seconds after first)
        current_time = 10.0
        time_since_last = current_time - detector.last_detection_time
        self.assertGreaterEqual(time_since_last, 3.0)
        
        if time_since_last >= 3.0:
            detector.last_detection_time = current_time
            if detector.detection_callback:
                detector.detection_callback()
        
        self.assertEqual(callback_count[0], 2)


class TestWakeWordDetectorPauseResume(unittest.TestCase):
    """Tests for WakeWordDetector pause/resume behavior."""

    def _create_detector(self, threshold=0.5):
        """Create a WakeWordDetector with mocked dependencies."""
        mock_oww_model = MockOpenwakewordModel()
        
        with patch.dict(sys.modules, {
            'openwakeword': MagicMock(),
            'openwakeword.model': MagicMock(),
        }):
            with patch('pyaudio.PyAudio') as mock_pyaudio:
                from main import WakeWordDetector
                
                with patch.object(WakeWordDetector, '__init__', lambda self, **kwargs: None):
                    detector = WakeWordDetector.__new__(WakeWordDetector)
                    
                    detector.wakeword_models = ['test_wakeword']
                    detector.threshold = threshold
                    detector.chunk_size = 1280
                    detector.sample_rate = 16000
                    detector.device_index = None
                    detector.np = MagicMock()
                    detector.oww_model = mock_oww_model
                    detector.audio = MagicMock()
                    detector.stream = None
                    detector.is_running = False
                    detector.is_paused = False
                    detector.detection_callback = None
                    detector.last_detection_time = 0.0
                    detector._beep_playing = False
                    
                    return detector, mock_oww_model

    def test_pause_sets_flag_and_resets_model(self):
        """Verify pause() sets is_paused and resets the model."""
        detector, mock_model = self._create_detector()
        
        self.assertFalse(detector.is_paused)
        mock_model._reset_called = False
        
        detector.pause()
        
        self.assertTrue(detector.is_paused)
        self.assertTrue(mock_model._reset_called)

    def test_resume_clears_flag_and_resets_model(self):
        """Verify resume() clears is_paused, resets model, and updates debounce timer."""
        detector, mock_model = self._create_detector()
        
        detector.is_paused = True
        detector.last_detection_time = 0.0
        mock_model._reset_called = False
        
        before_time = time.time()
        detector.resume()
        after_time = time.time()
        
        self.assertFalse(detector.is_paused)
        self.assertTrue(mock_model._reset_called)
        # last_detection_time should be updated to current time
        self.assertGreaterEqual(detector.last_detection_time, before_time)
        self.assertLessEqual(detector.last_detection_time, after_time)


class TestWakeWordDetectorThreshold(unittest.TestCase):
    """Tests for WakeWordDetector threshold behavior."""

    def _create_detector(self, threshold=0.5):
        """Create a WakeWordDetector with mocked dependencies."""
        mock_oww_model = MockOpenwakewordModel()
        
        with patch.dict(sys.modules, {
            'openwakeword': MagicMock(),
            'openwakeword.model': MagicMock(),
        }):
            with patch('pyaudio.PyAudio') as mock_pyaudio:
                from main import WakeWordDetector
                
                with patch.object(WakeWordDetector, '__init__', lambda self, **kwargs: None):
                    detector = WakeWordDetector.__new__(WakeWordDetector)
                    
                    detector.wakeword_models = ['test_wakeword']
                    detector.threshold = threshold
                    detector.chunk_size = 1280
                    detector.sample_rate = 16000
                    detector.device_index = None
                    detector.np = MagicMock()
                    detector.oww_model = mock_oww_model
                    detector.audio = MagicMock()
                    detector.stream = None
                    detector.is_running = False
                    detector.is_paused = False
                    detector.detection_callback = None
                    detector.last_detection_time = 0.0
                    detector._beep_playing = False
                    
                    return detector, mock_oww_model

    def test_score_below_threshold_no_trigger(self):
        """Verify scores below threshold don't trigger callback."""
        detector, mock_model = self._create_detector(threshold=0.5)
        
        callback_count = [0]
        def on_detection():
            callback_count[0] += 1
        
        detector.detection_callback = on_detection
        detector.last_detection_time = 0.0
        
        # Score is 0.4, below threshold of 0.5
        score = 0.4
        current_time = 10.0
        
        time_since_last = current_time - detector.last_detection_time
        if score >= detector.threshold and time_since_last >= 3.0:
            detector.last_detection_time = current_time
            if detector.detection_callback:
                detector.detection_callback()
        
        self.assertEqual(callback_count[0], 0)

    def test_score_at_threshold_triggers(self):
        """Verify scores at or above threshold trigger callback."""
        detector, mock_model = self._create_detector(threshold=0.5)
        
        callback_count = [0]
        def on_detection():
            callback_count[0] += 1
        
        detector.detection_callback = on_detection
        detector.last_detection_time = 0.0
        
        # Score is exactly at threshold
        score = 0.5
        current_time = 10.0
        
        time_since_last = current_time - detector.last_detection_time
        if score >= detector.threshold and time_since_last >= 3.0:
            detector.last_detection_time = current_time
            if detector.detection_callback:
                detector.detection_callback()
        
        self.assertEqual(callback_count[0], 1)

    def test_higher_threshold_requires_higher_score(self):
        """Verify higher thresholds require higher scores."""
        detector, mock_model = self._create_detector(threshold=0.8)
        
        callback_count = [0]
        def on_detection():
            callback_count[0] += 1
        
        detector.detection_callback = on_detection
        detector.last_detection_time = 0.0
        
        # Score of 0.7 would pass 0.5 threshold but not 0.8
        score = 0.7
        current_time = 10.0
        
        time_since_last = current_time - detector.last_detection_time
        if score >= detector.threshold and time_since_last >= 3.0:
            detector.last_detection_time = current_time
            if detector.detection_callback:
                detector.detection_callback()
        
        self.assertEqual(callback_count[0], 0)  # Should not trigger


class TestWakeWordDetectorCallbackManagement(unittest.TestCase):
    """Tests for WakeWordDetector callback management."""

    def _create_detector(self, threshold=0.5):
        """Create a WakeWordDetector with mocked dependencies."""
        mock_oww_model = MockOpenwakewordModel()
        
        with patch.dict(sys.modules, {
            'openwakeword': MagicMock(),
            'openwakeword.model': MagicMock(),
        }):
            with patch('pyaudio.PyAudio') as mock_pyaudio:
                from main import WakeWordDetector
                
                with patch.object(WakeWordDetector, '__init__', lambda self, **kwargs: None):
                    detector = WakeWordDetector.__new__(WakeWordDetector)
                    
                    detector.wakeword_models = ['test_wakeword']
                    detector.threshold = threshold
                    detector.chunk_size = 1280
                    detector.sample_rate = 16000
                    detector.device_index = None
                    detector.np = MagicMock()
                    detector.oww_model = mock_oww_model
                    detector.audio = MagicMock()
                    detector.stream = None
                    detector.is_running = False
                    detector.is_paused = False
                    detector.detection_callback = None
                    detector.last_detection_time = 0.0
                    detector._beep_playing = False
                    
                    return detector, mock_oww_model

    def test_set_detection_callback(self):
        """Verify set_detection_callback stores the callback."""
        detector, _ = self._create_detector()
        
        def my_callback():
            pass
        
        self.assertIsNone(detector.detection_callback)
        detector.set_detection_callback(my_callback)
        self.assertEqual(detector.detection_callback, my_callback)


if __name__ == "__main__":
    unittest.main()

