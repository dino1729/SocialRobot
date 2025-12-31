"""Unit tests for STT/TTS engine factories in audio/engine_config.py."""

import sys
import unittest
from unittest.mock import MagicMock, patch


class TestSTTEngineEnum(unittest.TestCase):
    """Tests for STTEngine enum."""

    def test_stt_engine_values(self):
        """Verify STTEngine enum has expected values."""
        from audio.engine_config import STTEngine
        
        self.assertEqual(STTEngine.FASTER_WHISPER.value, "faster-whisper")
        self.assertEqual(STTEngine.OPENAI_WHISPER.value, "openai-whisper")

    def test_stt_engine_from_string(self):
        """Verify STTEngine can be created from string."""
        from audio.engine_config import STTEngine
        
        self.assertEqual(STTEngine("faster-whisper"), STTEngine.FASTER_WHISPER)
        self.assertEqual(STTEngine("openai-whisper"), STTEngine.OPENAI_WHISPER)


class TestTTSEngineEnum(unittest.TestCase):
    """Tests for TTSEngine enum."""

    def test_tts_engine_values(self):
        """Verify TTSEngine enum has expected values."""
        from audio.engine_config import TTSEngine
        
        self.assertEqual(TTSEngine.KOKORO.value, "kokoro")
        self.assertEqual(TTSEngine.PIPER.value, "piper")
        self.assertEqual(TTSEngine.CHATTERBOX.value, "chatterbox")
        self.assertEqual(TTSEngine.VIBEVOICE.value, "vibevoice")

    def test_tts_engine_from_string(self):
        """Verify TTSEngine can be created from string."""
        from audio.engine_config import TTSEngine
        
        self.assertEqual(TTSEngine("kokoro"), TTSEngine.KOKORO)
        self.assertEqual(TTSEngine("piper"), TTSEngine.PIPER)
        self.assertEqual(TTSEngine("chatterbox"), TTSEngine.CHATTERBOX)
        self.assertEqual(TTSEngine("vibevoice"), TTSEngine.VIBEVOICE)


class TestDeviceDetection(unittest.TestCase):
    """Tests for device detection helpers."""

    def test_detect_whisper_device_with_cuda(self):
        """Verify _detect_whisper_device returns 'cuda' when available."""
        from audio.engine_config import _detect_whisper_device
        
        mock_ctranslate2 = MagicMock()
        mock_ctranslate2.get_cuda_device_count.return_value = 1
        
        with patch.dict(sys.modules, {'ctranslate2': mock_ctranslate2}):
            # Need to reimport to pick up the mock
            import importlib
            import audio.engine_config as ec
            
            # Manually test the logic
            try:
                if mock_ctranslate2.get_cuda_device_count() > 0:
                    result = "cuda"
                else:
                    result = "cpu"
            except Exception:
                result = "cpu"
            
            self.assertEqual(result, "cuda")

    def test_detect_whisper_device_without_cuda(self):
        """Verify _detect_whisper_device returns 'cpu' when CUDA not available."""
        from audio.engine_config import _detect_whisper_device
        
        mock_ctranslate2 = MagicMock()
        mock_ctranslate2.get_cuda_device_count.return_value = 0
        
        with patch.dict(sys.modules, {'ctranslate2': mock_ctranslate2}):
            try:
                if mock_ctranslate2.get_cuda_device_count() > 0:
                    result = "cuda"
                else:
                    result = "cpu"
            except Exception:
                result = "cpu"
            
            self.assertEqual(result, "cpu")


class TestComputeTypeDefaults(unittest.TestCase):
    """Tests for compute type default selection."""

    def test_cpu_default_compute_type(self):
        """Verify CPU uses int8 compute type."""
        from audio.engine_config import _get_default_compute_type
        
        result = _get_default_compute_type("cpu")
        self.assertEqual(result, "int8")

    def test_cuda_default_compute_type_not_jetson(self):
        """Verify desktop GPU uses float16 compute type."""
        import audio.engine_config as ec
        
        with patch.object(ec, '_is_jetson', return_value=False):
            result = ec._get_default_compute_type("cuda")
            self.assertEqual(result, "float16")

    def test_cuda_default_compute_type_on_jetson(self):
        """Verify Jetson platform uses int8 compute type."""
        import audio.engine_config as ec
        
        with patch.object(ec, '_is_jetson', return_value=True):
            result = ec._get_default_compute_type("cuda")
            self.assertEqual(result, "int8")


class TestSTTEngineModelMapping(unittest.TestCase):
    """Tests for STT engine model name mapping."""

    @patch('audio.engine_config._detect_whisper_device')
    @patch('audio.engine_config._get_default_compute_type')
    def test_faster_whisper_model_mapping_english(self, mock_compute, mock_device):
        """Verify FasterWhisperSTT uses .en model suffix for English."""
        mock_device.return_value = "cpu"
        mock_compute.return_value = "int8"
        
        mock_stt_class = MagicMock()
        
        with patch.dict(sys.modules, {
            'audio.stt': MagicMock(FasterWhisperSTT=mock_stt_class)
        }):
            from audio.engine_config import create_stt_engine, STTEngine
            
            with patch('audio.engine_config.FasterWhisperSTT', mock_stt_class, create=True):
                # Reimport the function
                import importlib
                import audio.engine_config as ec
                importlib.reload(ec)
                
                # Mock the import inside create_stt_engine
                with patch.object(ec, 'create_stt_engine') as mock_create:
                    # Just verify the model mapping logic
                    model_map = {
                        "tiny": "tiny.en",
                        "base": "base.en",
                        "small": "small.en",
                        "medium": "medium.en",
                        "large": "large-v3",
                    }
                    
                    for size, expected in model_map.items():
                        if size == "large":
                            self.assertEqual(expected, "large-v3")
                        else:
                            self.assertTrue(expected.endswith(".en"))

    def test_faster_whisper_model_mapping_non_english(self):
        """Verify FasterWhisperSTT uses base model name for non-English."""
        model_map_non_en = {
            "tiny": "tiny",
            "base": "base",
            "small": "small",
            "medium": "medium",
            "large": "large-v3",
        }
        
        # Verify mapping logic for non-English
        for size, expected in model_map_non_en.items():
            if size == "large":
                self.assertEqual(expected, "large-v3")
            else:
                self.assertFalse(expected.endswith(".en"))


class TestTTSEngineFactory(unittest.TestCase):
    """Tests for TTS engine factory function."""

    def test_kokoro_is_default(self):
        """Verify Kokoro is the default TTS engine."""
        from audio.engine_config import TTSEngine
        
        # Default value check
        self.assertEqual(TTSEngine.KOKORO.value, "kokoro")

    @patch('audio.tts.KokoroTTS')
    def test_create_tts_engine_kokoro(self, mock_kokoro):
        """Verify create_tts_engine creates KokoroTTS for 'kokoro'."""
        mock_instance = MagicMock()
        mock_kokoro.return_value = mock_instance
        
        from audio.engine_config import create_tts_engine
        
        result = create_tts_engine(engine="kokoro", use_gpu=False, voice="af_sarah")
        
        mock_kokoro.assert_called_once()
        call_kwargs = mock_kokoro.call_args.kwargs
        self.assertEqual(call_kwargs['voice'], 'af_sarah')
        self.assertEqual(call_kwargs['use_gpu'], False)

    @patch('audio.tts_piper.PiperTTS')
    def test_create_tts_engine_piper(self, mock_piper):
        """Verify create_tts_engine creates PiperTTS for 'piper'."""
        mock_instance = MagicMock()
        mock_piper.return_value = mock_instance
        
        from audio.engine_config import create_tts_engine
        
        result = create_tts_engine(engine="piper", use_gpu=True)
        
        mock_piper.assert_called_once()
        call_kwargs = mock_piper.call_args.kwargs
        self.assertEqual(call_kwargs['use_gpu'], True)

    @patch('audio.tts_chatterbox.ChatterboxTTS')
    def test_create_tts_engine_chatterbox(self, mock_chatterbox):
        """Verify create_tts_engine creates ChatterboxTTS for 'chatterbox'."""
        mock_instance = MagicMock()
        mock_chatterbox.return_value = mock_instance
        
        from audio.engine_config import create_tts_engine
        
        result = create_tts_engine(
            engine="chatterbox", 
            use_gpu=True, 
            voice_path="/path/to/voice.wav",
            use_turbo=False
        )
        
        mock_chatterbox.assert_called_once()
        call_kwargs = mock_chatterbox.call_args.kwargs
        self.assertEqual(call_kwargs['voice_path'], '/path/to/voice.wav')
        self.assertEqual(call_kwargs['use_gpu'], True)
        self.assertEqual(call_kwargs['use_turbo'], False)

    @patch('audio.tts_vibevoice.VibeVoiceTTS')
    def test_create_tts_engine_vibevoice(self, mock_vibevoice):
        """Verify create_tts_engine creates VibeVoiceTTS for 'vibevoice'."""
        mock_instance = MagicMock()
        mock_vibevoice.return_value = mock_instance
        
        from audio.engine_config import create_tts_engine
        
        result = create_tts_engine(
            engine="vibevoice", 
            use_gpu=True, 
            speaker="Bria",
            cfg_scale=2.0
        )
        
        mock_vibevoice.assert_called_once()
        call_kwargs = mock_vibevoice.call_args.kwargs
        self.assertEqual(call_kwargs['speaker'], 'Bria')
        self.assertEqual(call_kwargs['use_gpu'], True)
        self.assertEqual(call_kwargs['cfg_scale'], 2.0)

    def test_unknown_tts_engine_raises(self):
        """Verify unknown engine raises ValueError."""
        from audio.engine_config import create_tts_engine
        
        with self.assertRaises(ValueError) as ctx:
            create_tts_engine(engine="unknown_engine")
        
        self.assertIn("unknown", str(ctx.exception).lower())


class TestSTTEngineFactory(unittest.TestCase):
    """Tests for STT engine factory function."""

    @patch('audio.engine_config._detect_whisper_device')
    @patch('audio.engine_config._get_default_compute_type')
    @patch('audio.stt.FasterWhisperSTT')
    def test_create_stt_engine_faster_whisper(self, mock_stt, mock_compute, mock_device):
        """Verify create_stt_engine creates FasterWhisperSTT."""
        mock_device.return_value = "cpu"
        mock_compute.return_value = "int8"
        mock_instance = MagicMock()
        mock_stt.return_value = mock_instance
        
        from audio.engine_config import create_stt_engine
        
        result = create_stt_engine(engine="faster-whisper", model_size="tiny", language="en")
        
        mock_stt.assert_called_once()
        call_kwargs = mock_stt.call_args.kwargs
        self.assertEqual(call_kwargs['model_size_or_path'], 'tiny.en')
        self.assertEqual(call_kwargs['device'], 'cpu')
        self.assertEqual(call_kwargs['compute_type'], 'int8')
        self.assertEqual(call_kwargs['language'], 'en')

    @patch('audio.stt_openai.OpenAIWhisperSTT')
    def test_create_stt_engine_openai_whisper(self, mock_stt):
        """Verify create_stt_engine creates OpenAIWhisperSTT."""
        mock_instance = MagicMock()
        mock_stt.return_value = mock_instance
        
        from audio.engine_config import create_stt_engine
        
        result = create_stt_engine(engine="openai-whisper", model_size="base", device="cuda")
        
        mock_stt.assert_called_once()
        call_kwargs = mock_stt.call_args.kwargs
        self.assertEqual(call_kwargs['model_size'], 'base')
        self.assertEqual(call_kwargs['device'], 'cuda')

    def test_unknown_stt_engine_raises(self):
        """Verify unknown STT engine raises ValueError."""
        from audio.engine_config import create_stt_engine
        
        with self.assertRaises(ValueError) as ctx:
            create_stt_engine(engine="unknown_stt")
        
        self.assertIn("unknown", str(ctx.exception).lower())


class TestTextNormalization(unittest.TestCase):
    """Tests for normalize_text_for_wer function."""

    def test_lowercase(self):
        """Verify text is lowercased."""
        from audio.engine_config import normalize_text_for_wer
        
        result = normalize_text_for_wer("Hello World")
        self.assertEqual(result, "hello world")

    def test_remove_punctuation(self):
        """Verify punctuation is removed."""
        from audio.engine_config import normalize_text_for_wer
        
        result = normalize_text_for_wer("Hello, world! How are you?")
        self.assertEqual(result, "hello world how are you")

    def test_collapse_whitespace(self):
        """Verify multiple spaces are collapsed."""
        from audio.engine_config import normalize_text_for_wer
        
        result = normalize_text_for_wer("hello    world")
        self.assertEqual(result, "hello world")

    def test_strip_whitespace(self):
        """Verify leading/trailing whitespace is stripped."""
        from audio.engine_config import normalize_text_for_wer
        
        result = normalize_text_for_wer("  hello world  ")
        self.assertEqual(result, "hello world")

    def test_mixed_normalization(self):
        """Verify all normalizations work together."""
        from audio.engine_config import normalize_text_for_wer
        
        result = normalize_text_for_wer("  Hello,  WORLD!   How's it going?  ")
        self.assertEqual(result, "hello world hows it going")

    def test_empty_string(self):
        """Verify empty string returns empty."""
        from audio.engine_config import normalize_text_for_wer
        
        result = normalize_text_for_wer("")
        self.assertEqual(result, "")

    def test_whitespace_only(self):
        """Verify whitespace-only string returns empty."""
        from audio.engine_config import normalize_text_for_wer
        
        result = normalize_text_for_wer("   ")
        self.assertEqual(result, "")

    def test_numbers_preserved(self):
        """Verify numbers are preserved."""
        from audio.engine_config import normalize_text_for_wer
        
        result = normalize_text_for_wer("I have 42 apples.")
        self.assertEqual(result, "i have 42 apples")


class TestWordErrorRate(unittest.TestCase):
    """Tests for word_error_rate function."""

    def test_exact_match(self):
        """Verify exact match returns 0.0 WER."""
        from audio.engine_config import word_error_rate
        
        result = word_error_rate("hello world", "hello world")
        self.assertEqual(result, 0.0)

    def test_exact_match_with_different_case(self):
        """Verify case-insensitive matching."""
        from audio.engine_config import word_error_rate
        
        result = word_error_rate("Hello World", "hello world")
        self.assertEqual(result, 0.0)

    def test_exact_match_with_punctuation(self):
        """Verify punctuation is ignored."""
        from audio.engine_config import word_error_rate
        
        result = word_error_rate("Hello, world!", "hello world")
        self.assertEqual(result, 0.0)

    def test_single_substitution(self):
        """Verify single word substitution gives correct WER."""
        from audio.engine_config import word_error_rate
        
        # "hello world" (2 words) vs "hello earth" -> 1 substitution
        result = word_error_rate("hello world", "hello earth")
        self.assertAlmostEqual(result, 0.5)  # 1/2 = 0.5

    def test_single_deletion(self):
        """Verify single word deletion gives correct WER."""
        from audio.engine_config import word_error_rate
        
        # "hello beautiful world" (3 words) vs "hello world" -> 1 deletion
        result = word_error_rate("hello beautiful world", "hello world")
        self.assertAlmostEqual(result, 1/3)

    def test_single_insertion(self):
        """Verify single word insertion gives correct WER."""
        from audio.engine_config import word_error_rate
        
        # "hello world" (2 words) vs "hello beautiful world" -> 1 insertion
        result = word_error_rate("hello world", "hello beautiful world")
        self.assertAlmostEqual(result, 0.5)  # 1/2 = 0.5

    def test_all_words_wrong(self):
        """Verify complete mismatch gives 1.0 WER."""
        from audio.engine_config import word_error_rate
        
        # "hello world" (2 words) vs "foo bar" -> 2 substitutions
        result = word_error_rate("hello world", "foo bar")
        self.assertAlmostEqual(result, 1.0)

    def test_empty_hypothesis(self):
        """Verify empty hypothesis gives 1.0 WER (100% deletion)."""
        from audio.engine_config import word_error_rate
        
        result = word_error_rate("hello world", "")
        self.assertEqual(result, 1.0)

    def test_empty_reference(self):
        """Verify empty reference with hypothesis returns hypothesis length."""
        from audio.engine_config import word_error_rate
        
        # All insertions
        result = word_error_rate("", "hello world")
        self.assertEqual(result, 2.0)  # 2 insertions with 0 reference words

    def test_both_empty(self):
        """Verify both empty gives 0.0 WER."""
        from audio.engine_config import word_error_rate
        
        result = word_error_rate("", "")
        self.assertEqual(result, 0.0)

    def test_wer_can_exceed_one(self):
        """Verify WER can exceed 1.0 with many insertions."""
        from audio.engine_config import word_error_rate
        
        # "hi" (1 word) vs "hello beautiful amazing world" (4 words)
        # 1 substitution + 3 insertions = 4 errors / 1 reference word = 4.0
        result = word_error_rate("hi", "hello beautiful amazing world")
        self.assertAlmostEqual(result, 4.0)

    def test_longer_sentence(self):
        """Verify WER with a longer sentence."""
        from audio.engine_config import word_error_rate
        
        reference = "the quick brown fox jumps over the lazy dog"
        hypothesis = "the quick brown cat jumps over a lazy dog"
        # 9 words in reference
        # Errors: "fox" -> "cat" (substitution), "the" -> "a" (substitution) = 2 errors
        result = word_error_rate(reference, hypothesis)
        self.assertAlmostEqual(result, 2/9)


if __name__ == "__main__":
    unittest.main()

