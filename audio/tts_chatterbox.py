"""Text-to-speech using Chatterbox TTS with zero-shot voice cloning."""

from __future__ import annotations

import contextlib
import io
import logging
import os
import re
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from audio.suppress_warnings import get_pyaudio, suppress_stderr
pyaudio = get_pyaudio()

logger = logging.getLogger('socialrobot')


@contextlib.contextmanager
def suppress_chatterbox_logs():
    """Suppress verbose Chatterbox/torch logs and progress bars."""
    # Suppress warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        # Suppress root logger warnings
        root_logger = logging.getLogger()
        old_level = root_logger.level
        root_logger.setLevel(logging.ERROR)
        
        # Suppress specific loggers
        loggers_to_suppress = [
            'chatterbox', 'diffusers', 'transformers', 'torch',
            'torchaudio', 'huggingface_hub', 'filelock'
        ]
        old_levels = {}
        for name in loggers_to_suppress:
            logger = logging.getLogger(name)
            old_levels[name] = logger.level
            logger.setLevel(logging.ERROR)
        
        # Disable tqdm progress bars globally
        try:
            import tqdm
            old_tqdm_disable = getattr(tqdm.tqdm, '__init__', None)
            tqdm.tqdm.__init__.__globals__['disable'] = True
        except:
            pass
        
        # Capture stdout/stderr to suppress Chatterbox prints
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        try:
            yield
        finally:
            # Restore stdout/stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
            # Restore loggers
            root_logger.setLevel(old_level)
            for name, level in old_levels.items():
                logging.getLogger(name).setLevel(level)


class ChatterboxTTS:
    """Generate audio using Chatterbox TTS with zero-shot voice cloning.

    Chatterbox TTS enables high-quality voice cloning from a short audio sample.
    Simply provide a WAV file of the target voice and it will synthesize speech
    in that voice style.

    The turbo model (350M params) uses a one-step mel decoder for faster inference,
    making it ideal for CPU usage and real-time voice agents.
    """

    def __init__(
        self,
        voice_path: Optional[str] = None,
        device: str = "cpu",
        use_gpu: bool = False,
        use_turbo: bool = True,
    ) -> None:
        """Initialize Chatterbox TTS.

        Args:
            voice_path: Path to voice reference WAV file for zero-shot cloning.
                       If None, uses Chatterbox's default voice.
            device: Device to use ('cpu' or 'cuda'). Overridden by use_gpu if True.
            use_gpu: If True, forces CUDA device. Takes precedence over device param.
            use_turbo: If True, uses the faster turbo model (recommended for CPU).
        """
        self.voice_path = voice_path
        self.device = "cuda" if use_gpu else device
        self.use_turbo = use_turbo
        self.sample_rate = 24000  # Chatterbox default sample rate
        self._temp_voice_path = None  # Preprocessed voice file
        self._voice_prepared = False  # Whether voice conditionals are pre-computed

        # Validate voice path if provided
        if self.voice_path and not os.path.exists(self.voice_path):
            print(f"-> Warning: Voice file not found: {self.voice_path}")
            self.voice_path = None

        # Lazy-load model and torchaudio
        self._model = None
        self._torchaudio = None
        self._ChatterboxModel = None
        self._torch = None

        # Try to import Chatterbox (turbo or standard model)
        try:
            with suppress_chatterbox_logs():
                if self.use_turbo:
                    from chatterbox.tts_turbo import ChatterboxTurboTTS as ChatterboxModel
                else:
                    from chatterbox.tts import ChatterboxTTS as ChatterboxModel
                import torch
                import torchaudio
            model_type = "Turbo" if self.use_turbo else "Standard"
            print(f"-> Chatterbox {model_type} TTS library loaded")
            self._ChatterboxModel = ChatterboxModel
            self._torchaudio = torchaudio
            self._torch = torch
        except ImportError as e:
            print(f"-> Error: Chatterbox TTS not installed. Install with:")
            print("   pip install chatterbox-tts torchaudio")
            raise ImportError(
                "Chatterbox TTS not installed. Run: pip install chatterbox-tts torchaudio"
            ) from e

        # Preprocess voice file if provided (convert to float32 mono for compatibility)
        if self.voice_path:
            self._preprocess_voice_file()

        # Initialize the model
        self._initialize_model()

    def _preprocess_voice_file(self) -> None:
        """Preprocess voice file to ensure float32 mono format for Chatterbox compatibility."""
        import soundfile as sf

        try:
            # Load audio file
            data, sr = sf.read(self.voice_path, dtype='float32')

            # Convert stereo to mono if needed
            if len(data.shape) > 1:
                data = data.mean(axis=1).astype(np.float32)

            # Create temp file with preprocessed audio
            self._temp_voice_path = tempfile.NamedTemporaryFile(
                suffix='.wav', delete=False, prefix='chatterbox_voice_'
            ).name
            sf.write(self._temp_voice_path, data, sr, subtype='FLOAT')
            print(f"-> Preprocessed voice file: {sr}Hz, {len(data)/sr:.1f}s, float32 mono")

        except Exception as e:
            print(f"-> Warning: Could not preprocess voice file: {e}")
            self._temp_voice_path = None

    def _prepare_voice_conditionals(self) -> None:
        """Pre-compute voice conditionals with proper dtype handling."""
        if not self.voice_path or self._model is None:
            return

        try:
            import librosa

            voice_file = self._temp_voice_path or self.voice_path

            # Monkey-patch librosa.load to return float32
            original_librosa_load = librosa.load
            def float32_librosa_load(path, **kwargs):
                data, rate = original_librosa_load(path, **kwargs)
                return data.astype(np.float32), rate

            # Monkey-patch librosa.resample to return float32
            original_librosa_resample = librosa.resample
            def float32_librosa_resample(y, **kwargs):
                result = original_librosa_resample(y, **kwargs)
                return result.astype(np.float32)

            # Monkey-patch torch.from_numpy to convert float64 to float32
            original_from_numpy = self._torch.from_numpy
            def float32_from_numpy(arr):
                if hasattr(arr, 'dtype') and arr.dtype == np.float64:
                    arr = arr.astype(np.float32)
                return original_from_numpy(arr)

            # Monkey-patch torch.tensor to convert float64 to float32
            original_tensor = self._torch.tensor
            def float32_tensor(data, *args, **kwargs):
                if isinstance(data, np.ndarray) and data.dtype == np.float64:
                    data = data.astype(np.float32)
                return original_tensor(data, *args, **kwargs)

            # Apply patches
            librosa.load = float32_librosa_load
            librosa.resample = float32_librosa_resample
            self._torch.from_numpy = float32_from_numpy
            self._torch.tensor = float32_tensor

            try:
                with suppress_chatterbox_logs():
                    self._model.prepare_conditionals(voice_file)
                self._voice_prepared = True
            finally:
                # Restore original functions
                librosa.load = original_librosa_load
                librosa.resample = original_librosa_resample
                self._torch.from_numpy = original_from_numpy
                self._torch.tensor = original_tensor

        except Exception as e:
            print(f"-> Warning: Could not prepare voice conditionals: {e}")
            self._voice_prepared = False

    def _initialize_model(self) -> None:
        """Initialize the Chatterbox model."""
        if self._model is not None:
            return

        model_type = "Turbo" if self.use_turbo else "Standard"
        print(f"-> Loading Chatterbox {model_type} TTS model on {self.device.upper()}...", end="", flush=True)
        load_start = time.time()
        try:
            # Workaround for cuDNN sublibrary version mismatch on newer GPUs (e.g. RTX 50 series)
            # The error occurs in RNN flatten_parameters() when cuDNN versions don't match.
            # Temporarily disable cuDNN for RNN operations during model loading.
            with suppress_chatterbox_logs():
                if self.device == "cuda" and self._torch.cuda.is_available():
                    original_cudnn_enabled = self._torch.backends.cudnn.enabled
                    try:
                        # Disable cuDNN to avoid CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
                        self._torch.backends.cudnn.enabled = False
                        self._model = self._ChatterboxModel.from_pretrained(device=self.device)
                    finally:
                        # Re-enable cuDNN for other operations (convolutions benefit from it)
                        self._torch.backends.cudnn.enabled = original_cudnn_enabled
                else:
                    self._model = self._ChatterboxModel.from_pretrained(device=self.device)
            
            self.sample_rate = self._model.sr
            load_time = time.time() - load_start

            voice_info = f" with voice: {Path(self.voice_path).name}" if self.voice_path else ""
            print(f" done ({load_time:.1f}s){voice_info}")

            # Pre-compute voice conditionals if voice file provided
            if self.voice_path:
                self._prepare_voice_conditionals()

        except Exception as e:
            print(f"-> Error initializing Chatterbox TTS: {e}")
            raise

    def _split_text_by_sentences(self, text: str, max_chars: int = 250) -> List[str]:
        """Split text into chunks at sentence boundaries.
        
        Chatterbox works best with shorter text segments (~250 chars).
        This splits on sentence boundaries to maintain natural speech flow.
        
        Args:
            text: Text to split
            max_chars: Maximum characters per chunk
            
        Returns:
            List of text chunks
        """
        # Split by sentence-ending punctuation
        sentences = re.split(r'([.!?]+\s+)', text)
        
        chunks = []
        current_chunk = ""
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
            full_sentence = sentence + punctuation
            
            # If adding this sentence would exceed max_chars, save current chunk
            if current_chunk and len(current_chunk) + len(full_sentence) > max_chars:
                chunks.append(current_chunk.strip())
                current_chunk = full_sentence
            else:
                current_chunk += full_sentence
        
        # Add remaining text
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio data as float32 numpy array
        """
        if not text.strip():
            return np.zeros(0, dtype=np.float32)
        
        # Ensure model is initialized
        if self._model is None:
            self._initialize_model()
        
        # Split long text into chunks
        text_chunks = self._split_text_by_sentences(text, max_chars=250)
        
        if len(text_chunks) > 1:
            logger.debug(f"Splitting text into {len(text_chunks)} chunks for synthesis")
        
        audio_arrays = []
        
        for i, chunk in enumerate(text_chunks):
            if len(text_chunks) > 1:
                logger.debug(f"Synthesizing chunk {i+1}/{len(text_chunks)}: {chunk[:40]}...")
            
            try:
                # Generate audio with or without voice cloning
                # If conditionals are pre-computed, don't pass audio_prompt_path
                with suppress_chatterbox_logs():
                    if self._voice_prepared:
                        wav_tensor = self._model.generate(chunk)
                    elif self.voice_path:
                        # Fallback to file path (may have dtype issues)
                        voice_file = self._temp_voice_path or self.voice_path
                        wav_tensor = self._model.generate(chunk, audio_prompt_path=voice_file)
                    else:
                        wav_tensor = self._model.generate(chunk)
                
                # Convert torch tensor to numpy array
                # wav_tensor shape is typically (1, num_samples) or (num_samples,)
                wav_np = wav_tensor.squeeze().cpu().numpy().astype(np.float32)
                audio_arrays.append(wav_np)
                
            except Exception as e:
                logger.warning(f"Error synthesizing chunk {i+1}: {e}")
                continue
        
        if not audio_arrays:
            return np.zeros(0, dtype=np.float32)
        
        # Concatenate all audio chunks with small silence gaps
        if len(audio_arrays) == 1:
            return audio_arrays[0]
        
        # Add 50ms silence between chunks for natural flow
        silence_samples = int(self.sample_rate * 0.05)
        silence = np.zeros(silence_samples, dtype=np.float32)
        
        result = audio_arrays[0]
        for chunk_audio in audio_arrays[1:]:
            result = np.concatenate([result, silence, chunk_audio])
        
        return result

    def _chunk_audio(self, audio_data: np.ndarray, chunk_size: int) -> List[np.ndarray]:
        """Split audio into chunks for playback with amplitude monitoring."""
        return [audio_data[i : i + chunk_size] for i in range(0, len(audio_data), chunk_size)]

    def play_audio_with_amplitude(
        self,
        audio_data: np.ndarray,
        amplitude_callback: Optional[Callable[[float], None]] = None,
        chunk_duration: float = 0.02,
    ) -> None:
        """Play audio with optional amplitude callback for animation.
        
        Args:
            audio_data: Audio data as float32 numpy array
            amplitude_callback: Optional callback receiving amplitude levels (0.0-1.0)
            chunk_duration: Duration of each chunk in seconds for amplitude updates
        """
        if audio_data is None or len(audio_data) == 0:
            if amplitude_callback:
                amplitude_callback(0.0)
            return

        audio_float = audio_data.astype(np.float32, copy=False)
        chunk_size = max(1, int(self.sample_rate * chunk_duration))
        chunks = self._chunk_audio(audio_float, chunk_size)
        if not chunks:
            return

        # Pre-calculate RMS values for amplitude visualization
        rms_values = [float(np.sqrt(np.mean(np.square(chunk)) + 1e-8)) for chunk in chunks]
        max_rms = max(rms_values) or 1.0
        normalized_levels = [min(rms / max_rms, 1.0) for rms in rms_values]

        # Convert to int16 for playback
        audio_int16 = np.clip(audio_float * 32767.0, -32768, 32767).astype(np.int16)

        pa = pyaudio.PyAudio()
        stream = None

        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=4096,
            )

            cursor = 0
            for chunk, level in zip(chunks, normalized_levels):
                frames = audio_int16[cursor : cursor + len(chunk)]
                stream.write(frames.tobytes())
                cursor += len(chunk)
                if amplitude_callback:
                    amplitude_callback(level)
        finally:
            if stream is not None:
                try:
                    output_latency = stream.get_output_latency()
                except Exception:
                    output_latency = 0.0

                drain_wait = (
                    max(output_latency, chunk_duration)
                    if output_latency and output_latency > 0
                    else chunk_duration
                )
                if drain_wait and drain_wait > 0:
                    time.sleep(drain_wait)

                if amplitude_callback:
                    amplitude_callback(0.0)
                stream.stop_stream()
                stream.close()
            elif amplitude_callback:
                amplitude_callback(0.0)
            pa.terminate()


__all__ = ["ChatterboxTTS"]

