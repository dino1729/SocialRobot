"""Text-to-speech using Chatterbox TTS with zero-shot voice cloning."""

from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pyaudio


class ChatterboxTTS:
    """Generate audio using Chatterbox TTS with zero-shot voice cloning.
    
    Chatterbox TTS enables high-quality voice cloning from a short audio sample.
    Simply provide a WAV file of the target voice and it will synthesize speech
    in that voice style.
    """

    def __init__(
        self,
        voice_path: Optional[str] = None,
        device: str = "cpu",
        use_gpu: bool = False,
    ) -> None:
        """Initialize Chatterbox TTS.
        
        Args:
            voice_path: Path to voice reference WAV file for zero-shot cloning.
                       If None, uses Chatterbox's default voice.
            device: Device to use ('cpu' or 'cuda'). Overridden by use_gpu if True.
            use_gpu: If True, forces CUDA device. Takes precedence over device param.
        """
        self.voice_path = voice_path
        self.device = "cuda" if use_gpu else device
        self.sample_rate = 24000  # Chatterbox default sample rate
        
        # Validate voice path if provided
        if self.voice_path and not os.path.exists(self.voice_path):
            print(f"-> Warning: Voice file not found: {self.voice_path}")
            self.voice_path = None
        
        # Lazy-load model and torchaudio
        self._model = None
        self._torchaudio = None
        self._ChatterboxModel = None
        
        # Try to import Chatterbox
        try:
            from chatterbox.tts import ChatterboxTTS as ChatterboxModel
            import torchaudio
            self._ChatterboxModel = ChatterboxModel
            self._torchaudio = torchaudio
            print("-> Chatterbox TTS library loaded successfully")
        except ImportError as e:
            print(f"-> Error: Chatterbox TTS not installed. Install with:")
            print("   pip install chatterbox-tts torchaudio torchcodec")
            raise ImportError(
                "Chatterbox TTS not installed. Run: pip install chatterbox-tts torchaudio torchcodec"
            ) from e
        
        # Initialize the model
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the Chatterbox model."""
        if self._model is not None:
            return
        
        print(f"-> Loading Chatterbox TTS model on device: {self.device}")
        try:
            self._model = self._ChatterboxModel.from_pretrained(device=self.device)
            self.sample_rate = self._model.sr
            
            voice_info = f" with voice: {self.voice_path}" if self.voice_path else " (default voice)"
            print(f"-> Using Chatterbox TTS for speech synthesis on {self.device.upper()}{voice_info}.")
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
            print(f"-> Splitting text into {len(text_chunks)} chunks for synthesis")
        
        audio_arrays = []
        
        for i, chunk in enumerate(text_chunks):
            if len(text_chunks) > 1:
                print(f"-> Synthesizing chunk {i+1}/{len(text_chunks)}: {chunk[:40]}...")
            
            try:
                # Generate audio with or without voice cloning
                if self.voice_path:
                    wav_tensor = self._model.generate(chunk, audio_prompt_path=self.voice_path)
                else:
                    wav_tensor = self._model.generate(chunk)
                
                # Convert torch tensor to numpy array
                # wav_tensor shape is typically (1, num_samples) or (num_samples,)
                wav_np = wav_tensor.squeeze().cpu().numpy().astype(np.float32)
                audio_arrays.append(wav_np)
                
            except Exception as e:
                print(f"-> Error synthesizing chunk {i+1}: {e}")
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

