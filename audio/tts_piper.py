"""Text-to-speech using Piper TTS."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pyaudio
import time
from piper import PiperVoice


class PiperTTS:
    """Generate audio using Piper neural TTS model."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        use_gpu: bool = False,
    ) -> None:
        """Initialize Piper TTS.
        
        Args:
            model_path: Path to Piper ONNX model file
            config_path: Path to Piper config JSON file
            use_gpu: Enable GPU acceleration (requires CUDA-enabled ONNX Runtime)
        """
        self.use_gpu = use_gpu
        self.sample_rate = 22050  # Piper default sample rate
        
        # Download default model if not provided
        if model_path is None:
            model_path = self._download_default_model()
        
        if config_path is None:
            # Piper expects .onnx.json not just .json
            config_path = model_path + '.json'
        
        print(f"-> Loading Piper TTS model from {model_path}")
        
        # Load Piper voice
        # Try GPU if requested, but fall back to CPU if unavailable
        actual_use_cuda = False
        if use_gpu:
            try:
                import onnxruntime as ort
                available_providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in available_providers or 'TensorrtExecutionProvider' in available_providers:
                    print(f"-> Attempting to use GPU for Piper TTS")
                    actual_use_cuda = True
                else:
                    print(f"-> Warning: GPU requested but no GPU provider available, using CPU")
            except Exception as e:
                print(f"-> Warning: GPU check failed ({e}), using CPU")
        
        self.voice = PiperVoice.load(model_path, config_path, use_cuda=actual_use_cuda)
        device_str = "GPU" if actual_use_cuda else "CPU"
        
        print(f"-> Using Piper TTS for speech synthesis on {device_str}.")

    def _download_default_model(self) -> str:
        """Download default Piper model if not exists."""
        import requests
        
        model_dir = Path.home() / ".cache" / "piper_tts"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_file = model_dir / "en_US-lessac-medium.onnx"
        config_file = model_dir / "en_US-lessac-medium.onnx.json"
        
        if not model_file.exists():
            print("-> Downloading Piper model (first time only)...")
            
            # Download model
            model_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
            response = requests.get(model_url, stream=True, timeout=120)
            response.raise_for_status()
            
            with open(model_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Download config
            config_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
            response = requests.get(config_url, timeout=60)
            response.raise_for_status()
            
            with open(config_file, 'wb') as f:
                f.write(response.content)
            
            print(f"-> Downloaded Piper model to {model_file}")
        
        return str(model_file)

    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio data as float32 numpy array
        """
        if not text.strip():
            return np.zeros(0, dtype=np.float32)

        # Synthesize with Piper - returns a generator of AudioChunk objects
        audio_arrays = []
        for audio_chunk in self.voice.synthesize(text):
            # AudioChunk has audio_float_array property
            audio_arrays.append(audio_chunk.audio_float_array)
        
        if not audio_arrays:
            return np.zeros(0, dtype=np.float32)
        
        # Concatenate all chunks
        audio_data = np.concatenate(audio_arrays)
        
        return audio_data.astype(np.float32)

    def _chunk_audio(self, audio_data: np.ndarray, chunk_size: int) -> List[np.ndarray]:
        """Split audio into chunks."""
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
            amplitude_callback: Optional callback for amplitude levels
            chunk_duration: Duration of each chunk in seconds
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

        rms_values = [float(np.sqrt(np.mean(np.square(chunk)) + 1e-8)) for chunk in chunks]
        max_rms = max(rms_values) or 1.0
        normalized_levels = [min(rms / max_rms, 1.0) for rms in rms_values]

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


__all__ = ["PiperTTS"]

