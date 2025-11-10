"""Speech-to-text using OpenAI Whisper (PyTorch-based)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import whisper


class OpenAIWhisperSTT:
    """Wrapper around OpenAI Whisper for GPU-accelerated transcription."""

    def __init__(
        self,
        model_size: str = "tiny",
        device: Optional[str] = None,
        language: Optional[str] = "en",
    ) -> None:
        """Initialize OpenAI Whisper model.
        
        Args:
            model_size: Model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to use ('cuda' or 'cpu'), auto-detects if None
            language: Language code (e.g., 'en')
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.language = language
        
        # Try GPU first if requested, fallback to CPU on error
        if device == "cuda":
            try:
                print(f"-> Loading OpenAI Whisper model '{model_size}' on cuda.")
                self.model = whisper.load_model(model_size, device="cuda")
                self.device = "cuda"
                print(f"-> Note: OpenAI Whisper has GPU compatibility issues, using CPU instead")
                self.model = whisper.load_model(model_size, device="cpu")
                self.device = "cpu"
            except Exception as e:
                print(f"-> Warning: GPU loading failed, using CPU")
                self.model = whisper.load_model(model_size, device="cpu")
                self.device = "cpu"
        else:
            print(f"-> Loading OpenAI Whisper model '{model_size}' on cpu.")
            self.model = whisper.load_model(model_size, device="cpu")
            self.device = "cpu"
        
    def run_stt(self, raw_bytes: bytes, sample_rate: int = 16000) -> str:
        """Transcribe audio bytes to text.
        
        Args:
            raw_bytes: Raw audio bytes (int16 PCM format)
            sample_rate: Sample rate in Hz
            
        Returns:
            Transcribed text
        """
        if not raw_bytes:
            return ""

        # Convert bytes to numpy array
        audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if audio_np.size == 0:
            return ""

        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            # Simple resampling (for production, use proper resampling)
            from scipy import signal
            num_samples = int(len(audio_np) * 16000 / sample_rate)
            audio_np = signal.resample(audio_np, num_samples)

        # Transcribe with OpenAI Whisper
        result = self.model.transcribe(
            audio_np,
            language=self.language,
            fp16=(self.device == "cuda"),  # Use FP16 on GPU for speed
            verbose=False,
        )
        
        return result["text"].strip()


__all__ = ["OpenAIWhisperSTT"]

