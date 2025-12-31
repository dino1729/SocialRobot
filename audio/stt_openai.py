"""Speech-to-text using OpenAI Whisper (PyTorch-based)."""

from __future__ import annotations

import os
import ssl
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
import urllib.request

import numpy as np
import torch
import whisper


class OpenAIWhisperSTT:
    """Wrapper around OpenAI Whisper for GPU-accelerated transcription."""

    @staticmethod
    @contextmanager
    def _temporary_https_context(context_factory):
        previous = ssl._create_default_https_context
        previous_opener = urllib.request._opener
        ssl._create_default_https_context = context_factory
        # urllib caches a global opener (and its SSL context) after the first call.
        # Reset it so subsequent urlopen() calls pick up our temporary context.
        urllib.request._opener = None
        try:
            yield
        finally:
            ssl._create_default_https_context = previous
            urllib.request._opener = previous_opener

    @staticmethod
    def _is_cert_verify_failure(error: BaseException) -> bool:
        # urllib wraps SSL failures like: URLError("<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] ...>")
        message = f"{error!r} {error}"
        return "CERTIFICATE_VERIFY_FAILED" in message or "certificate verify failed" in message.lower()

    @staticmethod
    def _get_ca_bundle_from_env() -> Optional[str]:
        for variable_name in (
            "OPENAI_WHISPER_CA_BUNDLE",
            "SSL_CERT_FILE",
            "REQUESTS_CA_BUNDLE",
            "CURL_CA_BUNDLE",
        ):
            value = os.environ.get(variable_name)
            if value:
                return value
        return None

    @classmethod
    def _load_model(cls, model_size: str, device: str):
        download_root = os.environ.get("OPENAI_WHISPER_DOWNLOAD_ROOT")
        ssl_no_verify = os.environ.get("OPENAI_WHISPER_SSL_NO_VERIFY") in (
            "1",
            "true",
            "TRUE",
            "yes",
            "YES",
        )

        def load():
            return whisper.load_model(
                model_size,
                device=device,
                download_root=download_root,
            )

        if ssl_no_verify:
            with cls._temporary_https_context(ssl._create_unverified_context):
                return load()

        ca_bundle = cls._get_ca_bundle_from_env()
        if ca_bundle:
            ca_path = Path(ca_bundle).expanduser()
            if ca_path.exists():
                with cls._temporary_https_context(
                    lambda: ssl.create_default_context(cafile=str(ca_path))
                ):
                    return load()

        try:
            return load()
        except Exception as e:
            if not cls._is_cert_verify_failure(e):
                raise

            try:
                import certifi  # type: ignore

                with cls._temporary_https_context(
                    lambda: ssl.create_default_context(cafile=certifi.where())
                ):
                    return load()
            except Exception as e2:
                if cls._is_cert_verify_failure(e2) and ssl_no_verify:
                    with cls._temporary_https_context(ssl._create_unverified_context):
                        return load()

                raise RuntimeError(
                    "OpenAI Whisper model download failed due to TLS certificate verification.\n"
                    "If you're behind a proxy / MITM TLS, set `OPENAI_WHISPER_CA_BUNDLE=/path/to/ca.pem`.\n"
                    "As a last resort (insecure), set `OPENAI_WHISPER_SSL_NO_VERIFY=1` to disable verification."
                ) from e2

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
                self.model = self._load_model(model_size=model_size, device="cuda")
                self.device = "cuda"
            except Exception as e:
                print(f"-> Warning: GPU loading failed, using CPU")
                self.model = self._load_model(model_size=model_size, device="cpu")
                self.device = "cpu"
        else:
            print(f"-> Loading OpenAI Whisper model '{model_size}' on cpu.")
            self.model = self._load_model(model_size=model_size, device="cpu")
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
