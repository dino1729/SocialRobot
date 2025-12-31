"""Lightweight voice activity detection utilities."""

import collections
import threading
import time
from dataclasses import dataclass
from typing import Callable, Deque, Optional, Tuple

from audio.suppress_warnings import get_pyaudio, suppress_stderr
pyaudio = get_pyaudio()
import webrtcvad


Frame = Tuple[bytes, bool]


@dataclass
class VADConfig:
    """Parameters that control the WebRTC VAD behaviour."""

    sample_rate: int = 16000
    frame_duration_ms: int = 30
    padding_duration_ms: int = 300
    activation_ratio: float = 0.6
    deactivation_ratio: float = 0.85
    aggressiveness: int = 2  # 0-3, larger is more aggressive


class VADListener:
    """Continuously captures audio and emits speech segments."""

    def __init__(
        self,
        config: VADConfig = VADConfig(),
        device_index: Optional[int] = None,
        on_speech_callback: Optional[Callable[[bytes], None]] = None,
    ) -> None:
        self.config = config
        self.sample_rate = config.sample_rate
        self.frame_duration_ms = config.frame_duration_ms
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        self.padding_frames = max(1, int(config.padding_duration_ms / self.frame_duration_ms))
        self.activation_count = max(1, int(self.padding_frames * config.activation_ratio))
        self.deactivation_count = max(1, int(self.padding_frames * config.deactivation_ratio))

        self.device_index = device_index
        self.on_speech_callback = on_speech_callback

        self._vad = webrtcvad.Vad(config.aggressiveness)
        self._pa = pyaudio.PyAudio()
        self._stream = None

        self._stop_flag = threading.Event()
        self._vad_enabled = threading.Event()
        self._vad_enabled.set()
        self._frames_to_skip = 0
        self._reset_buffers = False

    def start(self) -> None:
        """Start capturing audio and running VAD (blocking)."""

        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.frame_size,
            input_device_index=self.device_index,
        )

        ring_buffer: Deque[Frame] = collections.deque(maxlen=self.padding_frames)
        triggered = False
        voiced_frames: Deque[bytes] = collections.deque()

        print("-> VAD listening loop started (WebRTC).")

        frame_interval = self.frame_duration_ms / 1000.0

        while not self._stop_flag.is_set():
            if not self._vad_enabled.is_set():
                if self._stream is not None and self._stream.is_active():
                    self._stream.stop_stream()
                ring_buffer.clear()
                voiced_frames.clear()
                triggered = False
                time.sleep(frame_interval)
                continue

            if self._stream is not None and not self._stream.is_active():
                self._stream.start_stream()

            frame = self._stream.read(self.frame_size, exception_on_overflow=False)

            if self._reset_buffers:
                ring_buffer.clear()
                voiced_frames.clear()
                triggered = False
                self._reset_buffers = False

            if self._frames_to_skip > 0:
                self._frames_to_skip -= 1
                continue

            is_speech = self._vad.is_speech(frame, self.sample_rate)
            ring_buffer.append((frame, is_speech))

            if not triggered:
                if len(ring_buffer) == ring_buffer.maxlen:
                    num_voiced = sum(1 for _, speech in ring_buffer if speech)
                    if num_voiced >= self.activation_count:
                        triggered = True
                        while ring_buffer:
                            voiced_frames.append(ring_buffer.popleft()[0])
            else:
                voiced_frames.append(frame)
                if len(ring_buffer) == ring_buffer.maxlen:
                    num_unvoiced = sum(1 for _, speech in ring_buffer if not speech)
                    if num_unvoiced >= self.deactivation_count:
                        triggered = False
                        speech_audio = b"".join(voiced_frames)
                        voiced_frames.clear()
                        ring_buffer.clear()
                        if speech_audio and self.on_speech_callback:
                            print("-> Speech segment detected ({} bytes).".format(len(speech_audio)))
                            self.on_speech_callback(speech_audio)

        self._stream.stop_stream()
        self._stream.close()
        self._pa.terminate()

    def stop(self) -> None:
        self._stop_flag.set()

    def enable_vad(self) -> None:
        skip_frames = max(1, int(0.4 / (self.frame_duration_ms / 1000.0)))
        self._frames_to_skip = max(skip_frames, self.padding_frames)
        self._reset_buffers = True
        if self._stream is not None and not self._stream.is_active():
            self._stream.start_stream()
        self._vad_enabled.set()

    def disable_vad(self) -> None:
        self._frames_to_skip = 0
        self._reset_buffers = True
        if self._stream is not None and self._stream.is_active():
            self._stream.stop_stream()
        self._vad_enabled.clear()


class FixedTimeListener:
    """Captures audio for a fixed duration instead of using VAD.
    
    This is useful when VAD doesn't work well with certain audio setups
    or when users prefer predictable recording durations.
    """

    def __init__(
        self,
        listen_seconds: float = 5.0,
        sample_rate: int = 16000,
        device_index: Optional[int] = None,
        on_speech_callback: Optional[Callable[[bytes], None]] = None,
    ) -> None:
        self.listen_seconds = listen_seconds
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.on_speech_callback = on_speech_callback

        self._pa = pyaudio.PyAudio()
        self._stream = None
        self._stop_flag = threading.Event()
        self._listening_enabled = threading.Event()
        self._listening_enabled.set()
        
        # Frame size for reading (same as VAD for consistency)
        self.frame_duration_ms = 30
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)

    def start(self) -> None:
        """Start the fixed-time listening loop (blocking)."""
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.frame_size,
            input_device_index=self.device_index,
        )

        print(f"-> Fixed-time listening started ({self.listen_seconds}s per segment).")

        while not self._stop_flag.is_set():
            if not self._listening_enabled.is_set():
                time.sleep(0.1)
                continue

            # Capture audio for the fixed duration
            print(f"-> Recording for {self.listen_seconds}s...")
            frames = []
            frames_needed = int(self.sample_rate * self.listen_seconds / self.frame_size)
            
            for _ in range(frames_needed):
                if self._stop_flag.is_set() or not self._listening_enabled.is_set():
                    break
                frame = self._stream.read(self.frame_size, exception_on_overflow=False)
                frames.append(frame)
            
            if frames and self._listening_enabled.is_set():
                audio_data = b"".join(frames)
                print(f"-> Captured {len(audio_data)} bytes of audio.")
                
                if self.on_speech_callback:
                    self.on_speech_callback(audio_data)

        self._stream.stop_stream()
        self._stream.close()
        self._pa.terminate()

    def stop(self) -> None:
        """Stop the listener."""
        self._stop_flag.set()

    def enable_vad(self) -> None:
        """Enable listening (named for API compatibility with VADListener)."""
        self._listening_enabled.set()

    def disable_vad(self) -> None:
        """Disable listening (named for API compatibility with VADListener)."""
        self._listening_enabled.clear()


__all__ = ["VADListener", "VADConfig", "FixedTimeListener"]
