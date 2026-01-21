"""Shared test fixtures and configuration."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Generator, List

import numpy as np
import pytest

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_rate() -> int:
    """Standard sample rate for test audio."""
    return 16000


@pytest.fixture
def short_audio(sample_rate: int) -> np.ndarray:
    """1 second of test audio (sine wave)."""
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # 440 Hz sine wave at moderate volume
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def silent_audio(sample_rate: int) -> np.ndarray:
    """1 second of silence."""
    return np.zeros(sample_rate, dtype=np.float32)


@pytest.fixture
def noisy_audio(sample_rate: int) -> np.ndarray:
    """1 second of audio with noise."""
    duration = 1.0
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, endpoint=False)
    # Speech-like signal plus noise
    signal = 0.3 * np.sin(2 * np.pi * 200 * t)  # Low frequency speech
    noise = 0.05 * np.random.randn(samples)  # Background noise
    return (signal + noise).astype(np.float32)


@pytest.fixture
def clipped_audio(sample_rate: int) -> np.ndarray:
    """Audio with transient peaks/clipping."""
    duration = 1.0
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)
    # Add some transient peaks
    audio[int(0.25 * samples)] = 0.95
    audio[int(0.5 * samples)] = -0.95
    audio[int(0.75 * samples)] = 0.9
    return audio.astype(np.float32)


@pytest.fixture
def audio_with_dc_offset(sample_rate: int) -> np.ndarray:
    """Audio with DC offset."""
    duration = 1.0
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, endpoint=False)
    # Sine wave with DC offset
    return (0.3 * np.sin(2 * np.pi * 440 * t) + 0.1).astype(np.float32)


@pytest.fixture
def audio_with_echo(sample_rate: int) -> np.ndarray:
    """Audio with artificial echo."""
    duration = 1.0
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, endpoint=False)
    # Original signal
    signal = np.zeros(samples, dtype=np.float32)
    # Impulse at the beginning
    signal[0:100] = 0.5 * np.sin(2 * np.pi * 440 * t[0:100])
    # Delayed echo
    delay_samples = int(0.06 * sample_rate)  # 60ms delay
    signal[delay_samples:delay_samples + 100] += 0.3 * np.sin(2 * np.pi * 440 * t[0:100])
    return signal


@pytest.fixture
def temp_audio_file(tmp_path: Path, short_audio: np.ndarray, sample_rate: int) -> Path:
    """Create a temporary audio file."""
    import soundfile as sf
    audio_path = tmp_path / "test_audio.wav"
    sf.write(audio_path, short_audio, sample_rate)
    return audio_path


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Create a temporary cache directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def valid_cache_data() -> Dict:
    """Valid cache data structure."""
    from app.main import CACHE_SCHEMA_VERSION, ENHANCEMENT_VERSION
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "audio_hash": "abc123",
        "model": "tiny",
        "language": "en",
        "enhancement_version": ENHANCEMENT_VERSION,
        "enhancement_enabled": True,
        "enhancement_preset": "general",
        "enhancement_backend": "soap",
        "segments": [
            {"start": 0.0, "end": 1.0, "speaker": 1, "text": "Hello world"},
            {"start": 1.0, "end": 2.0, "speaker": 2, "text": "Hi there"},
        ],
    }


@pytest.fixture
def mock_whisper_model():
    """Mock Whisper model for testing."""
    class MockWhisperModel:
        def transcribe(self, audio_path: str, language: str = "en", verbose: bool = False):
            return {
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "Test transcription segment one."},
                    {"start": 1.0, "end": 2.5, "text": "This is segment two."},
                ]
            }
    return MockWhisperModel()


@pytest.fixture
def mock_diarization_result():
    """Mock diarization result (list of turns)."""
    return [
        (0.0, 1.0, "SPEAKER_00"),
        (1.0, 2.5, "SPEAKER_01"),
    ]


class MockTurn:
    """Mock pyannote Turn object."""
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


class MockDiarization:
    """Mock pyannote diarization result."""
    def __init__(self, turns: List[tuple]):
        self._turns = turns

    def itertracks(self, yield_label: bool = False):
        for start, end, speaker in self._turns:
            yield MockTurn(start, end), None, speaker


@pytest.fixture
def mock_pipeline():
    """Mock pyannote pipeline for testing."""
    class MockPipeline:
        def __call__(self, input_data):
            return MockDiarization([
                (0.0, 1.0, "SPEAKER_00"),
                (1.0, 2.5, "SPEAKER_01"),
            ])

        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            return cls()

    return MockPipeline()


# Markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
