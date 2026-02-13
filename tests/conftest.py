"""Shared test fixtures and configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def sample_segments() -> List[Dict]:
    return [
        {"start": 0.0, "end": 1.5, "speaker": 1, "text": "Hello world"},
        {"start": 1.5, "end": 3.0, "speaker": 2, "text": "Hi there"},
        {"start": 3.0, "end": 5.0, "speaker": 1, "text": "How are you doing today?"},
    ]


@pytest.fixture
def valid_cache_data(sample_segments) -> Dict:
    from app.cache import CACHE_SCHEMA_VERSION
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "audio_hash": "abc123",
        "segments": sample_segments,
    }


@pytest.fixture
def temp_audio_file(tmp_path: Path) -> Path:
    """Create a minimal WAV file for testing."""
    audio_path = tmp_path / "test_audio.wav"
    # Minimal valid WAV header + silence
    import struct
    sample_rate = 16000
    num_samples = sample_rate  # 1 second
    data_size = num_samples * 2  # 16-bit
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size, b'WAVE',
        b'fmt ', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b'data', data_size,
    )
    audio_path.write_bytes(header + b'\x00' * data_size)
    return audio_path


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


def mock_voxtral_response(segments=None):
    """Create a mock Voxtral API response."""
    if segments is None:
        segments = [
            MagicMock(start=0.0, end=1.5, text="Hello world", speaker_id="S0"),
            MagicMock(start=1.5, end=3.0, text="Hi there", speaker_id="S1"),
        ]
    response = MagicMock()
    response.segments = segments
    return response


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
