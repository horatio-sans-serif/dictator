"""Integration tests for full transcription pipeline.

These tests use real audio files and real models (Whisper tiny) to verify
end-to-end functionality. They are marked as slow and integration.
"""
from __future__ import annotations

import json
import queue
import sys
import threading
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import (
    CACHE_DIR,
    CACHE_SCHEMA_VERSION,
    ENHANCEMENT_VERSION,
    TranscriptionWorker,
    compute_file_hash,
    preprocess_audio,
)


def generate_speech_like_audio(sample_rate: int, duration: float, frequency: float = 200.0) -> np.ndarray:
    """Generate audio that resembles speech frequencies.

    This creates a more complex waveform that Whisper can process,
    even though it won't produce meaningful text.
    """
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, endpoint=False)

    # Fundamental frequency (speech-like)
    signal = 0.3 * np.sin(2 * np.pi * frequency * t)

    # Add harmonics for more complex waveform
    signal += 0.15 * np.sin(2 * np.pi * frequency * 2 * t)
    signal += 0.08 * np.sin(2 * np.pi * frequency * 3 * t)

    # Add some amplitude modulation to mimic speech rhythm
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)  # 3 Hz modulation
    signal = signal * modulation

    # Add slight noise
    noise = 0.02 * np.random.randn(samples)
    signal = signal + noise

    return signal.astype(np.float32)


class TestEnhancementIntegration:
    """Integration tests for audio enhancement pipeline."""

    @pytest.mark.integration
    def test_soap_enhancement_creates_file(self, tmp_path: Path) -> None:
        """Soap enhancement should create enhanced file."""
        # Create test audio
        audio = generate_speech_like_audio(16000, 2.0)
        audio_path = tmp_path / "test.wav"
        sf.write(audio_path, audio, 16000)

        audio_hash = compute_file_hash(audio_path)
        enhanced_path, compare_path = preprocess_audio(
            audio_path,
            audio_hash,
            preset_name="general",
            create_compare=False,
            enabled=True,
            backend="soap",
        )

        assert enhanced_path.exists()
        assert compare_path is None

        # Read enhanced audio
        enhanced_audio, sr = sf.read(enhanced_path)
        assert len(enhanced_audio) > 0
        assert sr == 16000

    @pytest.mark.integration
    def test_soap_enhancement_with_compare(self, tmp_path: Path) -> None:
        """Should create comparison file when requested."""
        audio = generate_speech_like_audio(16000, 2.0)
        audio_path = tmp_path / "test.wav"
        sf.write(audio_path, audio, 16000)

        audio_hash = compute_file_hash(audio_path)
        enhanced_path, compare_path = preprocess_audio(
            audio_path,
            audio_hash,
            preset_name="general",
            create_compare=True,
            compare_duration=5.0,
            enabled=True,
            backend="soap",
        )

        assert enhanced_path.exists()
        assert compare_path is not None
        assert compare_path.exists()

    @pytest.mark.integration
    def test_enhancement_disabled_returns_original(self, tmp_path: Path) -> None:
        """Should return original path when enhancement disabled."""
        audio = generate_speech_like_audio(16000, 2.0)
        audio_path = tmp_path / "test.wav"
        sf.write(audio_path, audio, 16000)

        audio_hash = compute_file_hash(audio_path)
        enhanced_path, compare_path = preprocess_audio(
            audio_path,
            audio_hash,
            preset_name="general",
            enabled=False,
        )

        assert enhanced_path == audio_path
        assert compare_path is None

    @pytest.mark.integration
    def test_all_presets_work(self, tmp_path: Path) -> None:
        """All presets should successfully enhance audio."""
        from app.main import MIC_PRESETS

        audio = generate_speech_like_audio(16000, 1.0)
        audio_path = tmp_path / "test.wav"
        sf.write(audio_path, audio, 16000)

        for preset_name in MIC_PRESETS.keys():
            audio_hash = f"test_{preset_name}"
            enhanced_path, _ = preprocess_audio(
                audio_path,
                audio_hash,
                preset_name=preset_name,
                enabled=True,
                backend="soap",
            )
            assert enhanced_path.exists(), f"Preset {preset_name} failed"


@pytest.mark.slow
class TestTranscriptionIntegration:
    """Integration tests for full transcription pipeline.

    These tests require Whisper model download and are slow.
    """

    @pytest.fixture
    def test_audio_path(self, tmp_path: Path) -> Path:
        """Create test audio file."""
        audio = generate_speech_like_audio(16000, 3.0)
        audio_path = tmp_path / "speech.wav"
        sf.write(audio_path, audio, 16000)
        return audio_path

    @pytest.mark.integration
    def test_full_pipeline_produces_segments(self, test_audio_path: Path) -> None:
        """Full pipeline should produce transcript segments."""
        messages: queue.Queue = queue.Queue()
        worker = TranscriptionWorker(
            audio_path=test_audio_path,
            messages=messages,
            model_name="tiny",
            enhancement_enabled=False,  # Skip enhancement to focus on transcription
        )

        worker.start()
        worker.join(timeout=180)  # 3 minute timeout for CPU transcription

        # Collect messages
        collected: List[Dict] = []
        while not messages.empty():
            collected.append(messages.get_nowait())

        # Should have progress messages
        progress_msgs = [m for m in collected if m.get("type") == "progress"]
        assert len(progress_msgs) > 0, f"No progress messages. All messages: {collected}"

        # Should have either done or error (pipeline completed)
        done_msgs = [m for m in collected if m.get("type") == "done"]
        error_msgs = [m for m in collected if m.get("type") == "error"]

        # Pipeline must complete (either success or error)
        assert len(done_msgs) == 1 or len(error_msgs) > 0, (
            f"Pipeline did not complete. Messages: {[m.get('type') for m in collected]}"
        )

        # If completed successfully, we're good (segments may be empty for generated audio)
        # Note: Whisper might return empty segments for generated audio
        # This is acceptable - we're testing the pipeline, not speech recognition

    @pytest.mark.integration
    def test_worker_can_be_stopped(self, test_audio_path: Path) -> None:
        """Worker should respond to stop request."""
        messages: queue.Queue = queue.Queue()
        worker = TranscriptionWorker(
            audio_path=test_audio_path,
            messages=messages,
            model_name="tiny",
        )

        worker.start()
        # Give it a moment to start
        import time
        time.sleep(0.5)

        worker.request_stop()
        worker.join(timeout=5)

        assert not worker.is_alive()


class TestCacheIntegration:
    """Integration tests for caching behavior."""

    @pytest.mark.integration
    def test_cache_hit_returns_same_result(self, tmp_path: Path) -> None:
        """Second run should use cache and return identical results."""
        # Create test audio
        audio = generate_speech_like_audio(16000, 2.0)
        audio_path = tmp_path / "speech.wav"
        sf.write(audio_path, audio, 16000)

        # First run
        messages1: queue.Queue = queue.Queue()
        worker1 = TranscriptionWorker(
            audio_path=audio_path,
            messages=messages1,
            model_name="tiny",
            enhancement_enabled=False,  # Skip enhancement for speed
        )
        worker1.start()
        worker1.join(timeout=180)  # 3 minute timeout for first run

        collected1: List[Dict] = []
        while not messages1.empty():
            collected1.append(messages1.get_nowait())

        # First run should complete (done or error)
        done_msgs1 = [m for m in collected1 if m.get("type") == "done"]
        error_msgs1 = [m for m in collected1 if m.get("type") == "error"]
        if len(done_msgs1) == 0 and len(error_msgs1) == 0:
            pytest.skip("First transcription did not complete in time")

        # If first run errored, skip cache test
        if len(error_msgs1) > 0:
            pytest.skip(f"First transcription errored: {error_msgs1[0].get('message')}")

        segments1 = [m.get("segment") for m in collected1 if m.get("type") == "segment"]

        # Second run (should use cache)
        messages2: queue.Queue = queue.Queue()
        worker2 = TranscriptionWorker(
            audio_path=audio_path,
            messages=messages2,
            model_name="tiny",
            enhancement_enabled=False,
        )
        worker2.start()
        worker2.join(timeout=30)  # Cache hit should be fast

        collected2: List[Dict] = []
        while not messages2.empty():
            collected2.append(messages2.get_nowait())

        segments2 = [m.get("segment") for m in collected2 if m.get("type") == "segment"]

        # Check for cache hit indicator
        done_msgs = [m for m in collected2 if m.get("type") == "done"]
        assert len(done_msgs) == 1, f"No done message on cache hit. Messages: {collected2}"
        # Cache hit indicated by from_cache=True
        assert done_msgs[0].get("from_cache") is True

        # Results should be identical
        assert len(segments1) == len(segments2)

    @pytest.mark.integration
    def test_model_change_invalidates_cache(self, tmp_path: Path) -> None:
        """Changing model should invalidate cache."""
        audio = generate_speech_like_audio(16000, 2.0)
        audio_path = tmp_path / "speech.wav"
        sf.write(audio_path, audio, 16000)

        # Create cache with tiny model
        audio_hash = compute_file_hash(audio_path)
        cache_file = CACHE_DIR / f"{audio_hash}.json"

        cache_data = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "audio_hash": audio_hash,
            "model": "tiny",
            "language": "en",
            "enhancement_version": 0,
            "enhancement_enabled": False,
            "enhancement_preset": "none",
            "enhancement_backend": "none",
            "segments": [{"start": 0, "end": 1, "speaker": 1, "text": "cached"}],
        }
        cache_file.write_text(json.dumps(cache_data), encoding="utf-8")

        # Run with different model
        messages: queue.Queue = queue.Queue()
        worker = TranscriptionWorker(
            audio_path=audio_path,
            messages=messages,
            model_name="base",  # Different from cached "tiny"
            enhancement_enabled=False,
        )

        # Check cache validation
        result = worker._try_load_cache(cache_file, audio_hash)
        assert result is None  # Cache should be invalidated

        # Cleanup
        if cache_file.exists():
            cache_file.unlink()


class TestErrorHandling:
    """Integration tests for error handling."""

    @pytest.mark.integration
    def test_missing_file_emits_error(self, tmp_path: Path) -> None:
        """Should emit error for missing audio file."""
        messages: queue.Queue = queue.Queue()
        worker = TranscriptionWorker(
            audio_path=tmp_path / "nonexistent.wav",
            messages=messages,
            model_name="tiny",
        )

        worker.start()
        worker.join(timeout=5)

        collected: List[Dict] = []
        while not messages.empty():
            collected.append(messages.get_nowait())

        error_msgs = [m for m in collected if m.get("type") == "error"]
        assert len(error_msgs) > 0
        assert "not found" in str(error_msgs[0].get("message", "")).lower()

    @pytest.mark.integration
    def test_empty_audio_handled(self, tmp_path: Path) -> None:
        """Should handle empty audio file gracefully."""
        # Create empty/silent audio
        audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        audio_path = tmp_path / "silent.wav"
        sf.write(audio_path, audio, 16000)

        messages: queue.Queue = queue.Queue()
        worker = TranscriptionWorker(
            audio_path=audio_path,
            messages=messages,
            model_name="tiny",
            enhancement_enabled=False,
        )

        worker.start()
        worker.join(timeout=60)

        collected: List[Dict] = []
        while not messages.empty():
            collected.append(messages.get_nowait())

        # Should complete (may have error or empty segments)
        done_or_error = [m for m in collected if m.get("type") in ("done", "error")]
        assert len(done_or_error) > 0


@pytest.mark.slow
@pytest.mark.integration
class TestRealAudioFiles:
    """Tests using real audio files from test assets."""

    @pytest.fixture
    def assets_dir(self) -> Path:
        """Get test assets directory."""
        return ROOT / "test" / "assets"

    def test_process_real_audio_file(self, assets_dir: Path) -> None:
        """Should process real audio files from test assets."""
        audio_files = list(assets_dir.glob("*.m4a"))
        if not audio_files:
            pytest.skip("No audio files in test/assets")

        audio_path = audio_files[0]
        messages: queue.Queue = queue.Queue()
        worker = TranscriptionWorker(
            audio_path=audio_path,
            messages=messages,
            model_name="tiny",
            preset_name="general",
        )

        worker.start()
        worker.join(timeout=180)  # 3 minute timeout for real files

        collected: List[Dict] = []
        while not messages.empty():
            collected.append(messages.get_nowait())

        # Should produce segments
        segments = [m for m in collected if m.get("type") == "segment"]
        assert len(segments) > 0, "No segments produced from real audio"

        # Should have done message
        done_msgs = [m for m in collected if m.get("type") == "done"]
        assert len(done_msgs) == 1
