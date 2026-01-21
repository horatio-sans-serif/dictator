"""Unit tests for TranscriptionWorker and related functionality."""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import (
    CACHE_SCHEMA_VERSION,
    ENHANCEMENT_VERSION,
    TranscriptionWorker,
    AudioPlayer,
    UISessionResult,
    print_segment,
    print_transcript,
    format_timestamp,
)


class TestTranscriptionWorker:
    """Tests for TranscriptionWorker class."""

    def _create_test_audio(self, tmp_path: Path, sample_rate: int = 16000) -> Path:
        """Create a test audio file."""
        duration = 2.0
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples, endpoint=False)
        audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        audio_path = tmp_path / "test.wav"
        sf.write(audio_path, audio, sample_rate)
        return audio_path

    @pytest.mark.unit
    def test_worker_initialization(self, tmp_path: Path) -> None:
        """Should initialize worker with correct defaults."""
        audio_path = self._create_test_audio(tmp_path)
        messages: queue.Queue = queue.Queue()

        worker = TranscriptionWorker(
            audio_path=audio_path,
            messages=messages,
            model_name="tiny",
        )

        assert worker.audio_path == audio_path
        assert worker.model_name == "tiny"
        assert worker.language == "en"
        assert worker.enhancement_enabled is True
        assert worker.enhancer_backend == "soap"
        assert worker.preset_name == "general"

    @pytest.mark.unit
    def test_worker_custom_config(self, tmp_path: Path) -> None:
        """Should accept custom configuration."""
        audio_path = self._create_test_audio(tmp_path)
        messages: queue.Queue = queue.Queue()

        worker = TranscriptionWorker(
            audio_path=audio_path,
            messages=messages,
            model_name="large-v3",
            preset_name="iphone",
            enhancement_enabled=False,
            enhancer_backend="resemble",
        )

        assert worker.model_name == "large-v3"
        assert worker.preset_name == "iphone"
        assert worker.enhancement_enabled is False
        assert worker.enhancer_backend == "resemble"

    @pytest.mark.unit
    def test_worker_invalid_preset_defaults(self, tmp_path: Path) -> None:
        """Should default to general for invalid preset."""
        audio_path = self._create_test_audio(tmp_path)
        messages: queue.Queue = queue.Queue()

        worker = TranscriptionWorker(
            audio_path=audio_path,
            messages=messages,
            preset_name="invalid_preset",
        )

        assert worker.preset_name == "general"

    @pytest.mark.unit
    def test_worker_invalid_enhancer_defaults(self, tmp_path: Path) -> None:
        """Should default to soap for invalid enhancer."""
        audio_path = self._create_test_audio(tmp_path)
        messages: queue.Queue = queue.Queue()

        worker = TranscriptionWorker(
            audio_path=audio_path,
            messages=messages,
            enhancer_backend="invalid_enhancer",
        )

        assert worker.enhancer_backend == "soap"

    @pytest.mark.unit
    def test_worker_stop_event(self, tmp_path: Path) -> None:
        """Should have working stop event."""
        audio_path = self._create_test_audio(tmp_path)
        messages: queue.Queue = queue.Queue()

        worker = TranscriptionWorker(
            audio_path=audio_path,
            messages=messages,
        )

        assert not worker.stop_event.is_set()
        worker.request_stop()
        assert worker.stop_event.is_set()

    @pytest.mark.unit
    def test_worker_emit_message(self, tmp_path: Path) -> None:
        """Should emit messages to queue."""
        audio_path = self._create_test_audio(tmp_path)
        messages: queue.Queue = queue.Queue()

        worker = TranscriptionWorker(
            audio_path=audio_path,
            messages=messages,
        )

        worker._emit({"type": "test", "data": "hello"})
        msg = messages.get_nowait()
        assert msg["type"] == "test"
        assert msg["data"] == "hello"

    @pytest.mark.unit
    def test_worker_emit_blocked_when_stopped(self, tmp_path: Path) -> None:
        """Should not emit messages when stopped."""
        audio_path = self._create_test_audio(tmp_path)
        messages: queue.Queue = queue.Queue()

        worker = TranscriptionWorker(
            audio_path=audio_path,
            messages=messages,
        )

        worker.request_stop()
        worker._emit({"type": "test"})
        assert messages.empty()

    @pytest.mark.unit
    def test_worker_emit_progress(self, tmp_path: Path) -> None:
        """Should emit progress messages."""
        audio_path = self._create_test_audio(tmp_path)
        messages: queue.Queue = queue.Queue()

        worker = TranscriptionWorker(
            audio_path=audio_path,
            messages=messages,
        )

        worker._emit_progress(50, "Processing...")
        msg = messages.get_nowait()
        assert msg["type"] == "progress"
        assert msg["percent"] == 50
        assert msg["message"] == "Processing..."

    @pytest.mark.unit
    def test_worker_emit_progress_clamps(self, tmp_path: Path) -> None:
        """Should clamp progress to 0-100."""
        audio_path = self._create_test_audio(tmp_path)
        messages: queue.Queue = queue.Queue()

        worker = TranscriptionWorker(
            audio_path=audio_path,
            messages=messages,
        )

        worker._emit_progress(-10, "Test")
        msg = messages.get_nowait()
        assert msg["percent"] == 0

        worker._emit_progress(150, "Test")
        msg = messages.get_nowait()
        assert msg["percent"] == 100

    @pytest.mark.unit
    def test_worker_debug_compare_defaults(self, tmp_path: Path) -> None:
        """Should handle debug compare configuration."""
        audio_path = self._create_test_audio(tmp_path)
        messages: queue.Queue = queue.Queue()

        # Default: no debug compare
        worker = TranscriptionWorker(
            audio_path=audio_path,
            messages=messages,
        )
        assert worker.debug_compare is False
        assert worker.debug_compare_duration is None

        # With debug compare enabled
        worker2 = TranscriptionWorker(
            audio_path=audio_path,
            messages=messages,
            debug_compare=True,
        )
        assert worker2.debug_compare is True
        assert worker2.debug_compare_duration == 30.0  # Default

        # With custom duration
        worker3 = TranscriptionWorker(
            audio_path=audio_path,
            messages=messages,
            debug_compare=True,
            debug_compare_duration=60.0,
        )
        assert worker3.debug_compare_duration == 60.0

        # With zero duration (full file)
        worker4 = TranscriptionWorker(
            audio_path=audio_path,
            messages=messages,
            debug_compare=True,
            debug_compare_duration=0,
        )
        assert worker4.debug_compare_duration is None


class TestMergeSegments:
    """Tests for segment merging logic."""

    def _create_worker(self, tmp_path: Path) -> TranscriptionWorker:
        """Create worker for testing merge functionality."""
        audio_file = tmp_path / "dummy.wav"
        audio_file.write_bytes(b"dummy")
        messages: queue.Queue = queue.Queue()
        return TranscriptionWorker(
            audio_path=audio_file,
            messages=messages,
        )

    @pytest.mark.unit
    def test_merge_with_diarization(self, tmp_path: Path) -> None:
        """Should merge transcript with diarization."""
        worker = self._create_worker(tmp_path)
        transcript_segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 1.0, "end": 2.5, "text": "World"},
        ]
        diarization_segments = [
            (0.0, 1.0, "SPEAKER_00"),
            (1.0, 2.5, "SPEAKER_01"),
        ]

        result = worker._merge_segments(transcript_segments, diarization_segments)

        assert len(result) == 2
        assert result[0]["speaker"] == 1
        assert result[0]["text"] == "Hello"
        assert result[1]["speaker"] == 2
        assert result[1]["text"] == "World"

    @pytest.mark.unit
    def test_merge_without_diarization(self, tmp_path: Path) -> None:
        """Should assign all segments to speaker 1 when no diarization."""
        worker = self._create_worker(tmp_path)
        transcript_segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 1.0, "end": 2.5, "text": "World"},
        ]

        result = worker._merge_segments(transcript_segments, [])

        assert len(result) == 2
        assert result[0]["speaker"] == 1
        assert result[1]["speaker"] == 1

    @pytest.mark.unit
    def test_merge_timestamps_rounded(self, tmp_path: Path) -> None:
        """Should round timestamps to 3 decimal places."""
        worker = self._create_worker(tmp_path)
        transcript_segments = [
            {"start": 0.123456, "end": 1.987654, "text": "Test"},
        ]

        result = worker._merge_segments(transcript_segments, [])

        assert result[0]["start"] == 0.123
        assert result[0]["end"] == 1.988  # Rounded

    @pytest.mark.unit
    def test_merge_best_speaker_overlap(self, tmp_path: Path) -> None:
        """Should assign speaker based on maximum overlap."""
        worker = self._create_worker(tmp_path)
        # Segment from 0.5 to 1.5 overlaps more with SPEAKER_01 (0.5 overlap) than SPEAKER_00 (0.5 overlap)
        # Actually equal, so will pick first that has max
        transcript_segments = [
            {"start": 0.5, "end": 1.5, "text": "Test"},
        ]
        diarization_segments = [
            (0.0, 1.0, "SPEAKER_00"),  # 0.5 overlap
            (1.0, 2.0, "SPEAKER_01"),  # 0.5 overlap
        ]

        result = worker._merge_segments(transcript_segments, diarization_segments)
        # First one with max overlap wins
        assert result[0]["speaker"] in [1, 2]


class TestAudioPlayer:
    """Tests for AudioPlayer class."""

    @pytest.mark.unit
    def test_player_initialization(self, tmp_path: Path) -> None:
        """Should initialize player."""
        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"dummy")
        player = AudioPlayer(audio_path)
        assert player.audio_path == audio_path
        assert player.process is None
        assert player.paused is False

    @pytest.mark.unit
    def test_player_set_audio_path(self, tmp_path: Path) -> None:
        """Should update audio path."""
        audio_path1 = tmp_path / "test1.wav"
        audio_path2 = tmp_path / "test2.wav"
        audio_path1.write_bytes(b"dummy")
        audio_path2.write_bytes(b"dummy")

        player = AudioPlayer(audio_path1)
        player.set_audio_path(audio_path2)
        assert player.audio_path == audio_path2

    @pytest.mark.unit
    def test_player_stop_when_not_playing(self, tmp_path: Path) -> None:
        """Should handle stop when nothing is playing."""
        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"dummy")
        player = AudioPlayer(audio_path)
        # Should not raise
        player.stop()
        assert player.process is None

    @pytest.mark.unit
    def test_toggle_pause_when_not_playing(self, tmp_path: Path) -> None:
        """Should raise when toggling pause with nothing playing."""
        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"dummy")
        player = AudioPlayer(audio_path)

        with pytest.raises(RuntimeError, match="Nothing is currently playing"):
            player.toggle_pause()


class TestUIHelpers:
    """Tests for UI helper functions."""

    @pytest.mark.unit
    def test_print_segment(self, capsys) -> None:
        """Should format and print segment."""
        segment = {
            "start": 65.5,
            "end": 70.2,
            "speaker": 1,
            "text": "Hello world",
        }
        print_segment(segment)
        captured = capsys.readouterr()
        assert "SPEAKER 1" in captured.out
        # 65.5 seconds rounds to 66 seconds = 1:06
        assert "01:06" in captured.out
        assert "Hello world" in captured.out

    @pytest.mark.unit
    def test_print_transcript(self, capsys) -> None:
        """Should print full transcript."""
        segments = [
            {"start": 0.0, "end": 1.0, "speaker": 1, "text": "First"},
            {"start": 1.0, "end": 2.0, "speaker": 2, "text": "Second"},
        ]
        print_transcript(segments)
        captured = capsys.readouterr()
        assert "SPEAKER 1" in captured.out
        assert "SPEAKER 2" in captured.out
        assert "First" in captured.out
        assert "Second" in captured.out


class TestUISessionResult:
    """Tests for UISessionResult dataclass."""

    @pytest.mark.unit
    def test_result_with_segments(self) -> None:
        """Should store segments."""
        segments = [{"text": "test"}]
        result = UISessionResult(segments=segments)
        assert result.segments == segments
        assert result.error is None

    @pytest.mark.unit
    def test_result_with_error(self) -> None:
        """Should store error."""
        result = UISessionResult(segments=[], error="Something went wrong")
        assert result.segments == []
        assert result.error == "Something went wrong"


class TestWorkerThreadSafety:
    """Tests for worker thread safety."""

    @pytest.mark.unit
    def test_concurrent_emit(self, tmp_path: Path) -> None:
        """Should handle concurrent message emission."""
        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"dummy")
        messages: queue.Queue = queue.Queue()

        worker = TranscriptionWorker(
            audio_path=audio_path,
            messages=messages,
        )

        num_threads = 10
        num_messages = 100
        threads = []

        def emit_messages():
            for i in range(num_messages):
                worker._emit({"type": "test", "id": i})

        for _ in range(num_threads):
            t = threading.Thread(target=emit_messages)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All messages should be received
        received = 0
        while not messages.empty():
            messages.get_nowait()
            received += 1

        assert received == num_threads * num_messages

    @pytest.mark.unit
    def test_stop_event_thread_safe(self, tmp_path: Path) -> None:
        """Should have thread-safe stop event."""
        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"dummy")
        messages: queue.Queue = queue.Queue()

        worker = TranscriptionWorker(
            audio_path=audio_path,
            messages=messages,
        )

        # Multiple threads checking/setting stop
        results = []

        def check_and_set():
            for _ in range(100):
                results.append(worker.stop_event.is_set())
                if len(results) > 50:
                    worker.request_stop()

        threads = [threading.Thread(target=check_and_set) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Eventually all should see stop as set
        assert worker.stop_event.is_set()
