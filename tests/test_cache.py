"""Unit tests for cache validation and management."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import (
    CACHE_SCHEMA_VERSION,
    ENHANCEMENT_VERSION,
    TranscriptionWorker,
)


class TestCacheValidation:
    """Tests for cache validation logic."""

    def _create_worker(
        self,
        tmp_path: Path,
        model_name: str = "tiny",
        preset_name: str = "general",
        enhancement_enabled: bool = True,
        enhancer_backend: str = "soap",
    ) -> TranscriptionWorker:
        """Create a worker for testing cache validation."""
        import queue
        audio_file = tmp_path / "dummy.wav"
        audio_file.write_bytes(b"dummy")
        messages = queue.Queue()
        return TranscriptionWorker(
            audio_path=audio_file,
            messages=messages,
            model_name=model_name,
            preset_name=preset_name,
            enhancement_enabled=enhancement_enabled,
            enhancer_backend=enhancer_backend,
        )

    @pytest.mark.unit
    def test_valid_cache_accepted(self, tmp_path: Path, valid_cache_data: Dict) -> None:
        """Should accept valid cache data."""
        worker = self._create_worker(tmp_path)
        valid, reason = worker._cache_is_valid(valid_cache_data, "abc123")
        assert valid is True
        assert reason == ""

    @pytest.mark.unit
    def test_schema_version_mismatch(self, tmp_path: Path, valid_cache_data: Dict) -> None:
        """Should reject cache with wrong schema version."""
        worker = self._create_worker(tmp_path)
        valid_cache_data["schema_version"] = CACHE_SCHEMA_VERSION - 1
        valid, reason = worker._cache_is_valid(valid_cache_data, "abc123")
        assert valid is False
        assert "schema version" in reason.lower()

    @pytest.mark.unit
    def test_audio_hash_mismatch(self, tmp_path: Path, valid_cache_data: Dict) -> None:
        """Should reject cache with wrong audio hash."""
        worker = self._create_worker(tmp_path)
        valid, reason = worker._cache_is_valid(valid_cache_data, "different_hash")
        assert valid is False
        assert "hash" in reason.lower()

    @pytest.mark.unit
    def test_model_mismatch(self, tmp_path: Path, valid_cache_data: Dict) -> None:
        """Should reject cache with different model."""
        worker = self._create_worker(tmp_path, model_name="large-v3")
        valid, reason = worker._cache_is_valid(valid_cache_data, "abc123")
        assert valid is False
        assert "model" in reason.lower()

    @pytest.mark.unit
    def test_language_mismatch(self, tmp_path: Path, valid_cache_data: Dict) -> None:
        """Should reject cache with different language."""
        worker = self._create_worker(tmp_path)
        valid_cache_data["language"] = "de"
        valid, reason = worker._cache_is_valid(valid_cache_data, "abc123")
        assert valid is False
        assert "language" in reason.lower()

    @pytest.mark.unit
    def test_enhancement_version_mismatch(self, tmp_path: Path, valid_cache_data: Dict) -> None:
        """Should reject cache with different enhancement version."""
        worker = self._create_worker(tmp_path)
        valid_cache_data["enhancement_version"] = ENHANCEMENT_VERSION - 1
        valid, reason = worker._cache_is_valid(valid_cache_data, "abc123")
        assert valid is False
        assert "enhancement" in reason.lower()

    @pytest.mark.unit
    def test_enhancement_enabled_mismatch(self, tmp_path: Path, valid_cache_data: Dict) -> None:
        """Should reject cache when enhancement enabled flag differs."""
        worker = self._create_worker(tmp_path, enhancement_enabled=False)
        valid, reason = worker._cache_is_valid(valid_cache_data, "abc123")
        assert valid is False
        assert "enhancement" in reason.lower()

    @pytest.mark.unit
    def test_preset_mismatch(self, tmp_path: Path, valid_cache_data: Dict) -> None:
        """Should reject cache with different preset."""
        worker = self._create_worker(tmp_path, preset_name="iphone")
        valid, reason = worker._cache_is_valid(valid_cache_data, "abc123")
        assert valid is False
        assert "preset" in reason.lower()

    @pytest.mark.unit
    def test_backend_mismatch(self, tmp_path: Path, valid_cache_data: Dict) -> None:
        """Should reject cache with different backend."""
        worker = self._create_worker(tmp_path, enhancer_backend="resemble")
        valid, reason = worker._cache_is_valid(valid_cache_data, "abc123")
        assert valid is False
        assert "backend" in reason.lower()

    @pytest.mark.unit
    def test_missing_model(self, tmp_path: Path, valid_cache_data: Dict) -> None:
        """Should reject cache without model field."""
        worker = self._create_worker(tmp_path)
        del valid_cache_data["model"]
        valid, reason = worker._cache_is_valid(valid_cache_data, "abc123")
        assert valid is False
        assert "missing" in reason.lower() or "metadata" in reason.lower()

    @pytest.mark.unit
    def test_missing_language(self, tmp_path: Path, valid_cache_data: Dict) -> None:
        """Should reject cache without language field."""
        worker = self._create_worker(tmp_path)
        del valid_cache_data["language"]
        valid, reason = worker._cache_is_valid(valid_cache_data, "abc123")
        assert valid is False

    @pytest.mark.unit
    def test_missing_enhancement_metadata(self, tmp_path: Path, valid_cache_data: Dict) -> None:
        """Should reject cache without enhancement metadata."""
        worker = self._create_worker(tmp_path)
        del valid_cache_data["enhancement_version"]
        valid, reason = worker._cache_is_valid(valid_cache_data, "abc123")
        assert valid is False
        assert "enhancement" in reason.lower()

    @pytest.mark.unit
    def test_non_dict_rejected(self, tmp_path: Path) -> None:
        """Should reject non-dict data."""
        worker = self._create_worker(tmp_path)
        valid, reason = worker._cache_is_valid("not a dict", "abc123")
        assert valid is False
        assert "malformed" in reason.lower()

    @pytest.mark.unit
    def test_enhancement_disabled_cache(self, tmp_path: Path) -> None:
        """Should validate cache when enhancement is disabled."""
        worker = self._create_worker(tmp_path, enhancement_enabled=False)
        cache_data = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "audio_hash": "abc123",
            "model": "tiny",
            "language": "en",
            "enhancement_version": 0,
            "enhancement_enabled": False,
            "enhancement_preset": "none",
            "enhancement_backend": "none",
            "segments": [],
        }
        valid, reason = worker._cache_is_valid(cache_data, "abc123")
        assert valid is True


class TestCacheLoading:
    """Tests for cache loading logic."""

    def _create_worker_for_load(self, tmp_path: Path) -> TranscriptionWorker:
        """Create worker for cache loading tests."""
        import queue
        audio_file = tmp_path / "dummy.wav"
        audio_file.write_bytes(b"dummy")
        messages = queue.Queue()
        return TranscriptionWorker(
            audio_path=audio_file,
            messages=messages,
            model_name="tiny",
        )

    @pytest.mark.unit
    def test_load_nonexistent_cache(self, tmp_path: Path) -> None:
        """Should return None for nonexistent cache file."""
        worker = self._create_worker_for_load(tmp_path)
        cache_file = tmp_path / "nonexistent.json"
        result = worker._try_load_cache(cache_file, "abc123")
        assert result is None

    @pytest.mark.unit
    def test_load_corrupt_json(self, tmp_path: Path) -> None:
        """Should return None for corrupt JSON."""
        worker = self._create_worker_for_load(tmp_path)
        cache_file = tmp_path / "corrupt.json"
        cache_file.write_text("not valid json {{{", encoding="utf-8")
        result = worker._try_load_cache(cache_file, "abc123")
        assert result is None

    @pytest.mark.unit
    def test_load_valid_cache(self, tmp_path: Path, valid_cache_data: Dict) -> None:
        """Should return segments for valid cache."""
        worker = self._create_worker_for_load(tmp_path)
        cache_file = tmp_path / "valid.json"
        cache_file.write_text(json.dumps(valid_cache_data), encoding="utf-8")
        result = worker._try_load_cache(cache_file, "abc123")
        assert result is not None
        assert len(result) == 2
        assert result[0]["text"] == "Hello world"

    @pytest.mark.unit
    def test_load_cache_missing_segments(self, tmp_path: Path, valid_cache_data: Dict) -> None:
        """Should return empty list if segments field is missing (defaults to [])."""
        worker = self._create_worker_for_load(tmp_path)
        del valid_cache_data["segments"]
        cache_file = tmp_path / "no_segments.json"
        cache_file.write_text(json.dumps(valid_cache_data), encoding="utf-8")
        result = worker._try_load_cache(cache_file, "abc123")
        # The code returns empty list when segments missing via .get("segments", [])
        # which is valid but empty
        assert result == []

    @pytest.mark.unit
    def test_load_cache_non_list_segments(self, tmp_path: Path, valid_cache_data: Dict) -> None:
        """Should return None if segments is not a list."""
        worker = self._create_worker_for_load(tmp_path)
        valid_cache_data["segments"] = "not a list"
        cache_file = tmp_path / "bad_segments.json"
        cache_file.write_text(json.dumps(valid_cache_data), encoding="utf-8")
        result = worker._try_load_cache(cache_file, "abc123")
        assert result is None


class TestCachePaths:
    """Tests for cache path generation."""

    @pytest.mark.unit
    def test_enhancement_paths_soap(self) -> None:
        """Should generate correct paths for soap backend."""
        from app.main import _enhancement_paths, ENHANCEMENT_VERSION
        enhanced, compare = _enhancement_paths("abc123", "soap", "general", create_compare=True)
        assert "abc123" in str(enhanced)
        assert "soap" in str(enhanced)
        assert "general" in str(enhanced)
        assert f"v{ENHANCEMENT_VERSION}" in str(enhanced)
        assert compare is not None
        assert "compare" in str(compare)

    @pytest.mark.unit
    def test_enhancement_paths_resemble(self) -> None:
        """Should generate correct paths for resemble backend."""
        from app.main import _enhancement_paths
        enhanced, compare = _enhancement_paths("abc123", "resemble", "iphone", create_compare=False)
        assert "resemble" in str(enhanced)
        assert "iphone" in str(enhanced)
        assert compare is None

    @pytest.mark.unit
    def test_enhancement_paths_no_compare(self) -> None:
        """Should return None compare path when not requested."""
        from app.main import _enhancement_paths
        enhanced, compare = _enhancement_paths("abc123", "soap", "general", create_compare=False)
        assert enhanced is not None
        assert compare is None
