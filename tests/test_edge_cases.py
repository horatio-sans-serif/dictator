"""Comprehensive edge case tests for security, performance, and input validation."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import (
    AudioProcessingError,
    _file_hash_cache,
    _wrap_text_cached,
    apply_bandpass_filter,
    compute_file_hash,
    get_auth_token,
    validate_audio_path,
)


class TestValidateAudioPath:
    """Tests for validate_audio_path security function."""

    @pytest.mark.unit
    def test_non_existent_file_raises_error(self, tmp_path: Path) -> None:
        """Should raise ValueError for non-existent file."""
        non_existent = tmp_path / "does_not_exist.wav"
        with pytest.raises(ValueError, match="Audio file does not exist"):
            validate_audio_path(non_existent)

    @pytest.mark.unit
    def test_directory_path_raises_error(self, tmp_path: Path) -> None:
        """Should raise ValueError for directory path."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        with pytest.raises(ValueError, match="Path is not a regular file"):
            validate_audio_path(test_dir)

    @pytest.mark.unit
    def test_invalid_extension_raises_error(self, tmp_path: Path) -> None:
        """Should raise ValueError for invalid file extension."""
        invalid_file = tmp_path / "test.txt"
        invalid_file.touch()
        with pytest.raises(ValueError, match="Invalid audio file extension"):
            validate_audio_path(invalid_file)

    @pytest.mark.unit
    def test_valid_wav_file_returns_resolved_path(self, tmp_path: Path) -> None:
        """Should return resolved absolute path for valid .wav file."""
        valid_file = tmp_path / "test.wav"
        valid_file.touch()
        result = validate_audio_path(valid_file)
        assert result.is_absolute()
        assert result == valid_file.resolve()
        assert result.exists()

    @pytest.mark.unit
    def test_valid_m4a_file_accepted(self, tmp_path: Path) -> None:
        """Should accept valid .m4a file extension."""
        valid_file = tmp_path / "test.m4a"
        valid_file.touch()
        result = validate_audio_path(valid_file)
        assert result.is_absolute()
        assert result.exists()

    @pytest.mark.unit
    def test_valid_mp3_file_accepted(self, tmp_path: Path) -> None:
        """Should accept valid .mp3 file extension."""
        valid_file = tmp_path / "test.mp3"
        valid_file.touch()
        result = validate_audio_path(valid_file)
        assert result.is_absolute()
        assert result.exists()

    @pytest.mark.unit
    def test_valid_flac_file_accepted(self, tmp_path: Path) -> None:
        """Should accept valid .flac file extension."""
        valid_file = tmp_path / "test.flac"
        valid_file.touch()
        result = validate_audio_path(valid_file)
        assert result.is_absolute()
        assert result.exists()

    @pytest.mark.unit
    def test_case_insensitive_extension(self, tmp_path: Path) -> None:
        """Should handle uppercase extensions."""
        valid_file = tmp_path / "test.WAV"
        valid_file.touch()
        result = validate_audio_path(valid_file)
        assert result.is_absolute()
        assert result.exists()

    @pytest.mark.unit
    def test_resolves_relative_path(self, tmp_path: Path) -> None:
        """Should resolve relative paths to absolute."""
        valid_file = tmp_path / "test.wav"
        valid_file.touch()
        # Change to temp directory and use relative path
        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            result = validate_audio_path(Path("test.wav"))
            assert result.is_absolute()
            assert result == valid_file.resolve()
        finally:
            os.chdir(original_cwd)

    @pytest.mark.unit
    def test_expands_user_path(self, tmp_path: Path) -> None:
        """Should expand ~ in paths."""
        # Create a file in a known location
        valid_file = tmp_path / "test.wav"
        valid_file.touch()

        # Mock expanduser to return our test path
        with patch.object(Path, "expanduser", return_value=valid_file):
            result = validate_audio_path(Path("~/test.wav"))
            assert result.is_absolute()


class TestGetAuthToken:
    """Tests for get_auth_token security function."""

    @pytest.mark.unit
    def test_returns_none_when_env_var_not_set(self) -> None:
        """Should return None when environment variable is not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = get_auth_token("NONEXISTENT_VAR")
            assert result is None

    @pytest.mark.unit
    def test_returns_none_for_empty_token(self) -> None:
        """Should return None for empty string token."""
        with patch.dict(os.environ, {"TEST_TOKEN": ""}):
            result = get_auth_token("TEST_TOKEN")
            assert result is None

    @pytest.mark.unit
    def test_returns_none_for_whitespace_token(self) -> None:
        """Should return None for whitespace-only token."""
        with patch.dict(os.environ, {"TEST_TOKEN": "   \t\n   "}):
            result = get_auth_token("TEST_TOKEN")
            assert result is None

    @pytest.mark.unit
    def test_returns_stripped_token_for_valid_token(self) -> None:
        """Should return stripped token for valid token."""
        with patch.dict(os.environ, {"TEST_TOKEN": "  valid_token_123  "}):
            result = get_auth_token("TEST_TOKEN")
            assert result == "valid_token_123"

    @pytest.mark.unit
    def test_returns_token_without_leading_trailing_whitespace(self) -> None:
        """Should strip leading and trailing whitespace."""
        with patch.dict(os.environ, {"TEST_TOKEN": "\n\tmy_secret_key\n\t"}):
            result = get_auth_token("TEST_TOKEN")
            assert result == "my_secret_key"
            assert not result.startswith("\n")
            assert not result.endswith("\t")

    @pytest.mark.unit
    def test_preserves_internal_whitespace(self) -> None:
        """Should preserve whitespace within the token."""
        with patch.dict(os.environ, {"TEST_TOKEN": "  token with spaces  "}):
            result = get_auth_token("TEST_TOKEN")
            assert result == "token with spaces"

    @pytest.mark.unit
    def test_default_env_var_name(self) -> None:
        """Should use PYANNOTE_AUTH_TOKEN as default."""
        with patch.dict(os.environ, {"PYANNOTE_AUTH_TOKEN": "default_token"}):
            result = get_auth_token()
            assert result == "default_token"


class TestComputeFileHash:
    """Tests for compute_file_hash performance caching."""

    @pytest.mark.unit
    def test_returns_same_hash_for_same_file(self, temp_audio_file: Path) -> None:
        """Should return same hash for same file when called multiple times."""
        # Clear cache to ensure clean test
        _file_hash_cache.clear()

        hash1 = compute_file_hash(temp_audio_file)
        hash2 = compute_file_hash(temp_audio_file)
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex digest length

    @pytest.mark.unit
    def test_uses_cache_on_repeated_calls(self, temp_audio_file: Path) -> None:
        """Should use cache on repeated calls (check _file_hash_cache)."""
        # Clear cache to ensure clean test
        _file_hash_cache.clear()

        # First call should populate cache
        hash1 = compute_file_hash(temp_audio_file)

        # Verify cache contains the entry
        stat = temp_audio_file.stat()
        cache_key = (str(temp_audio_file), stat.st_mtime, stat.st_size)
        assert cache_key in _file_hash_cache
        assert _file_hash_cache[cache_key] == hash1

        # Second call should use cache
        hash2 = compute_file_hash(temp_audio_file)
        assert hash2 == hash1

        # Cache should still contain only one entry for this file
        assert _file_hash_cache[cache_key] == hash1

    @pytest.mark.unit
    def test_different_files_have_different_hashes(self, tmp_path: Path, sample_rate: int) -> None:
        """Should return different hashes for different files."""
        import soundfile as sf

        _file_hash_cache.clear()

        # Create two different audio files
        file1 = tmp_path / "audio1.wav"
        file2 = tmp_path / "audio2.wav"

        audio1 = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sample_rate))
        audio2 = np.sin(2 * np.pi * 880 * np.linspace(0, 1, sample_rate))

        sf.write(file1, audio1, sample_rate)
        sf.write(file2, audio2, sample_rate)

        hash1 = compute_file_hash(file1)
        hash2 = compute_file_hash(file2)

        assert hash1 != hash2

    @pytest.mark.unit
    def test_cache_limit_enforced(self, tmp_path: Path) -> None:
        """Should limit cache size by removing oldest entries."""
        import soundfile as sf

        _file_hash_cache.clear()

        # Create more files than the cache limit (100)
        # We'll create 105 files to test eviction
        files = []
        for i in range(105):
            file_path = tmp_path / f"audio_{i}.wav"
            # Create minimal valid audio
            audio = np.array([0.1, 0.2, 0.1], dtype=np.float32)
            sf.write(file_path, audio, 16000)
            files.append(file_path)

        # Hash all files
        for file_path in files:
            compute_file_hash(file_path)

        # Cache should not exceed max size
        assert len(_file_hash_cache) <= 100


class TestWrapTextCached:
    """Tests for _wrap_text_cached performance caching."""

    @pytest.mark.unit
    def test_returns_consistent_results(self) -> None:
        """Should return consistent results for same input."""
        text = "This is a test string that needs to be wrapped at a certain width."
        width = 40

        result1 = _wrap_text_cached(text, width)
        result2 = _wrap_text_cached(text, width)

        assert result1 == result2
        assert isinstance(result1, tuple)
        assert all(isinstance(line, str) for line in result1)

    @pytest.mark.unit
    def test_caches_results(self) -> None:
        """Should cache results (check cache_info)."""
        # Clear cache first
        _wrap_text_cached.cache_clear()

        text = "This is a unique test string for cache verification."
        width = 50

        # First call - cache miss
        info_before = _wrap_text_cached.cache_info()
        result1 = _wrap_text_cached(text, width)
        info_after_first = _wrap_text_cached.cache_info()

        # Should have one cache miss
        assert info_after_first.misses == info_before.misses + 1

        # Second call - cache hit
        result2 = _wrap_text_cached(text, width)
        info_after_second = _wrap_text_cached.cache_info()

        # Should have one cache hit
        assert info_after_second.hits == info_after_first.hits + 1
        assert result1 == result2

    @pytest.mark.unit
    def test_different_widths_cached_separately(self) -> None:
        """Should cache different widths separately."""
        _wrap_text_cached.cache_clear()

        text = "Same text different widths"

        result1 = _wrap_text_cached(text, 20)
        result2 = _wrap_text_cached(text, 40)

        # Results should be different
        assert result1 != result2

        # Both should be cached
        info = _wrap_text_cached.cache_info()
        assert info.currsize >= 2

    @pytest.mark.unit
    def test_empty_string_handling(self) -> None:
        """Should handle empty strings."""
        result = _wrap_text_cached("", 80)
        assert result == ("",)


class TestApplyBandpassFilterValidation:
    """Tests for apply_bandpass_filter input validation."""

    @pytest.mark.unit
    def test_zero_sample_rate_raises_error(self, short_audio: np.ndarray) -> None:
        """Should raise ValueError for zero sample_rate."""
        with pytest.raises(ValueError, match="sample_rate must be between"):
            apply_bandpass_filter(short_audio, 0)

    @pytest.mark.unit
    def test_negative_sample_rate_raises_error(self, short_audio: np.ndarray) -> None:
        """Should raise ValueError for negative sample_rate."""
        with pytest.raises(ValueError, match="sample_rate must be between"):
            apply_bandpass_filter(short_audio, -1000)

    @pytest.mark.unit
    def test_too_low_sample_rate_raises_error(self, short_audio: np.ndarray) -> None:
        """Should raise ValueError for sample_rate below minimum."""
        with pytest.raises(ValueError, match="sample_rate must be between"):
            apply_bandpass_filter(short_audio, 500)  # Below MIN_SAMPLE_RATE (1000)

    @pytest.mark.unit
    def test_too_high_sample_rate_raises_error(self, short_audio: np.ndarray) -> None:
        """Should raise ValueError for sample_rate above maximum."""
        with pytest.raises(ValueError, match="sample_rate must be between"):
            apply_bandpass_filter(short_audio, 200000)  # Above MAX_SAMPLE_RATE (192000)

    @pytest.mark.unit
    def test_minimum_valid_sample_rate(self, short_audio: np.ndarray) -> None:
        """Should accept minimum valid sample_rate."""
        # MIN_SAMPLE_RATE is 1000
        result = apply_bandpass_filter(short_audio, 1000)
        assert result is not None
        assert len(result) == len(short_audio)

    @pytest.mark.unit
    def test_maximum_valid_sample_rate(self, short_audio: np.ndarray) -> None:
        """Should accept maximum valid sample_rate."""
        # MAX_SAMPLE_RATE is 192000
        result = apply_bandpass_filter(short_audio, 192000)
        assert result is not None
        assert len(result) == len(short_audio)

    @pytest.mark.unit
    def test_typical_sample_rate_works(self, short_audio: np.ndarray) -> None:
        """Should work with typical sample rate."""
        result = apply_bandpass_filter(short_audio, 16000)
        assert result is not None
        assert len(result) == len(short_audio)


class TestAudioProcessingError:
    """Tests for AudioProcessingError exception class."""

    @pytest.mark.unit
    def test_can_be_raised(self) -> None:
        """Should be able to raise AudioProcessingError."""
        with pytest.raises(AudioProcessingError):
            raise AudioProcessingError("Test error message")

    @pytest.mark.unit
    def test_can_be_caught(self) -> None:
        """Should be able to catch AudioProcessingError."""
        try:
            raise AudioProcessingError("Test error")
        except AudioProcessingError as e:
            assert str(e) == "Test error"

    @pytest.mark.unit
    def test_inherits_from_exception(self) -> None:
        """Should inherit from Exception."""
        assert issubclass(AudioProcessingError, Exception)

    @pytest.mark.unit
    def test_can_be_caught_as_exception(self) -> None:
        """Should be catchable as generic Exception."""
        try:
            raise AudioProcessingError("Generic catch test")
        except Exception as e:
            assert isinstance(e, AudioProcessingError)
            assert str(e) == "Generic catch test"

    @pytest.mark.unit
    def test_preserves_error_message(self) -> None:
        """Should preserve error message."""
        message = "Detailed error description with context"
        error = AudioProcessingError(message)
        assert str(error) == message

    @pytest.mark.unit
    def test_can_be_raised_without_message(self) -> None:
        """Should allow raising without message."""
        with pytest.raises(AudioProcessingError):
            raise AudioProcessingError()
