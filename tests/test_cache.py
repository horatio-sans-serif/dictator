"""Tests for caching logic."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.cache import (
    CACHE_SCHEMA_VERSION,
    compute_file_hash,
    load_cache,
)


class TestComputeFileHash:
    def test_consistent_hash(self, temp_audio_file):
        h1 = compute_file_hash(temp_audio_file)
        h2 = compute_file_hash(temp_audio_file)
        assert h1 == h2

    def test_different_files_different_hash(self, tmp_path):
        f1 = tmp_path / "a.wav"
        f2 = tmp_path / "b.wav"
        f1.write_bytes(b"content1")
        f2.write_bytes(b"content2")
        assert compute_file_hash(f1) != compute_file_hash(f2)

    def test_hash_is_sha256(self, temp_audio_file):
        h = compute_file_hash(temp_audio_file)
        assert len(h) == 64


class TestCacheLoading:
    def test_valid_cache_loaded(self, tmp_path, sample_segments):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")
        audio_hash = compute_file_hash(audio_file)

        cache_file = tmp_path / f"{audio_hash}.json"
        cache_data = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "audio_hash": audio_hash,
            "segments": sample_segments,
        }
        cache_file.write_text(json.dumps(cache_data))

        result = load_cache(cache_file, audio_hash)
        assert result == sample_segments

    def test_cache_version_mismatch(self, tmp_path, sample_segments):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")
        audio_hash = compute_file_hash(audio_file)

        cache_file = tmp_path / "cache.json"
        cache_data = {
            "schema_version": CACHE_SCHEMA_VERSION - 1,
            "audio_hash": audio_hash,
            "segments": sample_segments,
        }
        cache_file.write_text(json.dumps(cache_data))

        result = load_cache(cache_file, audio_hash)
        assert result is None

    def test_cache_hash_mismatch(self, tmp_path, sample_segments):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"fake audio data")

        cache_file = tmp_path / "cache.json"
        cache_data = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "audio_hash": "wrong_hash",
            "segments": sample_segments,
        }
        cache_file.write_text(json.dumps(cache_data))

        result = load_cache(cache_file, "correct_hash")
        assert result is None

    def test_corrupt_cache(self, tmp_path):
        cache_file = tmp_path / "cache.json"
        cache_file.write_text("not valid json{{{")

        result = load_cache(cache_file, "anyhash")
        assert result is None

    def test_missing_cache(self, tmp_path):
        cache_file = tmp_path / "nonexistent.json"

        result = load_cache(cache_file, "anyhash")
        assert result is None
