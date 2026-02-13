"""Tests for the Voxtral transcription function."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from app.transcribe import transcribe, convert_to_mp3
from tests.conftest import mock_voxtral_response


def _patch_mistral(mock_client):
    """Context manager to patch the lazy `from mistralai import Mistral`."""
    fake_mod = MagicMock()
    fake_mod.Mistral = MagicMock(return_value=mock_client)
    return patch.dict("sys.modules", {"mistralai": fake_mod})


class TestTranscribe:
    def test_missing_api_key(self, temp_audio_file):
        fake_mod = MagicMock()
        with patch.dict("sys.modules", {"mistralai": fake_mod}):
            with patch.dict(os.environ, {"MISTRAL_API_KEY": ""}, clear=False):
                with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
                    transcribe(temp_audio_file)

    def test_successful_transcription(self, temp_audio_file):
        mock_response = mock_voxtral_response()
        mock_client = MagicMock()
        mock_client.audio.transcriptions.complete.return_value = mock_response

        with _patch_mistral(mock_client):
            with patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"}, clear=False):
                segments = transcribe(temp_audio_file)

        assert len(segments) == 2
        assert segments[0]["text"] == "Hello world"
        assert segments[0]["speaker"] == 1
        assert segments[1]["text"] == "Hi there"
        assert segments[1]["speaker"] == 2

    def test_speaker_ids_sequential(self, temp_audio_file):
        segs = [
            MagicMock(start=0.0, end=1.0, text="A", speaker_id="X"),
            MagicMock(start=1.0, end=2.0, text="B", speaker_id="Y"),
            MagicMock(start=2.0, end=3.0, text="C", speaker_id="X"),
        ]
        mock_response = mock_voxtral_response(segs)
        mock_client = MagicMock()
        mock_client.audio.transcriptions.complete.return_value = mock_response

        with _patch_mistral(mock_client):
            with patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"}, clear=False):
                segments = transcribe(temp_audio_file)

        assert segments[0]["speaker"] == 1
        assert segments[1]["speaker"] == 2
        assert segments[2]["speaker"] == 1

    def test_empty_segments_raises(self, temp_audio_file):
        mock_response = mock_voxtral_response([])
        mock_client = MagicMock()
        mock_client.audio.transcriptions.complete.return_value = mock_response

        with _patch_mistral(mock_client):
            with patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"}, clear=False):
                with pytest.raises(RuntimeError, match="did not return"):
                    transcribe(temp_audio_file)

    def test_blank_text_segments_skipped(self, temp_audio_file):
        segs = [
            MagicMock(start=0.0, end=1.0, text="  ", speaker_id="S0"),
            MagicMock(start=1.0, end=2.0, text="Real text", speaker_id="S0"),
        ]
        mock_response = mock_voxtral_response(segs)
        mock_client = MagicMock()
        mock_client.audio.transcriptions.complete.return_value = mock_response

        with _patch_mistral(mock_client):
            with patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"}, clear=False):
                segments = transcribe(temp_audio_file)

        assert len(segments) == 1
        assert segments[0]["text"] == "Real text"


class TestConvertToMp3:
    def test_mp3_not_converted(self, tmp_path):
        mp3 = tmp_path / "test.mp3"
        mp3.write_bytes(b"fake mp3")
        assert convert_to_mp3(mp3) == mp3

    def test_wav_not_converted(self, tmp_path):
        wav = tmp_path / "test.wav"
        wav.write_bytes(b"fake wav")
        assert convert_to_mp3(wav) == wav

    def test_m4a_needs_conversion(self, tmp_path):
        m4a = tmp_path / "test.m4a"
        m4a.write_bytes(b"fake m4a")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = convert_to_mp3(m4a)
            assert result.suffix == ".mp3"
            mock_run.assert_called_once()
