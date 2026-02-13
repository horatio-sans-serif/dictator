"""Tests for CLI argument parsing and main entrypoint."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from app.cli import build_parser, main
from app.transcribe import validate_audio_path
from app.ui import format_timestamp


class TestBuildParser:
    def test_audio_arg_optional(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.audio is None

    def test_audio_arg_provided(self):
        parser = build_parser()
        args = parser.parse_args(["test.mp3"])
        assert args.audio == "test.mp3"

    def test_transcribe_only_flag(self):
        parser = build_parser()
        args = parser.parse_args(["-T", "test.mp3"])
        assert args.transcribe_only is True

    def test_serve_flag(self):
        parser = build_parser()
        args = parser.parse_args(["--serve"])
        assert args.serve is True

    def test_port_default(self):
        parser = build_parser()
        args = parser.parse_args(["--serve"])
        assert args.port == 8377

    def test_port_custom(self):
        parser = build_parser()
        args = parser.parse_args(["--serve", "--port", "9000"])
        assert args.port == 9000


class TestValidateAudioPath:
    def test_valid_wav(self, temp_audio_file):
        result = validate_audio_path(temp_audio_file)
        assert result == temp_audio_file.resolve()

    def test_nonexistent_file(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            validate_audio_path(tmp_path / "nope.wav")

    def test_not_a_file(self, tmp_path):
        with pytest.raises(ValueError, match="not a regular file"):
            validate_audio_path(tmp_path)

    def test_invalid_extension(self, tmp_path):
        bad = tmp_path / "test.xyz"
        bad.write_bytes(b"data")
        with pytest.raises(ValueError, match="Invalid audio file extension"):
            validate_audio_path(bad)

    def test_webm_allowed(self, tmp_path):
        webm = tmp_path / "test.webm"
        webm.write_bytes(b"data")
        result = validate_audio_path(webm)
        assert result == webm.resolve()


class TestFormatTimestamp:
    def test_zero(self):
        assert format_timestamp(0) == "00:00"

    def test_seconds(self):
        assert format_timestamp(45) == "00:45"

    def test_minutes(self):
        assert format_timestamp(125) == "02:05"

    def test_hours(self):
        assert format_timestamp(3661) == "01:01:01"

    def test_negative(self):
        assert format_timestamp(-5) == "00:00"


class TestMainServe:
    def test_serve_calls_web_server(self):
        mock_web = MagicMock()
        with patch.dict("sys.modules", {"app.web": mock_web}):
            main(["--serve"])
            mock_web.run_server.assert_called_once_with(port=8377)
