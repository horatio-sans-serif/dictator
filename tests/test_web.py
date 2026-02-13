"""Tests for the web server endpoints."""
from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest
from webtest import TestApp
from bottle import default_app

import app.web  # triggers route registration


@pytest.fixture
def web_client():
    return TestApp(default_app())


class TestWebTranscribe:
    def test_index_returns_html(self, web_client):
        resp = web_client.get("/")
        assert resp.status_int == 200
        assert b"Dictator" in resp.body

    def test_transcribe_no_file(self, web_client):
        resp = web_client.post("/transcribe", expect_errors=True)
        assert resp.status_int == 400
        data = json.loads(resp.body)
        assert "error" in data

    def test_transcribe_success(self, web_client):
        sample_segments = [
            {"start": 0.0, "end": 1.5, "speaker": 1, "text": "Hello"},
        ]

        with patch("app.web.transcribe", return_value=sample_segments):
            with patch("app.web.convert_to_mp3", side_effect=lambda p: p):
                resp = web_client.post(
                    "/transcribe",
                    upload_files=[("file", "test.mp3", b"fake audio data")],
                )

        assert resp.status_int == 200
        data = json.loads(resp.body)
        assert "segments" in data
        assert len(data["segments"]) == 1
        assert data["segments"][0]["text"] == "Hello"

    def test_transcribe_error(self, web_client):
        with patch("app.web.transcribe", side_effect=RuntimeError("API error")):
            with patch("app.web.convert_to_mp3", side_effect=lambda p: p):
                resp = web_client.post(
                    "/transcribe",
                    upload_files=[("file", "test.mp3", b"fake audio data")],
                    expect_errors=True,
                )

        assert resp.status_int == 500
        data = json.loads(resp.body)
        assert "error" in data
        assert "API error" in data["error"]
