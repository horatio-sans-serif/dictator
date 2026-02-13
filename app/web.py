"""Bottle web server for browser-based transcription."""
from __future__ import annotations

import tempfile
from pathlib import Path

import bottle

from app.transcribe import convert_to_mp3, transcribe

STATIC_DIR = Path(__file__).resolve().parent / "static"


@bottle.route("/")
def index():
    return bottle.static_file("index.html", root=str(STATIC_DIR))


@bottle.route("/static/<filepath:path>")
def static(filepath):
    return bottle.static_file(filepath, root=str(STATIC_DIR))


@bottle.post("/transcribe")
def transcribe_upload():
    upload = bottle.request.files.get("file")
    if not upload:
        bottle.response.status = 400
        return {"error": "No file uploaded"}

    suffix = Path(upload.filename or "audio.webm").suffix.lower() or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        upload.save(f)
        tmp_path = Path(f.name)

    try:
        audio_path = convert_to_mp3(tmp_path)
        try:
            segments = transcribe(audio_path)
        finally:
            if audio_path != tmp_path:
                audio_path.unlink(missing_ok=True)
        return {"segments": segments}
    except Exception as exc:
        bottle.response.status = 500
        return {"error": str(exc)}
    finally:
        tmp_path.unlink(missing_ok=True)


def run_server(port: int = 8377) -> None:
    print(f"Dictator web UI: http://localhost:{port}")
    bottle.run(host="0.0.0.0", port=port, quiet=False)
