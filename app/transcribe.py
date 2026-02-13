"""Voxtral API transcription with speaker diarization."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

ALLOWED_AUDIO_EXTENSIONS = frozenset(
    {".wav", ".m4a", ".mp3", ".flac", ".ogg", ".aac", ".wma", ".aiff", ".aif", ".webm"}
)

VOXTRAL_NATIVE_FORMATS = frozenset({".mp3", ".wav", ".flac", ".ogg", ".webm"})


def validate_audio_path(audio_path: Path) -> Path:
    resolved = audio_path.expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"Audio file does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Path is not a regular file: {resolved}")
    suffix = resolved.suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise ValueError(
            f"Invalid audio file extension '{suffix}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"
        )
    return resolved


def convert_to_mp3(audio_path: Path) -> Path:
    if audio_path.suffix.lower() in VOXTRAL_NATIVE_FORMATS:
        return audio_path
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to convert audio files")
    fd, out_name = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    out = Path(out_name)
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path), "-q:a", "2", str(out)],
        capture_output=True,
    )
    if result.returncode != 0:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr.decode()}")
    return out


def transcribe(audio_path: Path) -> List[Dict[str, object]]:
    from mistralai import Mistral

    api_key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY environment variable is required. "
            "Get one at https://console.mistral.ai/"
        )

    upload_path = convert_to_mp3(audio_path)
    try:
        client = Mistral(api_key=api_key)
        with open(upload_path, "rb") as f:
            response = client.audio.transcriptions.complete(
                model="voxtral-mini-latest",
                file={"content": f, "file_name": upload_path.name},
                diarize=True,
                timestamp_granularities=["segment"],
            )
    finally:
        if upload_path != audio_path:
            upload_path.unlink(missing_ok=True)

    speaker_map: Dict[str, int] = {}
    next_id = 1
    segments: List[Dict[str, object]] = []

    for seg in response.segments or []:
        text = (seg.text or "").strip()
        if not text:
            continue
        raw_speaker = seg.speaker_id or "SPEAKER_00"
        if raw_speaker not in speaker_map:
            speaker_map[raw_speaker] = next_id
            next_id += 1
        segments.append({
            "start": round(float(seg.start), 3),
            "end": round(float(seg.end), 3),
            "speaker": speaker_map[raw_speaker],
            "text": text,
        })

    if not segments:
        raise RuntimeError("Voxtral did not return any transcript segments")

    return segments
