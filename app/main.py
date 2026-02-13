"""Backward-compatibility shim. Import from specific modules instead."""
from app.cache import CACHE_SCHEMA_VERSION, compute_file_hash
from app.cli import build_parser, main
from app.player import AudioPlayer
from app.transcribe import (
    ALLOWED_AUDIO_EXTENSIONS,
    VOXTRAL_NATIVE_FORMATS,
    convert_to_mp3,
    transcribe,
    validate_audio_path,
)
from app.ui import (
    format_timestamp,
    print_transcript,
    render_screen,
    run_transcribe_only,
    run_ui,
    safe_addstr,
)
from app.worker import TranscriptionWorker

__all__ = [
    "ALLOWED_AUDIO_EXTENSIONS",
    "AudioPlayer",
    "CACHE_SCHEMA_VERSION",
    "TranscriptionWorker",
    "VOXTRAL_NATIVE_FORMATS",
    "build_parser",
    "compute_file_hash",
    "convert_to_mp3",
    "format_timestamp",
    "main",
    "print_transcript",
    "render_screen",
    "run_transcribe_only",
    "run_ui",
    "safe_addstr",
    "transcribe",
    "validate_audio_path",
]

if __name__ == "__main__":
    import sys
    sys.exit(main())
