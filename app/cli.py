"""CLI entrypoint for Dictator."""
from __future__ import annotations

import argparse
import curses
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from app.transcribe import validate_audio_path
from app.ui import print_transcript, run_transcribe_only, run_ui


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dictator - Speech transcription with speaker diarization"
    )
    parser.add_argument(
        "audio",
        type=str,
        nargs="?",
        help="Path to the audio file to transcribe (reads from stdin if not provided)",
    )
    parser.add_argument(
        "-T",
        "--transcribe-only",
        action="store_true",
        help="Skip the curses UI and stream diarized transcript to stdout",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the web server for browser-based transcription",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8377,
        help="Port for the web server (default: 8377)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.serve:
        from app.web import run_server
        run_server(port=args.port)
        return 0

    audio_path: Path
    temp_file: Optional[Path] = None

    if args.audio:
        try:
            audio_path = validate_audio_path(Path(args.audio))
        except ValueError as e:
            parser.error(str(e))
    else:
        import atexit

        audio_data = sys.stdin.buffer.read()
        if not audio_data:
            parser.error("No audio data received from stdin")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            audio_path = Path(f.name)
            temp_file = audio_path

        print(
            f"Read {len(audio_data)} bytes from stdin to {audio_path}",
            file=sys.stderr,
        )

        atexit.register(lambda: temp_file and temp_file.unlink(missing_ok=True))

    if args.transcribe_only:
        return run_transcribe_only(audio_path)

    try:
        result = curses.wrapper(lambda stdscr: run_ui(stdscr, audio_path))
    except curses.error as exc:
        print(f"Curses error: {exc}", file=sys.stderr)
        return 1

    if result.error and not result.segments:
        print(f"Error: {result.error}", file=sys.stderr)
        return 1

    if result.segments:
        print_transcript(result.segments)
    else:
        print("No transcript segments produced.", file=sys.stderr)
        return 1

    return 0


def main_cli() -> None:
    sys.exit(main())
