"""Curses terminal UI and CLI streaming mode."""
from __future__ import annotations

import curses
import queue
import sys
import textwrap
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.player import AudioPlayer
from app.worker import TranscriptionWorker


def format_timestamp(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def safe_addstr(
    window: "curses._CursesWindow", y: int, x: int, text: str, attr: int = 0
) -> None:
    height, width = window.getmaxyx()
    if y < 0 or y >= height or x >= width:
        return
    try:
        window.addnstr(y, x, text, max(0, width - x - 1), attr)
    except curses.error:
        pass


@dataclass
class UISessionResult:
    segments: List[Dict[str, object]]
    error: Optional[str] = None


def run_ui(
    stdscr: "curses._CursesWindow",
    audio_path: Path,
) -> UISessionResult:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)

    messages: "queue.Queue[Dict[str, object]]" = queue.Queue()
    worker = TranscriptionWorker(audio_path, messages)
    worker.start()

    player = AudioPlayer(audio_path)
    segments: List[Dict[str, object]] = []

    current_index = 0
    progress_percent = 0
    progress_message = "Preparing..."
    status_message = "J/K navigate | Enter play | Space pause | Q quit"
    error_message: Optional[str] = None
    auto_play_started = False

    try:
        while True:
            try:
                msg = messages.get_nowait()
            except queue.Empty:
                msg = None
                if (
                    not worker.is_alive()
                    and progress_percent < 100
                    and not error_message
                ):
                    error_message = "Worker thread stopped unexpectedly"
                    progress_message = "Processing failed"
                    progress_percent = 100

            if msg:
                mtype = msg.get("type")
                if mtype == "progress":
                    progress_percent = int(msg.get("percent", progress_percent))
                    progress_message = str(msg.get("message", progress_message))
                elif mtype == "segment":
                    segment = msg.get("segment")
                    if isinstance(segment, dict):
                        segments.append(segment)
                        current_index = min(current_index, len(segments) - 1)
                        if not auto_play_started and segments:
                            try:
                                player.play_from(float(segments[0]["start"]))
                                status_message = f"Playing from {format_timestamp(float(segments[0]['start']))}"
                                auto_play_started = True
                            except RuntimeError as exc:
                                error_message = str(exc)
                elif mtype == "status":
                    status_message = str(msg.get("message", status_message))
                elif mtype == "error":
                    error_message = str(msg.get("message", "Unknown error"))
                    progress_message = "Processing failed"
                    progress_percent = 100
                elif mtype == "done":
                    progress_percent = 100
                    if progress_message.lower() != "transcription complete":
                        progress_message = "Transcription complete"

            render_screen(
                stdscr,
                segments,
                current_index,
                progress_percent,
                progress_message,
                status_message,
                error_message,
            )

            ch = stdscr.getch()
            if ch in (ord("q"), ord("Q")):
                worker.request_stop()
                break
            elif ch in (ord("j"), curses.KEY_DOWN):
                if segments:
                    current_index = min(len(segments) - 1, current_index + 1)
            elif ch in (ord("k"), curses.KEY_UP):
                if segments:
                    current_index = max(0, current_index - 1)
            elif ch in (curses.KEY_ENTER, 10, 13):
                if segments:
                    start_time = float(segments[current_index]["start"])
                    try:
                        player.play_from(start_time)
                        status_message = f"Playing from {format_timestamp(start_time)}"
                    except RuntimeError as exc:
                        error_message = str(exc)
            elif ch == ord(" "):
                try:
                    player.toggle_pause()
                    status_message = "Paused" if player.paused else "Playing"
                except RuntimeError as exc:
                    error_message = str(exc)

            if (
                error_message
                and progress_message == "Processing failed"
                and not segments
            ):
                time.sleep(1.0)
                worker.request_stop()
                break

            time.sleep(0.05)
    except KeyboardInterrupt:
        worker.request_stop()
    finally:
        player.stop()
        worker.join(timeout=1.0)

    return UISessionResult(segments=segments, error=error_message)


def run_transcribe_only(audio_path: Path) -> int:
    messages: "queue.Queue[Dict[str, object]]" = queue.Queue()
    worker = TranscriptionWorker(audio_path, messages)
    worker.start()
    exiting = False
    exit_code = 0
    try:
        while not exiting:
            try:
                msg = messages.get(timeout=0.1)
            except queue.Empty:
                if not worker.is_alive():
                    break
                continue

            mtype = msg.get("type")
            if mtype == "segment":
                segment = msg.get("segment")
                if isinstance(segment, dict):
                    _print_segment(segment)
            elif mtype == "status":
                status = msg.get("message")
                if status:
                    print(f"[status] {status}", file=sys.stderr)
            elif mtype == "progress":
                percent = msg.get("percent")
                message = msg.get("message", "")
                if percent is not None:
                    print(f"[progress] {int(percent)}% {message}", file=sys.stderr)
            elif mtype == "error":
                error_message = msg.get("message", "Unknown error")
                print(f"[error] {error_message}", file=sys.stderr)
                exit_code = 1
                exiting = True
            elif mtype == "done":
                exiting = True
    finally:
        worker.request_stop()
        worker.join(timeout=1.0)
    return exit_code


@lru_cache(maxsize=256)
def _wrap_text_cached(text: str, width: int) -> Tuple[str, ...]:
    wrapped = textwrap.wrap(text, width=width) or [""]
    return tuple(wrapped)


def render_screen(
    stdscr: "curses._CursesWindow",
    segments: List[Dict[str, object]],
    current_index: int,
    progress_percent: int,
    progress_message: str,
    status_message: str,
    error_message: Optional[str],
) -> None:
    stdscr.erase()
    height, width = stdscr.getmaxyx()

    safe_addstr(stdscr, 0, 0, "Dictator - Speech Transcription")

    bar_width = max(10, min(width - 20, 40))
    filled = int(bar_width * progress_percent / 100)
    bar = "[" + "#" * filled + "-" * (bar_width - filled) + "]"
    safe_addstr(
        stdscr, 1, 0, f"Progress {progress_percent:3d}% {bar} {progress_message}"
    )

    safe_addstr(stdscr, 2, 0, status_message)
    if error_message:
        safe_addstr(stdscr, 3, 0, f"Error: {error_message}", curses.A_BOLD)

    content_start_row = 5 if error_message else 4
    if content_start_row >= height:
        stdscr.refresh()
        return

    if not segments:
        safe_addstr(stdscr, content_start_row, 0, "Waiting for transcript segments...")
        stdscr.refresh()
        return

    available_rows = height - content_start_row - 1
    if available_rows <= 0:
        stdscr.refresh()
        return

    start_index = current_index
    total_rows = 0
    while start_index > 0:
        rows = _segment_row_count(segments[start_index - 1], width)
        if total_rows + rows > available_rows:
            break
        start_index -= 1
        total_rows += rows

    row = content_start_row
    for idx in range(start_index, len(segments)):
        if row >= height - 1:
            break
        segment = segments[idx]
        attr = curses.A_REVERSE if idx == current_index else curses.A_NORMAL
        header = f"SPEAKER {segment['speaker']} ({format_timestamp(float(segment['start']))}-{format_timestamp(float(segment['end']))}):"
        safe_addstr(stdscr, row, 0, header, attr)
        row += 1
        wrapped = _wrap_text_cached(str(segment["text"]), width - 4)
        for line in wrapped:
            if row >= height - 1:
                break
            safe_addstr(stdscr, row, 0, f"  {line}", attr)
            row += 1
        if row >= height - 1:
            break
    stdscr.refresh()


def _segment_row_count(segment: Dict[str, object], width: int) -> int:
    wrapped = _wrap_text_cached(str(segment["text"]), width - 4)
    return 1 + len(wrapped)


def print_transcript(segments: List[Dict[str, object]]) -> None:
    for seg in segments:
        start = format_timestamp(float(seg["start"]))
        end = format_timestamp(float(seg["end"]))
        speaker = int(seg["speaker"])
        print(f"SPEAKER {speaker} ({start}-{end}):")
        wrapped = textwrap.wrap(str(seg["text"]).strip(), width=100) or [""]
        for line in wrapped:
            print(f"  {line}")
        print()


def _print_segment(segment: Dict[str, object]) -> None:
    start = format_timestamp(float(segment.get("start", 0.0)))
    end = format_timestamp(float(segment.get("end", 0.0)))
    speaker = int(segment.get("speaker", 0))
    text = str(segment.get("text", "")).strip()
    print(f"SPEAKER {speaker} ({start}-{end}): {text}")
