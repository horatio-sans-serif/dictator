"""Thread-safe audio playback via ffplay."""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Optional


class AudioPlayer:
    def __init__(self, audio_path: Path) -> None:
        self.audio_path = audio_path
        self.process: Optional[subprocess.Popen[bytes]] = None
        self.lock = threading.Lock()
        self.paused = False

    def set_audio_path(self, audio_path: Path) -> None:
        with self.lock:
            self.audio_path = audio_path
            self.stop()

    def play_from(self, start_time: float) -> None:
        with self.lock:
            self.stop()
            audio_path_resolved = self.audio_path.resolve()
            if not audio_path_resolved.exists():
                raise RuntimeError(f"Audio file does not exist: {audio_path_resolved}")
            if not audio_path_resolved.is_file():
                raise RuntimeError(f"Path is not a regular file: {audio_path_resolved}")
            cmd = [
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-ss",
                f"{max(0.0, start_time):.2f}",
                str(audio_path_resolved),
            ]
            proc = None
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.process = proc
                self.paused = False
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "ffplay (from ffmpeg) is required for playback"
                ) from exc
            except Exception:
                if proc and proc.poll() is None:
                    proc.terminate()
                raise

    def toggle_pause(self) -> None:
        with self.lock:
            if (
                not self.process
                or self.process.poll() is not None
                or not self.process.stdin
            ):
                raise RuntimeError("Nothing is currently playing")
            try:
                self.process.stdin.write(b"p")
                self.process.stdin.flush()
            except BrokenPipeError:
                self.stop()
                raise RuntimeError("Playback process ended unexpectedly") from None
            self.paused = not self.paused

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        self.paused = False
