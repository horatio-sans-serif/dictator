"""Background transcription worker thread."""
from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Dict

from app.cache import (
    cache_path_for,
    compute_file_hash,
    load_cache,
    save_cache,
)
from app.transcribe import transcribe


class TranscriptionWorker(threading.Thread):
    def __init__(
        self,
        audio_path: Path,
        messages: "queue.Queue[Dict[str, object]]",
    ) -> None:
        super().__init__(daemon=True)
        self.audio_path = audio_path
        self.messages = messages
        self.stop_event = threading.Event()

    def request_stop(self) -> None:
        self.stop_event.set()

    def _emit(self, payload: Dict[str, object]) -> None:
        if not self.stop_event.is_set():
            try:
                self.messages.put(payload, timeout=1.0)
            except queue.Full:
                pass

    def _emit_progress(self, percent: int, message: str) -> None:
        percent = max(0, min(100, percent))
        self._emit({"type": "progress", "percent": percent, "message": message})

    def run(self) -> None:
        try:
            if not self.audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {self.audio_path}")

            audio_hash = compute_file_hash(self.audio_path)
            cache_file = cache_path_for(audio_hash)
            cached_segments = load_cache(cache_file, audio_hash)
            if cached_segments is not None:
                self._emit_progress(100, "Loaded cached transcription")
                for seg in cached_segments:
                    self._emit({"type": "segment", "segment": seg})
                self._emit({"type": "done", "from_cache": True})
                return

            self._emit_progress(10, "Transcribing with Voxtral...")

            segments = transcribe(self.audio_path)

            if self.stop_event.is_set():
                return

            save_cache(cache_file, audio_hash, segments)
            self._emit_progress(100, "Transcription complete")

            for seg in segments:
                if self.stop_event.is_set():
                    break
                self._emit({"type": "segment", "segment": seg})

            self._emit({"type": "done", "from_cache": False})
        except Exception as exc:
            self._emit({"type": "error", "message": str(exc)})
