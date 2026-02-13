# Repository Guidelines

## Project Structure & Module Organization

- `app/cli.py` hosts the CLI entrypoint, argument parsing, and dispatch to UI/web/transcribe-only modes.
- `app/transcribe.py` contains the Voxtral API integration, audio format validation, and ffmpeg conversion.
- `app/cache.py` handles SHA-256 file hashing and JSON cache load/save with schema versioning.
- `app/player.py` wraps ffplay for thread-safe audio playback with pause/resume.
- `app/worker.py` runs transcription in a background thread, communicating via a message queue.
- `app/ui.py` implements the curses TUI, CLI streaming mode, and screen rendering.
- `app/web.py` is the Bottle web server with file upload and transcription endpoints.
- `app/static/index.html` is the browser UI with drag-and-drop and mic recording.
- `app/main.py` is a backward-compatibility shim re-exporting symbols from the above modules.
- `app/__main__.py` enables `python -m app` invocation.
- `data/` is reserved for user assets. The app writes cache files to `data/cache/<audio_sha>.json`.
- `etc/` contains the launchd plist template for daemon mode.
- `scripts/` contains install/uninstall scripts for the launchd service.
- Configuration lives in `pyproject.toml`. Add optional tooling under `[dependency-groups.dev]`.

## Build, Test, and Development Commands

- `uv sync` installs runtime dependencies (mistralai, bottle).
- `uv run dictator path/to/audio.m4a` runs the TUI.
- `uv run dictator path/to/audio.m4a -T` streams diarized transcript to stdout.
- `uv run dictator --serve` starts the web server on port 8377.
- `uv run python -m pytest` runs the test suite.

## Environment Variables

- `MISTRAL_API_KEY` (required) - Mistral API key for Voxtral transcription.

## Architecture

- Transcription uses Mistral's Voxtral API (`voxtral-mini-latest` model) with `diarize=True`.
- The `transcribe()` function in `app/transcribe.py` does a lazy import of `mistralai` to avoid requiring the API key at import time.
- Non-native audio formats (m4a, aac, wma, aiff) are converted to mp3 via ffmpeg before upload.
- `TranscriptionWorker` in `app/worker.py` runs transcription in a background thread, communicating via a message queue.
- Cache is keyed by SHA-256 hash of the audio file. Cache schema version bumps invalidate old entries.
- Module dependency graph: cli -> ui -> worker -> transcribe + cache; player is standalone.

## Coding Style & Naming Conventions

- Follow PEP 8 with 4-space indentation.
- Keep user-facing strings concise; UI lines should fit an 80-column terminal.
- Use f-strings for formatting.
- Log through the UI queue in worker threads, not bare `print`.

## Testing Guidelines

- Tests mock the Voxtral API via `patch.dict("sys.modules", {"mistralai": fake_mod})` for the lazy import.
- Use `webtest.TestApp` for web endpoint tests.
- Tests run without API keys and complete in under a second.
- Organize tests by feature area (`test_cli.py`, `test_cache.py`, `test_transcribe.py`, `test_web.py`).

## Commit & Pull Request Guidelines

- Write imperative, prefix-style commits (`feat: add web UI`, `fix: handle empty segments`).
- Document new CLI flags or environment variables in `README.md` and this file.
