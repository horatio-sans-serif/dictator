# Repository Guidelines

## Project Structure & Module Organization
- `app/main.py` hosts the CLI entrypoint, curses UI, and transcription pipeline. Keep UI helpers at the bottom and processing helpers near the top so newcomers can scan setup → worker → interface in order.
- `data/` is reserved for user assets. The app writes cache files to `data/cache/<audio_sha>.json`; avoid committing personal audio.
- Configuration lives in `pyproject.toml`. Add optional tooling under `[dependency-groups.dev]` instead of the main dependency list.

## Build, Test, and Development Commands
- `uv sync` or `pip install -e .` installs runtime dependencies (Whisper, librosa, pyannote).
- `uv run python -m app.main path/to/audio.m4a` runs the application. Use `--model tiny` when smoke testing to minimise GPU/CPU load.
  - The default Whisper checkpoint is `large-v3`; expect long downloads the first time. Override with `--model` (e.g., `--model tiny`) for quicker dev loops.
  - `-T/--transcribe-only` bypasses curses and prints segments directly; keep CLI docs in sync when changing output format.
  - `--preset` toggles Soap-inspired enhancement curves (see `MIC_PRESETS`) including the new `iphone` preset.
  - `--enhancer` switches between the Soap chain (`soap`) and the Resemble Enhance neural backend (`resemble`); document any behavioural changes when touching either path.
  - `--no-enhancement` bypasses the processing chain entirely if users want the raw file.
  - `--debug-enhanced-compare` writes a WAV that alternates original/enhanced 5 s chunks for QA, complete with a high beep before each enhanced segment and a double-beep separator.
  - `--debug-enhanced-compare-duration` controls how many seconds of audio land in that compare WAV (0 = full track; default 30 s).
- `uv run pytest` executes the regression suite:
  - `tests/test_assets.py` confirms every `test/assets/*.m4a` clip has a matching `.txt` transcript.
  - `tests/test_compare_audio.py` verifies the debug compare exporter enforces duration caps and audible beeps.
  - `tests/test_transcription_quality.py` re-runs Whisper Tiny on baseline vs. enhanced audio for each labeled asset, comparing word error rate and cosine similarity to ensure enhancements never regress transcripts beyond the per-clip tolerances. Update the tolerances or fixture transcripts in the same commit when enhancement behaviour changes.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation. Prefer descriptive helper names (`format_timestamp`, `segment_row_count`) over abbreviations.
- Keep user-facing strings concise; UI lines should fit an 80-column terminal.
- When adding modules, expose CLI entrypoints through `python -m package.module` instead of baking shebang scripts.
- Use f-strings for formatting and avoid bare `print` debugging inside worker threads—log through the UI queue.

## Testing Guidelines
- Organize tests by feature area (`tests/test_cli.py`, `tests/test_transcription.py`, etc.).
- Use `pytest` markers already declared in `pytest.ini` (`@pytest.mark.slow`, `@pytest.mark.integration`) to separate device-heavy runs.
- Mock Whisper/Pyannote calls for unit tests; reserve real-model executions for `slow` suites with cached fixtures.
- Ensure new behaviour toggles (e.g., playback fallbacks) ship with regression tests that assert user-visible messages.
- When adding new reference clips to `test/assets/`, include a ground-truth `.txt` transcript and extend `AUDIO_CASES` in `tests/test_transcription_quality.py` with preset + margin values so CI watches for real transcription changes.
- Keep test commands documented with their `uv run ...` prefix so contributors don’t skip the project venv.

## Commit & Pull Request Guidelines
- Write imperative, prefix-style commits (`feat: stream segments during diarization`, `fix: guard ffplay errors`). Squash noisy WIP commits before raising a PR.
- PRs should describe motivation, outline UI or UX shifts, and link related issues. Include terminal recordings or asciicasts when changing interactive flows.
- Document new CLI flags or environment variables in `README.md` and `AGENTS.md` as part of the same change.

## Configuration & Troubleshooting Tips
- Whisper downloads models to the default cache (`~/.cache/whisper`). Mention this if a change alters model sizes or defaults.
- Audio enhancement writes mono 16-bit WAV copies to `data/cache/<hash>-enhanced-<backend>-<preset>-vX.wav`; reuse the helper in `app/main.py` when adding new preprocessing steps so cache keys stay consistent with backend + preset metadata.
- The Resemble Enhance backend downloads its weights via `git` + `git-lfs` into `site-packages/resemble_enhance/model_repo`; call that out in docs when changing its behaviour or prerequisites.
- The enhancement chain now includes dynamic compression tuned for human speech (roughly ages 13–60); adjust `apply_dynamic_compression` if future work needs different targets.
- Mic bump suppression cross-fades sub-80 ms spikes (see `suppress_mic_bumps`); keep it in sync with `remove_transient_peaks` when tuning thresholds.
- Soap-style presets live in `MIC_PRESETS`; update `--preset` handling and cache metadata whenever you add/remove presets.
- Preset values only affect the Soap backend—if you add new Resemble modes, gate Soap-specific options accordingly so cache metadata stays truthful.
- Audio playback now defaults to the enhanced WAV so users hear exactly what Whisper ingests; keep `AudioPlayer.set_audio_path` in sync if you change preprocessing.
- Cache metadata now encodes the Whisper model, language, enhancement enabled flag, version, and preset—bump `CACHE_SCHEMA_VERSION` when altering compatibility rules.
- Pyannote diarization now feeds an in-memory waveform to avoid torchcodec/ffmpeg dylib issues; prefer extending that path before adding new system requirements.
- CLI preloads Whisper/pyannote before curses starts so download output remains visible; keep that flow intact when touching `prepare_dependencies`/`run_ui`.
- If playback support changes, note any `ffmpeg` requirements and test on macOS + Linux at minimum.
