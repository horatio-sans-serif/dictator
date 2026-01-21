# Dictator CLI

An interactive terminal application that transcribes local audio, assigns speaker labels, and lets you scrub through the conversation from the keyboard. Transcriptions stream into the interface, can be replayed segment-by-segment, and are cached so repeat runs are instant.

## Requirements

- Python 3.11+
- `ffmpeg` with `ffplay` available on `PATH` (for audio playback)
- Model weights for [`openai-whisper`](https://github.com/openai/whisper) (downloaded on first use)
- Optional: set `PYANNOTE_AUTH_TOKEN` for multi-speaker diarization
- Optional: `git` + `git-lfs` if you plan to use the Resemble Enhance backend (it downloads from Hugging Face)

Install Python dependencies with [uv](https://github.com/astral-sh/uv) or `pip`:

```bash
uv sync
# or
pip install -e .
```

## Usage

```bash
# Interactive UI
uv run python -m app.main path/to/audio.m4a

# Non-interactive (stream segments to stdout)
uv run python -m app.main path/to/audio.m4a -T
```

- `--model` selects the Whisper checkpoint (`large-v3` by default for maximum accuracy).
- `-T/--transcribe-only` skips the curses UI and streams segments as they are produced with speaker labels and timestamps (handy for regression tests).
- `--preset` picks a Soap-inspired enhancement profile (`general`, `sm7b`, `sm58`, `podmic`, `usb-condenser`, `iphone`).
- `--enhancer` switches between the Soap chain (`soap`, default) and the Resemble Enhance neural backend (`resemble`). The latter downloads a checkpoint on first use and runs much faster on GPUs than CPUs.
- `--debug-enhanced-compare` exports a WAV that alternates 5 s of original audio with 5 s of enhanced audio so you can A/B the processing.
- `--debug-enhanced-compare-duration` limits that WAV to the first N seconds (default 30) so you don’t have to audition an hour-long file.
- `--no-enhancement` bypasses every enhancement backend if you want Whisper to hear the untouched original.
- The terminal UI opens immediately. Progress updates appear along the top.
- Segments render as soon as the transcript is ready. Cached runs display instantly.
- On first run (or when changing models), the CLI preloads Whisper/pyannote *before* curses starts so you can watch their download progress in the normal terminal output.

### Keyboard Controls

- `J` / `K` – move to the next / previous segment
- `Enter` – play from the start time of the highlighted segment
- `Space` – toggle pause/resume on the current playback
- `Q` – quit the session (transcript is printed to stdout on exit)

Playback begins automatically from the first segment once transcription finishes. Audio commands require `ffplay`; if it is unavailable, playback controls report an error but the transcript remains usable.

## Audio Enhancement

Before transcription starts, the app creates an enhanced copy of the source audio (cached under `data/cache/<hash>-enhanced-<backend>-<preset>-v6.wav`). `backend` is `soap` by default; set `--enhancer resemble` to opt into the Resemble Enhance neural model. Compare WAVs are written next to the cache as `...-compare-<backend>-<preset>-v6.wav`.

### Soap Chain (default)

- DC offset is removed to keep the waveform centered.
- A tunable band-pass filter emphasizes human speech frequencies per preset.
- A spectral gate reduces steady background noise and blends a little of the dry signal back in (see each preset’s `denoise_wet`).
- Short-delay echo energy is reduced.
- Single-sample clicks and pops are replaced with local medians to avoid skewing normalization.
- Sub-80 ms mic bumps are detected and cross-faded out so physical knocks don’t saturate the model input.
- A speech-tuned dynamic compressor evens out 13–60 year-old voice ranges so quiet speakers get a boost before normalization.
- High-frequency “Suds” boosts and low-frequency “D-Mud” cuts emulate the Soap Audacity plugin so podcasters get the same tonal polish.
- The signal is normalized so Whisper receives a consistent level.

### Resemble Enhance (optional)

`--enhancer resemble` swaps the Soap chain for [Resemble Enhance](https://github.com/resemble-ai/resemble-enhance), a two-stage neural denoiser/enhancer trained on 44.1 kHz speech. The first run downloads weights via `git-lfs` (~1 GB) and subsequent runs reuse the cached model under `site-packages/resemble_enhance/model_repo`. On GPUs it runs close to real-time; on CPUs it automatically drops to 32 solver steps to stay manageable. This backend ignores `--preset` (the metadata still records the last selected preset so caches stay distinct).

If enhancement fails for any reason, the original audio is used automatically. Once enhancement succeeds, the UI also plays that enhanced copy so you hear the exact audio Whisper processes. Use `--debug-enhanced-compare` to generate A/B WAVs (`...-compare-<backend>-<preset>-v6.wav`) that alternate 5 s of the original followed by 5 s of the enhanced signal so you can quickly spot issues. A short single beep indicates the switch to the enhanced segment and a double-beep separator marks each round. On macOS you can listen with `afplay path/to/compare.wav`; on other platforms use `ffplay` or any media player.

### Voice Presets

`--preset` determines which Soap-style EQ/comp curve to apply when `--enhancer soap` is active:

- `general` – Balanced default for most setups.
- `sm7b` – Broadcast-style compression/EQ for Shure SM7B dynamics.
- `sm58` – Lighter presence and low-cut shaping for handheld SM58-style mics.
- `podmic` – Adds mid clarity to darker dynamic mics like the Rode PodMic.
- `usb-condenser` – Subtle shaping for already-bright USB condensers.
- `iphone` – Adaptive Soap-inspired preset tuned for recent iPhones: it keeps denoise/compression subtle by default, but re-enables EQ and dynamics automatically on very quiet or high-crest recordings so transcripts stay intelligible without over-processing bright clips.

## Caching

Audio files are hashed (SHA-256) and cached under `data/cache/<hash>.json`, plus an enhanced WAV next to it.
Cache entries record the Whisper model, language, and enhancement version used to create them; if any of those settings change (or the schema bumps), the transcript is automatically regenerated so you never have to clear cache files by hand.

## Testing & Regression

Run the full suite with:

```bash
uv run pytest
```

That command covers three areas:

- `tests/test_assets.py` ensures every fixture under `test/assets/` has a matching `.txt` transcript (ground truth for quality checks).
- `tests/test_compare_audio.py` guards the debug compare exporter so each generated WAV respects the requested duration cap and includes the audible single- and double-beep cues.
- `tests/test_transcription_quality.py` re-runs Whisper Tiny on each labeled clip twice (baseline vs. enhanced) and asserts the enhanced output never regresses beyond the per-clip word-error-rate / cosine-similarity margins. Update the `.txt` transcripts and the `AUDIO_CASES` table whenever you add new fixtures or change enhancement expectations.

For manual spot checks without the curses UI, pair `-T/--transcribe-only` with `--preset ...` and compare the streamed output before/after your change.

## Speaker Diarization

Diarization is attempted with `pyannote/speaker-diarization-3.1`. Provide a Hugging Face access token via `PYANNOTE_AUTH_TOKEN`. If diarization fails or the token is absent, every segment is assigned to `SPEAKER 1`.
Audio is loaded into memory before running the pipeline, so ffmpeg/torchcodec quirks on macOS will not block diarization.

## Project Layout

```
.
├── app/                  # CLI implementation
├── data/                 # User audio files and cached transcripts
├── pyproject.toml        # Project metadata and dependencies
└── pytest.ini            # Test configuration (placeholders for future suites)
```
