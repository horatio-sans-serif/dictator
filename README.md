# Dictator

Speech transcription with speaker diarization, powered by Mistral's Voxtral API. Available as a terminal UI (curses), CLI, or web interface with drag-and-drop and microphone recording.

## Requirements

- Python 3.11+
- `ffmpeg` on PATH (for audio format conversion and playback via `ffplay`)
- `MISTRAL_API_KEY` environment variable (get one at https://console.mistral.ai/)

Install dependencies:

```bash
uv sync
```

## Usage

### Terminal UI (default)

```bash
uv run dictator path/to/audio.m4a
```

The curses UI shows progress, renders segments with speaker labels and timestamps, and auto-plays from the first segment.

#### Keyboard Controls

- `J` / `K` - next / previous segment
- `Enter` - play from highlighted segment
- `Space` - toggle pause/resume
- `Q` - quit (transcript printed to stdout on exit)

### CLI Output

```bash
uv run dictator path/to/audio.m4a -T
```

Streams diarized segments to stdout as they arrive. Progress and status go to stderr.

### Stdin

```bash
cat audio.wav | uv run dictator
```

### Web UI

```bash
uv run dictator --serve
# or with custom port:
uv run dictator --serve --port 9000
```

Opens a web server at `http://localhost:8377` with:

- Drag-and-drop zone for audio/video files
- Microphone recording button (records then transcribes)
- Results with color-coded speaker labels and timestamps

### Daemon Mode (macOS)

Keep the web server running at login:

```bash
# Install (requires MISTRAL_API_KEY in environment)
./scripts/install-launchd.sh

# Uninstall
./scripts/uninstall-launchd.sh

# Check status
launchctl list | grep dictator
tail -f /tmp/dictator-server.log
```

## Environment Variables

| Variable          | Required | Description                               |
| ----------------- | -------- | ----------------------------------------- |
| `MISTRAL_API_KEY` | Yes      | Mistral API key for Voxtral transcription |

## Supported Formats

Audio files in mp3, wav, flac, ogg, and webm are sent directly to the API. Other formats (m4a, aac, wma, aiff) are auto-converted to mp3 via ffmpeg.

## Caching

Audio files are hashed (SHA-256) and cached under `data/cache/<hash>.json`. If the same file is transcribed again, the cached result is returned instantly. Cache version bumps invalidate old entries automatically.

## Testing

```bash
uv run python -m pytest
```

Tests mock the Voxtral API so they run without an API key and complete in under a second.

## Project Layout

```
.
├── app/
│   ├── __init__.py      # Package marker
│   ├── __main__.py      # python -m app support
│   ├── cli.py           # Argument parsing, main entrypoint
│   ├── transcribe.py    # Voxtral API, format conversion, audio validation
│   ├── cache.py         # SHA-256 hashing, cache load/save
│   ├── player.py        # AudioPlayer (ffplay wrapper)
│   ├── worker.py        # TranscriptionWorker (background thread)
│   ├── ui.py            # Curses TUI, CLI streaming, screen rendering
│   ├── web.py           # Bottle web server
│   ├── main.py          # Backward-compatibility shim
│   └── static/
│       └── index.html   # Web UI (drag-drop, mic recording)
├── etc/
│   └── com.dictator.server.plist  # launchd template
├── scripts/
│   ├── install-launchd.sh
│   └── uninstall-launchd.sh
├── tests/
│   ├── conftest.py
│   ├── test_cache.py
│   ├── test_cli.py
│   ├── test_transcribe.py
│   └── test_web.py
├── data/cache/          # Transcription cache (gitignored)
├── pyproject.toml
└── pytest.ini
```
