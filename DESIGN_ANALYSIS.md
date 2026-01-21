# Design Analysis Report: dictator

## Executive Summary

This report analyzes the architectural design and outstanding issues in the dictator audio transcription system. The system is functional but suffers from several design problems that impact maintainability, testability, and extensibility.

---

## 1. Current Architecture Overview

### System Purpose

An interactive CLI tool that:

1. Transcribes audio files using OpenAI Whisper
2. Performs speaker diarization using pyannote
3. Enhances audio quality via DSP chain or neural model
4. Provides curses-based interactive playback UI
5. Caches results for fast repeat access

### High-Level Flow

```
Audio File
    |
    v
[Audio Hash] --> [Cache Check]
    |                  |
    v                  v (cache hit)
[Enhancement]     [Load Cached]
    |                  |
    v                  |
[Transcription]        |
    |                  |
    v                  |
[Diarization]          |
    |                  |
    v                  |
[Merge Segments] <-----+
    |
    v
[Cache Write]
    |
    v
[UI/Output]
```

---

## 2. Critical Design Issues

### 2.1 Monolithic Architecture (CRITICAL)

**Problem**: All 1,525 lines of code reside in a single file (`app/main.py`).

**Impact**:

- Testing individual components requires importing the entire module
- Cannot mock or replace subsystems independently
- Code navigation is difficult
- Changes to one component risk breaking others
- No clear dependency boundaries

**Recommended Refactoring**:

```
app/
├── __init__.py
├── main.py              # CLI entrypoint only (~100 lines)
├── audio/
│   ├── __init__.py
│   ├── enhancement.py   # Soap chain, Resemble backend
│   ├── player.py        # AudioPlayer class
│   └── utils.py         # load_waveform, compute_hash
├── transcription/
│   ├── __init__.py
│   ├── whisper.py       # Whisper integration
│   ├── diarization.py   # pyannote integration
│   └── worker.py        # TranscriptionWorker
├── cache/
│   ├── __init__.py
│   └── manager.py       # Cache validation, read/write
├── ui/
│   ├── __init__.py
│   ├── curses_ui.py     # Interactive UI
│   └── stream_ui.py     # Non-interactive output
└── config.py            # Constants, presets
```

### 2.2 Incomplete Dependency Declaration (HIGH)

**Problem**: `soundfile` is imported but not in `pyproject.toml`.

**Impact**: Fresh installs may fail or produce confusing errors.

**Fix**: Add to pyproject.toml:

```toml
dependencies = [
    "soundfile>=0.12.1",
    # ... existing deps
]
```

### 2.3 Tight Coupling Between Components (HIGH)

**Problem**: `TranscriptionWorker` manages everything: enhancement, transcription, diarization, caching, and UI messaging.

**Impact**:

- Cannot test transcription without enhancement logic
- Cannot reuse components in other contexts
- Worker class has too many responsibilities

**Recommended Fix**: Apply dependency injection:

```python
class TranscriptionWorker:
    def __init__(
        self,
        audio_path: Path,
        enhancer: AudioEnhancer,
        transcriber: Transcriber,
        diarizer: Diarizer,
        cache: CacheManager,
        messenger: MessageQueue,
    ):
        # ...
```

### 2.4 Magic Numbers Throughout (MEDIUM)

**Problem**: Hardcoded values scattered across audio processing functions.

**Examples**:

- `hop_length=512`, `win_length=2048` (reduce_background_noise)
- `delay_ms=60.0`, `attenuation=0.35` (suppress_echo)
- `threshold_db=-5.0`, `window=256` (remove_transient_peaks)
- `threshold_multiplier=6.0` (suppress_mic_bumps)

**Impact**: Difficult to tune or override; unclear what values mean.

**Recommended Fix**: Move to preset configuration or named constants:

```python
@dataclass
class NoiseGateConfig:
    hop_length: int = 512
    win_length: int = 2048
    noise_frames_ratio: float = 1/6
    noise_threshold_multiplier: float = 1.5
```

### 2.5 Limited Error Recovery (MEDIUM)

**Problem**: Enhancement failures are caught but not tracked in cache.

**Impact**: If enhancement fails, next run attempts it again. No way to detect persistent failures.

**Recommended Fix**: Track enhancement status in cache metadata:

```json
{
    "enhancement_status": "success" | "failed" | "skipped",
    "enhancement_error": "optional error message"
}
```

### 2.6 Progress Estimation is Crude (LOW)

**Problem**: Fixed multipliers for time estimation (1.2x for transcription, 0.6x for diarization).

**Impact**: Progress bar is inaccurate; user experience suffers on varying hardware.

**Recommended Fix**: Adaptive estimation based on:

- Historical run times per model/file size
- Real-time rate calculation during processing

---

## 3. Testing Gaps

### Current Test Coverage

| Area              | Coverage | Tests                   |
| ----------------- | -------- | ----------------------- |
| Audio enhancement | Partial  | 1 test (compare export) |
| Transcription     | Partial  | 5 quality tests         |
| Diarization       | None     | 0 tests                 |
| UI/curses         | None     | 0 tests                 |
| CLI parsing       | None     | 0 tests                 |
| Cache validation  | None     | 0 tests                 |
| Error handling    | None     | 0 tests                 |
| Worker threading  | None     | 0 tests                 |

### Missing Test Categories

1. **Unit Tests for Audio Functions**
    - `remove_dc_offset()`
    - `apply_bandpass_filter()`
    - `reduce_background_noise()`
    - `suppress_echo()`
    - `remove_transient_peaks()`
    - `normalize_volume()`
    - `apply_dynamic_compression()`
    - `apply_high_shelf()`
    - `suppress_mic_bumps()`

2. **Unit Tests for Cache Logic**
    - `_cache_is_valid()` with various invalidation scenarios
    - `_try_load_cache()` with corrupt files
    - Cache key generation

3. **Integration Tests**
    - Full pipeline with real audio
    - Enhancement → transcription → diarization flow
    - Cache hit/miss scenarios

4. **Error Path Tests**
    - Missing audio file
    - Corrupted audio
    - Invalid Whisper model
    - pyannote token issues
    - ffplay unavailable

5. **UI Tests**
    - Keyboard input handling
    - Screen rendering
    - Message queue processing

---

## 4. Security Considerations

### 4.1 Path Traversal

**Status**: Low risk - audio path is resolved to absolute path.

### 4.2 Shell Injection

**Status**: Mitigated - ffplay is called with subprocess array, not shell string.

### 4.3 Sensitive Data

**Status**: `PYANNOTE_AUTH_TOKEN` read from environment (good practice).

---

## 5. Performance Considerations

### 5.1 File Hashing on Every Run

**Issue**: SHA-256 computed even when cache will be invalidated.
**Impact**: Adds latency for large files.
**Fix**: Quick metadata check before full hash (file size + mtime).

### 5.2 Model Loading Overhead

**Issue**: Whisper model loaded before cache check.
**Impact**: Unnecessary GPU/memory usage for cache hits.
**Fix**: Lazy load models only when needed.

### 5.3 Full Audio Load for Diarization

**Issue**: Entire waveform loaded into memory.
**Impact**: Memory pressure for long files.
**Status**: Acceptable trade-off for torchcodec compatibility.

---

## 6. Recommendations Summary

### Immediate (Critical)

1. Add `soundfile` to dependencies
2. Add comprehensive unit tests for audio functions
3. Add cache validation tests

### Short-term (High Priority)

4. Split monolithic file into modules
5. Extract `TranscriptionWorker` responsibilities
6. Add integration tests with real audio

### Medium-term

7. Implement dependency injection
8. Move magic numbers to configuration
9. Add adaptive progress estimation
10. Track enhancement failures in cache

### Long-term

11. Add async support for concurrent processing
12. Support streaming transcription
13. Add plugin system for custom enhancers

---

## 7. TTS Integration Considerations

Based on analysis of the patientquestions project and recent TTS options:

### Recommended TTS Options

1. **Pocket TTS** (Kyutai)
    - 100M parameters, CPU-optimized
    - Voice cloning support
    - `pip install pocket-tts`
    - Good for offline use

2. **Kokoro** (hexgrad)
    - 82M parameters
    - 54 voices, 8 languages
    - `pip install kokoro>=0.9.4`
    - Production-ready

### Integration Approach

If TTS is added for read-back functionality:

```python
# Abstract provider pattern (from patientquestions)
class TTSProvider(ABC):
    def generate_speech(self, text: str, voice: str) -> Generator[np.ndarray, None, None]: ...
    def supports_streaming(self) -> bool: ...
```

---

## Appendix: File Structure Analysis

### Current Structure

```
dictator/
├── app/
│   └── main.py          # 1,525 lines - EVERYTHING
├── data/
│   └── cache/           # Transcription + enhanced audio cache
├── test/
│   └── assets/          # 5 audio files with transcripts
├── tests/
│   ├── test_assets.py           # Asset validation (1 test)
│   ├── test_compare_audio.py    # Compare export (1 test)
│   └── test_transcription_quality.py  # Quality regression (5 tests)
├── CLAUDE.md
├── README.md
├── pyproject.toml
└── pytest.ini
```

### Lines of Code Distribution

- Audio Processing: 457 lines (30%)
- Transcription Worker: 350 lines (23%)
- UI/Rendering: 290 lines (19%)
- Utilities/Helpers: 250 lines (16%)
- CLI/Main: 178 lines (12%)
