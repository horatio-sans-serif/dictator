# TODO List: dictator

Last Updated: 2026-01-19

## Priority Legend

- **P0**: Critical - Blocks functionality or testing
- **P1**: High - Important for code quality/maintainability
- **P2**: Medium - Improves user experience or developer experience
- **P3**: Low - Nice to have

## Dependencies

```
[Dependency Graph]

P0.1 (Fix dependencies) <-- P0.2 (Unit tests) <-- P0.3 (Integration tests)
                                    ^
                                    |
P1.1 (Modularize) <-----------------+
        |
        v
P1.2 (Extract components) <-- P1.3 (Dependency injection)
        |
        v
P2.1 (Error handling) <-- P2.2 (Enhanced caching)
```

---

## P0: Critical

### P0.1 Fix Incomplete Dependencies

**Status**: Not Started
**Dependencies**: None
**Effort**: 5 minutes

Add missing `soundfile` dependency to `pyproject.toml`:

```toml
dependencies = [
    "soundfile>=0.12.1",
    # ... existing
]
```

**Acceptance Criteria**:

- [ ] `uv sync` installs soundfile
- [ ] Fresh install in clean venv succeeds
- [ ] All imports work without errors

---

### P0.2 Create Comprehensive Unit Test Suite

**Status**: Not Started
**Dependencies**: P0.1
**Effort**: 4-6 hours

Create unit tests for all audio processing functions:

**Files to Create**:

- [ ] `tests/test_audio_processing.py`
- [ ] `tests/test_cache.py`
- [ ] `tests/test_cli.py`
- [ ] `tests/test_worker.py`

**Test Coverage Targets**:

| Module                     | Target |
| -------------------------- | ------ |
| Audio processing functions | 90%    |
| Cache validation           | 100%   |
| CLI argument parsing       | 100%   |
| Worker message handling    | 80%    |

**Acceptance Criteria**:

- [ ] All audio functions have tests with edge cases
- [ ] Cache validation covers all invalidation scenarios
- [ ] Tests run in < 30 seconds (excluding slow markers)
- [ ] Code coverage >= 80%

---

### P0.3 Create Integration Test Suite

**Status**: Not Started
**Dependencies**: P0.2
**Effort**: 2-3 hours

Test full pipeline with real audio:

**Files to Create**:

- [ ] `tests/test_integration.py`
- [ ] `tests/fixtures/` (generated audio with known content)

**Test Scenarios**:

- [ ] Enhancement -> Transcription -> Diarization flow
- [ ] Cache hit returns identical results
- [ ] Cache invalidation triggers recompute
- [ ] Multiple speakers detected correctly
- [ ] Error recovery (missing files, bad audio)

**Acceptance Criteria**:

- [ ] Tests use real Whisper (tiny model)
- [ ] Tests use real pyannote or mock appropriately
- [ ] Tests complete in < 5 minutes
- [ ] No flaky tests

---

## P1: High Priority

### P1.1 Modularize Codebase

**Status**: Not Started
**Dependencies**: P0.2 (tests provide safety net)
**Effort**: 8-12 hours

Split `app/main.py` into logical modules:

**Target Structure**:

```
app/
├── __init__.py
├── main.py           # CLI entrypoint only
├── audio/
│   ├── __init__.py
│   ├── enhancement.py
│   ├── player.py
│   └── presets.py
├── transcription/
│   ├── __init__.py
│   ├── whisper.py
│   ├── diarization.py
│   └── worker.py
├── cache/
│   ├── __init__.py
│   └── manager.py
├── ui/
│   ├── __init__.py
│   ├── curses_ui.py
│   └── stream_ui.py
└── config.py
```

**Acceptance Criteria**:

- [ ] No single file > 300 lines
- [ ] All existing tests pass
- [ ] No circular imports
- [ ] Clear import paths

---

### P1.2 Extract Component Interfaces

**Status**: Not Started
**Dependencies**: P1.1
**Effort**: 4-6 hours

Define clean interfaces for major components:

```python
# Enhancer interface
class AudioEnhancer(Protocol):
    def enhance(self, audio_path: Path, preset: str) -> Path: ...
    def supports_preset(self, preset: str) -> bool: ...

# Transcriber interface
class Transcriber(Protocol):
    def transcribe(self, audio_path: Path, language: str) -> List[Segment]: ...

# Diarizer interface
class Diarizer(Protocol):
    def diarize(self, audio_path: Path) -> List[SpeakerTurn]: ...
```

**Acceptance Criteria**:

- [ ] Interfaces defined with Protocol classes
- [ ] Implementations conform to interfaces
- [ ] Can substitute mock implementations in tests

---

### P1.3 Implement Dependency Injection

**Status**: Not Started
**Dependencies**: P1.2
**Effort**: 3-4 hours

Refactor `TranscriptionWorker` to accept dependencies:

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
        ...
```

**Acceptance Criteria**:

- [ ] Worker no longer instantiates its own dependencies
- [ ] Tests can inject mocks for all components
- [ ] Real components wired in main.py

---

## P2: Medium Priority

### P2.1 Improve Error Handling

**Status**: Not Started
**Dependencies**: P1.1
**Effort**: 2-3 hours

Add proper error handling throughout:

- [ ] Define custom exception hierarchy
- [ ] Catch and report enhancement failures with context
- [ ] Track persistent failures to avoid retry loops
- [ ] Improve error messages for user-facing issues

**Custom Exceptions**:

```python
class TranscriberError(Exception): ...
class AudioEnhancementError(TranscriberError): ...
class TranscriptionError(TranscriberError): ...
class DiarizationError(TranscriberError): ...
class CacheError(TranscriberError): ...
```

---

### P2.2 Enhanced Cache Metadata

**Status**: Not Started
**Dependencies**: P2.1
**Effort**: 2-3 hours

Track more information in cache:

```json
{
    "schema_version": 6,
    "audio_hash": "...",
    "enhancement_status": "success",
    "enhancement_error": null,
    "processing_time_ms": 45000,
    "whisper_timing": {"transcription_ms": 30000, "model": "large-v3"},
    "diarization_timing": {"duration_ms": 15000, "speakers_found": 2}
}
```

**Acceptance Criteria**:

- [ ] Cache stores enhancement success/failure
- [ ] Cache stores timing information
- [ ] Cache bump when format changes
- [ ] Migration path for old caches

---

### P2.3 Move Magic Numbers to Configuration

**Status**: Not Started
**Dependencies**: P1.1
**Effort**: 2-3 hours

Extract hardcoded values:

```python
@dataclass
class AudioProcessingConfig:
    # Noise reduction
    noise_hop_length: int = 512
    noise_win_length: int = 2048
    noise_frames_ratio: float = 1/6
    noise_threshold: float = 1.5

    # Echo suppression
    echo_delay_ms: float = 60.0
    echo_attenuation: float = 0.35

    # Transient removal
    transient_threshold_db: float = -5.0
    transient_window: int = 256
```

---

### P2.4 Add Progress Callback Support

**Status**: Not Started
**Dependencies**: P1.2
**Effort**: 2-3 hours

Allow custom progress reporting:

```python
ProgressCallback = Callable[[int, str], None]

def transcribe(
    audio_path: Path,
    on_progress: Optional[ProgressCallback] = None,
) -> List[Segment]:
    ...
```

---

## P3: Low Priority

### P3.1 Add TTS Read-Back Support

**Status**: Not Started
**Dependencies**: P1.2, P2.4
**Effort**: 6-8 hours

Add text-to-speech for transcript read-back:

**Options**:

1. Pocket TTS (Kyutai) - CPU-friendly, voice cloning
2. Kokoro (hexgrad) - Multi-language, 54 voices

**Implementation**:

```python
class TTSProvider(Protocol):
    def generate(self, text: str, voice: str) -> np.ndarray: ...
    def list_voices(self) -> List[str]: ...

class PocketTTSProvider(TTSProvider): ...
class KokoroTTSProvider(TTSProvider): ...
```

---

### P3.2 Adaptive Progress Estimation

**Status**: Not Started
**Dependencies**: P2.2
**Effort**: 3-4 hours

Learn from historical runs:

- Store timing in cache metadata
- Build model: time = f(file_size, model, hardware)
- Update estimates in real-time during processing

---

### P3.3 Async/Streaming Transcription

**Status**: Not Started
**Dependencies**: P1.3
**Effort**: 8-12 hours

Add support for streaming results:

```python
async def transcribe_stream(
    audio_path: Path,
) -> AsyncIterator[Segment]:
    ...
```

---

### P3.4 Plugin System for Enhancers

**Status**: Not Started
**Dependencies**: P1.2
**Effort**: 4-6 hours

Allow external enhancement plugins:

```python
# Entry point in pyproject.toml
[project.entry-points."wswn.enhancers"]
soap = "app.audio.enhancement:SoapEnhancer"
resemble = "app.audio.enhancement:ResembleEnhancer"
my_custom = "my_package:CustomEnhancer"
```

---

## Test Suite Roadmap

### Phase 1: Unit Tests (P0.2)

```
tests/
├── test_audio_processing.py    # Audio DSP functions
├── test_cache.py               # Cache validation logic
├── test_cli.py                 # Argument parsing
├── test_worker.py              # Worker thread behavior
└── conftest.py                 # Shared fixtures
```

### Phase 2: Integration Tests (P0.3)

```
tests/
├── test_integration.py         # Full pipeline tests
├── fixtures/
│   ├── generate_test_audio.py  # Script to create test files
│   ├── single_speaker.wav      # Generated single speaker
│   ├── two_speakers.wav        # Generated dialogue
│   └── noisy_audio.wav         # Generated noisy audio
```

### Phase 3: End-to-End Tests

```
tests/
├── test_e2e.py                 # Full CLI tests
├── test_ui.py                  # Curses UI tests (with pyte)
```

---

## Technical Debt Tracking

| Item                         | Priority | Effort | Status      |
| ---------------------------- | -------- | ------ | ----------- |
| Missing soundfile dependency | P0       | 5m     | Not Started |
| Monolithic main.py           | P1       | 8-12h  | Not Started |
| No unit tests for audio      | P0       | 4-6h   | Not Started |
| No cache validation tests    | P0       | 2h     | Not Started |
| Magic numbers                | P2       | 2-3h   | Not Started |
| Tight coupling               | P1       | 4-6h   | Not Started |
| Progress estimation          | P3       | 3-4h   | Not Started |

---

## Changelog

### 2026-01-19

- Initial TODO.md created
- Identified 16 tasks across 4 priority levels
- Defined test suite roadmap
- Mapped dependencies between tasks
