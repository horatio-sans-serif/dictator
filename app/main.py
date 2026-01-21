from __future__ import annotations

import argparse
import curses
import hashlib
import json
import os
import queue
import subprocess
import sys
import textwrap
import threading
import time
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
import whisper
from scipy.signal import butter, sosfiltfilt, lfilter


class AudioProcessingError(Exception):
    """Base exception for audio processing failures."""

    pass


# Audio processing constants
MIN_FRAME_LENGTH = 1024
MAX_FRAME_LENGTH = 4096
FRAME_DURATION_S = 0.08

# Transcription time estimation constants
MIN_TRANSCRIBE_TIME_S = 15.0
TRANSCRIBE_TIME_MULTIPLIER = 1.2

# Diarization time estimation constants
MIN_DIARIZATION_TIME_S = 8.0
DIARIZATION_TIME_MULTIPLIER = 0.6

# Sample rate validation constants
MIN_SAMPLE_RATE = 1000
MAX_SAMPLE_RATE = 192000

warnings.filterwarnings(
    "ignore",
    message="torchcodec is not installed correctly so built-in audio decoding will fail.*",
    category=UserWarning,
    module=r"pyannote\.audio\.core\.io",
)
warnings.filterwarnings(
    "ignore",
    message=r"invalid escape sequence '\\s'",
    category=SyntaxWarning,
    module=r"pyannote\.database\..*",
)

try:
    from pyannote.audio import Pipeline  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Pipeline = None  # type: ignore

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
CACHE_DIR = DATA / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
ENHANCEMENT_VERSION = 6
CACHE_SCHEMA_VERSION = 5
DEFAULT_PRESET = "general"
DEFAULT_ENHANCER = "soap"
SUPPORTED_ENHANCERS = {
    "soap": "Built-in Soap-inspired EQ/compression chain",
    "resemble": "Resemble Enhance neural model",
}
# Allowed audio file extensions for path validation (security)
ALLOWED_AUDIO_EXTENSIONS = frozenset(
    {".wav", ".m4a", ".mp3", ".flac", ".ogg", ".aac", ".wma", ".aiff", ".aif"}
)


def validate_audio_path(audio_path: Path) -> Path:
    """Validate and sanitize an audio file path.

    Security checks:
    - Resolves to absolute path to prevent path traversal
    - Verifies the path exists and is a regular file
    - Validates file extension is an allowed audio format

    Args:
        audio_path: Path to validate

    Returns:
        Resolved absolute Path if valid

    Raises:
        ValueError: If the path fails validation
    """
    resolved = audio_path.expanduser().resolve()

    if not resolved.exists():
        raise ValueError(f"Audio file does not exist: {resolved}")

    if not resolved.is_file():
        raise ValueError(f"Path is not a regular file: {resolved}")

    suffix = resolved.suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise ValueError(
            f"Invalid audio file extension '{suffix}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"
        )

    return resolved


def get_auth_token(env_var: str = "PYANNOTE_AUTH_TOKEN") -> Optional[str]:
    """Retrieve and validate an authentication token from environment.

    Security checks:
    - Returns None if the environment variable is not set
    - Returns None if the value is empty or contains only whitespace
    - Strips leading/trailing whitespace from valid tokens

    Args:
        env_var: Name of the environment variable to read

    Returns:
        Validated token string, or None if not set or invalid
    """
    token = os.environ.get(env_var)
    if token is None:
        return None
    token = token.strip()
    if not token:
        return None
    return token


MIC_PRESETS: Dict[str, Dict[str, float]] = {
    "general": {
        "low_cut_hz": 70.0,
        "high_cut_hz": 16000.0,
        "presence_boost_db": 1.5,
        "presence_freq_hz": 3500.0,
        "air_boost_db": 0.8,
        "air_freq_hz": 12000.0,
        "compression_target_db": -23.0,
        "compression_ratio": 2.4,
        "max_boost_db": 12.0,
        "max_cut_db": -8.0,
        "denoise_wet": 0.5,
        "use_eq": True,
        "use_compression": True,
        "use_bandpass": True,
        "use_bump_suppression": True,
        "use_transient_clipper": True,
        "use_echo_suppression": True,
        "use_normalize": True,
    },
    "sm7b": {
        "low_cut_hz": 90.0,
        "high_cut_hz": 16000.0,
        "presence_boost_db": 3.0,
        "presence_freq_hz": 3600.0,
        "air_boost_db": 1.0,
        "air_freq_hz": 12000.0,
        "compression_target_db": -24.0,
        "compression_ratio": 2.6,
        "max_boost_db": 14.0,
        "max_cut_db": -10.0,
        "denoise_wet": 0.55,
        "use_eq": True,
        "use_compression": True,
        "use_bandpass": True,
        "use_bump_suppression": True,
        "use_transient_clipper": True,
        "use_echo_suppression": True,
        "use_normalize": True,
    },
    "sm58": {
        "low_cut_hz": 80.0,
        "high_cut_hz": 15500.0,
        "presence_boost_db": 1.4,
        "presence_freq_hz": 3200.0,
        "air_boost_db": 0.5,
        "air_freq_hz": 11000.0,
        "compression_target_db": -22.0,
        "compression_ratio": 2.2,
        "max_boost_db": 11.0,
        "max_cut_db": -8.0,
        "denoise_wet": 0.45,
        "use_eq": True,
        "use_compression": True,
        "use_bandpass": True,
        "use_bump_suppression": True,
        "use_transient_clipper": True,
        "use_echo_suppression": True,
        "use_normalize": True,
    },
    "podmic": {
        "low_cut_hz": 85.0,
        "high_cut_hz": 16000.0,
        "presence_boost_db": 2.0,
        "presence_freq_hz": 3600.0,
        "air_boost_db": 0.9,
        "air_freq_hz": 12500.0,
        "compression_target_db": -23.5,
        "compression_ratio": 2.4,
        "max_boost_db": 13.0,
        "max_cut_db": -9.0,
        "denoise_wet": 0.5,
        "use_eq": True,
        "use_compression": True,
        "use_bandpass": True,
        "use_bump_suppression": True,
        "use_transient_clipper": True,
        "use_echo_suppression": True,
        "use_normalize": True,
    },
    "usb-condenser": {
        "low_cut_hz": 65.0,
        "high_cut_hz": 18000.0,
        "presence_boost_db": 0.8,
        "presence_freq_hz": 2600.0,
        "air_boost_db": 0.3,
        "air_freq_hz": 13000.0,
        "compression_target_db": -21.5,
        "compression_ratio": 2.0,
        "max_boost_db": 10.0,
        "max_cut_db": -7.0,
        "denoise_wet": 0.4,
        "use_eq": True,
        "use_compression": True,
        "use_bandpass": True,
        "use_bump_suppression": True,
        "use_transient_clipper": True,
        "use_echo_suppression": True,
        "use_normalize": True,
    },
    "iphone": {
        "low_cut_hz": 80.0,
        "high_cut_hz": 17000.0,
        "presence_boost_db": 0.8,
        "presence_freq_hz": 3200.0,
        "air_boost_db": 0.25,
        "air_freq_hz": 13500.0,
        "compression_target_db": -21.5,
        "compression_ratio": 1.6,
        "max_boost_db": 8.0,
        "max_cut_db": -3.0,
        "denoise_wet": 0.25,
        "transient_threshold_db": -0.5,
        "bump_threshold_multiplier": 12.0,
        "bump_absolute_floor": 0.25,
        "bump_max_duration_ms": 60.0,
        "bump_pad_ms": 6.0,
        "use_eq": False,
        "eq_rms_threshold": 0.02,
        "eq_crest_threshold": 35.0,
        "use_compression": False,
        "comp_rms_threshold": 0.02,
        "comp_crest_threshold": 35.0,
        "use_bandpass": True,
        "use_bump_suppression": True,
        "use_transient_clipper": True,
        "use_echo_suppression": False,
        "use_normalize": True,
    },
}

# In-memory cache for file hashes keyed by (path, mtime, size)
_file_hash_cache: Dict[Tuple[str, float, int], str] = {}
_FILE_HASH_CACHE_MAX_SIZE = 100


def compute_file_hash(path: Path) -> str:
    stat = path.stat()
    cache_key = (str(path), stat.st_mtime, stat.st_size)
    if cache_key in _file_hash_cache:
        return _file_hash_cache[cache_key]
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    result = digest.hexdigest()
    # Limit cache size by removing oldest entry if needed
    if len(_file_hash_cache) >= _FILE_HASH_CACHE_MAX_SIZE:
        _file_hash_cache.pop(next(iter(_file_hash_cache)))
    _file_hash_cache[cache_key] = result
    return result


def format_timestamp(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def preprocess_audio(
    audio_path: Path,
    audio_hash: str,
    preset_name: str = DEFAULT_PRESET,
    create_compare: bool = False,
    compare_duration: Optional[float] = None,
    enabled: bool = True,
    backend: str = DEFAULT_ENHANCER,
) -> Tuple[Path, Optional[Path]]:
    if not enabled:
        return audio_path, None
    backend_key = backend.lower().strip()
    if backend_key not in SUPPORTED_ENHANCERS:
        backend_key = DEFAULT_ENHANCER
    if backend_key == "resemble":
        return preprocess_audio_resemble(
            audio_path,
            audio_hash,
            preset_name,
            create_compare=create_compare,
            compare_duration=compare_duration,
        )
    return preprocess_audio_soap(
        audio_path,
        audio_hash,
        preset_name,
        create_compare=create_compare,
        compare_duration=compare_duration,
    )


def preprocess_audio_soap(
    audio_path: Path,
    audio_hash: str,
    preset_name: str,
    create_compare: bool,
    compare_duration: Optional[float],
) -> Tuple[Path, Optional[Path]]:
    enhanced_path, compare_path = _enhancement_paths(
        audio_hash, "soap", preset_name, create_compare
    )
    compare_ready = compare_path and compare_path.exists()
    if enhanced_path.exists() and (not create_compare or compare_ready):
        return enhanced_path, compare_path if compare_ready else None

    samples, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
    if samples.size == 0:
        raise RuntimeError("Audio file contained no samples")

    preset = MIC_PRESETS.get(preset_name, MIC_PRESETS[DEFAULT_PRESET])
    if not bool(preset.get("process", True)):
        return audio_path, None
    low_cut = float(preset.get("low_cut_hz", 80.0))
    high_cut = float(preset.get("high_cut_hz", 8000.0))

    processed = remove_dc_offset(samples)
    rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    crest_factor = peak / (rms + 1e-9) if rms > 0 else 0.0
    eq_enabled = bool(preset.get("use_eq", True))
    comp_enabled = bool(preset.get("use_compression", True))
    if not eq_enabled:
        eq_rms_threshold = float(preset.get("eq_rms_threshold", 0.0))
        eq_crest_threshold = float(preset.get("eq_crest_threshold", 0.0))
        if (eq_rms_threshold > 0 and rms < eq_rms_threshold) or (
            eq_crest_threshold > 0 and crest_factor > eq_crest_threshold
        ):
            eq_enabled = True
    if not comp_enabled:
        comp_rms_threshold = float(preset.get("comp_rms_threshold", 0.0))
        comp_crest_threshold = float(preset.get("comp_crest_threshold", 0.0))
        if (comp_rms_threshold > 0 and rms < comp_rms_threshold) or (
            comp_crest_threshold > 0 and crest_factor > comp_crest_threshold
        ):
            comp_enabled = True
    if preset.get("use_bandpass", True):
        processed = apply_bandpass_filter(
            processed, sample_rate, lowcut=low_cut, highcut=high_cut
        )
    processed = reduce_background_noise(processed, sample_rate)
    denoise_wet = float(preset.get("denoise_wet", 0.6))
    denoise_wet = min(1.0, max(0.0, denoise_wet))
    processed = denoise_wet * processed + (1 - denoise_wet) * samples
    if preset.get("use_echo_suppression", True):
        processed = suppress_echo(
            processed,
            sample_rate,
            delay_ms=float(preset.get("echo_delay_ms", 60.0)),
            attenuation=float(preset.get("echo_attenuation", 0.35)),
        )
    if preset.get("use_transient_clipper", True):
        processed = remove_transient_peaks(
            processed,
            threshold_db=float(preset.get("transient_threshold_db", -5.0)),
            window=int(preset.get("transient_window", 256)),
        )
    if preset.get("use_bump_suppression", True):
        processed = suppress_mic_bumps(
            processed,
            sample_rate,
            threshold_multiplier=float(preset.get("bump_threshold_multiplier", 6.0)),
            absolute_floor=float(preset.get("bump_absolute_floor", 0.12)),
            max_duration_ms=float(preset.get("bump_max_duration_ms", 80.0)),
            pad_ms=float(preset.get("bump_pad_ms", 10.0)),
        )
    if eq_enabled:
        processed = apply_high_shelf(
            processed,
            sample_rate,
            freq_hz=float(preset.get("presence_freq_hz", 3200.0)),
            gain_db=float(preset.get("presence_boost_db", 0.0)),
        )
        processed = apply_high_shelf(
            processed,
            sample_rate,
            freq_hz=float(preset.get("air_freq_hz", 9500.0)),
            gain_db=float(preset.get("air_boost_db", 0.0)),
        )
    if comp_enabled:
        processed = apply_dynamic_compression(
            processed,
            sample_rate,
            target_db=float(preset.get("compression_target_db", -23.0)),
            max_boost_db=float(preset.get("max_boost_db", 18.0)),
            max_cut_db=float(preset.get("max_cut_db", -12.0)),
            compression_ratio=float(preset.get("compression_ratio", 3.0)),
        )
    if preset.get("use_normalize", True):
        processed = normalize_volume(processed)

    sf.write(enhanced_path, processed, sample_rate, subtype="FLOAT")

    produced_compare: Optional[Path] = None
    if create_compare and compare_path is not None:
        produced_compare = export_debug_compare(
            samples,
            processed,
            sample_rate,
            compare_path,
            max_duration_s=compare_duration,
        )
        del samples

    return enhanced_path, produced_compare


def preprocess_audio_resemble(
    audio_path: Path,
    audio_hash: str,
    preset_name: str,
    create_compare: bool,
    compare_duration: Optional[float],
) -> Tuple[Path, Optional[Path]]:
    enhanced_path, compare_path = _enhancement_paths(
        audio_hash, "resemble", preset_name, create_compare
    )
    compare_ready = compare_path and compare_path.exists()
    if enhanced_path.exists() and (not create_compare or compare_ready):
        return enhanced_path, compare_path if compare_ready else None

    try:
        from resemble_enhance.enhancer.inference import enhance as resemble_enhance  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Resemble Enhance is unavailable; run `uv sync` to install dependencies"
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
    if samples.size == 0:
        raise RuntimeError("Audio file contained no samples")
    waveform = torch.from_numpy(samples).to(torch.float32)

    # CPU runs are expensive, so drop to 32 steps to keep latency reasonable.
    nfe = 32 if device.type == "cpu" else 64
    enhanced_tensor, enhanced_sr = resemble_enhance(
        dwav=waveform,
        sr=int(sample_rate),
        device=device,
        nfe=nfe,
        solver="midpoint",
        lambd=1.0,
        tau=0.5,
    )
    enhanced = enhanced_tensor.detach().cpu().numpy()
    sf.write(enhanced_path, enhanced, int(enhanced_sr), subtype="PCM_16")

    produced_compare: Optional[Path] = None
    if create_compare and compare_path is not None:
        original = samples.copy()
        if int(sample_rate) != int(enhanced_sr):
            original = librosa.resample(
                original, orig_sr=int(sample_rate), target_sr=int(enhanced_sr)
            )
        produced_compare = export_debug_compare(
            original,
            enhanced,
            int(enhanced_sr),
            compare_path,
            max_duration_s=compare_duration,
        )

    return enhanced_path, produced_compare


def _enhancement_paths(
    audio_hash: str,
    backend: str,
    preset_name: str,
    create_compare: bool,
) -> Tuple[Path, Optional[Path]]:
    backend_slug = backend.replace(" ", "-")
    preset_slug = (preset_name or "none").replace(" ", "-")
    enhanced_path = (
        CACHE_DIR
        / f"{audio_hash}-enhanced-{backend_slug}-{preset_slug}-v{ENHANCEMENT_VERSION}.wav"
    )
    compare_path = (
        CACHE_DIR
        / f"{audio_hash}-compare-{backend_slug}-{preset_slug}-v{ENHANCEMENT_VERSION}.wav"
        if create_compare
        else None
    )
    return enhanced_path, compare_path


def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    mean_value = float(np.mean(audio))
    if abs(mean_value) < 1e-7:
        return audio
    return audio - mean_value


def apply_bandpass_filter(
    audio: np.ndarray, sample_rate: int, lowcut: float = 80.0, highcut: float = 8000.0
) -> np.ndarray:
    if not MIN_SAMPLE_RATE <= sample_rate <= MAX_SAMPLE_RATE:
        raise ValueError(
            f"sample_rate must be between {MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE}, got {sample_rate}"
        )
    nyquist = 0.5 * sample_rate
    low = max(lowcut / nyquist, 0.0)
    high = min(highcut / nyquist, 0.999)
    if not 0 < low < high:
        return audio
    sos = butter(6, [low, high], btype="band", output="sos")
    return sosfiltfilt(sos, audio)


def reduce_background_noise(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    hop_length = 512
    win_length = 2048
    stft = librosa.stft(audio, n_fft=win_length, hop_length=hop_length)
    magnitude, phase = librosa.magphase(stft)
    noise_frames = max(
        1, min(magnitude.shape[1] // 6, int(sample_rate * 0.3 / hop_length))
    )
    noise_profile = np.median(magnitude[:, :noise_frames], axis=1, keepdims=True)
    noise_threshold = noise_profile * 1.5
    attenuation = np.where(magnitude > noise_threshold, 1.0, 0.35)
    magnitude_denoised = magnitude * attenuation
    stft_denoised = magnitude_denoised * phase
    reduced = librosa.istft(stft_denoised, hop_length=hop_length, length=len(audio))
    return reduced


def suppress_echo(
    audio: np.ndarray,
    sample_rate: int,
    delay_ms: float = 60.0,
    attenuation: float = 0.35,
) -> np.ndarray:
    delay_samples = int(sample_rate * delay_ms / 1000.0)
    if delay_samples <= 0 or delay_samples >= len(audio):
        return audio
    output = np.copy(audio)
    output[delay_samples:] -= attenuation * audio[:-delay_samples]
    return output


def remove_transient_peaks(
    audio: np.ndarray, threshold_db: float = -5.0, window: int = 256
) -> np.ndarray:
    if audio.size == 0:
        return audio
    clipped = np.copy(audio)
    abs_audio = np.abs(audio)
    rms = float(np.sqrt(np.mean(abs_audio**2)))
    if rms < 1e-6:
        return audio
    target = 10 ** (threshold_db / 20.0)
    dynamic_threshold = max(target, rms * 6)
    spike_indices = np.where(abs_audio > dynamic_threshold)[0]
    if spike_indices.size == 0:
        return audio
    half_window = max(1, window // 2)
    for idx in spike_indices:
        start = max(0, idx - half_window)
        end = min(len(audio), idx + half_window)
        neighborhood = np.concatenate((audio[start:idx], audio[idx + 1 : end]))
        replacement = float(np.median(neighborhood)) if neighborhood.size else 0.0
        clipped[idx] = replacement
    return clipped


def normalize_volume(
    audio: np.ndarray, target_rms_db: float = -20.0, peak: float = 0.98
) -> np.ndarray:
    rms = float(np.sqrt(np.mean(audio**2)))
    max_val = float(np.max(np.abs(audio))) if audio.size else 0.0
    if rms < 1e-6 or max_val == 0.0:
        return audio
    target_linear = 10 ** (target_rms_db / 20.0)
    gain_rms = target_linear / (rms + 1e-9)
    gain_peak = peak / (max_val + 1e-9)
    gain = min(gain_rms, gain_peak)
    normalized = audio * gain
    return np.clip(normalized, -1.0, 1.0)


def apply_dynamic_compression(
    audio: np.ndarray,
    sample_rate: int,
    target_db: float = -23.0,
    max_boost_db: float = 18.0,
    max_cut_db: float = -12.0,
    compression_ratio: float = 3.0,
) -> np.ndarray:
    if audio.size == 0:
        return audio

    frame_length = min(
        MAX_FRAME_LENGTH, max(MIN_FRAME_LENGTH, int(sample_rate * FRAME_DURATION_S))
    )
    hop_length = frame_length // 4
    rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]
    rms = np.maximum(rms, 1e-7)
    env_db = 20.0 * np.log10(rms)

    gains_db = np.empty_like(env_db)
    for idx, level_db in enumerate(env_db):
        delta = target_db - level_db
        if delta > 0:
            gains_db[idx] = min(delta, max_boost_db)
        else:
            gains_db[idx] = max(delta / compression_ratio, max_cut_db)

    gains_linear = 10.0 ** (gains_db / 20.0)
    frame_centers = np.arange(len(gains_linear)) * hop_length + frame_length / 2.0
    sample_positions = np.arange(len(audio))
    gain_curve = np.interp(
        sample_positions,
        frame_centers,
        gains_linear,
        left=gains_linear[0],
        right=gains_linear[-1],
    )
    processed = audio * gain_curve
    return np.clip(processed, -1.0, 1.0)


def apply_high_shelf(
    audio: np.ndarray,
    sample_rate: int,
    freq_hz: float,
    gain_db: float,
    slope: float = 0.75,
) -> np.ndarray:
    if audio.size == 0 or abs(gain_db) < 0.1:
        return audio
    freq_hz = max(10.0, min(freq_hz, sample_rate / 2 - 100.0))
    w0 = 2 * np.pi * freq_hz / sample_rate
    A = 10 ** (gain_db / 40.0)
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / 2 * np.sqrt((A + 1 / A) * (1 / slope - 1) + 2)

    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha

    b = np.array([b0 / a0, b1 / a0, b2 / a0])
    a = np.array([1.0, a1 / a0, a2 / a0])
    filtered = lfilter(b, a, audio)
    return np.clip(filtered, -1.0, 1.0)


def suppress_mic_bumps(
    audio: np.ndarray,
    sample_rate: int,
    threshold_multiplier: float = 6.0,
    absolute_floor: float = 0.12,
    max_duration_ms: float = 80.0,
    pad_ms: float = 10.0,
) -> np.ndarray:
    if audio.size == 0:
        return audio

    rms = float(np.sqrt(np.mean(audio**2)))
    threshold = max(absolute_floor, rms * threshold_multiplier)
    mask = np.abs(audio) >= threshold
    if not np.any(mask):
        return audio

    processed = np.copy(audio)
    max_len = max(1, int(sample_rate * max_duration_ms / 1000.0))
    pad = max(1, int(sample_rate * pad_ms / 1000.0))
    length = len(audio)
    idx = 0

    while idx < length:
        if not mask[idx]:
            idx += 1
            continue
        start = idx
        while idx < length and mask[idx]:
            idx += 1
        end = idx
        segment_len = end - start
        if segment_len <= max_len:
            left = max(0, start - pad)
            right = min(length, end + pad)
            fill_len = right - left
            if fill_len <= 0:
                continue
            left_val = processed[left - 1] if left > 0 else 0.0
            right_val = processed[right] if right < length else processed[-1]
            interp = np.linspace(left_val, right_val, fill_len, endpoint=False)
            processed[left:right] = interp
        # longer bursts are left untouched to avoid erasing speech

    return processed


def generate_tone(
    sample_rate: int, frequency_hz: float, duration_s: float, amplitude: float = 0.2
) -> np.ndarray:
    samples = max(1, int(sample_rate * max(duration_s, 0.01)))
    t = np.linspace(0, duration_s, samples, endpoint=False)
    tone = amplitude * np.sin(2 * np.pi * frequency_hz * t)
    window = np.hanning(samples)
    return tone * window


def export_debug_compare(
    original: np.ndarray,
    enhanced: np.ndarray,
    sample_rate: int,
    out_path: Path,
    chunk_seconds: float = 5.0,
    switch_beep_hz: float = 2000.0,
    round_beep_hz: float = 1200.0,
    switch_duration_ms: float = 350.0,
    round_duration_ms: float = 220.0,
    round_beeps: int = 2,
    beep_gap_ms: float = 120.0,
    lead_silence_ms: float = 80.0,
    max_duration_s: Optional[float] = 30.0,
) -> Path:
    if original.size == 0 or enhanced.size == 0:
        raise RuntimeError("Cannot create compare track from empty audio")
    length = min(len(original), len(enhanced))
    chunk_samples = max(1, int(sample_rate * max(chunk_seconds, 0.1)))
    segments: List[np.ndarray] = []
    switch_tone = generate_tone(
        sample_rate, switch_beep_hz, switch_duration_ms / 1000.0
    )
    round_tone = generate_tone(sample_rate, round_beep_hz, round_duration_ms / 1000.0)
    gap = np.zeros(int(sample_rate * (beep_gap_ms / 1000.0)))
    lead = np.zeros(int(sample_rate * (lead_silence_ms / 1000.0)))
    round_seq = []
    for _ in range(max(0, round_beeps)):
        round_seq.append(lead)
        round_seq.append(round_tone)
        round_seq.append(gap)
    round_seq = np.concatenate(round_seq) if round_seq else np.zeros(0)

    max_samples = (
        None
        if not max_duration_s or max_duration_s <= 0
        else int(sample_rate * max_duration_s)
    )
    used_samples = 0

    for start in range(0, length, chunk_samples):
        end = min(length, start + chunk_samples)
        if max_samples is not None and used_samples >= max_samples:
            break
        if max_samples is not None and used_samples + (end - start) > max_samples:
            end = start + (max_samples - used_samples)
        segments.append(original[start:end])
        used_samples += end - start
        if max_samples is not None and used_samples >= max_samples:
            break
        segments.append(lead)
        segments.append(switch_tone)
        segments.append(lead)
        segments.append(enhanced[start:end])
        if round_seq.size:
            segments.append(round_seq)
    combined = np.concatenate(segments)
    sf.write(out_path, combined, sample_rate, subtype="FLOAT")
    return out_path


def load_waveform_for_pipeline(
    audio_path: Path, sample_rate: int = 16000
) -> Tuple["torch.Tensor", int]:
    samples, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    if samples.size == 0:
        raise RuntimeError("Audio file contained no samples")
    waveform = torch.from_numpy(samples).unsqueeze(0)
    return waveform, sample_rate


def prepare_dependencies(
    model_name: str,
) -> Tuple[Optional["whisper.Whisper"], Optional["Pipeline"], bool]:
    print(
        f"Preparing Whisper model '{model_name}' (downloads may take a while)...",
        flush=True,
    )
    whisper_model: Optional["whisper.Whisper"] = None
    diarization_pipeline: Optional["Pipeline"] = None
    pipeline_attempted = False
    try:
        whisper_model = whisper.load_model(model_name)
    except Exception as exc:
        print(f"Failed to load Whisper model {model_name}: {exc}", file=sys.stderr)
        raise

    if Pipeline is not None:
        token = get_auth_token()
        print("Preparing diarization pipeline...", flush=True)
        try:
            pipeline_kwargs = {}
            if token:
                pipeline_kwargs["use_auth_token"] = token
            pipeline_attempted = True
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", **pipeline_kwargs
            )
        except Exception as exc:
            print(
                f"Warning: could not initialize pyannote pipeline: {exc}",
                file=sys.stderr,
            )
    else:
        print("pyannote not available; diarization will be disabled.")
        pipeline_attempted = True

    return whisper_model, diarization_pipeline, pipeline_attempted


def safe_addstr(
    window: "curses._CursesWindow", y: int, x: int, text: str, attr: int = 0
) -> None:
    height, width = window.getmaxyx()
    if y < 0 or y >= height or x >= width:
        return
    try:
        window.addnstr(y, x, text, max(0, width - x - 1), attr)
    except curses.error:
        # Ignore curses errors caused by writing to the bottom-right cell.
        pass


class AudioPlayer:
    def __init__(self, audio_path: Path) -> None:
        self.audio_path = audio_path
        self.process: Optional[subprocess.Popen[bytes]] = None
        self.lock = threading.Lock()
        self.paused = False

    def set_audio_path(self, audio_path: Path) -> None:
        with self.lock:
            self.audio_path = audio_path
            # Stop any currently playing audio so future playbacks use the new file.
            self.stop()

    def play_from(self, start_time: float) -> None:
        with self.lock:
            self.stop()
            # Security: Validate path before passing to subprocess
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


class TranscriptionWorker(threading.Thread):
    def __init__(
        self,
        audio_path: Path,
        messages: "queue.Queue[Dict[str, object]]",
        model_name: str = "base",
        language: str = "en",
        whisper_model: Optional["whisper.Whisper"] = None,
        diarization_pipeline: Optional["Pipeline"] = None,
        pipeline_attempted: bool = False,
        preset_name: str = DEFAULT_PRESET,
        debug_compare: bool = False,
        debug_compare_duration: Optional[float] = None,
        enhancer_backend: str = DEFAULT_ENHANCER,
        enhancement_enabled: bool = True,
    ) -> None:
        super().__init__(daemon=True)
        self.audio_path = audio_path
        self.messages = messages
        self.model_name = model_name
        self.language = language
        self.stop_event = threading.Event()
        self.processed_audio_path = audio_path
        self.whisper_model = whisper_model
        self.diarization_pipeline = diarization_pipeline
        self._pipeline_checked = (
            diarization_pipeline is not None or Pipeline is None or pipeline_attempted
        )
        self.preset_name = preset_name if preset_name in MIC_PRESETS else DEFAULT_PRESET
        self.debug_compare = debug_compare
        if self.debug_compare:
            if debug_compare_duration is None:
                self.debug_compare_duration = 30.0
            elif debug_compare_duration > 0:
                self.debug_compare_duration = debug_compare_duration
            else:
                self.debug_compare_duration = None
        else:
            self.debug_compare_duration = None
        self.enhancement_enabled = enhancement_enabled
        if enhancer_backend not in SUPPORTED_ENHANCERS:
            enhancer_backend = DEFAULT_ENHANCER
        self.enhancer_backend = enhancer_backend

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

    def run(self) -> None:  # noqa: D401
        try:
            if not self.audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {self.audio_path}")

            audio_hash = compute_file_hash(self.audio_path)
            cache_file = CACHE_DIR / f"{audio_hash}.json"
            cached_segments = self._try_load_cache(cache_file, audio_hash)
            if cached_segments is not None:
                self._emit_progress(100, "Loaded cached transcription")
                for seg in cached_segments:
                    self._emit({"type": "segment", "segment": seg})
                self._emit({"type": "done", "from_cache": True})
                return

            processed_audio_path = self.audio_path
            compare_path: Optional[Path] = None
            if self.enhancement_enabled:
                try:
                    backend_label = (
                        "Resemble Enhance"
                        if self.enhancer_backend == "resemble"
                        else "Soap chain"
                    )
                    self._emit(
                        {
                            "type": "status",
                            "message": f"Enhancing audio with {backend_label}...",
                        }
                    )
                    processed_audio_path, compare_path = preprocess_audio(
                        self.audio_path,
                        audio_hash,
                        self.preset_name,
                        create_compare=self.debug_compare,
                        compare_duration=self.debug_compare_duration,
                        enabled=self.enhancement_enabled,
                        backend=self.enhancer_backend,
                    )
                    self._emit(
                        {"type": "processed_audio", "path": str(processed_audio_path)}
                    )
                    if compare_path is not None:
                        self._emit(
                            {
                                "type": "status",
                                "message": f"Debug compare saved to {compare_path}",
                            }
                        )
                    self._emit({"type": "status", "message": "Enhanced audio ready"})
                except Exception as exc:
                    self._emit(
                        {"type": "status", "message": f"Enhancement skipped: {exc}"}
                    )
                    processed_audio_path = self.audio_path
            else:
                self._emit(
                    {
                        "type": "status",
                        "message": "Enhancement disabled; using original audio",
                    }
                )

            self.processed_audio_path = processed_audio_path

            duration = 0.0
            try:
                duration = float(
                    librosa.get_duration(path=str(self.processed_audio_path))
                )
            except Exception:
                duration = 0.0

            transcribe_estimate = max(
                MIN_TRANSCRIBE_TIME_S, duration * TRANSCRIBE_TIME_MULTIPLIER
            )
            diarization_estimate = max(
                MIN_DIARIZATION_TIME_S, duration * DIARIZATION_TIME_MULTIPLIER
            )

            transcript_segments = self._run_stage(
                "Transcribing audio...",
                5,
                70,
                transcribe_estimate,
                self._transcribe,
            )
            if self.stop_event.is_set():
                return

            diarization_segments = self._run_stage(
                "Attributing speakers...",
                70,
                90,
                diarization_estimate,
                self._diarize,
            )
            if self.stop_event.is_set():
                return

            final_segments = self._merge_segments(
                transcript_segments, diarization_segments
            )
            cache_data = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "audio_hash": audio_hash,
                "model": self.model_name,
                "language": self.language,
                "enhancement_version": ENHANCEMENT_VERSION
                if self.enhancement_enabled
                else 0,
                "enhancement_enabled": self.enhancement_enabled,
                "enhancement_preset": self.preset_name
                if self.enhancement_enabled
                else "none",
                "enhancement_backend": self.enhancer_backend
                if self.enhancement_enabled
                else "none",
                "segments": final_segments,
            }
            cache_file.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")
            self._emit_progress(100, "Transcription complete")

            for seg in final_segments:
                if self.stop_event.is_set():
                    break
                self._emit({"type": "segment", "segment": seg})

            self._emit({"type": "done", "from_cache": False})
        except Exception as exc:
            self._emit({"type": "error", "message": str(exc)})

    def _run_stage(
        self,
        message: str,
        start_percent: int,
        end_percent: int,
        estimate_seconds: float,
        func: Callable[[], Iterable],
    ) -> Iterable:
        progress_done = threading.Event()
        start_time = time.time()
        last_emitted = start_percent

        def updater() -> None:
            nonlocal last_emitted
            while not progress_done.is_set():
                if estimate_seconds > 0:
                    elapsed = time.time() - start_time
                    fraction = min(0.99, elapsed / estimate_seconds)
                    percent = start_percent + int(
                        fraction * (end_percent - start_percent)
                    )
                    percent = max(start_percent, min(percent, end_percent - 1))
                else:
                    percent = start_percent
                if percent > last_emitted:
                    last_emitted = percent
                    self._emit_progress(percent, message)
                time.sleep(0.5)

        self._emit_progress(start_percent, message)
        updater_thread = threading.Thread(target=updater, daemon=True)
        updater_thread.start()
        try:
            result = func()
        finally:
            progress_done.set()
            updater_thread.join()
        self._emit_progress(end_percent, message)
        return result

    def _transcribe(self) -> List[Dict[str, object]]:
        if self.whisper_model is None:
            self.whisper_model = whisper.load_model(self.model_name)
        model = self.whisper_model
        result = model.transcribe(
            str(self.processed_audio_path),
            language=self.language,
            verbose=False,
        )
        segments = result.get("segments", [])
        cleaned: List[Dict[str, object]] = []
        for seg in segments:
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            cleaned.append(
                {
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                    "text": text,
                }
            )
        if not cleaned:
            raise RuntimeError("Whisper did not return any transcript segments")
        return cleaned

    def _diarize(self) -> List[Tuple[float, float, str]]:
        pipeline = self._get_diarization_pipeline()
        if pipeline is None:
            self._emit(
                {
                    "type": "status",
                    "message": "pyannote unavailable; using single speaker",
                }
            )
            return []

        try:
            try:
                waveform, sample_rate = load_waveform_for_pipeline(
                    self.processed_audio_path
                )
                diarization = pipeline(
                    {
                        "uri": str(self.processed_audio_path),
                        "waveform": waveform,
                        "sample_rate": sample_rate,
                    }
                )
            except Exception:
                diarization = pipeline(str(self.processed_audio_path))
            segments: List[Tuple[float, float, str]] = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append((float(turn.start), float(turn.end), speaker))
            segments.sort()
            return segments
        except Exception:
            self._emit(
                {
                    "type": "status",
                    "message": "Speaker diarization failed; using single speaker",
                }
            )
            return []

    def _get_diarization_pipeline(self) -> Optional["Pipeline"]:
        if Pipeline is None:
            return None
        if self.diarization_pipeline is not None:
            return self.diarization_pipeline
        if self._pipeline_checked:
            return None
        self._pipeline_checked = True
        token = get_auth_token()
        try:
            pipeline_kwargs = {}
            if token:
                pipeline_kwargs["use_auth_token"] = token
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", **pipeline_kwargs
            )
            self.diarization_pipeline = pipeline
            return pipeline
        except Exception:
            return None

    def _try_load_cache(
        self, cache_file: Path, audio_hash: str
    ) -> Optional[List[Dict[str, object]]]:
        if not cache_file.exists():
            return None
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception as exc:
            self._emit({"type": "status", "message": f"Ignoring corrupt cache: {exc}"})
            return None

        valid, reason = self._cache_is_valid(data, audio_hash)
        if not valid:
            if reason:
                self._emit(
                    {
                        "type": "status",
                        "message": f"Cache invalid: {reason}; recomputing",
                    }
                )
            return None

        segments = data.get("segments", [])
        if not isinstance(segments, list):
            self._emit({"type": "status", "message": "Cache invalid: segments missing"})
            return None
        return segments

    def _cache_is_valid(
        self, data: Dict[str, object], audio_hash: str
    ) -> Tuple[bool, str]:
        # Early exit checks ordered by most-likely-to-fail first
        if not isinstance(data, dict):
            return False, "malformed data"
        # Check audio hash first - most likely to differ for new files
        if data.get("audio_hash") != audio_hash:
            return False, "audio hash mismatch"
        # Schema version changes invalidate all caches
        if data.get("schema_version") != CACHE_SCHEMA_VERSION:
            return False, "schema version changed"
        # Check required metadata exists before comparing values
        model = data.get("model")
        language = data.get("language")
        if not model or not language:
            return False, "metadata missing"
        # Compare model and language
        if model != self.model_name:
            return False, f"model mismatch ({model} != {self.model_name})"
        if language != self.language:
            return False, "language changed"
        # Enhancement checks grouped together
        enhancement = data.get("enhancement_version")
        enabled_flag = data.get("enhancement_enabled")
        if enhancement is None or enabled_flag is None:
            return False, "enhancement metadata missing"
        expected_version = ENHANCEMENT_VERSION if self.enhancement_enabled else 0
        if (
            enhancement != expected_version
            or bool(enabled_flag) != self.enhancement_enabled
        ):
            return False, "enhancement configuration changed"
        expected_preset = self.preset_name if self.enhancement_enabled else "none"
        if data.get("enhancement_preset") != expected_preset:
            return False, "preset changed"
        expected_backend = self.enhancer_backend if self.enhancement_enabled else "none"
        if data.get("enhancement_backend") != expected_backend:
            return False, "enhancement backend changed"
        return True, ""

    def _merge_segments(
        self,
        transcript_segments: List[Dict[str, object]],
        diarization_segments: List[Tuple[float, float, str]],
    ) -> List[Dict[str, object]]:
        def best_speaker(start: float, end: float) -> str:
            label = "SPEAKER_00"
            best_overlap = 0.0
            for ds, de, speaker in diarization_segments:
                overlap = max(0.0, min(end, de) - max(start, ds))
                if overlap > best_overlap:
                    best_overlap = overlap
                    label = speaker
            return label

        speaker_labels = (
            [best_speaker(seg["start"], seg["end"]) for seg in transcript_segments]
            if diarization_segments
            else ["SPEAKER_00"] * len(transcript_segments)
        )

        speaker_ids: Dict[str, int] = {}
        next_id = 1
        output: List[Dict[str, object]] = []
        for seg, label in zip(transcript_segments, speaker_labels):
            if label not in speaker_ids:
                speaker_ids[label] = next_id
                next_id += 1
            output.append(
                {
                    "start": round(float(seg["start"]), 3),
                    "end": round(float(seg["end"]), 3),
                    "speaker": speaker_ids[label],
                    "text": seg["text"],
                }
            )
        return output


@dataclass
class UISessionResult:
    segments: List[Dict[str, object]]
    error: Optional[str] = None


def run_ui(
    stdscr: "curses._CursesWindow",
    audio_path: Path,
    model_name: str,
    whisper_model: Optional["whisper.Whisper"],
    diarization_pipeline: Optional["Pipeline"],
    pipeline_attempted: bool,
    preset_name: str,
    debug_compare: bool,
    debug_compare_duration: Optional[float],
    enhancement_enabled: bool,
    enhancer_backend: str,
) -> UISessionResult:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)

    messages: "queue.Queue[Dict[str, object]]" = queue.Queue()
    worker = TranscriptionWorker(
        audio_path,
        messages,
        model_name=model_name,
        whisper_model=whisper_model,
        diarization_pipeline=diarization_pipeline,
        pipeline_attempted=pipeline_attempted,
        preset_name=preset_name,
        debug_compare=debug_compare,
        debug_compare_duration=debug_compare_duration,
        enhancement_enabled=enhancement_enabled,
        enhancer_backend=enhancer_backend,
    )
    worker.start()

    player = AudioPlayer(audio_path)
    segments: List[Dict[str, object]] = []

    current_index = 0
    progress_percent = 0
    progress_message = "Preparing..."
    status_message = "J/K navigate · Enter play · Space pause · Q quit"
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
                elif mtype == "processed_audio":
                    raw_path = msg.get("path")
                    if isinstance(raw_path, str):
                        try:
                            player.set_audio_path(Path(raw_path))
                            status_message = "Playback source: enhanced audio"
                        except Exception:
                            pass
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
                # Allow a short pause for the error to be visible before exit.
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


def run_transcribe_only(
    audio_path: Path,
    model_name: str,
    whisper_model: Optional["whisper.Whisper"],
    diarization_pipeline: Optional["Pipeline"],
    pipeline_attempted: bool,
    preset_name: str,
    debug_compare: bool,
    debug_compare_duration: Optional[float],
    enhancement_enabled: bool,
    enhancer_backend: str,
) -> int:
    messages: "queue.Queue[Dict[str, object]]" = queue.Queue()
    worker = TranscriptionWorker(
        audio_path,
        messages,
        model_name=model_name,
        whisper_model=whisper_model,
        diarization_pipeline=diarization_pipeline,
        pipeline_attempted=pipeline_attempted,
        preset_name=preset_name,
        debug_compare=debug_compare,
        debug_compare_duration=debug_compare_duration,
        enhancement_enabled=enhancement_enabled,
        enhancer_backend=enhancer_backend,
    )
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
                    print_segment(segment)
            elif mtype == "status":
                status = msg.get("message")
                if status:
                    print(f"[status] {status}")
            elif mtype == "progress":
                percent = msg.get("percent")
                message = msg.get("message", "")
                if percent is not None:
                    print(f"[progress] {int(percent)}% {message}")
                else:
                    print(f"[progress] {message}")
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
    """Cache text wrapping results to avoid repeated computation during render."""
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

    safe_addstr(stdscr, 0, 0, "Simple Transcriber CLI")

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

    # Determine which segment index should be at the top of the view.
    start_index = current_index
    total_rows = 0
    while start_index > 0:
        rows = segment_row_count(segments[start_index - 1], width)
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


def segment_row_count(segment: Dict[str, object], width: int) -> int:
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


def print_segment(segment: Dict[str, object]) -> None:
    start = format_timestamp(float(segment.get("start", 0.0)))
    end = format_timestamp(float(segment.get("end", 0.0)))
    speaker = int(segment.get("speaker", 0))
    text = str(segment.get("text", "")).strip()
    print(f"SPEAKER {speaker} ({start}-{end}): {text}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive CLI transcription viewer")
    parser.add_argument(
        "audio",
        type=str,
        nargs="?",
        help="Path to the audio file to transcribe (reads from stdin if not provided)",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Whisper model name (default: large-v3)",
    )
    parser.add_argument(
        "-T",
        "--transcribe-only",
        action="store_true",
        help="Skip the curses UI and stream diarized transcript to stdout",
    )
    parser.add_argument(
        "--preset",
        default=DEFAULT_PRESET,
        choices=sorted(MIC_PRESETS.keys()),
        help="Voice enhancement preset for the Soap chain (default: general)",
    )
    parser.add_argument(
        "--enhancer",
        default=DEFAULT_ENHANCER,
        choices=sorted(SUPPORTED_ENHANCERS.keys()),
        help="Audio enhancement backend: soap (EQ/comp) or resemble (neural)",
    )
    parser.add_argument(
        "--debug-enhanced-compare",
        action="store_true",
        help="Export a WAV interleaving original/enhanced audio in 5s chunks",
    )
    parser.add_argument(
        "--debug-enhanced-compare-duration",
        type=float,
        default=30.0,
        help="Seconds of audio to include in the compare WAV (0 = entire file)",
    )
    parser.add_argument(
        "--no-enhancement",
        action="store_true",
        help="Bypass the audio enhancement pipeline and feed Whisper the original audio",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    audio_path: Path
    temp_file: Optional[Path] = None

    if args.audio:
        try:
            audio_path = validate_audio_path(Path(args.audio))
        except ValueError as e:
            parser.error(str(e))
    else:
        import tempfile
        import atexit

        audio_data = sys.stdin.buffer.read()
        if not audio_data:
            parser.error("No audio data received from stdin")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            audio_path = Path(f.name)
            temp_file = audio_path

        print(
            f"Read {len(audio_data)} bytes from stdin to {audio_path}", file=sys.stderr
        )

        atexit.register(lambda: temp_file and temp_file.unlink(missing_ok=True))

    try:
        whisper_model, diarization_pipeline, pipeline_attempted = prepare_dependencies(
            args.model
        )
    except Exception:
        return 1

    if args.transcribe_only:
        return run_transcribe_only(
            audio_path,
            args.model,
            whisper_model,
            diarization_pipeline,
            pipeline_attempted,
            args.preset,
            args.debug_enhanced_compare,
            args.debug_enhanced_compare_duration,
            not args.no_enhancement,
            args.enhancer,
        )

    try:
        result: UISessionResult = curses.wrapper(
            lambda stdscr: run_ui(
                stdscr,
                audio_path,
                args.model,
                whisper_model,
                diarization_pipeline,
                pipeline_attempted,
                args.preset,
                args.debug_enhanced_compare,
                args.debug_enhanced_compare_duration,
                not args.no_enhancement,
                args.enhancer,
            )
        )
    except curses.error as exc:  # pragma: no cover - terminal size errors
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


if __name__ == "__main__":
    sys.exit(main())
