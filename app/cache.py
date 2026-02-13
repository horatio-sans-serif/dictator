"""Transcription cache keyed by audio file SHA-256 hash."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
CACHE_DIR = DATA / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_SCHEMA_VERSION = 6

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
    if len(_file_hash_cache) >= _FILE_HASH_CACHE_MAX_SIZE:
        _file_hash_cache.pop(next(iter(_file_hash_cache)))
    _file_hash_cache[cache_key] = result
    return result


def cache_path_for(audio_hash: str) -> Path:
    return CACHE_DIR / f"{audio_hash}.json"


def load_cache(cache_file: Path, audio_hash: str) -> Optional[List[Dict[str, object]]]:
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("audio_hash") != audio_hash:
        return None
    if data.get("schema_version") != CACHE_SCHEMA_VERSION:
        return None
    segments = data.get("segments", [])
    if not isinstance(segments, list):
        return None
    return segments


def save_cache(cache_file: Path, audio_hash: str, segments: List[Dict[str, object]]) -> None:
    cache_data = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "audio_hash": audio_hash,
        "segments": segments,
    }
    cache_file.write_text(json.dumps(cache_data, indent=2), encoding="utf-8")
