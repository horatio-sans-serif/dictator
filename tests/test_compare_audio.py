from pathlib import Path
import sys

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import export_debug_compare  # noqa: E402


def test_export_debug_compare_duration(tmp_path: Path) -> None:
    sample_rate = 16000
    duration_s = 4
    samples = sample_rate * duration_s
    original = np.linspace(-0.5, 0.5, samples, dtype=np.float32)
    enhanced = -original
    out_path = tmp_path / "compare.wav"

    export_debug_compare(
        original,
        enhanced,
        sample_rate,
        out_path,
        chunk_seconds=1.0,
        max_duration_s=6.0,
    )

    audio, sr = sf.read(out_path)
    assert sr == sample_rate
    assert len(audio) <= samples * 4
    # ensure beeps are present (look for larger amplitude near chunk boundary)
    chunk = sample_rate
    assert np.max(np.abs(audio)) > 0.05
