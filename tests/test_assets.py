from pathlib import Path


def test_audio_assets_have_labels() -> None:
    assets_dir = Path("test/assets")
    assert assets_dir.exists(), "test/assets directory missing"

    audio_files = sorted(assets_dir.glob("*.m4a"))
    assert audio_files, "No .m4a files found under test/assets"

    for audio in audio_files:
        transcript = audio.with_suffix(".txt")
        assert transcript.exists(), f"Missing transcript for {audio.name}"
        assert transcript.stat().st_size > 0, f"Transcript {transcript.name} is empty"
