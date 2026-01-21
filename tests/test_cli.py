"""Unit tests for CLI argument parsing."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import build_parser, MIC_PRESETS, SUPPORTED_ENHANCERS, DEFAULT_PRESET, DEFAULT_ENHANCER


class TestBuildParser:
    """Tests for CLI argument parser."""

    @pytest.mark.unit
    def test_parser_creation(self) -> None:
        """Should create parser without errors."""
        parser = build_parser()
        assert parser is not None

    @pytest.mark.unit
    def test_required_audio_argument(self) -> None:
        """Should require audio argument."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    @pytest.mark.unit
    def test_audio_argument_parsed(self) -> None:
        """Should parse audio argument."""
        parser = build_parser()
        args = parser.parse_args(["test.m4a"])
        assert args.audio == "test.m4a"

    @pytest.mark.unit
    def test_default_model(self) -> None:
        """Should use large-v3 as default model."""
        parser = build_parser()
        args = parser.parse_args(["test.m4a"])
        assert args.model == "large-v3"

    @pytest.mark.unit
    def test_custom_model(self) -> None:
        """Should accept custom model."""
        parser = build_parser()
        args = parser.parse_args(["test.m4a", "--model", "tiny"])
        assert args.model == "tiny"

    @pytest.mark.unit
    def test_transcribe_only_flag(self) -> None:
        """Should parse -T flag."""
        parser = build_parser()
        args = parser.parse_args(["test.m4a", "-T"])
        assert args.transcribe_only is True

    @pytest.mark.unit
    def test_transcribe_only_long_flag(self) -> None:
        """Should parse --transcribe-only flag."""
        parser = build_parser()
        args = parser.parse_args(["test.m4a", "--transcribe-only"])
        assert args.transcribe_only is True

    @pytest.mark.unit
    def test_default_preset(self) -> None:
        """Should use general as default preset."""
        parser = build_parser()
        args = parser.parse_args(["test.m4a"])
        assert args.preset == DEFAULT_PRESET

    @pytest.mark.unit
    def test_valid_presets(self) -> None:
        """Should accept all defined presets."""
        parser = build_parser()
        for preset in MIC_PRESETS.keys():
            args = parser.parse_args(["test.m4a", "--preset", preset])
            assert args.preset == preset

    @pytest.mark.unit
    def test_invalid_preset_rejected(self) -> None:
        """Should reject invalid presets."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["test.m4a", "--preset", "invalid_preset"])

    @pytest.mark.unit
    def test_default_enhancer(self) -> None:
        """Should use soap as default enhancer."""
        parser = build_parser()
        args = parser.parse_args(["test.m4a"])
        assert args.enhancer == DEFAULT_ENHANCER

    @pytest.mark.unit
    def test_valid_enhancers(self) -> None:
        """Should accept all defined enhancers."""
        parser = build_parser()
        for enhancer in SUPPORTED_ENHANCERS.keys():
            args = parser.parse_args(["test.m4a", "--enhancer", enhancer])
            assert args.enhancer == enhancer

    @pytest.mark.unit
    def test_invalid_enhancer_rejected(self) -> None:
        """Should reject invalid enhancers."""
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["test.m4a", "--enhancer", "invalid_enhancer"])

    @pytest.mark.unit
    def test_debug_compare_flag(self) -> None:
        """Should parse --debug-enhanced-compare flag."""
        parser = build_parser()
        args = parser.parse_args(["test.m4a", "--debug-enhanced-compare"])
        assert args.debug_enhanced_compare is True

    @pytest.mark.unit
    def test_debug_compare_flag_default(self) -> None:
        """Debug compare should default to False."""
        parser = build_parser()
        args = parser.parse_args(["test.m4a"])
        assert args.debug_enhanced_compare is False

    @pytest.mark.unit
    def test_debug_compare_duration_default(self) -> None:
        """Debug compare duration should default to 30."""
        parser = build_parser()
        args = parser.parse_args(["test.m4a"])
        assert args.debug_enhanced_compare_duration == 30.0

    @pytest.mark.unit
    def test_debug_compare_duration_custom(self) -> None:
        """Should accept custom debug compare duration."""
        parser = build_parser()
        args = parser.parse_args(["test.m4a", "--debug-enhanced-compare-duration", "60"])
        assert args.debug_enhanced_compare_duration == 60.0

    @pytest.mark.unit
    def test_no_enhancement_flag(self) -> None:
        """Should parse --no-enhancement flag."""
        parser = build_parser()
        args = parser.parse_args(["test.m4a", "--no-enhancement"])
        assert args.no_enhancement is True

    @pytest.mark.unit
    def test_no_enhancement_default(self) -> None:
        """No enhancement should default to False."""
        parser = build_parser()
        args = parser.parse_args(["test.m4a"])
        assert args.no_enhancement is False

    @pytest.mark.unit
    def test_combined_flags(self) -> None:
        """Should handle multiple flags together."""
        parser = build_parser()
        args = parser.parse_args([
            "test.m4a",
            "--model", "medium",
            "--preset", "iphone",
            "--enhancer", "soap",
            "-T",
            "--no-enhancement",
        ])
        assert args.audio == "test.m4a"
        assert args.model == "medium"
        assert args.preset == "iphone"
        assert args.enhancer == "soap"
        assert args.transcribe_only is True
        assert args.no_enhancement is True


class TestPresetConfiguration:
    """Tests for preset configuration."""

    @pytest.mark.unit
    def test_all_presets_have_required_fields(self) -> None:
        """All presets should have required fields."""
        required_fields = [
            "low_cut_hz",
            "high_cut_hz",
            "use_bandpass",
            "use_normalize",
        ]
        for preset_name, preset in MIC_PRESETS.items():
            for field in required_fields:
                assert field in preset, f"Preset {preset_name} missing field {field}"

    @pytest.mark.unit
    def test_preset_values_reasonable(self) -> None:
        """Preset values should be within reasonable ranges."""
        for preset_name, preset in MIC_PRESETS.items():
            low_cut = preset.get("low_cut_hz", 0)
            high_cut = preset.get("high_cut_hz", 20000)
            assert 20 <= low_cut <= 200, f"Preset {preset_name} low_cut_hz out of range: {low_cut}"
            assert 8000 <= high_cut <= 22000, f"Preset {preset_name} high_cut_hz out of range: {high_cut}"
            assert low_cut < high_cut, f"Preset {preset_name} low_cut >= high_cut"

    @pytest.mark.unit
    def test_iphone_preset_has_adaptive_fields(self) -> None:
        """iPhone preset should have adaptive threshold fields."""
        iphone = MIC_PRESETS.get("iphone")
        assert iphone is not None
        # iPhone preset has conditional EQ/compression
        assert "eq_rms_threshold" in iphone or "use_eq" in iphone
        assert "comp_rms_threshold" in iphone or "use_compression" in iphone


class TestSupportedEnhancers:
    """Tests for enhancer configuration."""

    @pytest.mark.unit
    def test_soap_enhancer_defined(self) -> None:
        """Soap enhancer should be defined."""
        assert "soap" in SUPPORTED_ENHANCERS
        assert len(SUPPORTED_ENHANCERS["soap"]) > 0  # Has description

    @pytest.mark.unit
    def test_resemble_enhancer_defined(self) -> None:
        """Resemble enhancer should be defined."""
        assert "resemble" in SUPPORTED_ENHANCERS
        assert len(SUPPORTED_ENHANCERS["resemble"]) > 0  # Has description

    @pytest.mark.unit
    def test_default_enhancer_exists(self) -> None:
        """Default enhancer should exist in supported enhancers."""
        assert DEFAULT_ENHANCER in SUPPORTED_ENHANCERS

    @pytest.mark.unit
    def test_default_preset_exists(self) -> None:
        """Default preset should exist in presets."""
        assert DEFAULT_PRESET in MIC_PRESETS
