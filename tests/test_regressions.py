from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import click

from magic_hour_subtitles import ffmpeg
from magic_hour_subtitles import transcriber
from magic_hour_subtitles.cli import _validate_hex_color
from magic_hour_subtitles.compositor import (
    _compose_final,
    _normalise_timeline,
    _write_concat_file,
    compose,
)
from magic_hour_subtitles.layout import LayoutEngine
from magic_hour_subtitles.models import (
    LayoutConfig,
    Line,
    StyleConfig,
    SubtitleState,
    VideoInfo,
    Word,
)
from magic_hour_subtitles.transcript_align import (
    _smooth_timing,
    _trim_trailing_outliers,
)


ROOT = Path(__file__).resolve().parents[1]
FONT_PATH = ROOT / "magic_hour_subtitles" / "fonts" / "Montserrat-ExtraBold.ttf"


class FfmpegTests(unittest.TestCase):
    @patch("magic_hour_subtitles.ffmpeg.get_ffmpeg_exe", return_value="ffmpeg")
    @patch("magic_hour_subtitles.ffmpeg.subprocess.run")
    def test_duration_probe_accepts_ffmpeg_nonzero_exit(self, run, _get_exe):
        run.return_value = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout=b"",
            stderr=b"Duration: 00:01:02.50, start: 0.000000",
        )

        self.assertEqual(ffmpeg.get_media_duration("audio.mp3"), 62.5)

    def test_rotation_parser_handles_display_matrix(self):
        stderr = "displaymatrix: rotation of -90.00 degrees"
        self.assertEqual(ffmpeg._parse_rotation(stderr), -90.0)

    def test_fps_parser_falls_back_to_tbr(self):
        self.assertEqual(ffmpeg._parse_fps("Video: h264, 29.97 tbr"), 29.97)


class AlignmentTests(unittest.TestCase):
    def test_smoothing_preserves_long_speech_pauses(self):
        words = [
            Word("before", 0.0, 0.5),
            Word("after", 10.0, 10.5),
            Word("again", 10.6, 11.0),
        ]

        smoothed = _smooth_timing(words)

        self.assertEqual(
            [(word.start, word.end) for word in smoothed],
            [(0.0, 0.5), (10.0, 10.5), (10.6, 11.0)],
        )

    def test_trailing_filter_preserves_long_speech_pauses(self):
        words = [
            Word("before", 0.0, 0.5),
            Word("after", 10.0, 10.5),
        ]

        self.assertEqual(_trim_trailing_outliers(words), words)


class TimelineTests(unittest.TestCase):
    def test_timeline_sorts_clamps_and_removes_overlaps(self):
        states = [
            SubtitleState(start=4.0, end=10.0),
            SubtitleState(start=0.0, end=2.0),
            SubtitleState(start=1.0, end=3.0),
        ]

        result = _normalise_timeline(states, video_duration=5.0)

        self.assertEqual(
            [(state.start, state.end) for state in result],
            [(0.0, 1.0), (1.0, 3.0), (4.0, 5.0)],
        )

    @patch("magic_hour_subtitles.compositor._ensure_blank")
    def test_concat_keeps_small_gaps_and_exact_video_duration(self, _ensure_blank):
        states = [
            SubtitleState(start=0.02, end=0.10),
            SubtitleState(start=0.12, end=0.20),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            concat_path = temp / "concat.txt"
            png_paths = [temp / "one.png", temp / "two.png"]

            _write_concat_file(concat_path, states, png_paths, 0.25)

            durations = [
                float(line.split()[1])
                for line in concat_path.read_text(encoding="utf-8").splitlines()
                if line.startswith("duration ")
            ]
        self.assertAlmostEqual(sum(durations), 0.25)
        self.assertIn(0.02, durations)

    def test_compose_rejects_same_input_and_output(self):
        path = Path("same-video.mp4")
        with self.assertRaisesRegex(ValueError, "must be different"):
            compose(
                path,
                path,
                [],
                VideoInfo(width=320, height=240, fps=30.0, duration=1.0),
                StyleConfig(),
            )

    @patch("magic_hour_subtitles.compositor._compose_normal")
    @patch(
        "magic_hour_subtitles.compositor._compose_behind_subject",
        side_effect=RuntimeError("mask failed"),
    )
    def test_foreground_failure_falls_back_to_normal_overlay(
        self,
        compose_behind,
        compose_normal,
    ):
        decision = SimpleNamespace(enabled=True)
        paths = [Path("source.mp4"), Path("overlay.mkv"), Path("output.mp4")]

        _compose_final(
            *paths,
            Path("temp"),
            VideoInfo(320, 240, 30.0, 1.0),
            behind_subject=True,
            frame_analyses=[object()],
            occlusion_decisions=[decision],
            mask_dilate=2,
            mask_blur=5,
        )

        compose_behind.assert_called_once()
        compose_normal.assert_called_once_with(*paths)


class TranscriberTests(unittest.TestCase):
    def test_transcription_requests_word_timestamps_and_vad(self):
        model = Mock()
        model.transcribe.return_value = (
            iter([
                SimpleNamespace(words=[
                    SimpleNamespace(word=" hello ", start=0.1, end=0.5),
                    SimpleNamespace(word="world", start=0.6, end=1.0),
                ])
            ]),
            object(),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.mp3"
            audio_path.write_bytes(b"audio")
            words = transcriber._transcribe_with_model(
                model, audio_path, "en", "Magic Hour"
            )

        self.assertEqual(
            [(word.text, word.start, word.end) for word in words],
            [("hello", 0.1, 0.5), ("world", 0.6, 1.0)],
        )
        kwargs = model.transcribe.call_args.kwargs
        self.assertTrue(kwargs["word_timestamps"])
        self.assertTrue(kwargs["vad_filter"])
        self.assertEqual(kwargs["initial_prompt"], "Magic Hour")

    @patch("magic_hour_subtitles.transcriber._transcribe_with_model")
    @patch("magic_hour_subtitles.transcriber._load_model")
    @patch("magic_hour_subtitles.transcriber._cuda_available", return_value=True)
    def test_cuda_failure_falls_back_to_small_english_int8(
        self, _cuda_available, load_model, transcribe_with_model
    ):
        cpu_model = object()
        load_model.side_effect = [RuntimeError("CUDA unavailable"), cpu_model]
        transcribe_with_model.return_value = [Word("fallback", 0.0, 0.5)]

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.mp3"
            audio_path.write_bytes(b"audio")
            with self.assertWarns(RuntimeWarning):
                words = transcriber.transcribe(audio_path, language="en")

        self.assertEqual(words, [Word("fallback", 0.0, 0.5)])
        self.assertEqual(
            load_model.call_args_list,
            [
                call("distil-large-v3", "cuda", "float16"),
                call("small.en", "cpu", "int8"),
            ],
        )

    @patch("magic_hour_subtitles.transcriber._transcribe_with_model")
    @patch("magic_hour_subtitles.transcriber._load_model")
    @patch("magic_hour_subtitles.transcriber._cuda_available", return_value=False)
    def test_cpu_only_skips_gpu_model(
        self, _cuda_available, load_model, transcribe_with_model
    ):
        cpu_model = object()
        load_model.return_value = cpu_model
        transcribe_with_model.return_value = [Word("cpu", 0.0, 0.5)]

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.mp3"
            audio_path.write_bytes(b"audio")
            with self.assertWarns(RuntimeWarning):
                words = transcriber.transcribe(audio_path, language="en")

        self.assertEqual(words, [Word("cpu", 0.0, 0.5)])
        load_model.assert_called_once_with("small.en", "cpu", "int8")
        transcribe_with_model.assert_called_once_with(
            cpu_model, audio_path, "en", None
        )


class LayoutTests(unittest.TestCase):
    def _engine(self, position: str) -> LayoutEngine:
        return LayoutEngine(
            VideoInfo(width=400, height=300, fps=30.0, duration=5.0),
            StyleConfig(
                font_path=str(FONT_PATH),
                font_size=40,
                highlight_size=60,
            ),
            LayoutConfig(
                mode="karaoke",
                position=position,
                margin_x=20,
                margin_y=30,
            ),
        )

    def test_top_margin_positions_text_block_from_edge(self):
        engine = self._engine("top")
        rendered = engine._position_lines(
            [Line(words=[Word("hello", 0.0, 1.0)])],
            highlight_word_index=0,
        )

        effective_highlight_top = (
            rendered[0].y - engine._highlight_vertical_padding()
        )
        self.assertEqual(effective_highlight_top, 30)

    def test_highlight_slot_does_not_overlap_next_word(self):
        engine = self._engine("center")
        rendered = engine._position_lines(
            [Line(words=[
                Word("WIDE", 0.0, 1.0),
                Word("word", 1.0, 2.0),
            ])],
            highlight_word_index=0,
        )
        normal_width = engine.font.getlength("WIDE")
        highlight_width = engine.highlight_font.getlength("WIDE")
        highlighted_left = rendered[0].x - (highlight_width - normal_width) / 2
        highlighted_right = highlighted_left + highlight_width

        self.assertLessEqual(highlighted_right, rendered[1].x)


class CliValidationTests(unittest.TestCase):
    def test_hex_color_validation(self):
        option = click.Option(["--font-color"])
        self.assertEqual(
            _validate_hex_color(None, option, "a1b2c3"),
            "#A1B2C3",
        )
        with self.assertRaises(click.BadParameter):
            _validate_hex_color(None, option, "#GGGGGG")


if __name__ == "__main__":
    unittest.main()
