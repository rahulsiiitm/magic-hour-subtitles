from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import click

from killer_subtitles import ffmpeg
from killer_subtitles import transcriber
from killer_subtitles.cli import _validate_hex_color
from killer_subtitles.compositor import (
    _normalise_timeline,
    _write_concat_file,
    compose,
)
from killer_subtitles.layout import LayoutEngine
from killer_subtitles.models import (
    LayoutConfig,
    Line,
    StyleConfig,
    SubtitleState,
    VideoInfo,
    Word,
)
from killer_subtitles.transcript_align import (
    _smooth_timing,
    _trim_trailing_outliers,
)


ROOT = Path(__file__).resolve().parents[1]
FONT_PATH = ROOT / "killer_subtitles" / "fonts" / "Montserrat-ExtraBold.ttf"


class FfmpegTests(unittest.TestCase):
    @patch("killer_subtitles.ffmpeg.get_ffmpeg_exe", return_value="ffmpeg")
    @patch("killer_subtitles.ffmpeg.subprocess.run")
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

    @patch("killer_subtitles.compositor._ensure_blank")
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


class TranscriberTests(unittest.TestCase):
    @patch("killer_subtitles.transcriber._transcribe_single")
    @patch("killer_subtitles.ffmpeg.run_ffmpeg")
    @patch("killer_subtitles.ffmpeg.get_media_duration", return_value=10.0)
    def test_chunked_transcription_probes_and_offsets_chunks(
        self,
        _duration,
        run_ffmpeg,
        transcribe_single,
    ):
        def create_chunk(*args):
            Path(args[-1]).write_bytes(b"chunk")

        run_ffmpeg.side_effect = create_chunk
        transcribe_single.side_effect = [
            [Word("one", 0.0, 1.0)],
            [Word("two", 0.0, 1.0)],
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.mp3"
            audio_path.write_bytes(b"123456")
            with patch.object(transcriber, "TARGET_CHUNK_SIZE_BYTES", 3):
                words = transcriber._transcribe_chunked(
                    object(),
                    audio_path,
                    "en",
                    None,
                )

        self.assertEqual(
            [(word.text, word.start, word.end) for word in words],
            [("one", 0.0, 1.0), ("two", 5.0, 6.0)],
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
