from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from killer_subtitles.models import (
    LayoutConfig,
    PipelineConfig,
    StyleConfig,
    SubtitleState,
    VideoInfo,
    Word,
)
from killer_subtitles.pipeline import run_pipeline


class PipelineTests(unittest.TestCase):
    @patch("killer_subtitles.pipeline.compose")
    @patch("killer_subtitles.pipeline.LayoutEngine")
    @patch("killer_subtitles.pipeline.transcribe")
    @patch("killer_subtitles.pipeline.extract_audio")
    def test_pipeline_uses_legacy_layout_and_compositor(
        self, extract_audio, transcribe, layout_engine, compose
    ):
        words = [Word("hello", 0.0, 0.5)]
        states = [SubtitleState(start=0.0, end=0.5)]
        transcribe.return_value = words
        layout_engine.return_value.build_states.return_value = states

        def create_audio(_input_path, audio_path):
            Path(audio_path).write_bytes(b"audio")

        extract_audio.side_effect = create_audio

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_path = temp / "input.mp4"
            output_path = temp / "output.mp4"
            input_path.write_bytes(b"video")
            style = StyleConfig(font_path="font.ttf", font_size=40)
            layout = LayoutConfig(margin_x=20, margin_y=20)
            config = PipelineConfig(
                input_video=input_path,
                output_video=output_path,
                style=style,
                layout=layout,
            )
            video_info = VideoInfo(320, 240, 30.0, 1.0)

            result = run_pipeline(config, video_info=video_info)

        self.assertEqual(result, output_path)
        transcribe.assert_called_once()
        layout_engine.assert_called_once_with(video_info, style, layout)
        layout_engine.return_value.build_states.assert_called_once_with(words)
        layout_engine.return_value.build_dynamic_states.assert_not_called()
        compose.assert_called_once()
        self.assertEqual(compose.call_args.kwargs["states"], states)

    @patch("killer_subtitles.pipeline.compose")
    @patch("killer_subtitles.pipeline.LayoutEngine")
    @patch("killer_subtitles.pipeline.analyze_captions")
    @patch("killer_subtitles.pipeline.chunk_words")
    @patch("killer_subtitles.pipeline.transcribe")
    @patch("killer_subtitles.pipeline.extract_audio")
    def test_dynamic_pipeline_uses_caption_analysis(
        self,
        extract_audio,
        transcribe,
        chunk_words,
        analyze_captions,
        layout_engine,
        compose,
    ):
        words = [Word("dynamic", 0.0, 0.5)]
        captions = [object()]
        plans = [object()]
        states = [SubtitleState(start=0.0, end=0.5)]
        transcribe.return_value = words
        chunk_words.return_value = captions
        analyze_captions.return_value = plans
        layout_engine.return_value.build_dynamic_states.return_value = states

        def create_audio(_input_path, audio_path):
            Path(audio_path).write_bytes(b"audio")

        extract_audio.side_effect = create_audio

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_path = temp / "input.mp4"
            output_path = temp / "output.mp4"
            input_path.write_bytes(b"video")
            style = StyleConfig(font_path="font.ttf", font_size=40)
            layout = LayoutConfig(margin_x=20, margin_y=20)
            config = PipelineConfig(
                input_video=input_path,
                output_video=output_path,
                style=style,
                layout=layout,
                dynamic_captions=True,
            )
            video_info = VideoInfo(320, 240, 30.0, 1.0)

            run_pipeline(config, video_info=video_info)

        chunk_words.assert_called_once_with(words)
        analyze_captions.assert_called_once_with(captions)
        layout_engine.return_value.build_dynamic_states.assert_called_once_with(plans)
        layout_engine.return_value.build_states.assert_not_called()
        self.assertEqual(compose.call_args.kwargs["states"], states)


if __name__ == "__main__":
    unittest.main()
