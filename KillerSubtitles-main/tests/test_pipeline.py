from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from killer_subtitles.models import (
    Caption,
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
        captions = [Caption(words=words)]
        plans = [SimpleNamespace(caption=captions[0])]
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

            with patch("killer_subtitles.pipeline.VisionAnalyzer") as vision_analyzer:
                run_pipeline(config, video_info=video_info)

        chunk_words.assert_called_once_with(words, portrait=False)
        analyze_captions.assert_called_once_with(captions)
        layout_engine.return_value.build_dynamic_states.assert_called_once_with(plans)
        layout_engine.return_value.build_states.assert_not_called()
        vision_analyzer.assert_not_called()
        self.assertEqual(compose.call_args.kwargs["states"], states)

    @patch("killer_subtitles.pipeline.compose")
    @patch("killer_subtitles.pipeline.LayoutEngine")
    @patch("killer_subtitles.pipeline.analyze_captions")
    @patch("killer_subtitles.pipeline.chunk_words")
    @patch("killer_subtitles.pipeline.transcribe")
    @patch("killer_subtitles.pipeline.extract_audio")
    def test_portrait_pipeline_resolves_conservative_visual_config(
        self,
        extract_audio,
        transcribe,
        chunk_words,
        analyze_captions,
        layout_engine,
        compose,
    ):
        words = [Word("portrait", 0.0, 0.5)]
        captions = [Caption(words=words)]
        plans = [SimpleNamespace(caption=captions[0])]
        states = [SubtitleState(start=0.0, end=0.5)]
        transcribe.return_value = words
        chunk_words.return_value = captions
        analyze_captions.return_value = plans
        layout_engine.return_value.dynamic_line_count.return_value = 1
        layout_engine.return_value.build_dynamic_states.return_value = states
        extract_audio.side_effect = lambda _source, target: Path(target).write_bytes(b"audio")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_path = temp / "input.mp4"
            input_path.write_bytes(b"video")
            config = PipelineConfig(
                input_video=input_path,
                output_video=temp / "output.mp4",
                style=StyleConfig(font_path="font.ttf", font_size=32),
                layout=LayoutConfig(max_lines=3, margin_x=32, margin_y=32),
                dynamic_captions=True,
            )

            run_pipeline(config, video_info=VideoInfo(320, 640, 30.0, 1.0))

        chunk_words.assert_called_once_with(words, portrait=True)
        resolved_style = layout_engine.call_args.args[1]
        resolved_layout = layout_engine.call_args.args[2]
        self.assertEqual(resolved_style.font_size, 28)
        self.assertEqual(resolved_layout.max_lines, 2)
        self.assertEqual(resolved_layout.margin_x, 48)
        self.assertEqual(compose.call_args.kwargs["style"], resolved_style)

    @patch("killer_subtitles.pipeline.compose")
    @patch("killer_subtitles.pipeline.LayoutEngine")
    @patch("killer_subtitles.pipeline.analyze_captions")
    @patch("killer_subtitles.pipeline.chunk_words")
    @patch("killer_subtitles.pipeline.VisionAnalyzer")
    @patch("killer_subtitles.pipeline.transcribe")
    @patch("killer_subtitles.pipeline.extract_audio")
    def test_vision_failure_falls_back_to_phase2_position(
        self,
        extract_audio,
        transcribe,
        vision_analyzer,
        chunk_words,
        analyze_captions,
        layout_engine,
        compose,
    ):
        words = [Word("fallback", 0.0, 0.5)]
        captions = [Caption(words=words)]
        plans = [SimpleNamespace(caption=captions[0])]
        states = [SubtitleState(start=0.0, end=0.5)]
        transcribe.return_value = words
        chunk_words.return_value = captions
        analyze_captions.return_value = plans
        vision_analyzer.return_value.analyze.side_effect = RuntimeError("no vision")
        layout_engine.return_value.build_dynamic_states.return_value = states
        extract_audio.side_effect = lambda _source, target: Path(target).write_bytes(b"audio")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_path = temp / "input.mp4"
            output_path = temp / "output.mp4"
            input_path.write_bytes(b"video")
            config = PipelineConfig(
                input_video=input_path,
                output_video=output_path,
                style=StyleConfig(font_path="font.ttf", font_size=40),
                layout=LayoutConfig(margin_x=20, margin_y=20),
                dynamic_captions=True,
                smart_placement=True,
            )

            run_pipeline(config, video_info=VideoInfo(320, 240, 30.0, 1.0))

        vision_analyzer.return_value.analyze.assert_called_once()
        layout_engine.return_value.build_dynamic_states.assert_called_once_with(plans)
        self.assertEqual(compose.call_args.kwargs["states"], states)

    @patch("killer_subtitles.pipeline.compose")
    @patch("killer_subtitles.pipeline.LayoutEngine")
    @patch("killer_subtitles.pipeline.PlacementPlanner")
    @patch("killer_subtitles.pipeline.analyze_captions")
    @patch("killer_subtitles.pipeline.chunk_words")
    @patch("killer_subtitles.pipeline.VisionAnalyzer")
    @patch("killer_subtitles.pipeline.transcribe")
    @patch("killer_subtitles.pipeline.extract_audio")
    def test_smart_placement_uses_vision_and_positioned_plans(
        self,
        extract_audio,
        transcribe,
        vision_analyzer,
        chunk_words,
        analyze_captions,
        placement_planner,
        layout_engine,
        compose,
    ):
        words = [Word("placed", 0.0, 0.5)]
        captions = [Caption(words=words)]
        plans = [SimpleNamespace(caption=captions[0])]
        analyses = [object()]
        positioned_plans = [object()]
        states = [SubtitleState(start=0.0, end=0.5)]
        transcribe.return_value = words
        chunk_words.return_value = captions
        analyze_captions.return_value = plans
        vision_analyzer.return_value.analyze.return_value = analyses
        layout_engine.return_value.measure_dynamic_caption.return_value = (120, 50)
        placement_planner.return_value.plan.return_value = positioned_plans
        layout_engine.return_value.build_dynamic_states.return_value = states
        extract_audio.side_effect = lambda _source, target: Path(target).write_bytes(b"audio")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_path = temp / "input.mp4"
            output_path = temp / "output.mp4"
            input_path.write_bytes(b"video")
            layout = LayoutConfig(margin_x=20, margin_y=20)
            config = PipelineConfig(
                input_video=input_path,
                output_video=output_path,
                style=StyleConfig(font_path="font.ttf", font_size=40),
                layout=layout,
                smart_placement=True,
            )
            video_info = VideoInfo(320, 240, 30.0, 1.0)

            run_pipeline(config, video_info=video_info)

        vision_analyzer.return_value.analyze.assert_called_once_with(
            input_path,
            video_info,
        )
        placement_planner.assert_called_once_with(
            video_info,
            layout,
            hysteresis=config.vision.hysteresis,
        )
        placement_planner.return_value.plan.assert_called_once_with(
            plans,
            analyses,
            [(120, 50)],
        )
        layout_engine.return_value.build_dynamic_states.assert_called_once_with(
            plans,
            positioned_plans,
        )
        self.assertNotIn("behind_subject", compose.call_args.kwargs)

    @patch("killer_subtitles.pipeline.compose")
    @patch("killer_subtitles.pipeline.LayoutEngine")
    @patch("killer_subtitles.pipeline.OcclusionPlanner")
    @patch("killer_subtitles.pipeline.PlacementPlanner")
    @patch("killer_subtitles.pipeline.analyze_captions")
    @patch("killer_subtitles.pipeline.chunk_words")
    @patch("killer_subtitles.pipeline.VisionAnalyzer")
    @patch("killer_subtitles.pipeline.transcribe")
    @patch("killer_subtitles.pipeline.extract_audio")
    def test_behind_subject_reuses_phase3_analysis_and_reaches_compositor(
        self,
        extract_audio,
        transcribe,
        vision_analyzer,
        chunk_words,
        analyze_captions,
        placement_planner,
        occlusion_planner,
        layout_engine,
        compose,
    ):
        words = [Word("behind", 0.0, 0.5)]
        captions = [Caption(words=words)]
        plans = [SimpleNamespace(caption=captions[0])]
        analyses = [object()]
        positioned_plans = [object()]
        decisions = [object()]
        states = [SubtitleState(start=0.0, end=0.5)]
        transcribe.return_value = words
        chunk_words.return_value = captions
        analyze_captions.return_value = plans
        vision_analyzer.return_value.analyze.return_value = analyses
        layout_engine.return_value.measure_dynamic_caption.return_value = (120, 50)
        placement_planner.return_value.plan.return_value = positioned_plans
        layout_engine.return_value.build_dynamic_states.return_value = states
        occlusion_planner.return_value.plan.return_value = decisions
        extract_audio.side_effect = lambda _source, target: Path(target).write_bytes(b"audio")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_path = temp / "input.mp4"
            output_path = temp / "output.mp4"
            input_path.write_bytes(b"video")
            layout = LayoutConfig(margin_x=20, margin_y=20)
            config = PipelineConfig(
                input_video=input_path,
                output_video=output_path,
                style=StyleConfig(font_path="font.ttf", font_size=40),
                layout=layout,
                dynamic_captions=True,
                smart_placement=True,
                behind_subject=True,
            )
            video_info = VideoInfo(320, 240, 30.0, 1.0)

            run_pipeline(config, video_info=video_info)

        vision_analyzer.return_value.analyze.assert_called_once_with(
            input_path,
            video_info,
        )
        occlusion_planner.return_value.plan.assert_called_once_with(
            positioned_plans,
            states,
        )
        self.assertTrue(compose.call_args.kwargs["behind_subject"])
        self.assertIs(compose.call_args.kwargs["frame_analyses"], analyses)
        self.assertIs(compose.call_args.kwargs["occlusion_decisions"], decisions)


if __name__ == "__main__":
    unittest.main()
