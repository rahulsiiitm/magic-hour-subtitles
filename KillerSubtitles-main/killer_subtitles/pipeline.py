"""End-to-end subtitle pipeline with optional Phase 2/3 enhancements."""

from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable

from .caption_analysis import analyze_captions
from .caption_chunker import chunk_words
from .compositor import _normalise_timeline, compose
from .ffmpeg import extract_audio, get_video_info
from .layout import LayoutEngine
from .models import CaptionPlan, PipelineConfig, PlacementPlan, VideoInfo, Word
from .placement import PlacementPlanner
from .transcriber import transcribe
from .vision import VisionAnalyzer


ProgressCallback = Callable[[str, int, int], None]


def run_pipeline(
    config: PipelineConfig,
    *,
    video_info: VideoInfo | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Run the configured pipeline and return the final captioned MP4 path."""
    input_path = Path(config.input_video)
    output_path = Path(config.output_video)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output video paths must be different.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if video_info is None:
        _notify(progress_callback, "Reading video metadata", 0, 1)
        video_info = get_video_info(input_path)
        _notify(progress_callback, "Reading video metadata", 1, 1)

    with tempfile.TemporaryDirectory(prefix="killersubs_") as tmp:
        audio_path = Path(tmp) / "audio.mp3"

        _notify(progress_callback, "Extracting audio", 0, 1)
        extract_audio(input_path, audio_path)
        _notify(progress_callback, "Extracting audio", 1, 1)

        _notify(progress_callback, "Transcribing with faster-whisper", 0, 1)
        words = transcribe(
            audio_path,
            language=config.language,
            prompt=config.whisper_prompt,
            model_size=config.whisper_model,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
            cpu_model_size=config.cpu_model,
        )
        _notify(progress_callback, "Transcribing with faster-whisper", 1, 1)

    if config.caption_diagnostics:
        _print_word_stage("TRANSCRIBED", words)

    if config.transcript_path and words:
        from .transcript_align import align_to_script

        _notify(progress_callback, "Aligning to transcript", 0, 1)
        words = align_to_script(words, config.transcript_path)
        _notify(progress_callback, "Aligning to transcript", 1, 1)

    engine = LayoutEngine(video_info, config.style, config.layout)
    _notify(progress_callback, "Calculating layout", 0, 1)
    if config.dynamic_captions or config.smart_placement:
        captions = chunk_words(words)
        _validate_caption_coverage(words, captions)
        plans = analyze_captions(captions)
        _validate_plan_coverage(words, plans)
        placement_plans: list[PlacementPlan] = []
        vision_summary: tuple[str, int, float] | None = None
        if config.smart_placement and plans:
            _notify(progress_callback, "Analyzing video frames", 0, 1)
            try:
                _release_gpu_cache()
                analyzer = VisionAnalyzer(config.vision)
                analyses = analyzer.analyze(input_path, video_info)
                planner = PlacementPlanner(
                    video_info,
                    config.layout,
                    hysteresis=config.vision.hysteresis,
                )
                caption_sizes = [
                    engine.measure_dynamic_caption(plan) for plan in plans
                ]
                placement_plans = planner.plan(plans, analyses, caption_sizes)
                vision_summary = (
                    analyzer.device,
                    len(analyses),
                    analyzer.elapsed_seconds,
                )
            except Exception as exc:
                print(
                    "\nSmart placement unavailable; using fixed Phase 2 position: "
                    f"{exc}"
                )
            finally:
                _notify(progress_callback, "Analyzing video frames", 1, 1)

        if placement_plans:
            states = engine.build_dynamic_states(plans, placement_plans)
        else:
            states = engine.build_dynamic_states(plans)
        _validate_state_coverage(words, states)
        if config.caption_diagnostics:
            _print_dynamic_stage_diagnostics(
                words,
                captions,
                plans,
                states,
                video_info,
                config,
            )
            _print_caption_diagnostics(plans)
            if vision_summary is not None:
                device, frame_count, elapsed = vision_summary
                print(
                    f"\nVISION: device={device}, analyzed_frames={frame_count}, "
                    f"elapsed={elapsed:.2f}s"
                )
                _print_placement_diagnostics(placement_plans)
    else:
        states = engine.build_states(words)
    _notify(progress_callback, "Calculating layout", 1, 1)

    compose(
        source_video=input_path,
        output_path=output_path,
        states=states,
        video_info=video_info,
        style=config.style,
        progress_callback=progress_callback,
    )

    if config.export_srt:
        _export_srt(words, output_path.with_suffix(".srt"))

    return output_path


def _print_caption_diagnostics(plans: list[CaptionPlan]) -> None:
    for plan in plans:
        print(
            "\nCaption: " + repr(plan.caption.text)
            + f"\nTone: {plan.tone.value}"
            + f"\nKeywords: {plan.keywords}"
            + f"\nStart/end: {plan.caption.start:.3f} -> {plan.caption.end:.3f}"
        )


def _print_placement_diagnostics(plans: list[PlacementPlan]) -> None:
    for plan in plans:
        ranked = sorted(plan.scores.items(), key=lambda item: item[1], reverse=True)
        score_lines = "\n".join(
            f"  {name}: {score:.3f}" for name, score in ranked
        )
        selected_overlap = plan.person_overlaps.get(plan.placement.name, 0.0)
        print(
            "\nCaption: " + repr(plan.caption_plan.caption.text)
            + f"\nTone: {plan.caption_plan.tone.value}"
            + f"\nPlacement: {plan.placement.name}"
            + f"\nPerson overlap: {selected_overlap:.3f}"
            + f"\nBest raw candidate: {plan.best_raw_candidate}"
            + f"\nHysteresis: {plan.hysteresis_reason}"
            + "\nSafety override: "
            + ("yes" if plan.safety_override else "no")
            + f"\nScores:\n{score_lines}"
        )


def _release_gpu_cache() -> None:
    """Release optional framework caches between Whisper and YOLO."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _word_signature(word: Word) -> tuple[str, float, float]:
    return (word.text, word.start, word.end)


def _flatten_caption_words(captions: list) -> list[Word]:
    return [word for caption in captions for word in caption.words]


def _validate_caption_coverage(words: list[Word], captions: list) -> None:
    flattened = _flatten_caption_words(captions)
    expected_count = len(words)
    actual_count = sum(len(caption.words) for caption in captions)
    assert expected_count == actual_count, (
        f"Caption chunking lost words: input={expected_count}, chunked={actual_count}"
    )

    input_ids = Counter(id(word) for word in words)
    output_ids = Counter(id(word) for word in flattened)
    missing = [
        f"{index}:{word.text}@{word.start:.3f}-{word.end:.3f}"
        for index, word in enumerate(words)
        if output_ids[id(word)] == 0
    ]
    duplicated = [
        f"{index}:{word.text} x{output_ids[id(word)]}"
        for index, word in enumerate(words)
        if output_ids[id(word)] > input_ids[id(word)]
    ]
    assert not missing and not duplicated, (
        f"Caption object coverage failed; missing={missing}, duplicated={duplicated}"
    )

    before = [_word_signature(word) for word in words]
    after = [_word_signature(word) for word in flattened]
    assert before == after, "Caption chunking changed word order, text, or timestamps."


def _validate_plan_coverage(words: list[Word], plans: list[CaptionPlan]) -> None:
    planned_words = [
        word
        for plan in plans
        for word in plan.caption.words
    ]
    assert [_word_signature(word) for word in planned_words] == [
        _word_signature(word) for word in words
    ], "Caption analysis changed or lost the chunked word sequence."


def _validate_state_coverage(words: list[Word], states: list) -> None:
    assert len(states) == len(words), (
        f"Dynamic layout lost word states: words={len(words)}, states={len(states)}"
    )
    if words:
        assert states, "Dynamic layout produced no states for a non-empty transcript."
        assert states[0].start == words[0].start
        assert states[-1].end >= words[-1].end, (
            "Dynamic layout truncated the final timestamp: "
            f"word_end={words[-1].end}, state_end={states[-1].end}"
        )


def _print_word_stage(label: str, words: list[Word]) -> None:
    if not words:
        print(f"\n{label}: 0 words")
        return
    print(
        f"\n{label}: {len(words)} words"
        f"\n  first: {words[0].text!r} "
        f"{words[0].start:.3f}->{words[0].end:.3f}"
        f"\n  last:  {words[-1].text!r} "
        f"{words[-1].start:.3f}->{words[-1].end:.3f}"
    )


def _print_dynamic_stage_diagnostics(
    words: list[Word],
    captions: list,
    plans: list[CaptionPlan],
    states: list,
    video_info: VideoInfo,
    config: PipelineConfig,
) -> None:
    chunked_words = _flatten_caption_words(captions)
    planned_word_count = sum(len(plan.caption.words) for plan in plans)
    normalized = _normalise_timeline(states, video_info.duration)

    legacy_words = [Word(word.text, word.start, word.end) for word in words]
    legacy_engine = LayoutEngine(video_info, config.style, config.layout)
    legacy_states = legacy_engine.build_states(legacy_words)
    normalized_legacy = _normalise_timeline(legacy_states, video_info.duration)

    object_counts = Counter(id(word) for word in chunked_words)
    missing = [word.text for word in words if object_counts[id(word)] == 0]
    duplicated = [word.text for word in words if object_counts[id(word)] > 1]

    print(
        f"\nCHUNKED: {len(chunked_words)} words, {len(captions)} captions"
        f"\n  first caption start: "
        f"{captions[0].start:.3f}" if captions else "\nCHUNKED: 0 captions"
    )
    if captions:
        print(f"  last caption end: {captions[-1].end:.3f}")
    print(f"  missing objects: {missing or 'none'}")
    print(f"  duplicated objects: {duplicated or 'none'}")
    print(f"  ordered text/timestamps identical: yes")

    print(f"\nPLANNED: {planned_word_count} words, {len(plans)} plans")
    if states:
        print(
            f"\nSTATES: {len(states)} states"
            f"\n  first: {states[0].start:.3f}->{states[0].end:.3f}"
            f"\n  last:  {states[-1].start:.3f}->{states[-1].end:.3f}"
            f"\n  last transcription word end: {words[-1].end:.3f}"
        )
    else:
        print("\nSTATES: 0 states")

    print(
        f"\nTIMELINE NORMALIZATION: dynamic {len(states)}->{len(normalized)} states; "
        f"legacy {len(legacy_states)}->{len(normalized_legacy)} states"
    )
    if normalized:
        print(f"  normalized dynamic last end: {normalized[-1].end:.3f}")
    if normalized_legacy:
        print(f"  normalized legacy last end: {normalized_legacy[-1].end:.3f}")


def _export_srt(words: list[Word], srt_path: Path) -> None:
    """Write the existing basic eight-word SRT blocks."""
    def timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        whole_seconds = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return (
            f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},"
            f"{milliseconds:03d}"
        )

    lines: list[str] = []
    for index, offset in enumerate(range(0, len(words), 8), start=1):
        chunk = words[offset : offset + 8]
        lines.extend([
            str(index),
            f"{timestamp(chunk[0].start)} --> {timestamp(chunk[-1].end)}",
            " ".join(word.text for word in chunk),
            "",
        ])

    srt_path.write_text("\n".join(lines), encoding="utf-8")


def _notify(
    callback: ProgressCallback | None,
    phase: str,
    current: int,
    total: int,
) -> None:
    if callback is not None:
        callback(phase, current, total)
