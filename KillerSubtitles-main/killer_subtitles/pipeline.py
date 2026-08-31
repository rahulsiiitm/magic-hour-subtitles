"""Phase 1 end-to-end pipeline using the legacy layout and compositor."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

from .compositor import compose
from .ffmpeg import extract_audio, get_video_info
from .layout import LayoutEngine
from .models import PipelineConfig, VideoInfo, Word
from .transcriber import transcribe


ProgressCallback = Callable[[str, int, int], None]


def run_pipeline(
    config: PipelineConfig,
    *,
    video_info: VideoInfo | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Run Phase 1 and return the final captioned MP4 path."""
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

    if config.transcript_path and words:
        from .transcript_align import align_to_script

        _notify(progress_callback, "Aligning to transcript", 0, 1)
        words = align_to_script(words, config.transcript_path)
        _notify(progress_callback, "Aligning to transcript", 1, 1)

    _notify(progress_callback, "Calculating layout", 0, 1)
    states = LayoutEngine(video_info, config.style, config.layout).build_states(words)
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
