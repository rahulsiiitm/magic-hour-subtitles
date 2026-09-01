"""Video compositor -- stitches subtitle PNGs and overlays onto source.

Pipeline:
  1. Render each SubtitleState as a transparent PNG via the renderer.
  2. Write a concat demuxer file mapping each PNG to its display duration.
  3. FFmpeg creates an alpha-channel subtitle video from the concat list.
  4. FFmpeg overlays the subtitle video onto the source video.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

from .ffmpeg import run_ffmpeg
from .models import (
    FrameAnalysis,
    OcclusionDecision,
    StyleConfig,
    SubtitleState,
    VideoInfo,
)
from .occlusion import TemporalMaskProvider, iter_dense_masks
from .renderer import SubtitleRenderer


def compose(
    source_video: str | Path,
    output_path: str | Path,
    states: list[SubtitleState],
    video_info: VideoInfo,
    style: StyleConfig,
    *,
    progress_callback: Callable[[str, int, int], None] | None = None,
    behind_subject: bool = False,
    frame_analyses: list[FrameAnalysis] | None = None,
    occlusion_decisions: list[OcclusionDecision] | None = None,
    mask_dilate: int = 2,
    mask_blur: int = 5,
) -> Path:
    """Render subtitle states and burn them onto the source video.

    Parameters
    ----------
    source_video:
        Path to the original video file.
    output_path:
        Destination path for the final subtitled video.
    states:
        Ordered list of subtitle states with timing.
    video_info:
        Metadata about the source video.
    style:
        Visual styling configuration.
    progress_callback:
        Optional ``(phase, current, total)`` callback for progress updates.
    """
    source_video = Path(source_video)
    output_path = Path(output_path)
    if source_video.resolve() == output_path.resolve():
        raise ValueError("Input and output video paths must be different.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    states = _normalise_timeline(states, video_info.duration)
    renderer = SubtitleRenderer(video_info, style)

    with tempfile.TemporaryDirectory(prefix="magic_hour_subtitles_") as tmp_dir:
        tmp = Path(tmp_dir)

        # Render subtitle PNGs.
        _notify(progress_callback, "Rendering subtitle frames", 0, len(states))
        png_paths: list[Path] = []
        for i, state in enumerate(states):
            png_path = tmp / f"state_{i:05d}.png"
            renderer.render_to_file(state, png_path)
            png_paths.append(png_path)
            _notify(progress_callback, "Rendering subtitle frames", i + 1, len(states))

        if not png_paths:
            # No subtitles -- preserve the source when possible, otherwise
            # convert it so the requested output extension matches its data.
            if source_video.suffix.lower() == output_path.suffix.lower():
                import shutil
                shutil.copy2(str(source_video), str(output_path))
            else:
                run_ffmpeg(
                    "-y",
                    "-i", str(source_video),
                    "-map", "0:v:0",
                    "-map", "0:a?",
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "18",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-map_metadata", "0",
                    str(output_path),
                )
            return output_path

        # Write the concat demuxer list.
        concat_path = tmp / "concat.txt"
        _write_concat_file(concat_path, states, png_paths, video_info.duration)

        # Create the subtitle overlay video (PNG codec with alpha).
        overlay_video = tmp / "subtitle_overlay.mkv"
        _notify(progress_callback, "Creating subtitle overlay", 0, 1)

        run_ffmpeg(
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_path),
            "-c:v", "png",
            "-pix_fmt", "rgba",
            str(overlay_video),
        )
        _notify(progress_callback, "Creating subtitle overlay", 1, 1)

        # Final overlay, optionally restoring foreground person pixels.
        _notify(progress_callback, "Compositing final video", 0, 1)
        _compose_final(
            source_video,
            overlay_video,
            output_path,
            tmp,
            video_info,
            behind_subject=behind_subject,
            frame_analyses=frame_analyses or [],
            occlusion_decisions=occlusion_decisions or [],
            mask_dilate=mask_dilate,
            mask_blur=mask_blur,
        )
        _notify(progress_callback, "Compositing final video", 1, 1)

    return output_path


def _compose_final(
    source_video: Path,
    overlay_video: Path,
    output_path: Path,
    temp_dir: Path,
    video_info: VideoInfo,
    *,
    behind_subject: bool,
    frame_analyses: list[FrameAnalysis],
    occlusion_decisions: list[OcclusionDecision],
    mask_dilate: int,
    mask_blur: int,
) -> None:
    use_foreground = (
        behind_subject
        and bool(frame_analyses)
        and any(decision.enabled for decision in occlusion_decisions)
    )
    if use_foreground:
        try:
            _compose_behind_subject(
                source_video,
                overlay_video,
                output_path,
                temp_dir,
                video_info,
                frame_analyses,
                occlusion_decisions,
                mask_dilate,
                mask_blur,
            )
            return
        except Exception as exc:
            print(
                "\nBehind-subject compositing unavailable; "
                f"using normal captions: {exc}"
            )
    _compose_normal(source_video, overlay_video, output_path)


def _compose_normal(
    source_video: Path,
    overlay_video: Path,
    output_path: Path,
) -> None:
    run_ffmpeg(
        "-y",
        "-i", str(source_video),
        "-i", str(overlay_video),
        "-filter_complex",
        "[0:v:0][1:v:0]overlay=0:0:eof_action=pass:repeatlast=0[vout]",
        "-map", "[vout]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        "-map_metadata", "0",
        str(output_path),
    )


def _compose_behind_subject(
    source_video: Path,
    overlay_video: Path,
    output_path: Path,
    temp_dir: Path,
    video_info: VideoInfo,
    analyses: list[FrameAnalysis],
    decisions: list[OcclusionDecision],
    mask_dilate: int,
    mask_blur: int,
) -> None:
    from PIL import Image

    provider = TemporalMaskProvider(
        analyses,
        output_width=video_info.width,
        output_height=video_info.height,
        dilate=mask_dilate,
        blur=mask_blur,
    )
    mask_fps, frame_count, masks = iter_dense_masks(
        provider,
        decisions,
        video_info,
    )
    mask_dir = temp_dir / "foreground_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for frame_index, mask in enumerate(masks):
        Image.fromarray(mask, mode="L").save(
            mask_dir / f"mask_{frame_index:06d}.png",
            "PNG",
        )

    mask_video = temp_dir / "foreground_mask.mkv"
    run_ffmpeg(
        "-y",
        "-framerate", f"{mask_fps:.6f}",
        "-i", str(mask_dir / "mask_%06d.png"),
        "-frames:v", str(frame_count),
        "-c:v", "ffv1",
        "-pix_fmt", "gray",
        str(mask_video),
    )
    run_ffmpeg(
        "-y",
        "-i", str(source_video),
        "-i", str(overlay_video),
        "-i", str(mask_video),
        "-filter_complex",
        "[0:v:0]split=2[base][foreground];"
        "[base][1:v:0]overlay=0:0:eof_action=pass:repeatlast=0[captioned];"
        "[captioned]format=gbrp[captioned_rgb];"
        "[foreground]format=gbrp[foreground_rgb];"
        "[2:v:0]format=gbrp[mask_rgb];"
        "[captioned_rgb][foreground_rgb][mask_rgb]maskedmerge,"
        "format=yuv420p[vout]",
        "-map", "[vout]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-c:a", "aac",
        "-b:a", "192k",
        "-map_metadata", "0",
        str(output_path),
    )


def restore_foreground_pixels(original, captioned, mask):
    """Reference alpha formulation used by the FFmpeg masked merge."""
    import numpy as np

    if original.shape != captioned.shape or original.shape[:2] != mask.shape[:2]:
        raise ValueError("Original, captioned frame, and mask dimensions must match.")
    alpha = mask.astype(np.float32) / 255.0
    if alpha.ndim == 2:
        alpha = alpha[:, :, None]
    restored = original.astype(np.float32) * alpha + captioned.astype(np.float32) * (
        1.0 - alpha
    )
    return np.rint(restored).clip(0, 255).astype(np.uint8)


def _normalise_timeline(
    states: list[SubtitleState],
    video_duration: float,
) -> list[SubtitleState]:
    """Sort and clamp states into a non-overlapping absolute timeline."""
    if video_duration <= 0:
        return []

    ordered = sorted(states, key=lambda state: state.start)
    normalised: list[SubtitleState] = []

    for index, state in enumerate(ordered):
        start = max(0.0, state.start)
        end = min(video_duration, state.end)

        if index + 1 < len(ordered):
            next_start = max(0.0, ordered[index + 1].start)
            end = min(end, next_start)

        if end <= start:
            continue

        normalised.append(SubtitleState(
            rendered_words=state.rendered_words,
            start=start,
            end=end,
            caption_style=state.caption_style,
        ))

    return normalised


def _write_concat_file(
    concat_path: Path,
    states: list[SubtitleState],
    png_paths: list[Path],
    video_duration: float,
) -> None:
    """Build the FFmpeg concat demuxer input file.

    Each entry maps a PNG to a duration. Gaps between states (silence)
    are filled with the first state's blank frame (or a transparent frame
    rendered as state index 0 if the first state doesn't start at 0).
    """
    lines: list[str] = []

    blank_path = png_paths[0].parent / "blank.png"
    _ensure_blank(blank_path, png_paths[0])
    cursor = 0.0

    for state, png_path in zip(states, png_paths):
        gap = state.start - cursor
        if gap > 0:
            lines.append(f"file '{_escape_path(blank_path)}'")
            lines.append(f"duration {gap:.6f}")

        duration = state.end - state.start
        lines.append(f"file '{_escape_path(png_path)}'")
        lines.append(f"duration {duration:.6f}")
        cursor = state.end

    # After the last subtitle, show blank for the remainder of the video
    remaining = video_duration - cursor
    if remaining > 0:
        lines.append(f"file '{_escape_path(blank_path)}'")
        lines.append(f"duration {remaining:.6f}")

    # Concat demuxer requires a trailing entry without duration -- use blank
    lines.append(f"file '{_escape_path(blank_path)}'")

    concat_path.write_text("\n".join(lines), encoding="utf-8")


def _render_blank(blank_path: Path, _ref: object, sample_png: Path) -> None:
    """Create a transparent blank PNG matching the video dimensions."""
    if blank_path.exists():
        return
    from PIL import Image
    # Read sample to get dimensions
    with Image.open(str(sample_png)) as sample:
        w, h = sample.size
    blank = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    blank.save(str(blank_path), "PNG")


def _ensure_blank(blank_path: Path, sample_png: Path) -> None:
    if not blank_path.exists():
        _render_blank(blank_path, None, sample_png)


def _escape_path(p: Path) -> str:
    """Escape single quotes in paths for the concat demuxer."""
    return str(p).replace("'", "'\\''")


def _notify(
    cb: Callable[[str, int, int], None] | None,
    phase: str,
    current: int,
    total: int,
) -> None:
    if cb is not None:
        cb(phase, current, total)
