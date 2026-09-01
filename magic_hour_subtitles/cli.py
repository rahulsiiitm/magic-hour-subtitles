"""Click CLI for Magic Hour Dynamic Subtitles.

Usage:
    python -m magic_hour_subtitles INPUT_VIDEO -o OUTPUT_VIDEO [OPTIONS]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from . import __version__

console = Console()

# Resolve the bundled font path (works both as package and PyInstaller bundle)
if getattr(sys, "frozen", False):
    _PACKAGE_DIR = Path(sys._MEIPASS) / "magic_hour_subtitles"
else:
    _PACKAGE_DIR = Path(__file__).resolve().parent
_DEFAULT_FONT = _PACKAGE_DIR / "fonts" / "Montserrat-ExtraBold.ttf"


def _validate_hex_color(
    _ctx: click.Context,
    param: click.Parameter,
    value: str | None,
) -> str | None:
    if value is None:
        return None
    if not re.fullmatch(r"#?[0-9a-fA-F]{6}", value):
        raise click.BadParameter(
            "must be a six-digit hex color such as #FFFFFF",
            param=param,
        )
    return f"#{value.lstrip('#').upper()}"


def _default_font_path() -> str:
    if _DEFAULT_FONT.is_file():
        return str(_DEFAULT_FONT)
    raise click.ClickException(
        f"Bundled font not found at {_DEFAULT_FONT}. "
        "Please provide --font /path/to/font.ttf"
    )


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.argument("input_video", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o", "--output", "output_video",
    type=click.Path(dir_okay=False),
    default=None,
    help="Output video path. Defaults to <input>_subtitled.mp4.",
)
# -- Mode -----------------------------------------------------------
@click.option(
    "--mode", "mode",
    type=click.Choice(["karaoke", "word", "chunk"], case_sensitive=False),
    default=None,
    help="Subtitle display mode [default: karaoke].",
)
@click.option(
    "--words-per-line", type=click.IntRange(min=1), default=None,
    help="Fixed words per line (auto-calculated if omitted).",
)
@click.option(
    "--max-lines", type=click.IntRange(min=1), default=None,
    help="Max lines per page in karaoke mode [default: 3].",
)
@click.option(
    "--words-per-chunk", type=click.IntRange(min=1), default=None,
    help="Words per chunk in chunk mode [default: 3].",
)
# -- Style ----------------------------------------------------------
@click.option(
    "--font", "font_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to a .ttf font file [default: bundled Montserrat ExtraBold].",
)
@click.option(
    "--font-size", type=click.IntRange(min=1), default=None,
    help="Font size in pixels [default: ~5% of video height].",
)
@click.option("--font-color", callback=_validate_hex_color, default=None, help="Text color as hex [default: #FFFFFF].")
@click.option("--highlight-color", callback=_validate_hex_color, default=None, help="Current-word highlight color [default: #FFD700].")
@click.option("--outline-color", callback=_validate_hex_color, default=None, help="Outline color [default: #000000].")
@click.option("--outline-width", type=click.IntRange(min=0), default=None, help="Outline thickness in px [default: 5].")
@click.option("--shadow-color", callback=_validate_hex_color, default=None, help="Shadow color [default: #000000].")
@click.option("--shadow-offset", type=click.IntRange(min=0), default=None, help="Shadow offset in px [default: 2].")
@click.option("--highlight-size", type=click.IntRange(min=1), default=None, help="Font size in px for highlighted word (bigger = pop effect).")
@click.option("--uppercase", is_flag=True, default=False, help="Render all text in UPPERCASE.")
# -- Position -------------------------------------------------------
@click.option(
    "--position",
    type=click.Choice(["top", "upper", "center", "lower", "bottom"], case_sensitive=False),
    default=None,
    help="Vertical position [default: lower].",
)
@click.option("--margin-x", type=click.IntRange(min=0), default=None, help="Horizontal margin in px [default: 10% of width].")
@click.option("--margin-y", type=click.IntRange(min=0), default=None, help="Vertical margin in px (for top/bottom anchors).")
# -- Transcription --------------------------------------------------
@click.option("--language", default="en", help="ISO language code [default: en].")
@click.option("--whisper-prompt", default=None, help="Pronunciation guide for Whisper.")
@click.option(
    "--transcript", "transcript_path",
    type=click.Path(exists=True, dir_okay=False), default=None,
    help="Path to a text transcript to guide Whisper recognition.",
)
# -- Output ---------------------------------------------------------
@click.option("--export-srt", is_flag=True, default=False, help="Also export an .srt file.")
@click.option("--no-highlight", is_flag=True, default=False, help="Disable current-word highlighting.")
@click.option("--dynamic-captions", is_flag=True, default=False, help="Enable Phase 2 semantic chunks and tone styling.")
@click.option("--smart-placement", is_flag=True, default=False, help="Enable Phase 3 scene-aware caption placement.")
@click.option("--behind-subject", is_flag=True, default=False, help="Enable Phase 4 person foreground occlusion.")
@click.option("--caption-diagnostics", is_flag=True, default=False, help="Print caption tone, keywords, and timing.")
@click.option(
    "--preset",
    type=click.Choice(["tiktok", "reels", "shorts"], case_sensitive=False),
    default=None,
    help="Apply a platform preset (overrides style defaults).",
)
@click.version_option(__version__, prog_name="magic-hour-subtitles")
def main(
    input_video: str,
    output_video: str | None,
    mode: str | None,
    words_per_line: int | None,
    max_lines: int | None,
    words_per_chunk: int | None,
    font_path: str | None,
    font_size: int | None,
    font_color: str | None,
    highlight_color: str | None,
    outline_color: str | None,
    outline_width: int | None,
    shadow_color: str | None,
    shadow_offset: int | None,
    highlight_size: int | None,
    uppercase: bool,
    position: str | None,
    margin_x: int | None,
    margin_y: int | None,
    language: str,
    whisper_prompt: str | None,
    transcript_path: str | None,
    export_srt: bool,
    no_highlight: bool,
    dynamic_captions: bool,
    smart_placement: bool,
    behind_subject: bool,
    caption_diagnostics: bool,
    preset: str | None,
) -> None:
    """Generate TikTok-style subtitles for a video.

    Transcribes INPUT_VIDEO using local faster-whisper, then renders animated
    subtitles with thick outlines, shadows, and word highlighting.
    """
    from .ffmpeg import get_ffmpeg_exe, get_ffmpeg_version, get_video_info
    from .layout import resolve_visual_config
    from .models import LayoutConfig, PipelineConfig, StyleConfig
    from .pipeline import run_pipeline
    from .presets import resolve_preset

    # -- Banner & FFmpeg check --------------------------------------
    console.print(
        f"\n[bold]Magic Hour Dynamic Subtitles[/bold] v{__version__}",
        style="bold",
    )
    try:
        exe = get_ffmpeg_exe()
        ver = get_ffmpeg_version()
        console.print(f"  FFmpeg: {ver}", style="dim")
    except RuntimeError as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        raise SystemExit(1)

    # -- Video info -------------------------------------------------
    input_path = Path(input_video)
    try:
        with _spinner("Reading video metadata"):
            video_info = get_video_info(input_path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    console.print(
        f"  Video:  {video_info.width}x{video_info.height} "
        f"@ {video_info.fps:.1f}fps, {video_info.duration:.1f}s",
        style="dim",
    )

    # -- Resolve defaults via preset --------------------------------
    defaults = {}
    if preset:
        defaults = resolve_preset(preset, video_info.width, video_info.height)
        console.print(f"  Preset: {preset}", style="dim")

    def _val(explicit, key, fallback):
        if explicit is not None:
            return explicit
        return defaults.get(key, fallback)

    resolved_font = font_path or _default_font_path()
    resolved_font_size = max(
        1,
        _val(font_size, "font_size", int(video_info.height * 0.05)),
    )

    style = StyleConfig(
        font_path=resolved_font,
        font_size=resolved_font_size,
        font_color=_val(font_color, "font_color", "#FFFFFF"),
        highlight_color=_val(highlight_color, "highlight_color", "#FFD700"),
        outline_color=_val(outline_color, "outline_color", "#000000"),
        outline_width=_val(outline_width, "outline_width", 5),
        shadow_color=_val(shadow_color, "shadow_color", "#000000"),
        shadow_offset=_val(shadow_offset, "shadow_offset", 2),
        uppercase=uppercase,
        highlight_size=highlight_size if highlight_size is not None else 0,
    )

    if no_highlight:
        style.highlight_color = style.font_color

    resolved_margin_x = _val(
        margin_x, "margin_x", int(video_info.width * 0.10)
    )
    if resolved_margin_x * 2 >= video_info.width:
        raise click.ClickException(
            "--margin-x must leave a positive-width subtitle area."
        )
    resolved_margin_y = _val(
        margin_y, "margin_y", int(video_info.height * 0.05)
    )
    if resolved_margin_y >= video_info.height:
        raise click.ClickException("--margin-y must be less than the video height.")

    layout_cfg = LayoutConfig(
        mode=_val(mode, "mode", "karaoke"),
        words_per_line=words_per_line,
        max_lines=_val(max_lines, "max_lines", 3),
        words_per_chunk=_val(words_per_chunk, "words_per_chunk", 3),
        position=_val(position, "position", "lower"),
        margin_x=resolved_margin_x,
        margin_y=resolved_margin_y,
    )
    style, layout_cfg = resolve_visual_config(video_info, style, layout_cfg)
    if video_info.height > video_info.width:
        console.print(
            f"  Portrait polish: font={style.font_size}px, "
            f"max_lines={layout_cfg.max_lines}",
            style="dim",
        )

    # -- Output path ------------------------------------------------
    if not output_video:
        stem = input_path.stem
        output_path = input_path.with_name(f"{stem}_subtitled.mp4")
    else:
        output_path = Path(output_video)

    if input_path.resolve() == output_path.resolve():
        raise click.ClickException("Input and output video paths must be different.")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise click.ClickException(
            f"Could not create output directory: {output_path.parent}"
        ) from exc
    output_video = str(output_path)

    console.print(f"  Output: {output_video}", style="dim")
    console.print()

    pipeline_config = PipelineConfig(
        input_video=input_path,
        output_video=output_path,
        style=style,
        layout=layout_cfg,
        language=language,
        whisper_prompt=whisper_prompt,
        transcript_path=transcript_path,
        export_srt=export_srt,
        dynamic_captions=dynamic_captions,
        smart_placement=smart_placement,
        behind_subject=behind_subject,
        caption_diagnostics=caption_diagnostics,
    )

    # -- Pipeline ---------------------------------------------------
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    )
    task_ids: dict[str, int] = {}

    def _progress_cb(phase: str, current: int, total: int) -> None:
        if phase not in task_ids:
            task_ids[phase] = progress.add_task(phase, total=total)
        progress.update(task_ids[phase], completed=current, total=total)

    with progress:
        try:
            run_pipeline(
                pipeline_config,
                video_info=video_info,
                progress_callback=_progress_cb,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise click.ClickException(str(exc)) from exc

    console.print(f"\n[bold green]Done![/bold green] {output_video}\n")

    if export_srt:
        srt_path = Path(output_video).with_suffix(".srt")
        console.print(f"  SRT exported: {srt_path}")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _spinner(message: str):
    """Context manager that shows a rich spinner while work runs."""
    return console.status(f"  {message}...", spinner="dots")
