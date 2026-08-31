"""FFmpeg binary resolution and subprocess helpers.

Locates an FFmpeg binary from imageio-ffmpeg (bundled), static-ffmpeg
(downloaded on first use), or the system PATH -- in that order.
Provides helpers for extracting audio, reading video metadata, and
running arbitrary FFmpeg commands.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from .models import VideoInfo


@lru_cache(maxsize=1)
def get_ffmpeg_exe() -> str:
    """Return the path to a usable ffmpeg binary.

    Resolution order:
      1. imageio-ffmpeg bundled binary
      2. static-ffmpeg (downloads on first use)
      3. System ffmpeg on PATH
    """
    # 1. imageio-ffmpeg
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and _is_valid(exe):
            return exe
    except Exception:
        pass

    # 2. static-ffmpeg
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
        if _is_valid("ffmpeg"):
            return "ffmpeg"
    except Exception:
        pass

    # 3. System ffmpeg
    if _is_valid("ffmpeg"):
        return "ffmpeg"

    raise RuntimeError(
        "No FFmpeg binary found. Install one of:\n"
        "  pip install imageio-ffmpeg   (recommended, bundles ffmpeg)\n"
        "  pip install static-ffmpeg    (downloads ffmpeg on first use)\n"
        "Or install ffmpeg on your system and ensure it is on PATH."
    )


def get_ffmpeg_version() -> str:
    """Return the version string of the resolved FFmpeg binary."""
    exe = get_ffmpeg_exe()
    out = subprocess.check_output(
        [exe, "-version"], stderr=subprocess.STDOUT, **_popen_kwargs()
    )
    first_line = out.decode(errors="ignore").split("\n", 1)[0]
    match = re.search(r"version\s+(\S+)", first_line)
    return match.group(1) if match else "unknown"


def get_video_info(video_path: str | Path) -> VideoInfo:
    """Extract width, height, fps, and duration from a video file.

    Uses ``ffmpeg -i`` stderr parsing since imageio-ffmpeg does not
    bundle ffprobe.
    """
    exe = get_ffmpeg_exe()
    result = subprocess.run(
        [exe, "-i", str(video_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **_popen_kwargs(),
    )
    stderr = result.stderr.decode(errors="ignore")

    width, height = _parse_dimensions(stderr)
    rotation = _parse_rotation(stderr)
    if round(abs(rotation)) % 180 == 90:
        width, height = height, width
    fps = _parse_fps(stderr)
    duration = _parse_duration(stderr)

    return VideoInfo(width=width, height=height, fps=fps, duration=duration)


def get_media_duration(media_path: str | Path) -> float:
    """Read media duration without treating FFmpeg's probe exit as a failure."""
    exe = get_ffmpeg_exe()
    result = subprocess.run(
        [exe, "-i", str(media_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **_popen_kwargs(),
    )
    stderr = result.stderr.decode(errors="ignore")
    return _parse_duration(stderr)


def extract_audio(
    video_path: str | Path,
    audio_path: str | Path,
    sample_rate: int = 16000,
) -> Path:
    """Extract audio from video as mono MP3 at the given sample rate."""
    exe = get_ffmpeg_exe()
    audio_path = Path(audio_path)
    _run(
        exe, "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "mp3",
        "-ar", str(sample_rate),
        "-ac", "1",
        str(audio_path),
    )
    return audio_path


def run_ffmpeg(*args: str) -> subprocess.CompletedProcess:
    """Run an arbitrary FFmpeg command, raising on failure."""
    exe = get_ffmpeg_exe()
    return _run(exe, *args)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _run(*cmd: str) -> subprocess.CompletedProcess:
    kwargs = _popen_kwargs()
    result = subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **kwargs,
    )
    if result.returncode != 0:
        stderr_text = result.stderr.decode(errors="ignore").strip()
        raise RuntimeError(
            f"FFmpeg command failed (exit {result.returncode}):\n"
            f"  {' '.join(cmd)}\n{stderr_text}"
        )
    return result


def _is_valid(exe: str) -> bool:
    try:
        subprocess.check_call(
            [exe, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **_popen_kwargs(),
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _popen_kwargs() -> dict:
    """Platform-specific subprocess kwargs (suppress console flash on Windows)."""
    kwargs: dict = {}
    if sys.platform.startswith("win"):
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        kwargs["startupinfo"] = si
    return kwargs


def _parse_dimensions(stderr: str) -> tuple[int, int]:
    match = re.search(r"(\d{2,5})x(\d{2,5})[\s,]", stderr)
    if not match:
        raise ValueError("Could not determine video dimensions from FFmpeg output.")
    return int(match.group(1)), int(match.group(2))


def _parse_fps(stderr: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:fps|tbr)", stderr)
    if not match:
        raise ValueError("Could not determine video FPS from FFmpeg output.")
    return float(match.group(1))


def _parse_rotation(stderr: str) -> float:
    display_matrix = re.search(
        r"rotation of\s+(-?\d+(?:\.\d+)?)\s+degrees",
        stderr,
        flags=re.IGNORECASE,
    )
    if display_matrix:
        return float(display_matrix.group(1))

    rotate_tag = re.search(
        r"\brotate\s*:\s*(-?\d+(?:\.\d+)?)",
        stderr,
        flags=re.IGNORECASE,
    )
    return float(rotate_tag.group(1)) if rotate_tag else 0.0


def _parse_duration(stderr: str) -> float:
    match = re.search(
        r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr
    )
    if not match:
        raise ValueError("Could not determine video duration from FFmpeg output.")
    h, m, s = match.group(1), match.group(2), match.group(3)
    return int(h) * 3600 + int(m) * 60 + float(s)
