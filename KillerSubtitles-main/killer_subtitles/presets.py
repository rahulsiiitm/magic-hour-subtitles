"""Platform presets for common short-form video formats.

Each preset provides sensible defaults for font size, position, margins,
and styling that look good on the target platform. Values that depend on
the actual video dimensions (font_size, margin_x) are expressed as
fractions and resolved at runtime by ``resolve_preset``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Preset:
    """A named collection of default overrides."""

    name: str
    font_size_frac: float       # fraction of video height
    position: str
    margin_x_frac: float        # fraction of video width
    outline_width: int
    shadow_offset: int
    font_color: str
    highlight_color: str
    outline_color: str
    shadow_color: str
    max_lines: int
    mode: str


PRESETS: dict[str, Preset] = {
    "tiktok": Preset(
        name="tiktok",
        font_size_frac=0.05,
        position="lower",
        margin_x_frac=0.08,
        outline_width=5,
        shadow_offset=2,
        font_color="#FFFFFF",
        highlight_color="#FFD700",
        outline_color="#000000",
        shadow_color="#000000",
        max_lines=3,
        mode="karaoke",
    ),
    "reels": Preset(
        name="reels",
        font_size_frac=0.048,
        position="lower",
        margin_x_frac=0.10,
        outline_width=4,
        shadow_offset=2,
        font_color="#FFFFFF",
        highlight_color="#00E5FF",
        outline_color="#000000",
        shadow_color="#000000",
        max_lines=3,
        mode="karaoke",
    ),
    "shorts": Preset(
        name="shorts",
        font_size_frac=0.045,
        position="center",
        margin_x_frac=0.10,
        outline_width=4,
        shadow_offset=2,
        font_color="#FFFFFF",
        highlight_color="#FFEB3B",
        outline_color="#000000",
        shadow_color="#000000",
        max_lines=2,
        mode="karaoke",
    ),
}


def resolve_preset(
    preset_name: str,
    video_width: int,
    video_height: int,
) -> dict[str, Any]:
    """Return a dict of resolved default values for the given preset.

    Fractional sizes are converted to pixel values based on actual
    video dimensions. The returned dict keys match CLI option names.
    """
    p = PRESETS.get(preset_name)
    if p is None:
        raise ValueError(
            f"Unknown preset: {preset_name!r}. "
            f"Choose from: {', '.join(PRESETS)}"
        )
    return {
        "font_size": int(video_height * p.font_size_frac),
        "position": p.position,
        "margin_x": int(video_width * p.margin_x_frac),
        "outline_width": p.outline_width,
        "shadow_offset": p.shadow_offset,
        "font_color": p.font_color,
        "highlight_color": p.highlight_color,
        "outline_color": p.outline_color,
        "shadow_color": p.shadow_color,
        "max_lines": p.max_lines,
        "mode": p.mode,
    }
