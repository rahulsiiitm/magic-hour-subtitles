"""Pillow-based subtitle renderer.

Renders each SubtitleState as a transparent RGBA PNG with thick outlines,
drop shadows, and optional word highlighting -- matching TikTok subtitle
quality.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .models import RenderedWord, StyleConfig, SubtitleState, VideoInfo


def _hex_to_rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    """Convert ``#RRGGBB`` hex string to an RGBA tuple."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (r, g, b, alpha)


class SubtitleRenderer:
    """Renders SubtitleState objects to transparent PNG images."""

    def __init__(self, video: VideoInfo, style: StyleConfig) -> None:
        self.width = video.width
        self.height = video.height
        self.style = style
        self.font = ImageFont.truetype(style.font_path, style.font_size)

        hl_size = style.highlight_size if style.highlight_size > 0 else style.font_size
        self.highlight_font = ImageFont.truetype(style.font_path, hl_size)
        self._has_highlight_size = style.highlight_size > 0

        self.font_color = _hex_to_rgba(style.font_color)
        self.highlight_color = _hex_to_rgba(style.highlight_color)
        self.outline_color = _hex_to_rgba(style.outline_color)
        self.shadow_color = _hex_to_rgba(style.shadow_color, alpha=160)

    def render(self, state: SubtitleState) -> Image.Image:
        """Render a subtitle state to a transparent RGBA ``Image``."""
        img = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        if not state.rendered_words:
            return img

        draw = ImageDraw.Draw(img)

        for rw in state.rendered_words:
            fill = self.highlight_color if rw.is_current else self.font_color
            self._draw_word(draw, rw, fill)

        return img

    def render_to_file(self, state: SubtitleState, path: Path) -> Path:
        """Render a subtitle state and save as PNG."""
        img = self.render(state)
        img.save(str(path), "PNG")
        return path

    def _draw_word(
        self,
        draw: ImageDraw.ImageDraw,
        rw: RenderedWord,
        fill: tuple[int, int, int, int],
    ) -> None:
        x, y = rw.x, rw.y
        text = rw.text
        ow = self.style.outline_width
        so = self.style.shadow_offset

        use_big = rw.is_current and self._has_highlight_size
        font = self.highlight_font if use_big else self.font

        if use_big:
            # Use font design metrics (ascent+descent) for accurate centering
            normal_asc, normal_desc = self.font.getmetrics()
            big_asc, big_desc = self.highlight_font.getmetrics()
            normal_total = normal_asc + normal_desc
            big_total = big_asc + big_desc

            # Shift up by half the height difference so it's vertically centred
            y = y - (big_total - normal_total) // 2

            # Horizontally center the larger glyph over the normal position
            normal_w = self.font.getlength(text)
            big_w = self.highlight_font.getlength(text)
            x = x - int((big_w - normal_w) / 2)

        # Shadow pass (offset, semi-transparent)
        if so > 0:
            draw.text(
                (x + so, y + so),
                text,
                font=font,
                fill=self.shadow_color,
                stroke_width=ow,
                stroke_fill=self.shadow_color,
            )

        # Main text with thick outline
        draw.text(
            (x, y),
            text,
            font=font,
            fill=fill,
            stroke_width=ow,
            stroke_fill=self.outline_color,
        )
