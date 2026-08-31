"""Data models for KillerSubtitles pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Word:
    """A single transcribed word with timing information from Whisper."""

    text: str
    start: float
    end: float


@dataclass
class RenderedWord:
    """A word positioned on the subtitle canvas, ready for rendering."""

    text: str
    x: int
    y: int
    is_current: bool = False


@dataclass
class Line:
    """A line of words that fits within the subtitle area width."""

    words: list[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def start(self) -> float:
        return self.words[0].start if self.words else 0.0

    @property
    def end(self) -> float:
        return self.words[-1].end if self.words else 0.0


@dataclass
class Page:
    """A group of lines displayed together on screen (cleared as a unit)."""

    lines: list[Line] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.lines[0].start if self.lines else 0.0

    @property
    def end(self) -> float:
        return self.lines[-1].end if self.lines else 0.0

    @property
    def all_words(self) -> list[Word]:
        return [w for line in self.lines for w in line.words]


@dataclass
class SubtitleState:
    """
    A single snapshot of what the subtitle overlay should look like.

    Each state maps to one transparent PNG. In karaoke mode, a new state is
    produced for every word (showing accumulated text with the current word
    highlighted). In word mode, each state is a single word. In chunk mode,
    each state is a block of N words.
    """

    rendered_words: list[RenderedWord] = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0


@dataclass
class VideoInfo:
    """Metadata extracted from the source video via FFmpeg."""

    width: int
    height: int
    fps: float
    duration: float


@dataclass
class StyleConfig:
    """All visual styling parameters for subtitle rendering."""

    font_path: str = ""
    font_size: int = 0
    font_color: str = "#FFFFFF"
    highlight_color: str = "#FFD700"
    outline_color: str = "#000000"
    outline_width: int = 5
    shadow_color: str = "#000000"
    shadow_offset: int = 2
    uppercase: bool = False
    highlight_size: int = 0  # 0 = same as font_size; >0 = px size for highlighted word


@dataclass
class LayoutConfig:
    """Parameters controlling text wrapping and positioning."""

    mode: str = "karaoke"
    words_per_line: int | None = None
    max_lines: int = 3
    words_per_chunk: int = 3
    position: str = "lower"
    margin_x: int = 0
    margin_y: int = 0
