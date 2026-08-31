"""Data models for KillerSubtitles pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


@dataclass
class Word:
    """A single transcribed word with timing information from Whisper."""

    text: str
    start: float
    end: float


@dataclass
class Caption:
    """A natural group of timed words displayed together."""

    words: list[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(word.text for word in self.words)

    @property
    def start(self) -> float:
        return self.words[0].start if self.words else 0.0

    @property
    def end(self) -> float:
        return self.words[-1].end if self.words else 0.0


class Tone(str, Enum):
    NEUTRAL = "neutral"
    EXCITED = "excited"
    SERIOUS = "serious"
    QUESTION = "question"


@dataclass(frozen=True)
class CaptionStyle:
    """Restrained tone-specific overrides for one caption."""

    font_color: str
    active_color: str
    keyword_color: str
    outline_width_delta: int = 0
    keyword_scale: float = 1.10
    active_scale: float = 1.15
    combined_scale: float = 1.20
    active_y_offset_frac: float = 0.0


@dataclass(frozen=True)
class CaptionPlan:
    """Analyzed caption ready for dynamic layout and rendering."""

    caption: Caption
    tone: Tone
    keyword_indices: tuple[int, ...]
    style: CaptionStyle

    @property
    def keywords(self) -> list[str]:
        return [self.caption.words[index].text for index in self.keyword_indices]


@dataclass(frozen=True)
class VisionConfig:
    """Small, Colab-friendly controls for sampled frame analysis."""

    model_name: str = "yolo11n-seg.pt"
    analysis_fps: float = 7.5
    long_side: int = 640
    map_long_side: int = 160
    confidence: float = 0.35
    person_dilation: int = 5
    scene_cut_threshold: float = 0.35
    device: str | None = None
    hysteresis: float = 0.12


@dataclass(frozen=True)
class FrameAnalysis:
    """Reduced occupancy/statistic maps for one sampled video frame."""

    timestamp: float
    frame_index: int
    map_width: int
    map_height: int
    person_map: Any
    clutter_map: Any
    motion_map: Any
    person_confidence: float = 0.0
    scene_cut: bool = False


@dataclass(frozen=True)
class Placement:
    """One measured caption rectangle in source-video coordinates."""

    name: str
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class PlacementPlan:
    """A caption plan paired with one stable scene-aware placement."""

    caption_plan: CaptionPlan
    placement: Placement
    scores: dict[str, float] = field(default_factory=dict)
    person_overlaps: dict[str, float] = field(default_factory=dict)
    best_raw_candidate: str = ""
    hysteresis_applied: bool = False
    hysteresis_reason: str = "not applicable"
    safety_override: bool = False
    previous_person_overlap: float | None = None


@dataclass
class RenderedWord:
    """A word positioned on the subtitle canvas, ready for rendering."""

    text: str
    x: int
    y: int
    is_current: bool = False
    is_important: bool = False
    scale: float = 1.0
    y_offset: int = 0


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
    caption_style: CaptionStyle | None = None


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


@dataclass
class PipelineConfig:
    """Resolved inputs for the Phase 1 subtitle pipeline."""

    input_video: str | Path
    output_video: str | Path
    style: StyleConfig
    layout: LayoutConfig
    language: str = "en"
    whisper_prompt: str | None = None
    transcript_path: str | Path | None = None
    export_srt: bool = False
    whisper_model: str = "distil-large-v3"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    cpu_model: str | None = None
    dynamic_captions: bool = False
    caption_diagnostics: bool = False
    smart_placement: bool = False
    vision: VisionConfig = field(default_factory=VisionConfig)
