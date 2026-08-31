"""Layout engine for subtitle text wrapping and positioning.

Uses Pillow font metrics to measure word widths, then groups words into
lines, lines into pages, and pages into timed SubtitleState snapshots
for all three display modes (karaoke, word, chunk).
"""

from __future__ import annotations

from pathlib import Path

from PIL import ImageFont

from .models import (
    CaptionPlan,
    LayoutConfig,
    Line,
    Page,
    Placement,
    PlacementPlan,
    RenderedWord,
    StyleConfig,
    SubtitleState,
    VideoInfo,
    Word,
)


# Vertical position anchors as fractions of video height
POSITION_ANCHORS: dict[str, float] = {
    "top": 0.10,
    "upper": 0.25,
    "center": 0.50,
    "lower": 0.75,
    "bottom": 0.90,
}

MIN_DYNAMIC_STATE_DURATION = 0.10


class LayoutEngine:
    """Calculates text wrapping and produces positioned SubtitleStates."""

    def __init__(
        self,
        video: VideoInfo,
        style: StyleConfig,
        layout: LayoutConfig,
    ) -> None:
        self.video = video
        self.style = style
        self.layout = layout

        self.font = ImageFont.truetype(style.font_path, style.font_size)
        highlight_size = (
            style.highlight_size if style.highlight_size > 0 else style.font_size
        )
        self.highlight_font = ImageFont.truetype(style.font_path, highlight_size)
        self._dynamic_fonts: dict[int, ImageFont.FreeTypeFont] = {}
        self.space_width = self.font.getlength(" ")

        # Subtitle area
        self.area_width = video.width - 2 * layout.margin_x
        self.area_x_start = layout.margin_x

        # Vertical anchor (center of the text block)
        anchor_frac = POSITION_ANCHORS.get(layout.position, 0.75)
        self.anchor_y = int(video.height * anchor_frac)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_states(self, words: list[Word]) -> list[SubtitleState]:
        """Convert transcribed words into a list of timed SubtitleStates."""
        if not words:
            return []

        if self.style.uppercase:
            for w in words:
                w.text = w.text.upper()

        mode = self.layout.mode
        if mode == "karaoke":
            return self._build_karaoke(words)
        elif mode == "word":
            return self._build_word(words)
        elif mode == "chunk":
            return self._build_chunk(words)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

    def build_dynamic_states(
        self,
        plans: list[CaptionPlan],
        placement_plans: list[PlacementPlan] | None = None,
    ) -> list[SubtitleState]:
        """Build stable, full-caption states with per-word dynamic emphasis."""
        states: list[SubtitleState] = []
        placements = {
            id(item.caption_plan): item.placement
            for item in (placement_plans or [])
        }
        for plan_index, plan in enumerate(plans):
            if not plan.caption.words:
                continue

            positioned = self._position_dynamic_caption(
                plan,
                placements.get(id(plan)),
            )
            keywords = set(plan.keyword_indices)
            words = plan.caption.words

            for active_index, active_word in enumerate(words):
                rendered: list[RenderedWord] = []
                for word_index, template in positioned:
                    important = word_index in keywords
                    current = word_index == active_index
                    if important and current:
                        scale = plan.style.combined_scale
                    elif current:
                        scale = plan.style.active_scale
                    elif important:
                        scale = plan.style.keyword_scale
                    else:
                        scale = 1.0

                    rendered.append(RenderedWord(
                        text=template.text,
                        x=template.x,
                        y=template.y,
                        is_current=current,
                        is_important=important,
                        scale=scale,
                        y_offset=(
                            int(round(
                                self.style.font_size
                                * plan.style.active_y_offset_frac
                            ))
                            if current
                            else 0
                        ),
                    ))

                has_next_word = active_index + 1 < len(words)
                next_start = (
                    words[active_index + 1].start
                    if has_next_word
                    else self._next_caption_start(plans, plan_index)
                )
                nominal_end = next_start if has_next_word else active_word.end
                end = self._dynamic_state_end(
                    active_word,
                    nominal_end,
                    next_start,
                )
                states.append(SubtitleState(
                    rendered_words=rendered,
                    start=active_word.start,
                    end=end,
                    caption_style=plan.style,
                ))
        return states

    def _dynamic_state_end(
        self,
        word: Word,
        nominal_end: float,
        boundary: float | None,
    ) -> float:
        """Return a positive state end without crossing a later boundary."""
        if word.end > word.start:
            return nominal_end

        minimum_end = word.start + MIN_DYNAMIC_STATE_DURATION
        end = minimum_end
        if boundary is not None and boundary > word.start:
            end = min(end, boundary)
        return end

    @staticmethod
    def _next_caption_start(
        plans: list[CaptionPlan],
        plan_index: int,
    ) -> float | None:
        for next_plan in plans[plan_index + 1:]:
            if next_plan.caption.words:
                return next_plan.caption.start
        return None

    # ------------------------------------------------------------------
    # Mode: karaoke (build-up)
    # ------------------------------------------------------------------

    def _build_karaoke(self, words: list[Word]) -> list[SubtitleState]:
        pages = self._paginate(words)
        states: list[SubtitleState] = []

        for page in pages:
            all_page_words = page.all_words
            for word_idx, current_word in enumerate(all_page_words):
                visible_words = all_page_words[: word_idx + 1]
                visible_lines = self._wrap_words(visible_words)

                start = current_word.start
                # End time: next word's start, or last word's end for final word
                if word_idx + 1 < len(all_page_words):
                    end = all_page_words[word_idx + 1].start
                else:
                    end = current_word.end

                rendered = self._position_lines(
                    visible_lines,
                    highlight_word_index=word_idx,
                    total_lines_hint=len(self._wrap_words(all_page_words)),
                )
                states.append(SubtitleState(
                    rendered_words=rendered, start=start, end=end,
                ))

        return states

    # ------------------------------------------------------------------
    # Mode: word (one at a time)
    # ------------------------------------------------------------------

    def _build_word(self, words: list[Word]) -> list[SubtitleState]:
        states: list[SubtitleState] = []
        for w in words:
            rw = self._center_single_word(w.text)
            states.append(SubtitleState(
                rendered_words=[rw], start=w.start, end=w.end,
            ))
        return states

    # ------------------------------------------------------------------
    # Mode: chunk (N words at a time)
    # ------------------------------------------------------------------

    def _build_chunk(self, words: list[Word]) -> list[SubtitleState]:
        n = max(1, self.layout.words_per_chunk)
        states: list[SubtitleState] = []

        for i in range(0, len(words), n):
            chunk = words[i : i + n]
            chunk_end = chunk[-1].end

            for word_idx, current_word in enumerate(chunk):
                start = current_word.start
                end = chunk[word_idx + 1].start if word_idx + 1 < len(chunk) else chunk_end

                lines = self._wrap_words(chunk)
                rendered = self._position_lines(
                    lines, highlight_word_index=word_idx,
                )
                states.append(SubtitleState(
                    rendered_words=rendered, start=start, end=end,
                ))

        return states

    # ------------------------------------------------------------------
    # Text wrapping
    # ------------------------------------------------------------------

    def _paginate(self, words: list[Word]) -> list[Page]:
        """Split words into pages of max_lines each."""
        all_lines = self._wrap_words(words)
        max_l = max(1, self.layout.max_lines)
        pages: list[Page] = []

        for i in range(0, len(all_lines), max_l):
            page_lines = all_lines[i : i + max_l]
            pages.append(Page(lines=page_lines))

        return pages

    def _wrap_words(self, words: list[Word]) -> list[Line]:
        """Wrap words into lines that fit the subtitle area width.

        If ``words_per_line`` is set, use a fixed word count per line.
        Otherwise, greedily fill lines based on pixel width.
        """
        if self.layout.words_per_line is not None:
            return self._wrap_fixed(words)
        return self._wrap_auto(words)

    def _wrap_auto(self, words: list[Word]) -> list[Line]:
        lines: list[Line] = []
        current_line: list[Word] = []
        current_width = 0.0

        for word in words:
            word_width = self._word_slot_width(word.text)
            needed = word_width + (self.space_width if current_line else 0)

            if current_width + needed > self.area_width * 0.95 and current_line:
                lines.append(Line(words=list(current_line)))
                current_line = [word]
                current_width = word_width
            else:
                current_line.append(word)
                current_width += needed

        if current_line:
            lines.append(Line(words=list(current_line)))

        return lines

    def _wrap_fixed(self, words: list[Word]) -> list[Line]:
        n = max(1, self.layout.words_per_line or 1)
        lines: list[Line] = []
        for i in range(0, len(words), n):
            lines.append(Line(words=words[i : i + n]))
        return lines

    def _position_dynamic_caption(
        self,
        plan: CaptionPlan,
        placement: Placement | None = None,
    ) -> list[tuple[int, RenderedWord]]:
        """Position a caption using maximum-scale slots to prevent reflow."""
        lines, reserve_font, line_height, block_width, block_height = (
            self._dynamic_caption_geometry(plan)
        )
        keyword_indices = set(plan.keyword_indices)

        if placement is not None:
            block_left = placement.x
            block_top = placement.y
        else:
            block_left = self.area_x_start + (self.area_width - block_width) / 2
            if self.layout.position == "top":
                block_top = self.layout.margin_y
            elif self.layout.position == "bottom":
                block_top = self.video.height - self.layout.margin_y - block_height
            else:
                block_top = self.anchor_y - block_height // 2
        block_left = max(0, min(block_left, self.video.width - block_width))
        block_top = max(0, min(block_top, self.video.height - block_height))

        normal_total = sum(self.font.getmetrics())
        reserve_total = sum(reserve_font.getmetrics())
        vertical_padding = max(0, (reserve_total - normal_total) // 2)
        positioned: list[tuple[int, RenderedWord]] = []

        for line_index, line in enumerate(lines):
            widths = [
                self._dynamic_slot_width(
                    self._display_text(word.text),
                    plan,
                    word_index in keyword_indices,
                )
                for word_index, word in line
            ]
            line_width = sum(widths) + self.space_width * max(0, len(line) - 1)
            cursor_x = block_left + (block_width - line_width) / 2
            line_y = block_top + line_index * line_height + vertical_padding

            for (word_index, word), slot_width in zip(line, widths):
                text = self._display_text(word.text)
                normal_width = self.font.getlength(text)
                positioned.append((
                    word_index,
                    RenderedWord(
                        text=text,
                        x=int(cursor_x + (slot_width - normal_width) / 2),
                        y=int(line_y),
                        is_important=word_index in keyword_indices,
                    ),
                ))
                cursor_x += slot_width + self.space_width

        return positioned

    def measure_dynamic_caption(self, plan: CaptionPlan) -> tuple[int, int]:
        """Measure the stable maximum-scale box used by every word state."""
        _lines, _font, _line_height, width, height = (
            self._dynamic_caption_geometry(plan)
        )
        return (int(round(width)), int(round(height)))

    def _dynamic_caption_geometry(
        self,
        plan: CaptionPlan,
    ) -> tuple[
        list[list[tuple[int, Word]]],
        ImageFont.FreeTypeFont,
        int,
        float,
        int,
    ]:
        indexed_words = list(enumerate(plan.caption.words))
        keyword_indices = set(plan.keyword_indices)
        lines: list[list[tuple[int, Word]]] = []
        current_line: list[tuple[int, Word]] = []
        current_width = 0.0

        for word_index, word in indexed_words:
            text = self._display_text(word.text)
            slot_width = self._dynamic_slot_width(
                text,
                plan,
                word_index in keyword_indices,
            )
            needed = slot_width + (self.space_width if current_line else 0)
            if current_width + needed > self.area_width * 0.95 and current_line:
                lines.append(current_line)
                current_line = [(word_index, word)]
                current_width = slot_width
            else:
                current_line.append((word_index, word))
                current_width += needed
        if current_line:
            lines.append(current_line)

        reserve_scale = max(
            plan.style.active_scale,
            plan.style.keyword_scale,
            plan.style.combined_scale,
        )
        reserve_font = self._dynamic_font(reserve_scale)
        reserve_bbox = reserve_font.getbbox("Ayg|")
        line_height = int((reserve_bbox[3] - reserve_bbox[1]) * 1.35)
        offset_room = abs(int(self.style.font_size * plan.style.active_y_offset_frac))
        line_height += offset_room
        block_height = len(lines) * line_height
        line_widths = [
            sum(
                self._dynamic_slot_width(
                    self._display_text(word.text),
                    plan,
                    word_index in keyword_indices,
                )
                for word_index, word in line
            ) + self.space_width * max(0, len(line) - 1)
            for line in lines
        ]
        block_width = max(line_widths, default=1.0)
        return lines, reserve_font, line_height, block_width, block_height

    def _dynamic_slot_width(
        self,
        text: str,
        plan: CaptionPlan,
        important: bool,
    ) -> float:
        max_scale = (
            plan.style.combined_scale
            if important
            else plan.style.active_scale
        )
        return max(
            self.font.getlength(text),
            self._dynamic_font(max_scale).getlength(text),
        )

    def _dynamic_font(self, scale: float) -> ImageFont.FreeTypeFont:
        size = max(1, int(round(self.style.font_size * scale)))
        if size not in self._dynamic_fonts:
            self._dynamic_fonts[size] = ImageFont.truetype(
                self.style.font_path,
                size,
            )
        return self._dynamic_fonts[size]

    def _display_text(self, text: str) -> str:
        return text.upper() if self.style.uppercase else text

    # ------------------------------------------------------------------
    # Positioning
    # ------------------------------------------------------------------

    def _position_lines(
        self,
        lines: list[Line],
        highlight_word_index: int | None = None,
        total_lines_hint: int | None = None,
    ) -> list[RenderedWord]:
        """Compute pixel (x, y) for every word across multiple lines.

        The text block is vertically centred on ``self.anchor_y``.
        Each line is horizontally centred within the subtitle area.
        ``total_lines_hint`` reserves vertical space for future lines
        in karaoke mode so text doesn't jump around as lines are added.
        """
        line_height = self._line_height()
        display_lines = total_lines_hint if total_lines_hint else len(lines)
        block_height = display_lines * line_height
        if self.layout.position == "top":
            block_top = self.layout.margin_y
        elif self.layout.position == "bottom":
            block_top = self.video.height - self.layout.margin_y - block_height
        else:
            block_top = self.anchor_y - block_height // 2
        block_top = max(0, min(block_top, self.video.height - block_height))

        rendered: list[RenderedWord] = []
        global_word_idx = 0

        for line_idx, line in enumerate(lines):
            line_width = sum(self._word_slot_width(w.text) for w in line.words)
            line_width += self.space_width * max(0, len(line.words) - 1)
            line_x = self.area_x_start + int((self.area_width - line_width) / 2)
            line_y = (
                block_top
                + line_idx * line_height
                + self._highlight_vertical_padding()
            )

            cursor_x = line_x
            for w in line.words:
                normal_width = self.font.getlength(w.text)
                slot_width = self._word_slot_width(w.text)
                is_current = (
                    highlight_word_index is not None
                    and global_word_idx == highlight_word_index
                )
                rendered.append(RenderedWord(
                    text=w.text,
                    x=int(cursor_x + (slot_width - normal_width) / 2),
                    y=int(line_y),
                    is_current=is_current,
                ))
                cursor_x += slot_width + self.space_width
                global_word_idx += 1

        return rendered

    def _center_single_word(self, text: str) -> RenderedWord:
        """Position a single word at the centre of the subtitle anchor."""
        word_width = self.font.getlength(text)
        bbox = self.font.getbbox(text)
        highlight_bbox = self.highlight_font.getbbox(text)
        word_height = max(
            bbox[3] - bbox[1],
            highlight_bbox[3] - highlight_bbox[1],
        )
        vertical_padding = self._highlight_vertical_padding()

        x = self.area_x_start + int((self.area_width - word_width) / 2)
        if self.layout.position == "top":
            y = self.layout.margin_y + vertical_padding
        elif self.layout.position == "bottom":
            y = (
                self.video.height
                - self.layout.margin_y
                - word_height
                + vertical_padding
            )
        else:
            y = self.anchor_y - word_height // 2 + vertical_padding
        y = max(
            vertical_padding,
            min(y, self.video.height - word_height + vertical_padding),
        )

        return RenderedWord(text=text, x=x, y=y, is_current=True)

    def _line_height(self) -> int:
        """Compute line height from font metrics with comfortable spacing."""
        normal_bbox = self.font.getbbox("Ayg|")
        highlight_bbox = self.highlight_font.getbbox("Ayg|")
        raw_height = max(
            normal_bbox[3] - normal_bbox[1],
            highlight_bbox[3] - highlight_bbox[1],
        )
        return int(raw_height * 1.35)

    def _word_slot_width(self, text: str) -> float:
        """Reserve enough horizontal room for the word's highlighted size."""
        return max(
            self.font.getlength(text),
            self.highlight_font.getlength(text),
        )

    def _highlight_vertical_padding(self) -> int:
        normal_asc, normal_desc = self.font.getmetrics()
        highlight_asc, highlight_desc = self.highlight_font.getmetrics()
        return max(
            0,
            ((highlight_asc + highlight_desc) - (normal_asc + normal_desc)) // 2,
        )
