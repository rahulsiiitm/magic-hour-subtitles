"""Deterministic natural caption chunking over word-level timestamps."""

from __future__ import annotations

import re
from collections.abc import Callable

from .models import Caption, Word


TARGET_MIN_WORDS = 3
TARGET_MAX_WORDS = 6
MAX_WORDS = 8
TARGET_MIN_DURATION = 0.8
TARGET_MAX_DURATION = 2.4
PAUSE_BREAK_SECONDS = 0.45

PORTRAIT_TARGET_MIN_WORDS = 2
PORTRAIT_TARGET_MAX_WORDS = 5
PORTRAIT_MAX_WORDS = 6
PORTRAIT_TARGET_MAX_DURATION = 2.1
PREFERRED_MIN_CAPTION_DURATION = 0.55
MERGED_MAX_DURATION = 2.4

AVOID_END_WORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "for", "with",
}

_STRONG_END = re.compile(r"[.!?;][\"')\]]*$")
_COMMA_END = re.compile(r",[\"')\]]*$")


def chunk_words(words: list[Word], *, portrait: bool = False) -> list[Caption]:
    """Group words naturally while preserving the original Word objects."""
    if not words:
        return []

    target_min_words = (
        PORTRAIT_TARGET_MIN_WORDS if portrait else TARGET_MIN_WORDS
    )
    target_max_words = (
        PORTRAIT_TARGET_MAX_WORDS if portrait else TARGET_MAX_WORDS
    )
    max_words = PORTRAIT_MAX_WORDS if portrait else MAX_WORDS
    target_max_duration = (
        PORTRAIT_TARGET_MAX_DURATION if portrait else TARGET_MAX_DURATION
    )

    captions: list[Caption] = []
    pending: list[Word] = []

    for index, word in enumerate(words):
        pending.append(word)
        next_word = words[index + 1] if index + 1 < len(words) else None
        duration = pending[-1].end - pending[0].start
        pause = (
            max(0.0, next_word.start - word.end)
            if next_word is not None
            else 0.0
        )

        if _STRONG_END.search(word.text) or pause > PAUSE_BREAK_SECONDS:
            _emit_all(captions, pending)
        elif (
            _COMMA_END.search(word.text)
            and len(pending) >= target_min_words
            and duration >= TARGET_MIN_DURATION
        ):
            _emit_all(captions, pending)
        elif len(pending) >= max_words or duration >= target_max_duration:
            split_at = _best_hard_split(
                pending,
                target_min_words,
                target_max_words,
                max_words,
            )
            captions.append(Caption(words=list(pending[:split_at])))
            pending = pending[split_at:]
        elif (
            len(pending) >= target_max_words
            and duration >= TARGET_MIN_DURATION
            and not _avoid_end(word.text)
        ):
            _emit_all(captions, pending)

    if pending:
        captions.append(Caption(words=list(pending)))

    _rebalance_trailing_single(
        captions,
        max_words=max_words,
        max_duration=target_max_duration,
    )
    return captions


def fit_captions_to_line_limit(
    captions: list[Caption],
    *,
    max_lines: int,
    line_count: Callable[[Caption], int],
) -> list[Caption]:
    """Split only visually overflowing captions, preserving every Word object."""
    fitted: list[Caption] = []
    pending = list(captions)
    while pending:
        caption = pending.pop(0)
        if len(caption.words) <= 1 or line_count(caption) <= max_lines:
            fitted.append(caption)
            continue

        split_at = _best_visual_split(caption, line_count)
        left = Caption(words=list(caption.words[:split_at]))
        right = Caption(words=list(caption.words[split_at:]))
        pending[0:0] = [left, right]
    return fitted


def merge_micro_captions(
    captions: list[Caption],
    *,
    line_count: Callable[[Caption], int],
    max_lines: int = 2,
    max_words: int = PORTRAIT_MAX_WORDS,
    max_duration: float = MERGED_MAX_DURATION,
) -> list[Caption]:
    """Merge flashing captions only when an adjacent visual phrase stays safe."""
    merged = [Caption(words=list(caption.words)) for caption in captions]
    while True:
        changed = False
        for index, caption in enumerate(merged):
            if caption.end - caption.start >= PREFERRED_MIN_CAPTION_DURATION:
                continue

            candidates: list[tuple[tuple[float, ...], int, Caption]] = []
            if index > 0:
                previous = merged[index - 1]
                combined = Caption(words=[*previous.words, *caption.words])
                if _safe_micro_merge(
                    previous,
                    caption,
                    combined,
                    line_count=line_count,
                    max_lines=max_lines,
                    max_words=max_words,
                    max_duration=max_duration,
                ):
                    pause = max(0.0, caption.start - previous.end)
                    balance = abs(len(previous.words) - len(caption.words))
                    candidates.append(((pause, balance, 1.0), -1, combined))
            if index + 1 < len(merged):
                following = merged[index + 1]
                combined = Caption(words=[*caption.words, *following.words])
                if _safe_micro_merge(
                    caption,
                    following,
                    combined,
                    line_count=line_count,
                    max_lines=max_lines,
                    max_words=max_words,
                    max_duration=max_duration,
                ):
                    pause = max(0.0, following.start - caption.end)
                    balance = abs(len(following.words) - len(caption.words))
                    candidates.append(((pause, balance, 0.0), 1, combined))

            if not candidates:
                continue
            _score, direction, combined = min(candidates)
            if direction < 0:
                merged[index - 1:index + 1] = [combined]
            else:
                merged[index:index + 2] = [combined]
            changed = True
            break
        if not changed:
            return merged


def _safe_micro_merge(
    left: Caption,
    right: Caption,
    combined: Caption,
    *,
    line_count: Callable[[Caption], int],
    max_lines: int,
    max_words: int,
    max_duration: float,
) -> bool:
    if not left.words or not right.words:
        return False
    if _STRONG_END.search(left.words[-1].text):
        return False
    if right.start - left.end > PAUSE_BREAK_SECONDS:
        return False
    if len(combined.words) > max_words:
        return False
    if combined.end - combined.start > max_duration:
        return False
    return line_count(combined) <= max_lines


def _best_visual_split(
    caption: Caption,
    line_count: Callable[[Caption], int],
) -> int:
    candidates: list[tuple[tuple[float, ...], int]] = []
    midpoint = len(caption.words) / 2
    for split_at in range(1, len(caption.words)):
        left = Caption(words=list(caption.words[:split_at]))
        right = Caption(words=list(caption.words[split_at:]))
        left_lines = line_count(left)
        right_lines = line_count(right)
        awkward_end = 1.0 if _avoid_end(left.words[-1].text) else 0.0
        candidates.append((
            (
                float(max(left_lines, right_lines)),
                awkward_end,
                float(left_lines + right_lines),
                abs(split_at - midpoint),
            ),
            split_at,
        ))
    return min(candidates)[1]


def _emit_all(captions: list[Caption], pending: list[Word]) -> None:
    if pending:
        captions.append(Caption(words=list(pending)))
        pending.clear()


def _best_hard_split(
    words: list[Word],
    target_min_words: int,
    target_max_words: int,
    max_words: int,
) -> int:
    upper = min(target_max_words, len(words))
    for split_at in range(upper, target_min_words - 1, -1):
        if not _avoid_end(words[split_at - 1].text):
            return split_at
    return min(max_words, len(words))


def _rebalance_trailing_single(
    captions: list[Caption],
    *,
    max_words: int,
    max_duration: float,
) -> None:
    if len(captions) < 2 or len(captions[-1].words) != 1:
        return

    previous = captions[-2]
    trailing = captions[-1]
    boundary_pause = trailing.start - previous.end
    if (
        _STRONG_END.search(previous.words[-1].text)
        or boundary_pause > PAUSE_BREAK_SECONDS
    ):
        return

    combined_count = len(previous.words) + 1
    combined_duration = trailing.end - previous.start
    if combined_count <= max_words and combined_duration <= max_duration + 0.4:
        previous.words.append(trailing.words[0])
        captions.pop()


def _avoid_end(text: str) -> bool:
    normalized = re.sub(r"[^\w']", "", text.lower())
    return normalized in AVOID_END_WORDS
