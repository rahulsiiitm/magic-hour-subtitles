"""Deterministic natural caption chunking over word-level timestamps."""

from __future__ import annotations

import re

from .models import Caption, Word


TARGET_MIN_WORDS = 3
TARGET_MAX_WORDS = 6
MAX_WORDS = 8
TARGET_MIN_DURATION = 0.8
TARGET_MAX_DURATION = 2.4
PAUSE_BREAK_SECONDS = 0.45

AVOID_END_WORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "for", "with",
}

_STRONG_END = re.compile(r"[.!?;][\"')\]]*$")
_COMMA_END = re.compile(r",[\"')\]]*$")


def chunk_words(words: list[Word]) -> list[Caption]:
    """Group words naturally while preserving the original Word objects."""
    if not words:
        return []

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
            and len(pending) >= TARGET_MIN_WORDS
            and duration >= TARGET_MIN_DURATION
        ):
            _emit_all(captions, pending)
        elif len(pending) >= MAX_WORDS or duration >= TARGET_MAX_DURATION:
            split_at = _best_hard_split(pending)
            captions.append(Caption(words=list(pending[:split_at])))
            pending = pending[split_at:]
        elif (
            len(pending) >= TARGET_MAX_WORDS
            and duration >= TARGET_MIN_DURATION
            and not _avoid_end(word.text)
        ):
            _emit_all(captions, pending)

    if pending:
        captions.append(Caption(words=list(pending)))

    _rebalance_trailing_single(captions)
    return captions


def _emit_all(captions: list[Caption], pending: list[Word]) -> None:
    if pending:
        captions.append(Caption(words=list(pending)))
        pending.clear()


def _best_hard_split(words: list[Word]) -> int:
    upper = min(TARGET_MAX_WORDS, len(words))
    for split_at in range(upper, TARGET_MIN_WORDS - 1, -1):
        if not _avoid_end(words[split_at - 1].text):
            return split_at
    return min(MAX_WORDS, len(words))


def _rebalance_trailing_single(captions: list[Caption]) -> None:
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
    if combined_count <= MAX_WORDS and combined_duration <= TARGET_MAX_DURATION + 0.4:
        previous.words.append(trailing.words[0])
        captions.pop()


def _avoid_end(text: str) -> bool:
    normalized = re.sub(r"[^\w']", "", text.lower())
    return normalized in AVOID_END_WORDS
