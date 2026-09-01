"""Deterministic tone classification, keyword ranking, and caption styles."""

from __future__ import annotations

import re
from collections import Counter

from .models import Caption, CaptionPlan, CaptionStyle, Tone


INTERROGATIVE_WORDS = {
    "why", "what", "how", "when", "where", "who",
}

QUESTION_AUXILIARIES = {
    "am", "are", "can", "could", "did", "do", "does", "had", "has",
    "have", "is", "may", "might", "must", "should", "was", "were",
    "will", "would",
}

EXCITED_WORDS = {
    "amazing", "incredible", "huge", "insane", "crazy", "powerful",
    "excited",
    "breakthrough", "everything", "fastest", "best", "love", "wow",
    "finally", "game-changing", "gamechanging",
}

SERIOUS_WORDS = {
    "warning", "dangerous", "failure", "fail", "critical", "serious",
    "risk", "problem", "issue", "mistake", "never", "important",
    "scared", "die", "dead", "death", "kill", "killing", "gun", "danger",
    "impossible", "trapped", "void", "terrifying",
}

TONE_ALIASES = {
    "critically": "critical",
    "dangerous": "danger",
    "excite": "excited",
    "exciting": "excited",
    "insanely": "insane",
    "power": "powerful",
    "powerfully": "powerful",
    "terrified": "terrifying",
    "terrify": "terrifying",
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can",
    "do", "does", "for", "from", "had", "has", "have", "he", "her",
    "his", "how", "i", "if", "in", "is", "it", "its", "me", "my",
    "of", "on", "or", "our", "she", "so", "that", "the", "their",
    "them", "they", "this", "to", "was", "we", "were", "what", "when",
    "where", "which", "who", "why", "will", "with", "would", "you", "your",
    "oh", "well", "did", "all", "these", "those", "about", "there", "here",
    "really",
}

EMPHASIS_WORDS = EXCITED_WORDS | SERIOUS_WORDS

TONE_STYLES: dict[Tone, CaptionStyle] = {
    Tone.NEUTRAL: CaptionStyle(
        font_color="#FFFFFF",
        active_color="#FFD166",
        keyword_color="#FFE29A",
        keyword_scale=1.09,
        active_scale=1.15,
        combined_scale=1.20,
    ),
    Tone.EXCITED: CaptionStyle(
        font_color="#FFFFFF",
        active_color="#FF6B4A",
        keyword_color="#FF8A65",
        outline_width_delta=1,
        keyword_scale=1.12,
        active_scale=1.17,
        combined_scale=1.21,
    ),
    Tone.SERIOUS: CaptionStyle(
        font_color="#FFFFFF",
        active_color="#4DD0E1",
        keyword_color="#80DEEA",
        outline_width_delta=2,
        keyword_scale=1.07,
        active_scale=1.08,
        combined_scale=1.12,
    ),
    Tone.QUESTION: CaptionStyle(
        font_color="#FFFFFF",
        active_color="#B8FF5A",
        keyword_color="#45E6E6",
        outline_width_delta=1,
        keyword_scale=1.09,
        active_scale=1.13,
        combined_scale=1.18,
        active_y_offset_frac=-0.04,
    ),
}


def analyze_captions(captions: list[Caption]) -> list[CaptionPlan]:
    """Analyze captions using transcript-wide frequencies and fixed rules."""
    frequencies = Counter(
        token
        for caption in captions
        for word in caption.words
        if (token := normalize_word(word.text))
    )

    plans: list[CaptionPlan] = []
    for caption in captions:
        tone = classify_tone(caption)
        keyword_indices = select_keyword_indices(caption, frequencies)
        plans.append(CaptionPlan(
            caption=caption,
            tone=tone,
            keyword_indices=keyword_indices,
            style=TONE_STYLES[tone],
        ))
    return plans


def classify_tone(caption: Caption) -> Tone:
    tokens = [canonical_tone_word(word.text) for word in caption.words]
    token_set = {token for token in tokens if token}
    text = caption.text

    if "?" in text or _is_strong_interrogative(tokens):
        return Tone.QUESTION
    if "!" in text or token_set & EXCITED_WORDS:
        return Tone.EXCITED
    if token_set & SERIOUS_WORDS:
        return Tone.SERIOUS
    return Tone.NEUTRAL


def select_keyword_indices(
    caption: Caption,
    transcript_frequencies: Counter[str],
) -> tuple[int, ...]:
    """Return stable word indices for at most one or two important words."""
    limit = 1 if len(caption.words) <= 4 else 2
    ranked: list[tuple[float, int]] = []

    for index, word in enumerate(caption.words):
        token = normalize_word(word.text)
        tone_token = canonical_tone_word(token)
        if not token or (
            token in STOPWORDS and tone_token not in EMPHASIS_WORDS
        ):
            continue

        frequency = max(1, transcript_frequencies.get(token, 1))
        rarity = 1.0 / frequency
        emphasis = 1.0 if tone_token in EMPHASIS_WORDS else 0.0
        length = min(len(token) / 10.0, 1.0)
        punctuation = 1.0 if re.search(r"[!?;]", word.text) else 0.0
        score = (
            0.45 * rarity
            + 0.30 * emphasis
            + 0.15 * length
            + 0.10 * punctuation
        )
        ranked.append((score, index))

    selected = sorted(ranked, key=lambda item: (-item[0], item[1]))[:limit]
    return tuple(sorted(index for _score, index in selected))


def normalize_word(text: str) -> str:
    normalized = text.strip().lower()
    return re.sub(r"^[^\w]+|[^\w]+$", "", normalized)


def canonical_tone_word(text: str) -> str:
    """Map a small set of safe lexical variants without general stemming."""
    token = normalize_word(text)
    return TONE_ALIASES.get(token, token)


def _is_strong_interrogative(tokens: list[str]) -> bool:
    if len(tokens) < 2:
        return False
    first, second = tokens[0], tokens[1]
    return (
        first in QUESTION_AUXILIARIES
        or (first in INTERROGATIVE_WORDS and second in QUESTION_AUXILIARIES)
    )
