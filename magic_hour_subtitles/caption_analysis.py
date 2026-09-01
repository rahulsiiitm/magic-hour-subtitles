"""Deterministic tone classification, keyword ranking, and caption styles."""

from __future__ import annotations

import re
from collections import Counter

from .display_text import format_display_tokens
from .models import Caption, CaptionPlan, CaptionStyle, ExpressionType, Tone


INTERROGATIVE_WORDS = {
    "why", "what", "how", "when", "where", "who",
}

QUESTION_AUXILIARIES = {
    "am", "are", "can", "could", "did", "do", "does", "had", "has",
    "have", "is", "may", "might", "must", "should", "was", "were",
    "will", "would",
}

QUESTION_SUBJECT_STARTERS = {
    "i", "you", "he", "she", "it", "we", "they", "this", "that",
    "these", "those", "there",
}

EXCITED_WORDS = {
    "amazing", "incredible", "huge", "massive", "largest", "insane",
    "crazy", "powerful", "excited",
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

QUESTION_CUE_WORDS = {
    "why", "how", "what", "who", "where", "when", "could", "would",
    "should",
}

CONTRAST_WORDS = {
    "but", "however", "actually", "instead", "even", "still", "yet",
    "although", "except",
}

CONTRAST_PHRASE_STARTS = {"that's", "thats", "here's", "heres"}

MAGNITUDE_WORDS = {
    "largest", "smallest", "fastest", "slowest", "million", "billion",
    "thousand", "tons", "percent",
}

MAX_EXPRESSIONS_PER_CAPTION = 2
_NUMERIC_TOKEN = re.compile(
    r"^[\$€£]?\d[\d,]*(?:\.\d+)?(?:%|x)?$",
    re.IGNORECASE,
)

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
        expression_types = classify_expressions(
            caption,
            tone,
            keyword_indices,
            ranked_keyword_indices=_rank_keyword_indices(
                caption,
                frequencies,
            ),
        )
        plans.append(CaptionPlan(
            caption=caption,
            tone=tone,
            keyword_indices=keyword_indices,
            style=TONE_STYLES[tone],
            expression_types=expression_types,
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
    selected = _rank_keyword_indices(caption, transcript_frequencies)[:limit]
    return tuple(sorted(selected))


def _rank_keyword_indices(
    caption: Caption,
    transcript_frequencies: Counter[str],
) -> list[int]:
    """Return every eligible keyword index in the existing score order."""
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

    return [
        index
        for _score, index in sorted(ranked, key=lambda item: (-item[0], item[1]))
    ]


def classify_expressions(
    caption: Caption,
    tone: Tone,
    keyword_indices: tuple[int, ...],
    *,
    ranked_keyword_indices: list[int] | None = None,
) -> tuple[ExpressionType, ...]:
    """Assign at most two deterministic local expressions by fixed priority."""
    count = len(caption.words)
    if count == 0:
        return ()

    tokens = [canonical_tone_word(word.text) for word in caption.words]
    normalized = [normalize_word(word.text) for word in caption.words]
    display_texts = [
        token.text
        for token in format_display_tokens(
            [word.text for word in caption.words]
        )
    ]
    numeric_indices = {
        index
        for index in range(count)
        if _is_numeric_magnitude(index, caption, normalized, display_texts)
    }
    question_index = _question_cue_index(tokens, tone)
    contrast_indices = _contrast_indices(normalized)
    candidates: list[tuple[int, int, ExpressionType]] = []

    for index, token in enumerate(tokens):
        expression = ExpressionType.NONE
        if (
            (tone is Tone.EXCITED and token in EXCITED_WORDS)
            or (tone is Tone.SERIOUS and token in SERIOUS_WORDS)
        ):
            expression = ExpressionType.TONE_TRIGGER
        elif index in numeric_indices:
            expression = ExpressionType.NUMERIC_MAGNITUDE
        elif index == question_index:
            expression = ExpressionType.QUESTION_CUE
        elif index in contrast_indices:
            expression = ExpressionType.CONTRAST_REVEAL
        if expression is not ExpressionType.NONE:
            candidates.append((_expression_priority(expression), index, expression))

    selected: dict[int, ExpressionType] = {}
    for _priority, index, expression in sorted(candidates):
        if len(selected) >= MAX_EXPRESSIONS_PER_CAPTION:
            break
        selected[index] = expression

    content_budget = min(
        len(keyword_indices),
        MAX_EXPRESSIONS_PER_CAPTION - len(selected),
    )
    content_candidates = ranked_keyword_indices or list(keyword_indices)
    for index in content_candidates:
        if content_budget <= 0:
            break
        if index in selected:
            continue
        selected[index] = ExpressionType.CONTENT_KEYWORD
        content_budget -= 1

    return tuple(
        selected.get(index, ExpressionType.NONE)
        for index in range(count)
    )


def _expression_priority(expression: ExpressionType) -> int:
    return {
        ExpressionType.TONE_TRIGGER: 1,
        ExpressionType.NUMERIC_MAGNITUDE: 2,
        ExpressionType.QUESTION_CUE: 3,
        ExpressionType.CONTRAST_REVEAL: 4,
        ExpressionType.CONTENT_KEYWORD: 5,
        ExpressionType.NONE: 6,
    }[expression]


def _question_cue_index(tokens: list[str], tone: Tone) -> int | None:
    if tone is not Tone.QUESTION:
        return None
    return next(
        (index for index, token in enumerate(tokens) if token in QUESTION_CUE_WORDS),
        None,
    )


def _contrast_indices(tokens: list[str]) -> set[int]:
    indices = {
        index for index, token in enumerate(tokens) if token in CONTRAST_WORDS
    }
    for index, token in enumerate(tokens[:-1]):
        if token in CONTRAST_PHRASE_STARTS and tokens[index + 1] == "why":
            indices.add(index)
    return indices


def _is_numeric_magnitude(
    index: int,
    caption: Caption,
    normalized: list[str],
    display_texts: list[str],
) -> bool:
    raw = (
        display_texts[index] or caption.words[index].text
    ).strip().strip("()[]{}\"'.,!?;:")
    token = normalized[index]
    if token in MAGNITUDE_WORDS:
        return any(
            0 <= neighbor < len(caption.words)
            and _looks_numeric(caption.words[neighbor].text)
            for neighbor in (index - 1, index + 1)
        )
    if not _looks_numeric(raw):
        return False
    if any(marker in raw.lower() for marker in ("$", "€", "£", ",", ".", "%")):
        return True
    if raw.lower().endswith("x"):
        return True
    digits = re.sub(r"\D", "", raw)
    if digits and int(digits) >= 100:
        return True
    return any(
        0 <= neighbor < len(normalized)
        and normalized[neighbor] in MAGNITUDE_WORDS
        for neighbor in (index - 1, index + 1)
    )


def _looks_numeric(text: str) -> bool:
    raw = text.strip().strip("()[]{}\"'.,!?;:")
    return bool(_NUMERIC_TOKEN.fullmatch(raw))


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
        (first in INTERROGATIVE_WORDS and second in QUESTION_AUXILIARIES)
        or (
            first in QUESTION_AUXILIARIES
            and second in QUESTION_SUBJECT_STARTERS
        )
    )
