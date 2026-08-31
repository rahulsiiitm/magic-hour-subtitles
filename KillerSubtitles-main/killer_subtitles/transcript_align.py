"""Post-processing alignment of Whisper output against a known script.

Corrects misheard words by aligning the Whisper transcript to the
ground-truth script using sequence matching, then replacing Whisper's
text with the script's text while preserving Whisper's timing.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from .models import Word


def align_to_script(
    whisper_words: list[Word],
    script_path: str | Path,
) -> list[Word]:
    """Align Whisper words to a ground-truth script and correct text.

    Returns a new list of Word objects with corrected text and original
    Whisper timing.  Hallucinated words (in Whisper but not in script)
    are dropped.  Words in the script that Whisper missed entirely are
    skipped (we have no timing for them).
    """
    script_text = Path(script_path).read_text(encoding="utf-8", errors="ignore")
    script_tokens = _tokenize(script_text)

    if not script_tokens or not whisper_words:
        return whisper_words

    # Normalised forms for matching
    w_norm = [_normalise(w.text) for w in whisper_words]
    s_norm = [_normalise(t) for t in script_tokens]

    matcher = SequenceMatcher(None, w_norm, s_norm, autojunk=False)
    opcodes = matcher.get_opcodes()

    result: list[Word] = []

    for tag, w_start, w_end, s_start, s_end in opcodes:
        w_slice = whisper_words[w_start:w_end]
        s_slice = script_tokens[s_start:s_end]

        if tag == "equal":
            # Words match -- keep Whisper timing, use script spelling
            for w_word, s_text in zip(w_slice, s_slice):
                result.append(Word(text=s_text, start=w_word.start, end=w_word.end))

        elif tag == "replace":
            # Whisper misheard -- redistribute timing across script words
            _merge_replace(result, w_slice, s_slice)

        elif tag == "delete":
            # Whisper hallucinated words not in script -- drop them
            pass

        elif tag == "insert":
            # Script has words Whisper missed -- no timing available, skip
            pass

    result = _smooth_timing(result)
    result = _trim_trailing_outliers(result)

    return result


def _smooth_timing(
    words: list[Word],
    max_word_gap: float = 1.5,
) -> list[Word]:
    """Repair invalid durations without collapsing legitimate speech pauses."""
    del max_word_gap  # retained for compatibility with older callers
    smoothed: list[Word] = []
    for word in words:
        end = word.end if word.end > word.start else word.start + 0.2
        smoothed.append(Word(text=word.text, start=word.start, end=end))
    return smoothed


def _trim_trailing_outliers(
    words: list[Word],
    gap_threshold: float = 4.0,
) -> list[Word]:
    """Drop timestamp reversals while preserving legitimate long pauses."""
    del gap_threshold  # retained for compatibility with older callers
    if len(words) < 2:
        return words

    # Detect timestamp reversals (later script word got an earlier timestamp)
    clean: list[Word] = [words[0]]
    for w in words[1:]:
        if w.start < clean[-1].start:
            # Timestamp went backwards -- this word is misaligned, skip it
            continue
        clean.append(w)

    return clean


def _merge_replace(
    result: list[Word],
    w_slice: list[Word],
    s_slice: list[str],
) -> None:
    """Handle a replace block by distributing Whisper timing over script words.

    If counts match (most common: 1-for-1 swap), it's a direct replacement.
    If counts differ, the total time span is divided evenly.
    """
    if not w_slice or not s_slice:
        return

    if len(w_slice) == len(s_slice):
        for w_word, s_text in zip(w_slice, s_slice):
            result.append(Word(text=s_text, start=w_word.start, end=w_word.end))
    else:
        # Distribute the full time span of the Whisper block evenly
        block_start = w_slice[0].start
        block_end = w_slice[-1].end
        duration = block_end - block_start
        n = len(s_slice)
        word_dur = duration / n

        for i, s_text in enumerate(s_slice):
            start = block_start + i * word_dur
            end = start + word_dur
            result.append(Word(text=s_text, start=round(start, 3), end=round(end, 3)))


def _tokenize(text: str) -> list[str]:
    """Split script text into words, preserving punctuation attached to words."""
    # Collapse whitespace and split
    words = text.split()
    # Filter out empty tokens and standalone punctuation
    return [w for w in words if any(c.isalnum() for c in w)]


def _normalise(text: str) -> str:
    """Normalise a word for comparison: lowercase, strip punctuation."""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\w]", "", text)
    return text
