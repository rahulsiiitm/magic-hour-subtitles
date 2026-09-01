"""Display-only cleanup for word-token spacing.

The formatter deliberately returns one display token per input token so word
timings and caption coverage remain untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


_NO_SPACE_BEFORE = re.compile(r"^(?:[,.;:!?%\]\)}]|['’](?:s|re|ve|ll|d|m|t)\b)", re.IGNORECASE)
_OPENING_END = re.compile(r"[(\[{]$")


@dataclass(frozen=True)
class DisplayToken:
    text: str
    space_before: bool


def format_display_tokens(
    tokens: list[str],
    *,
    uppercase: bool = False,
) -> list[DisplayToken]:
    """Normalize token whitespace while preserving the token count."""
    formatted: list[DisplayToken] = []
    previous_text = ""
    for index, raw in enumerate(tokens):
        text = _clean_token_text(raw)
        if uppercase:
            text = text.upper()
        space_before = index > 0
        if _NO_SPACE_BEFORE.search(text) and formatted:
            previous_index = _last_nonempty_index(formatted)
            previous = formatted[previous_index]
            formatted[previous_index] = DisplayToken(
                text=previous.text + text,
                space_before=previous.space_before,
            )
            formatted.append(DisplayToken(text="", space_before=False))
        elif _OPENING_END.search(previous_text) and formatted:
            previous = formatted[-1]
            formatted[-1] = DisplayToken(text="", space_before=previous.space_before)
            formatted.append(DisplayToken(
                text=previous.text + text,
                space_before=previous.space_before,
            ))
        elif (
            previous_text == ","
            and text[:1].isdigit()
            and _last_nonempty_text(formatted).rstrip(",").isdigit()
        ):
            previous_index = _last_nonempty_index(formatted)
            previous = formatted[previous_index]
            formatted[previous_index] = DisplayToken(
                text=previous.text + text,
                space_before=previous.space_before,
            )
            formatted.append(DisplayToken(text="", space_before=False))
        else:
            formatted.append(DisplayToken(text=text, space_before=space_before))
        previous_text = text
    return formatted


def _clean_token_text(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    return text


def _last_nonempty_index(tokens: list[DisplayToken]) -> int:
    for index in range(len(tokens) - 1, -1, -1):
        if tokens[index].text:
            return index
    return max(0, len(tokens) - 1)


def _last_nonempty_text(tokens: list[DisplayToken]) -> str:
    return tokens[_last_nonempty_index(tokens)].text if tokens else ""


def format_display_text(tokens: list[str], *, uppercase: bool = False) -> str:
    """Return cleaned display text without changing the source tokens."""
    parts: list[str] = []
    for token in format_display_tokens(tokens, uppercase=uppercase):
        if token.space_before and parts:
            parts.append(" ")
        parts.append(token.text)
    return "".join(parts)
