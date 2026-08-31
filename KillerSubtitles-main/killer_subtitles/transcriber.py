"""Whisper API integration for word-level transcription.

Handles OpenAI Whisper transcription with word-level timestamps,
optional pronunciation prompts, and automatic audio chunking for
files exceeding the 25 MB API limit.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

from openai import OpenAI

from .models import Word


MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB Whisper limit
TARGET_CHUNK_SIZE_BYTES = 24 * 1024 * 1024  # leave room for encoding overhead
PROMPT_TOKEN_CHAR_APPROX = 500  # ~224 tokens ≈ 500 chars for English


def transcribe(
    audio_path: str | Path,
    *,
    api_key: str | None = None,
    language: str = "en",
    prompt: str | None = None,
    transcript_path: str | Path | None = None,
) -> list[Word]:
    """Transcribe audio and return word-level timestamps.

    Parameters
    ----------
    audio_path:
        Path to an audio file (MP3, WAV, etc.).
    api_key:
        OpenAI API key.  Falls back to ``OPENAI_API_KEY`` env var.
    language:
        ISO-639-1 language code.
    prompt:
        Short pronunciation guide or key terms for Whisper.
    transcript_path:
        Optional path to a text file whose first ~500 chars are used as
        the Whisper prompt (ignored if *prompt* is also provided).
    """
    audio_path = Path(audio_path)
    client = _build_client(api_key)

    effective_prompt = _resolve_prompt(prompt, transcript_path)

    file_size = audio_path.stat().st_size
    if file_size <= MAX_FILE_SIZE_BYTES:
        return _transcribe_single(client, audio_path, language, effective_prompt)

    return _transcribe_chunked(client, audio_path, language, effective_prompt)


# ------------------------------------------------------------------
# Single-file transcription
# ------------------------------------------------------------------

def _transcribe_single(
    client: OpenAI,
    audio_path: Path,
    language: str,
    prompt: str | None,
) -> list[Word]:
    kwargs: dict = {
        "model": "whisper-1",
        "response_format": "verbose_json",
        "timestamp_granularities": ["word"],
        "language": language,
    }
    if prompt:
        kwargs["prompt"] = prompt

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(file=f, **kwargs)

    return [
        Word(text=w.word.strip(), start=w.start, end=w.end)
        for w in (response.words or [])
    ]


# ------------------------------------------------------------------
# Chunked transcription for files > 25 MB
# ------------------------------------------------------------------

def _transcribe_chunked(
    client: OpenAI,
    audio_path: Path,
    language: str,
    prompt: str | None,
) -> list[Word]:
    """Split audio into <=25 MB chunks and transcribe sequentially.

    Each chunk uses the previous chunk's transcript as a prompt to
    maintain continuity (Whisper's intended use for long audio).
    """
    from . import ffmpeg as ff

    file_size = audio_path.stat().st_size
    num_chunks = math.ceil(file_size / TARGET_CHUNK_SIZE_BYTES)

    try:
        total_duration = ff.get_media_duration(audio_path)
    except ValueError as exc:
        raise RuntimeError(
            "Could not determine audio duration for chunked transcription."
        ) from exc

    chunk_duration = total_duration / num_chunks
    all_words: list[Word] = []
    running_prompt = prompt

    with tempfile.TemporaryDirectory(prefix="killersubs_") as tmp:
        for i in range(num_chunks):
            start_time = i * chunk_duration
            chunk_path = Path(tmp) / f"chunk_{i:03d}.mp3"
            ff.run_ffmpeg(
                "-y",
                "-i", str(audio_path),
                "-ss", f"{start_time:.3f}",
                "-t", f"{chunk_duration:.3f}",
                "-acodec", "mp3",
                "-ar", "16000",
                "-ac", "1",
                str(chunk_path),
            )

            chunk_size = chunk_path.stat().st_size
            if chunk_size > MAX_FILE_SIZE_BYTES:
                raise RuntimeError(
                    f"Encoded audio chunk {i + 1} is {chunk_size / 1024 / 1024:.1f} MB, "
                    "which exceeds Whisper's 25 MB upload limit."
                )

            words = _transcribe_single(
                client, chunk_path, language, running_prompt
            )

            # Offset timestamps by chunk start
            for w in words:
                w.start += start_time
                w.end += start_time

            all_words.extend(words)

            # Use tail of this chunk's transcript as prompt for next
            if words:
                tail_text = " ".join(w.text for w in words[-30:])
                running_prompt = tail_text[-PROMPT_TOKEN_CHAR_APPROX:]

    return all_words


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_client(api_key: str | None) -> OpenAI:
    if api_key:
        return OpenAI(api_key=api_key)
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return OpenAI(api_key=env_key)
    raise RuntimeError(
        "No OpenAI API key found. Provide one via:\n"
        "  --api-key YOUR_KEY\n"
        "  OPENAI_API_KEY environment variable\n"
        "  .env file with OPENAI_API_KEY=YOUR_KEY"
    )


def _resolve_prompt(
    prompt: str | None,
    transcript_path: str | Path | None,
) -> str | None:
    if prompt:
        return prompt[:PROMPT_TOKEN_CHAR_APPROX]
    if transcript_path:
        path = Path(transcript_path)
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            return text[:PROMPT_TOKEN_CHAR_APPROX] if text else None
    return None
