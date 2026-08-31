"""Local faster-whisper integration with word-level timestamps."""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

from .models import Word


DEFAULT_GPU_MODEL = "distil-large-v3"
DEFAULT_CPU_MODEL_ENGLISH = "small.en"
DEFAULT_CPU_MODEL_MULTILINGUAL = "small"
PROMPT_CHAR_LIMIT = 500


def transcribe(
    audio_path: str | Path,
    *,
    language: str = "en",
    prompt: str | None = None,
    model_size: str = DEFAULT_GPU_MODEL,
    device: str = "cuda",
    compute_type: str = "float16",
    cpu_model_size: str | None = None,
) -> list[Word]:
    """Transcribe audio locally, falling back to CPU/int8 when CUDA fails."""
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if device.lower() == "cuda" and not _cuda_available():
        fallback_model = cpu_model_size or _default_cpu_model(language)
        warnings.warn(
            "CUDA is not available to CTranslate2. "
            f"Using {fallback_model!r} on CPU with int8.",
            RuntimeWarning,
            stacklevel=2,
        )
        model = _load_model(fallback_model, "cpu", "int8")
        return _transcribe_with_model(model, audio_path, language, prompt)

    try:
        model = _load_model(model_size, device, compute_type)
        return _transcribe_with_model(model, audio_path, language, prompt)
    except Exception as gpu_exc:
        if device.lower() == "cpu":
            raise

        fallback_model = cpu_model_size or _default_cpu_model(language)
        warnings.warn(
            f"faster-whisper CUDA transcription failed ({gpu_exc}). "
            f"Falling back to {fallback_model!r} on CPU with int8.",
            RuntimeWarning,
            stacklevel=2,
        )

        try:
            model = _load_model(fallback_model, "cpu", "int8")
            return _transcribe_with_model(model, audio_path, language, prompt)
        except Exception as cpu_exc:
            raise RuntimeError(
                "faster-whisper failed on both CUDA and CPU. "
                f"CUDA error: {gpu_exc}; CPU error: {cpu_exc}"
            ) from cpu_exc


@lru_cache(maxsize=4)
def _load_model(model_size: str, device: str, compute_type: str) -> Any:
    """Load and cache a faster-whisper model without importing it at module load."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Run: pip install -r requirements.txt"
        ) from exc

    return WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )


def _cuda_available() -> bool:
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except (ImportError, RuntimeError):
        return False


def _transcribe_with_model(
    model: Any,
    audio_path: Path,
    language: str,
    prompt: str | None,
) -> list[Word]:
    """Run faster-whisper and convert its lazy segment stream to Word models."""
    kwargs: dict[str, Any] = {
        "language": language,
        "word_timestamps": True,
        "vad_filter": True,
        "condition_on_previous_text": False,
        "beam_size": 5,
    }
    if prompt:
        kwargs["initial_prompt"] = prompt[:PROMPT_CHAR_LIMIT]

    segments, _info = model.transcribe(str(audio_path), **kwargs)

    words: list[Word] = []
    for segment in segments:
        for word in segment.words or []:
            text = word.word.strip()
            if not text or word.start is None or word.end is None:
                continue
            words.append(
                Word(text=text, start=float(word.start), end=float(word.end))
            )
    return words


def _default_cpu_model(language: str) -> str:
    return (
        DEFAULT_CPU_MODEL_ENGLISH
        if language.lower().startswith("en")
        else DEFAULT_CPU_MODEL_MULTILINGUAL
    )
