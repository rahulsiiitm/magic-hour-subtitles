# Phase 1: Local faster-whisper captions

Phase 1 keeps KillerSubtitles' existing karaoke/word/chunk layout, Pillow
renderer, and FFmpeg compositor. Transcription now runs locally with
faster-whisper using word-level timestamps and VAD.

## Defaults

- GPU: `distil-large-v3`, CUDA, float16
- CPU fallback for English: `small.en`, int8
- CPU fallback for other languages: `small`, int8

The first run downloads the selected Whisper model into the normal model cache.

## Local run

Python 3.9+ and a usable FFmpeg binary are required.

```bash
python -m pip install -r requirements.txt
python -m killer_subtitles input.mp4 -o output.mp4 --mode karaoke
```

For transcript correction, keep using `--transcript script.txt`.

## Google Colab

Select a GPU runtime, then run:

```bash
!git clone <REPOSITORY_URL> /content/magic-hour-subtitles
%cd /content/magic-hour-subtitles/KillerSubtitles-main
!python -m pip install -r requirements-colab.txt
!python -m killer_subtitles /content/input.mp4 -o /content/output.mp4
```

Upload `input.mp4` before the final command. If CUDA/CTranslate2 cannot start,
the transcriber warns and retries with the smaller CPU/int8 model.

## Tests

```bash
python -m unittest discover -s tests -v
```
