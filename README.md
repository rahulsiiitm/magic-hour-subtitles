# Magic Hour Dynamic Subtitles

Content-aware dynamic subtitle generation with semantic styling, scene-aware placement, and foreground-aware compositing.

This project generates subtitles that respond to both the spoken content and the visual composition of a video. It combines local speech transcription, semantic caption chunking, deterministic tone-aware styling, expression-aware word emphasis, scene-aware placement, person/object segmentation, conservative head-region protection, and optional behind-subject compositing. The complete pipeline runs locally without a paid API or API key.

## Try it in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rahulsiiitm/magic-hour-subtitles/blob/main/notebooks/Magic_Hour_Dynamic_Subtitles_Demo.ipynb)

The [reviewer notebook](notebooks/Magic_Hour_Dynamic_Subtitles_Demo.ipynb) provides the shortest path through the full system:

1. Run setup.
2. Upload a video.
3. Configure the high-level options.
4. Generate subtitles.
5. Preview the result.
6. Download the rendered video.

No API key is required. A GPU is recommended, but CPU fallback is supported.

## Key Features

### Local transcription

Audio is transcribed locally with `faster-whisper` and word-level timestamps. CUDA acceleration is used when available; if CUDA transcription is unavailable, the pipeline falls back to a smaller CPU model with integer quantization.

### Semantic caption planning

Speech is grouped into compact, readable caption units while preserving word order and timing. Lightweight deterministic semantic analysis classifies each caption as `neutral`, `excited`, `serious`, or `question`; it does not infer emotion from a speaker's voice.

### Expression-aware emphasis

Selected words can receive local emphasis as tone triggers, numeric magnitudes, contrast/reveal cues, question cues, or semantic keywords. Color and bounded scale changes create hierarchy while reserved layout geometry keeps line breaks stable and avoids reflow or jitter during active-word animation.

### Scene-aware placement

Portrait captions normally remain at a stable bottom-center baseline. Person or large foreground-object obstruction can trigger relocation to a safer candidate, while conservatively estimated upper-person/head regions receive stronger protection. Captions do not move toward subjects merely to manufacture an effect, and they return toward the baseline when an obstruction disappears.

### Behind-subject compositing

Placement is selected first. If the chosen caption region naturally intersects a safe part of an existing person or object mask, that foreground may be composited back over the text to create a text-behind-person/object effect. This behavior is opportunistic: it never requires moving a caption onto a person, and readability or head safety can reject it.

## Design Philosophy

```text
PLACEMENT FIRST
    -> SUBJECT / HEAD SAFETY
    -> OCCLUSION AT THE CHOSEN PLACEMENT
    -> RENDER
```

Placement and visual occlusion are deliberately separate decisions. Occlusion is not a placement objective, which prevents captions from "chasing" people across the frame simply to create a cinematic interaction.

## Architecture

```text
Video
├── Audio
│   └── faster-whisper
│       └── word timestamps
│           └── semantic caption planning
│               └── tone + expressions
│
└── Frames
    └── YOLO segmentation
        + clutter analysis
        + motion analysis
            └── placement planner
                └── optional foreground compositing
                    └── renderer
                        └── output video
```

- **Transcription:** `faster-whisper` produces timestamped words locally. An optional supplied transcript can correct text through alignment without replacing speech timing.
- **Caption intelligence:** words are chunked into readable phrases, assigned a deterministic tone, and annotated with a small set of expression types.
- **Visual analysis:** sampled frames provide segmentation, edge-density clutter, motion, and scene-cut signals without retaining full-resolution model tensors.
- **Placement:** candidate regions are evaluated for person, protected-head, foreground, clutter, motion, safe-zone, and stability constraints. A persistent baseline prevents caption-by-caption movement.
- **Compositing and rendering:** Pillow renders stable caption states, foreground masks are optionally layered over eligible text, and FFmpeg assembles the result with the original video and audio.

## Vision Pipeline

One Ultralytics YOLO segmentation pass produces both the person map and a conservative foreground map. Selected foreground classes are `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`, and `chair`; small non-person detections are ignored, and person-specific safety remains stronger than generic object handling.

Protected upper-person/head regions are approximated from the existing person segmentation map. This requires no additional face model or segmentation pass and should be understood as conservative visual protection, not precise face recognition.

## Content-Aware Styling and Typography

Each caption combines a tone style, active-word highlighting, selected semantic expressions, bounded scale changes, and restrained colors. The layout engine reserves the required emphasized-word space before rendering, so animation does not change line breaks or push neighboring words around.

Montserrat ExtraBold is bundled with the package. Typography scales with video dimensions; portrait video uses tighter sizing and line-count constraints, while measured wrapping aims for balanced, readable multi-line captions.

## Example Placement Behavior

```text
No obstruction                         -> stable bottom-center placement
Person blocks the normal area          -> relocate to a safer candidate
Foreground no longer obstructs         -> return toward the baseline
Safe overlap at the chosen placement   -> optional behind-subject compositing
Protected head region at risk          -> reject occlusion / prefer safety
```

## Local Installation

Python 3.9 or newer is required.

```bash
git clone https://github.com/rahulsiiitm/magic-hour-subtitles.git
cd magic-hour-subtitles
python -m pip install -e .
```

The editable install resolves the dependencies declared in `pyproject.toml` and installs the `magic-hour-subtitles` command. FFmpeg binaries and the default font are resolved through installed Python packages and bundled package data.

## Quick Start

Run the complete content-aware pipeline through the module:

```bash
python -m magic_hour_subtitles input.mp4 \
    -o output.mp4 \
    --dynamic-captions \
    --smart-placement \
    --behind-subject
```

Or use the installed command:

```bash
magic-hour-subtitles input.mp4 \
    -o output.mp4 \
    --dynamic-captions \
    --smart-placement \
    --behind-subject
```

Add `--caption-diagnostics` to inspect caption tone, selected expressions, placement, person/foreground overlap, protected-head status, and the behind-subject decision. Run `magic-hour-subtitles --help` for the complete option list.

Legacy `karaoke`, `word`, and `chunk` display modes remain available through `--mode`. Platform presets, transcript alignment, SRT export, and style overrides also remain available, but the three feature flags above define the primary take-home pipeline.

## Performance Notes

GPU execution is recommended for faster transcription and visual analysis. CPU fallback is fully supported but can be significantly slower. On first use, the runtime may download the selected `faster-whisper` model and YOLO segmentation weights; processing time varies with video length, resolution, hardware, and model-cache state.

## Limitations

- Transcription quality depends on speech clarity; singing and music-heavy audio can produce weaker results.
- Tone analysis is deterministic and does not infer a speaker's actual emotion.
- Foreground interactions depend on YOLO segmentation quality.
- Head protection is a conservative upper-person approximation, not face recognition.
- Behind-subject compositing is opportunistic and will not activate in every scene.
- Fast or complex motion can reduce interpolated-mask quality.
- CPU processing can be slow, especially on longer or higher-resolution videos.

## Tech Stack

- Python
- `faster-whisper`
- Ultralytics YOLO segmentation
- OpenCV
- Pillow
- FFmpeg
- NumPy

## Testing

The current suite contains 128 passing tests covering transcription integration, caption planning, expression styling, responsive layout, placement stability and safety, foreground occlusion, rendering fallbacks, and pipeline behavior.

```bash
pytest -q
```

## Project Structure

```text
magic-hour-subtitles/
├── magic_hour_subtitles/
├── notebooks/
│   └── Magic_Hour_Dynamic_Subtitles_Demo.ipynb
├── tests/
├── assets/
├── pyproject.toml
├── requirements.txt
├── requirements-colab.txt
└── README.md
```

## License

MIT License. See [LICENSE](LICENSE).
