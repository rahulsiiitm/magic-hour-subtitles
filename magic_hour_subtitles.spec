# PyInstaller spec for Magic Hour Dynamic Subtitles
#
# Bundles the CLI, font files, and the imageio-ffmpeg binary into a
# single standalone executable.
#
# Usage:
#   pip install pyinstaller
#   pyinstaller magic_hour_subtitles.spec
#
# Output lands in dist/magic-hour-subtitles (or dist/magic-hour-subtitles.exe on Windows)

import os
import sys
from pathlib import Path

import imageio_ffmpeg

block_cipher = None

# Locate the bundled FFmpeg binary from imageio-ffmpeg
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_binaries_dir = str(Path(ffmpeg_exe).parent)

# Locate our bundled font
font_dir = os.path.join("magic_hour_subtitles", "fonts")

a = Analysis(
    ["magic_hour_subtitles/__main__.py"],
    pathex=[],
    binaries=[],
    datas=[
        # Bundle our font files
        (font_dir, "magic_hour_subtitles/fonts"),
        # Bundle the imageio-ffmpeg binary
        (ffmpeg_binaries_dir, "imageio_ffmpeg/binaries"),
    ],
    hiddenimports=[
        "magic_hour_subtitles",
        "magic_hour_subtitles.caption_analysis",
        "magic_hour_subtitles.caption_chunker",
        "magic_hour_subtitles.cli",
        "magic_hour_subtitles.compositor",
        "magic_hour_subtitles.display_text",
        "magic_hour_subtitles.ffmpeg",
        "magic_hour_subtitles.layout",
        "magic_hour_subtitles.models",
        "magic_hour_subtitles.occlusion",
        "magic_hour_subtitles.pipeline",
        "magic_hour_subtitles.placement",
        "magic_hour_subtitles.presets",
        "magic_hour_subtitles.renderer",
        "magic_hour_subtitles.transcriber",
        "magic_hour_subtitles.transcript_align",
        "magic_hour_subtitles.vision",
        "imageio_ffmpeg",
        "imageio_ffmpeg.binaries",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="magic-hour-subtitles",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
