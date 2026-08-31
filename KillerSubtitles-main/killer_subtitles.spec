# PyInstaller spec for KillerSubtitles
#
# Bundles the CLI, font files, and the imageio-ffmpeg binary into a
# single standalone executable.
#
# Usage:
#   pip install pyinstaller
#   pyinstaller killer_subtitles.spec
#
# Output lands in dist/killer-subtitles (or dist/killer-subtitles.exe on Windows)

import os
import sys
from pathlib import Path

import imageio_ffmpeg

block_cipher = None

# Locate the bundled FFmpeg binary from imageio-ffmpeg
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_binaries_dir = str(Path(ffmpeg_exe).parent)

# Locate our bundled font
font_dir = os.path.join("killer_subtitles", "fonts")

a = Analysis(
    ["killer_subtitles/__main__.py"],
    pathex=[],
    binaries=[],
    datas=[
        # Bundle our font files
        (font_dir, "killer_subtitles/fonts"),
        # Bundle the imageio-ffmpeg binary
        (ffmpeg_binaries_dir, "imageio_ffmpeg/binaries"),
    ],
    hiddenimports=[
        "killer_subtitles",
        "killer_subtitles.cli",
        "killer_subtitles.compositor",
        "killer_subtitles.ffmpeg",
        "killer_subtitles.layout",
        "killer_subtitles.models",
        "killer_subtitles.presets",
        "killer_subtitles.renderer",
        "killer_subtitles.transcriber",
        "killer_subtitles.transcript_align",
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
    name="killer-subtitles",
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
