"""Entry point for ``python -m magic_hour_subtitles`` and PyInstaller bundles."""

try:
    from .cli import main
except ImportError:
    from magic_hour_subtitles.cli import main

main()
