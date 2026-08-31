"""Entry point for ``python -m killer_subtitles`` and PyInstaller bundles."""

try:
    from .cli import main
except ImportError:
    from killer_subtitles.cli import main

main()
