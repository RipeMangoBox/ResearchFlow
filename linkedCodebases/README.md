# Linked Codebases

Place local symlinks or junctions to external code repositories in this directory.

Recommended automated setup:

- macOS / Linux: `python3 scripts/link_codebase.py /path/to/your-codebase`
- Windows: `py -3 scripts\\link_codebase.py C:\\path\\to\\your-codebase`

If you want to override the folder name inside `linkedCodebases/`, add `--name your-codebase`.

Recommended pattern:

- macOS / Linux: `ln -s /path/to/your-codebase linkedCodebases/your-codebase`
- Windows PowerShell: `New-Item -ItemType Junction -Path .\\linkedCodebases\\your-codebase -Target C:\\path\\to\\your-codebase`

Keep the actual code repositories outside version control. This directory only serves as a local bridge so ResearchFlow can stay the active workspace while still seeing the target codebase.
