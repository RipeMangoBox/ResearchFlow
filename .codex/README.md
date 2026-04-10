# Generated Codex Compatibility Directory

`.claude/skills` is the single source of truth for the shared skill library.

This repository intentionally tracks only this README inside `.codex/`.
After cloning, generate the local Codex compatibility aliases from the repository root:

- macOS / Linux: `python3 scripts/setup_shared_skills.py`
- Windows: `py -3 scripts\\setup_shared_skills.py`

The setup script will create:

- `skills/`
- `skills-config.json`

Do not rely on committed `.codex/skills` aliases. Treat `.codex` as local setup state for the current platform.
