# Generated Agent Compatibility Directory

`.claude/skills` is the single source of truth for the shared skill library.

Run the cross-platform setup script from the repository root:

- macOS / Linux: `python3 scripts/setup_shared_skills.py`
- Windows: `py -3 scripts\\setup_shared_skills.py`

The script creates `skills/` and `skills-config.json` aliases here so Codex-compatible tooling can reuse the same skill library without copying files.
