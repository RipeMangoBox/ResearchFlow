# Agent Guide

> Architecture & data model: [ARCHITECTURE.md](researchflow-backend/ARCHITECTURE.md)
> Deployment: [DEPLOY.md](researchflow-backend/DEPLOY.md)

## Source of truth

**PostgreSQL is the only write target.** `paperAnalysis/`, `paperCollection/`, `obsidian-vault/` are read-only exports. For queries, use `/api/v1/search/*`, not local files.

## Connect Claude Code via MCP

ResearchFlow auto-discovers `.mcp.json` in the project root:

```json
{
  "mcpServers": {
    "researchflow-remote": {
      "url": "https://researchflow.xyz/sse"
    },
    "researchflow-local": {
      "command": "python",
      "args": ["-m", "backend.mcp.server"],
      "cwd": "researchflow-backend",
      "env": {"PYTHONPATH": "."}
    }
  }
}
```

## Skill routing

Full skill list and selection guide: [`.claude/skills/README.md`](.claude/skills/README.md)

## Rules

1. All writes go to backend API, never edit Markdown files as source
2. For queries, prefer `/api/v1/search/*` over reading local files
3. Analysis language default: `zh` (override per request)
4. Keep ResearchFlow as active workspace; link external repos under `linkedCodebases/`
5. Obsidian vault is auto-generated — do not edit files in `obsidian-vault/` directly
6. Pipeline steps are idempotent — already-completed steps are auto-skipped
7. Metadata observations are append-only — canonical resolver picks best value
