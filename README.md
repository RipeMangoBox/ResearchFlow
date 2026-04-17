<p align="center">
  <img src="./assets/LOGO.png" alt="ResearchFlow logo" width="280"/>
</p>

<h1 align="center">ResearchFlow</h1>

<p align="center"><strong>Give one paper → auto-build domain knowledge graph → track method evolution → smart exploration.</strong></p>

<p align="center">
  <a href="README.md">English</a> | <a href="README_CN.md">中文</a>
</p>

---

## What it does

```
You:  "Here's a paper about RLHF for VLM"
         ↓
Step 1:  Find the awesome-list for this domain → import 72 papers → score & prioritize
Step 2:  Download PDFs → analyze (L2 parse → L3 skim → L4 deep) → extract DeltaCards
Step 3:  Build method evolution DAG: GRPO → GRPO+LP → GDPO → GDPO+image_thinking
Step 4:  Classify: 3 structural changes, 5 plugins, 2 reward designs
Step 5:  You explore iteratively, system tracks pivots and suggests directions
```

**Core idea**: Methods build on methods in a DAG, not a flat list. The system tracks which improvements became new baselines, which are just plugins, and how paradigms evolve.

---

## Quick start

```bash
git clone https://github.com/RipeMangoBox/ResearchFlow.git
cd ResearchFlow/researchflow-backend
cp .env.example .env              # Set ANTHROPIC_API_KEY
make db && make migrate && make up
```

Then:
```bash
# Initialize a domain from an awesome-list
curl -X POST localhost:8000/api/v1/pipeline/init-domain \
  -H "Content-Type: application/json" \
  -d '{"domain": "RLHF VLM"}'

# Or give a single paper and build from there
curl -X POST localhost:8000/api/v1/import/links \
  -H "Content-Type: application/json" \
  -d '{"items": [{"url": "https://arxiv.org/abs/2402.03300"}]}'

# Run full pipeline (download → analyze → graph)
curl -X POST localhost:8000/api/v1/pipeline/{paper_id}/run

# Discover related papers via Semantic Scholar
curl -X POST localhost:8000/api/v1/pipeline/{paper_id}/discover
```

Web UI: `http://localhost:3000` | Claude Code: auto-discovers `.mcp.json`

---

## How the knowledge graph works

```
Paper → DeltaCard → IdeaDelta → GraphAssertions
          │
          ├── parent_delta_card_ids  (DAG: which methods this builds on)
          ├── method_category        (structural / plugin / reward / ...)
          ├── improvement_type       (fundamental_rethink / additive_plugin / ...)
          └── bottleneck_addressed   (auto-extracted research bottleneck)
```

A DeltaCard becomes an **established baseline** when 3+ papers build on it. When it's also structural, it can be **promoted to a new paradigm version**.

---

## System scale

| Component | Count |
|-----------|-------|
| DB tables | 31 |
| API routes | 81 |
| MCP tools | 18 |
| Services | 25 |
| Tests | 29 |

## Documentation

| Doc | What it covers |
|-----|---------------|
| [Architecture](researchflow-backend/ARCHITECTURE.md) | Knowledge graph structure, method evolution DAG, filtering, full pipeline |
| [Backend README](researchflow-backend/README.md) | Setup, features, API overview |
| [Deploy Guide](researchflow-backend/DEPLOY_GUIDE.md) | Cloud deployment |
| [Agent Guide](AGENTS.md) | MCP tools, skills, routing |

## License

MIT
