"""Frame assignment service — match a paper to the best ParadigmFrame.

Given a paper's category, tags, and content, determine which canonical
paradigm it should be evaluated against.

v3.1: Falls back to LLM-based dynamic discovery for unknown domains.
When no static mapping matches, uses the paper's abstract to identify
the domain, generate a new ParadigmTemplate with slots, and persist it.
"""

import json
import logging
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.analysis import ParadigmTemplate
from backend.models.graph import Slot

logger = logging.getLogger(__name__)

# Category → paradigm name mapping
CATEGORY_PARADIGM_MAP = {
    "Motion_Generation_Text_Speech_Music_Driven": "motion_generation_diffusion",
    "Human_Object_Interaction": "motion_generation_diffusion",
    "Human_Human_Interaction": "motion_generation_diffusion",
    "Human_Scene_Interaction": "motion_generation_diffusion",
    "Motion_Controlled_ImageVideo_Generation": "motion_generation_diffusion",
    "RL": "rl_standard",
    "Reinforcement_Learning": "rl_standard",
    "VLM": "vlm_standard",
    "Vision_Language_Model": "vlm_standard",
    "Multimodal_Interleaving_Reasoning": "vlm_standard",
    "Agent": "agent_standard",
    "Agentic_Reasoning": "agent_standard",
}

# Tag-based override (higher priority than category)
TAG_PARADIGM_MAP = {
    "reinforcement-learning": "rl_standard",
    "rl": "rl_standard",
    "ppo": "rl_standard",
    "vlm": "vlm_standard",
    "vision-language": "vlm_standard",
    "agent": "agent_standard",
    "tool-use": "agent_standard",
    "diffusion": "motion_generation_diffusion",
    "flow-matching": "motion_generation_diffusion",
    "motion-generation": "motion_generation_diffusion",
}

DISCOVER_SYSTEM = """You are a research domain analyst. Given a paper's title and abstract,
identify its domain and the canonical paradigm (standard method pipeline) for that domain.
Output JSON only."""

DISCOVER_PROMPT = """Analyze this paper and identify its research domain's canonical paradigm.

Title: {title}
Abstract: {abstract}
Category: {category}

Return a JSON object:
{{
  "domain": "short domain name (e.g., protein_folding, image_generation, nlp_reasoning)",
  "paradigm_name": "snake_case paradigm name (e.g., protein_folding_standard)",
  "paradigm_description": "One sentence describing the standard approach in this domain",
  "slots": [
    {{"name": "slot_name", "slot_type": "architecture|objective|data|inference", "description": "what this component does", "is_required": true}}
  ],
  "bottleneck": {{
    "title": "Main bottleneck this paper addresses",
    "description": "Why this is hard (2-3 sentences)"
  }}
}}

Rules:
- Identify 4-8 canonical slots that form the standard pipeline in this domain
- slot_type must be one of: architecture, objective, data, inference
- The paradigm should be the STANDARD approach, not what this specific paper proposes
- The bottleneck is what this paper is trying to solve

Return ONLY valid JSON."""


async def assign_paradigm(
    session: AsyncSession,
    category: str,
    tags: list[str] | None = None,
    title: str | None = None,
    abstract: str | None = None,
) -> tuple[ParadigmTemplate | None, list[dict]]:
    """Find the best matching ParadigmFrame for a paper.

    Priority: tags → category → existing DB paradigms by domain → LLM discovery.
    Returns (paradigm_template, slots_list) or (None, []) if all fail.
    """
    paradigm_name = None

    # 1. Check tags first (higher priority)
    if tags:
        for tag in tags:
            tag_lower = tag.lower().replace("_", "-")
            if tag_lower in TAG_PARADIGM_MAP:
                paradigm_name = TAG_PARADIGM_MAP[tag_lower]
                break

    # 2. Fall back to category
    if not paradigm_name:
        paradigm_name = CATEGORY_PARADIGM_MAP.get(category)

    # 3. If static mapping found, look up in DB
    if paradigm_name:
        paradigm, slots = await _fetch_paradigm(session, paradigm_name)
        if paradigm:
            return paradigm, slots

    # 4. Try fuzzy match on existing paradigms by category/domain
    paradigm, slots = await _fuzzy_match_paradigm(session, category)
    if paradigm:
        return paradigm, slots

    # 5. LLM-based dynamic discovery (fallback)
    if title or abstract:
        paradigm, slots = await _discover_paradigm_via_llm(
            session, title or "", abstract or "", category,
        )
        if paradigm:
            return paradigm, slots

    logger.info(f"No paradigm found for category={category}, tags={tags}")
    return None, []


async def _fetch_paradigm(
    session: AsyncSession,
    paradigm_name: str,
) -> tuple[ParadigmTemplate | None, list[dict]]:
    """Fetch a paradigm and its slots by name."""
    result = await session.execute(
        select(ParadigmTemplate).where(ParadigmTemplate.name == paradigm_name)
    )
    paradigm = result.scalar_one_or_none()
    if not paradigm:
        return None, []

    slots_result = await session.execute(
        text("SELECT id, name, description, slot_type, is_required FROM slots WHERE paradigm_id = :pid ORDER BY sort_order"),
        {"pid": paradigm.id},
    )
    slots = [dict(row._mapping) for row in slots_result.fetchall()]
    return paradigm, slots


async def _fuzzy_match_paradigm(
    session: AsyncSession,
    category: str,
) -> tuple[ParadigmTemplate | None, list[dict]]:
    """Try to match existing paradigms by domain similarity."""
    # Normalize category for matching
    cat_lower = category.lower().replace("_", " ").replace("-", " ")

    result = await session.execute(select(ParadigmTemplate))
    for paradigm in result.scalars():
        domain = (paradigm.domain or "").lower()
        name = paradigm.name.lower().replace("_", " ")
        if (cat_lower in domain or domain in cat_lower or
            cat_lower in name or name in cat_lower):
            _, slots = await _fetch_paradigm(session, paradigm.name)
            return paradigm, slots

    return None, []


async def _discover_paradigm_via_llm(
    session: AsyncSession,
    title: str,
    abstract: str,
    category: str,
) -> tuple[ParadigmTemplate | None, list[dict]]:
    """Use LLM to discover a new paradigm for an unknown domain.

    Creates ParadigmTemplate + Slots + optional Bottleneck in DB.
    """
    from backend.services.llm_service import call_llm

    prompt = DISCOVER_PROMPT.format(
        title=title[:500],
        abstract=(abstract or "No abstract available")[:2000],
        category=category,
    )

    try:
        resp = await call_llm(
            prompt=prompt,
            system=DISCOVER_SYSTEM,
            max_tokens=1024,
            session=session,
            prompt_version="paradigm_discover_v1",
        )

        data = _parse_json(resp.text)
        if not data or "paradigm_name" not in data:
            logger.warning(f"LLM paradigm discovery returned invalid data: {resp.text[:200]}")
            return None, []

        paradigm_name = data["paradigm_name"]
        domain = data.get("domain", category)

        # Check if this paradigm was already discovered
        existing, existing_slots = await _fetch_paradigm(session, paradigm_name)
        if existing:
            return existing, existing_slots

        # Create new paradigm
        raw_slots = data.get("slots", [])
        slots_dict = {s["name"]: s.get("description", "") for s in raw_slots if isinstance(s, dict)}

        paradigm = ParadigmTemplate(
            name=paradigm_name,
            version="v1",
            domain=domain,
            slots=slots_dict,
        )
        session.add(paradigm)
        await session.flush()

        # Create slot records
        slot_records = []
        for i, s in enumerate(raw_slots):
            if not isinstance(s, dict) or not s.get("name"):
                continue
            slot = Slot(
                paradigm_id=paradigm.id,
                name=s["name"],
                description=s.get("description"),
                slot_type=s.get("slot_type", "architecture"),
                is_required=s.get("is_required", True),
                sort_order=i,
            )
            session.add(slot)
            slot_records.append(slot)

        await session.flush()

        # Create bottleneck if provided
        bn_data = data.get("bottleneck")
        if bn_data and isinstance(bn_data, dict) and bn_data.get("title"):
            from backend.models.research import ProjectBottleneck
            bottleneck = ProjectBottleneck(
                title=bn_data["title"],
                description=bn_data.get("description"),
                domain=domain,
                paradigm_id=paradigm.id,
                status="active",
            )
            session.add(bottleneck)
            await session.flush()

        # Add category mapping for future use
        CATEGORY_PARADIGM_MAP[category] = paradigm_name

        logger.info(f"Discovered new paradigm: {paradigm_name} with {len(slot_records)} slots for domain={domain}")

        slots = [
            {"id": s.id, "name": s.name, "description": s.description,
             "slot_type": s.slot_type, "is_required": s.is_required}
            for s in slot_records
        ]
        return paradigm, slots

    except Exception as e:
        logger.error(f"LLM paradigm discovery failed: {e}")
        return None, []


def _parse_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None
