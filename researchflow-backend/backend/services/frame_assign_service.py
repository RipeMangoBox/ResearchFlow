"""Frame assignment service — match a paper to the best ParadigmFrame.

Given a paper's category, tags, and content, determine which canonical
paradigm it should be evaluated against.
"""

import logging
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.analysis import ParadigmTemplate

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


async def assign_paradigm(
    session: AsyncSession,
    category: str,
    tags: list[str] | None = None,
) -> tuple[ParadigmTemplate | None, list[dict]]:
    """Find the best matching ParadigmFrame for a paper.

    Returns (paradigm_template, slots_list) or (None, []) if no match.
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

    if not paradigm_name:
        logger.info(f"No paradigm match for category={category}, tags={tags}")
        return None, []

    # 3. Fetch from DB
    result = await session.execute(
        select(ParadigmTemplate).where(ParadigmTemplate.name == paradigm_name)
    )
    paradigm = result.scalar_one_or_none()
    if not paradigm:
        return None, []

    # 4. Fetch slots
    slots_result = await session.execute(
        text("SELECT id, name, description, slot_type, is_required FROM slots WHERE paradigm_id = :pid ORDER BY sort_order"),
        {"pid": paradigm.id},
    )
    slots = [dict(row._mapping) for row in slots_result.fetchall()]

    return paradigm, slots
