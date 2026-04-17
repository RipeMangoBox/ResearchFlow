"""Seed ParadigmFrames + Slots + MechanismFamilies for 3 domains.

Domains: RL, VLM, Agent/Agentic Reasoning
Also seeds common mechanism families.

Usage: python -m migration.seed_paradigm_frames [--db-url URL]
"""

import argparse
import sys
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.config import settings

# ── Domain definitions ──────────────────────────────────────────

PARADIGM_FRAMES = [
    {
        "name": "rl_standard",
        "domain": "RL",
        "slots": [
            {"name": "rollout", "type": "data", "desc": "Environment interaction / trajectory collection", "required": True},
            {"name": "reward", "type": "objective", "desc": "Reward signal design and shaping", "required": True},
            {"name": "credit_assignment", "type": "objective", "desc": "How credit/advantage is computed and attributed", "required": True},
            {"name": "policy_update", "type": "architecture", "desc": "Policy gradient / optimization rule", "required": True},
            {"name": "value_baseline", "type": "architecture", "desc": "Value function / baseline for variance reduction", "required": True},
            {"name": "exploration", "type": "inference", "desc": "Exploration strategy (epsilon-greedy, entropy, curiosity)", "required": False},
            {"name": "planner_memory", "type": "architecture", "desc": "Planning module / memory / world model", "required": False},
        ],
    },
    {
        "name": "vlm_standard",
        "domain": "VLM",
        "slots": [
            {"name": "vision_encoder", "type": "architecture", "desc": "Visual perception backbone (ViT, CLIP, SigLIP)", "required": True},
            {"name": "projector", "type": "architecture", "desc": "Vision-language connector / adapter", "required": True},
            {"name": "language_core", "type": "architecture", "desc": "Language model backbone (LLaMA, Qwen, etc.)", "required": True},
            {"name": "objective", "type": "objective", "desc": "Training objective (SFT, RLHF, DPO, contrastive)", "required": True},
            {"name": "data_mixture", "type": "data", "desc": "Pre-training / fine-tuning data composition", "required": True},
            {"name": "inference_planner", "type": "inference", "desc": "Inference strategy, chain-of-thought, tool use", "required": False},
        ],
    },
    {
        "name": "agent_standard",
        "domain": "Agent",
        "slots": [
            {"name": "perception", "type": "architecture", "desc": "Observation encoder (text, vision, multimodal)", "required": True},
            {"name": "planning", "type": "architecture", "desc": "Planning / reasoning module (CoT, tree search, graph)", "required": True},
            {"name": "action", "type": "architecture", "desc": "Action generation / execution module", "required": True},
            {"name": "memory", "type": "architecture", "desc": "Short-term / long-term / episodic memory", "required": False},
            {"name": "tool_use", "type": "inference", "desc": "External tool calling / API integration", "required": False},
            {"name": "reflection", "type": "objective", "desc": "Self-correction, self-evaluation, critique", "required": False},
        ],
    },
    {
        "name": "motion_generation_diffusion",
        "domain": "Motion Generation",
        "slots": [
            {"name": "motion_tokenizer", "type": "architecture", "desc": "Motion representation / VQ-VAE / continuous repr", "required": True},
            {"name": "denoiser", "type": "architecture", "desc": "Denoising network (U-Net, DiT, Transformer)", "required": True},
            {"name": "conditioning", "type": "architecture", "desc": "Conditioning mechanism (text, spatial, music)", "required": True},
            {"name": "objective", "type": "objective", "desc": "Training objective (DDPM, flow matching, score matching)", "required": True},
            {"name": "sampling", "type": "inference", "desc": "Sampling strategy (DDPM, DDIM, Euler, DPM-Solver)", "required": True},
            {"name": "physics_sim", "type": "architecture", "desc": "Physics simulation / RL controller (if any)", "required": False},
        ],
    },
]

MECHANISM_FAMILIES = [
    # Generative
    {"name": "diffusion", "domain": "Generative", "desc": "Denoising diffusion probabilistic models"},
    {"name": "flow_matching", "domain": "Generative", "desc": "Continuous normalizing flows / flow matching"},
    {"name": "masked_modeling", "domain": "Generative", "desc": "Masked token prediction (BERT-style)"},
    {"name": "autoregressive", "domain": "Generative", "desc": "Next-token prediction (GPT-style)"},
    {"name": "vq_vae", "domain": "Generative", "desc": "Vector-quantized variational autoencoder"},
    {"name": "gan", "domain": "Generative", "desc": "Generative adversarial network"},
    # RL
    {"name": "ppo", "domain": "RL", "desc": "Proximal policy optimization"},
    {"name": "grouped_reward", "domain": "RL", "desc": "Group-level reward / GRPO"},
    {"name": "reward_shaping", "domain": "RL", "desc": "Reward engineering and auxiliary rewards"},
    {"name": "curriculum_learning", "domain": "RL", "desc": "Curriculum / staged training"},
    # Architecture
    {"name": "transformer", "domain": "Architecture", "desc": "Self-attention based architecture"},
    {"name": "dit", "domain": "Architecture", "desc": "Diffusion Transformer"},
    {"name": "mamba", "domain": "Architecture", "desc": "Selective state space models"},
    # Alignment
    {"name": "rlhf", "domain": "Alignment", "desc": "RL from human feedback"},
    {"name": "dpo", "domain": "Alignment", "desc": "Direct preference optimization"},
    # Agent
    {"name": "chain_of_thought", "domain": "Agent", "desc": "Step-by-step reasoning"},
    {"name": "tool_calling", "domain": "Agent", "desc": "External tool / API invocation"},
    {"name": "self_reflection", "domain": "Agent", "desc": "Self-evaluation and correction"},
    {"name": "multi_agent", "domain": "Agent", "desc": "Multi-agent collaboration / debate"},
]


def seed(db_url: str) -> None:
    engine = create_engine(db_url)

    with Session(engine) as session:
        # Seed ParadigmFrames + Slots
        for frame_def in PARADIGM_FRAMES:
            # Check if already exists
            existing = session.execute(
                text("SELECT id FROM paradigm_templates WHERE name = :name"),
                {"name": frame_def["name"]},
            ).fetchone()

            if existing:
                paradigm_id = existing[0]
                print(f"  ParadigmFrame '{frame_def['name']}' already exists, skipping")
            else:
                # Create paradigm frame
                slots_json = {s["name"]: {"type": s["type"], "description": s["desc"], "required": s["required"]} for s in frame_def["slots"]}
                session.execute(
                    text("""
                        INSERT INTO paradigm_templates (name, version, domain, slots)
                        VALUES (:name, 'v1', :domain, :slots)
                    """),
                    {"name": frame_def["name"], "domain": frame_def["domain"], "slots": str(slots_json).replace("'", '"').replace("True", "true").replace("False", "false")},
                )
                session.flush()
                paradigm_id = session.execute(
                    text("SELECT id FROM paradigm_templates WHERE name = :name"),
                    {"name": frame_def["name"]},
                ).fetchone()[0]
                print(f"  Created ParadigmFrame '{frame_def['name']}'")

            # Seed Slots
            for i, slot_def in enumerate(frame_def["slots"]):
                existing_slot = session.execute(
                    text("SELECT id FROM slots WHERE paradigm_id = :pid AND name = :name"),
                    {"pid": paradigm_id, "name": slot_def["name"]},
                ).fetchone()
                if not existing_slot:
                    session.execute(
                        text("""
                            INSERT INTO slots (paradigm_id, name, description, slot_type, is_required, sort_order)
                            VALUES (:pid, :name, :desc, :type, :req, :order)
                        """),
                        {"pid": paradigm_id, "name": slot_def["name"], "desc": slot_def["desc"],
                         "type": slot_def["type"], "req": slot_def["required"], "order": i},
                    )

        # Seed MechanismFamilies
        for mf_def in MECHANISM_FAMILIES:
            existing = session.execute(
                text("SELECT id FROM mechanism_families WHERE name = :name"),
                {"name": mf_def["name"]},
            ).fetchone()
            if not existing:
                session.execute(
                    text("""
                        INSERT INTO mechanism_families (name, domain, description)
                        VALUES (:name, :domain, :desc)
                    """),
                    {"name": mf_def["name"], "domain": mf_def["domain"], "desc": mf_def["desc"]},
                )

        session.commit()

    # Summary
    with Session(engine) as session:
        pf_count = session.execute(text("SELECT count(*) FROM paradigm_templates")).scalar()
        slot_count = session.execute(text("SELECT count(*) FROM slots")).scalar()
        mf_count = session.execute(text("SELECT count(*) FROM mechanism_families")).scalar()
        print(f"\nSeed complete: {pf_count} paradigm frames, {slot_count} slots, {mf_count} mechanism families")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed paradigm frames and mechanism families")
    parser.add_argument("--db-url", default=settings.database_url_sync, help="Database URL")
    args = parser.parse_args()
    seed(args.db_url)
