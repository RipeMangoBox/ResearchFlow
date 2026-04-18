"""Seed taxonomy nodes with initial domain/modality/task/mechanism hierarchy.

Run: python -m migration.seed_taxonomy
"""

import asyncio
import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import async_session
from backend.models.taxonomy import TaxonomyNode, TaxonomyEdge


# ── Seed Data ──────────────────────────────────────────────────

SEED_NODES = [
    # ── Domains ──
    {"name": "LLM", "name_zh": "大语言模型", "dimension": "domain"},
    {"name": "VLM", "name_zh": "视觉语言模型", "dimension": "domain"},
    {"name": "Diffusion", "name_zh": "扩散模型", "dimension": "domain"},
    {"name": "Agent", "name_zh": "智能体", "dimension": "domain"},
    {"name": "Embodied AI", "name_zh": "具身智能", "dimension": "domain"},

    # ── Modalities ──
    {"name": "Image", "name_zh": "图像", "dimension": "modality"},
    {"name": "Video", "name_zh": "视频", "dimension": "modality"},
    {"name": "Text", "name_zh": "文本", "dimension": "modality"},
    {"name": "Audio", "name_zh": "音频", "dimension": "modality"},
    {"name": "3D", "name_zh": "三维", "dimension": "modality"},
    {"name": "Motion", "name_zh": "人体运动", "dimension": "modality"},
    {"name": "Multimodal", "name_zh": "多模态", "dimension": "modality"},

    # ── Tasks ──
    {"name": "Video Understanding", "name_zh": "视频理解", "dimension": "task"},
    {"name": "Image Understanding", "name_zh": "图像理解", "dimension": "task"},
    {"name": "Text Generation", "name_zh": "文本生成", "dimension": "task"},
    {"name": "Image Generation", "name_zh": "图像生成", "dimension": "task"},
    {"name": "Video Generation", "name_zh": "视频生成", "dimension": "task"},
    {"name": "Motion Generation", "name_zh": "运动生成", "dimension": "task"},
    {"name": "Object Detection", "name_zh": "目标检测", "dimension": "task"},
    {"name": "Segmentation", "name_zh": "分割", "dimension": "task"},
    {"name": "Reasoning", "name_zh": "推理", "dimension": "task"},
    {"name": "Code Generation", "name_zh": "代码生成", "dimension": "task"},
    {"name": "Math Reasoning", "name_zh": "数学推理", "dimension": "task"},
    {"name": "Dialogue", "name_zh": "对话", "dimension": "task"},

    # ── Subtasks ──
    {"name": "Long Video QA", "name_zh": "长视频问答", "dimension": "subtask"},
    {"name": "Streaming Video", "name_zh": "流视频理解", "dimension": "subtask"},
    {"name": "Video Captioning", "name_zh": "视频字幕", "dimension": "subtask"},
    {"name": "VQA", "name_zh": "视觉问答", "dimension": "subtask"},
    {"name": "Referring Segmentation", "name_zh": "指代分割", "dimension": "subtask"},
    {"name": "Text-to-Motion", "name_zh": "文本到运动", "dimension": "subtask"},
    {"name": "Music-to-Motion", "name_zh": "音乐到运动", "dimension": "subtask"},
    {"name": "Human-Object Interaction", "name_zh": "人物交互", "dimension": "subtask"},
    {"name": "Human-Scene Interaction", "name_zh": "人场景交互", "dimension": "subtask"},

    # ── Learning Paradigms ──
    {"name": "Reinforcement Learning", "name_zh": "强化学习", "dimension": "learning_paradigm",
     "aliases": ["RL"]},
    {"name": "Supervised Fine-Tuning", "name_zh": "监督微调", "dimension": "learning_paradigm",
     "aliases": ["SFT"]},
    {"name": "RLHF", "name_zh": "人类反馈强化学习", "dimension": "learning_paradigm"},
    {"name": "DPO", "name_zh": "直接偏好优化", "dimension": "learning_paradigm",
     "aliases": ["Direct Preference Optimization"]},
    {"name": "Self-Play", "name_zh": "自博弈", "dimension": "learning_paradigm"},
    {"name": "Pretraining", "name_zh": "预训练", "dimension": "learning_paradigm"},
    {"name": "Training-Free", "name_zh": "免训练", "dimension": "learning_paradigm"},
    {"name": "Distillation", "name_zh": "蒸馏", "dimension": "learning_paradigm"},
    {"name": "Continual Learning", "name_zh": "持续学习", "dimension": "learning_paradigm"},

    # ── Scenarios ──
    {"name": "Online / Streaming", "name_zh": "在线/流式", "dimension": "scenario"},
    {"name": "Offline", "name_zh": "离线", "dimension": "scenario"},
    {"name": "Zero-Shot", "name_zh": "零样本", "dimension": "scenario"},
    {"name": "Few-Shot", "name_zh": "少样本", "dimension": "scenario"},
    {"name": "Long Context", "name_zh": "长上下文", "dimension": "scenario"},

    # ── Mechanisms ──
    {"name": "Reward Design", "name_zh": "奖励设计", "dimension": "mechanism"},
    {"name": "Advantage Weighting", "name_zh": "优势权重", "dimension": "mechanism"},
    {"name": "KV Cache Compression", "name_zh": "KV缓存压缩", "dimension": "mechanism"},
    {"name": "Token Compression", "name_zh": "Token压缩", "dimension": "mechanism"},
    {"name": "Temporal Credit Assignment", "name_zh": "时序信用分配", "dimension": "mechanism"},
    {"name": "Architecture Optimization", "name_zh": "架构优化", "dimension": "mechanism"},
    {"name": "Latent Space Features", "name_zh": "隐空间特征提取", "dimension": "mechanism"},
    {"name": "Attention Mechanism", "name_zh": "注意力机制", "dimension": "mechanism"},
    {"name": "Memory Module", "name_zh": "记忆模块", "dimension": "mechanism"},
    {"name": "Flow Matching", "name_zh": "流匹配", "dimension": "mechanism"},
    {"name": "Diffusion Process", "name_zh": "扩散过程", "dimension": "mechanism"},
    {"name": "Tree Search", "name_zh": "树搜索", "dimension": "mechanism"},
    {"name": "Chain-of-Thought", "name_zh": "思维链", "dimension": "mechanism",
     "aliases": ["CoT"]},
    {"name": "Tool Use", "name_zh": "工具使用", "dimension": "mechanism"},
    {"name": "Speculative Decoding", "name_zh": "推测解码", "dimension": "mechanism"},
    {"name": "MoE", "name_zh": "混合专家", "dimension": "mechanism",
     "aliases": ["Mixture of Experts"]},

    # ── Method Baselines ──
    {"name": "GRPO", "name_zh": "GRPO", "dimension": "method_baseline",
     "aliases": ["Group Relative Policy Optimization"]},
    {"name": "PPO", "name_zh": "PPO", "dimension": "method_baseline",
     "aliases": ["Proximal Policy Optimization"]},
    {"name": "QwenVL", "name_zh": "通义千问VL", "dimension": "model_family",
     "aliases": ["Qwen-VL", "Qwen2-VL", "Qwen2.5-VL"]},
    {"name": "LLaVA", "name_zh": "LLaVA", "dimension": "model_family",
     "aliases": ["LLaVA-Video", "LLaVA-OneVision"]},
    {"name": "InternVL", "name_zh": "InternVL", "dimension": "model_family",
     "aliases": ["InternVL2", "InternVL2.5"]},
    {"name": "GPT-4", "name_zh": "GPT-4", "dimension": "model_family",
     "aliases": ["GPT-4o", "GPT-4V"]},
    {"name": "Claude", "name_zh": "Claude", "dimension": "model_family",
     "aliases": ["Claude 3.5", "Claude 4"]},

    # ── Constraints ──
    {"name": "Context Length", "name_zh": "上下文长度限制", "dimension": "constraint"},
    {"name": "Memory Cost", "name_zh": "显存开销", "dimension": "constraint"},
    {"name": "Latency", "name_zh": "延迟", "dimension": "constraint"},
    {"name": "Data Scarcity", "name_zh": "数据稀缺", "dimension": "constraint"},
    {"name": "Compute Budget", "name_zh": "算力预算", "dimension": "constraint"},
]

# ── Edges ──
SEED_EDGES = [
    # Task → Subtask (part_of)
    ("Video Understanding", "Long Video QA", "part_of"),
    ("Video Understanding", "Streaming Video", "part_of"),
    ("Video Understanding", "Video Captioning", "part_of"),
    ("Image Understanding", "VQA", "part_of"),
    ("Image Understanding", "Referring Segmentation", "part_of"),
    ("Motion Generation", "Text-to-Motion", "part_of"),
    ("Motion Generation", "Music-to-Motion", "part_of"),
    ("Motion Generation", "Human-Object Interaction", "part_of"),
    ("Motion Generation", "Human-Scene Interaction", "part_of"),

    # Paradigm relationships
    ("Reinforcement Learning", "RLHF", "is_a"),
    ("Reinforcement Learning", "DPO", "is_a"),
    ("Reinforcement Learning", "GRPO", "is_a"),
    ("Reinforcement Learning", "PPO", "is_a"),

    # Mechanism → Mechanism (uses/optimizes)
    ("KV Cache Compression", "Token Compression", "is_a"),
]


async def seed_taxonomy() -> dict:
    """Seed initial taxonomy nodes and edges."""
    async with async_session() as session:
        created_nodes = 0
        created_edges = 0
        node_ids: dict[str, uuid.UUID] = {}

        # Create nodes
        for node_data in SEED_NODES:
            # Check if already exists
            existing = await session.execute(
                select(TaxonomyNode).where(
                    TaxonomyNode.name == node_data["name"],
                    TaxonomyNode.dimension == node_data["dimension"],
                )
            )
            existing_node = existing.scalar_one_or_none()

            if existing_node:
                node_ids[node_data["name"]] = existing_node.id
                continue

            node = TaxonomyNode(
                name=node_data["name"],
                name_zh=node_data.get("name_zh"),
                dimension=node_data["dimension"],
                aliases=node_data.get("aliases"),
                status="canonical",  # seed nodes are canonical
            )
            session.add(node)
            await session.flush()
            node_ids[node_data["name"]] = node.id
            created_nodes += 1

        # Create edges
        for parent_name, child_name, relation in SEED_EDGES:
            parent_id = node_ids.get(parent_name)
            child_id = node_ids.get(child_name)

            if not parent_id or not child_id:
                continue

            # Check if edge exists
            existing = await session.execute(
                select(TaxonomyEdge).where(
                    TaxonomyEdge.parent_id == parent_id,
                    TaxonomyEdge.child_id == child_id,
                    TaxonomyEdge.relation_type == relation,
                )
            )
            if existing.scalar_one_or_none():
                continue

            edge = TaxonomyEdge(
                parent_id=parent_id,
                child_id=child_id,
                relation_type=relation,
            )
            session.add(edge)
            created_edges += 1

        await session.commit()

        return {
            "created_nodes": created_nodes,
            "created_edges": created_edges,
            "total_nodes": len(node_ids),
        }


if __name__ == "__main__":
    result = asyncio.run(seed_taxonomy())
    print(f"Seeded taxonomy: {result}")
