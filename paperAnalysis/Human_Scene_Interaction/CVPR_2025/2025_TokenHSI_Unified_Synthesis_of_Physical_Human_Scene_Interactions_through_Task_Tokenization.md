---
title: "TokenHSI: Unified Synthesis of Physical Human-Scene Interactions through Task Tokenization"
venue: CVPR
year: 2025
tags:
  - Human_Scene_Interaction
  - task/scene-interaction
  - task/physics-based-control
  - reinforcement-learning
  - dataset/PartNet-Mobility
  - repr/SMPL
  - opensource/full
core_operator: 任务token化统一策略（Task Tokenization）：将人体本体感受作为共享token，不同HSI任务作为独立任务token，通过掩码机制在单一Transformer中统一多技能并支持灵活组合
primary_logic: |
  场景几何 + 任务指令（坐下/搬运/跟随/开门等）
  → 人体本体感受编码为共享proprioception token
  → 各任务编码为独立task token（可变长度输入）
  → Transformer策略：proprioception token + task tokens → 关节力矩
  → 掩码机制：训练时随机掩码部分任务token → 学习跨技能共享知识
  → 技能组合：推理时拼接多个task token → 零样本组合新技能
claims:
  - "TokenHSI是首个在单一Transformer策略中统一多种物理人-场景交互技能的框架"
  - "任务token化+掩码机制使技能间知识共享，组合技能（如搬运+坐下）成功率72.1%，无需额外训练"
  - "支持物体/地形形状泛化和长时序任务完成，开门任务在未见门把手形状上成功率>90%"
pdf_ref: paperPDFs/Human_Scene_Interaction/CVPR_2025/2025_TokenHSI_Unified_Synthesis_of_Physical_Human_Scene_Interactions_through_Task_Tokenization.pdf
category: Human_Scene_Interaction
---

# TokenHSI: Unified Synthesis of Physical Human-Scene Interactions through Task Tokenization

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://liangpan99.github.io/TokenHSI/) · [CVPR 2025](https://arxiv.org/abs/2503.19901)
> - **Summary**: TokenHSI提出任务token化框架，将人体本体感受作为共享token、各HSI任务作为独立task token，在单一Transformer策略中统一多种物理人-场景交互技能，支持零样本技能组合、形状泛化和长时序任务完成。
> - **Key Performance**:
>   - 单一策略统一坐下/搬运/跟随/开门等多种HSI技能
>   - 技能组合（搬运+坐下）成功率**72.1%**，无需额外训练；开门泛化到未见形状成功率>90%

---

## Part I：问题与挑战

### 真正的卡点

物理人-场景交互合成面临**多技能统一**和**灵活组合**两大核心挑战：

- **技能孤岛**：现有方法为每种HSI任务（坐下、搬运、开门等）训练独立控制器，无法共享跨技能知识，部署和维护成本高
- **技能组合困难**：真实场景需要组合多种技能（如搬着箱子坐下），但独立训练的控制器无法直接组合——它们的状态空间和动作空间不兼容
- **形状泛化**：同一任务面对不同物体/地形形状（不同椅子、不同门把手）需要泛化能力，但RL策略容易过拟合训练时的特定形状

### 输入/输出接口

- 输入：场景几何（点云/SDF）+ 任务指令（任务类型 + 目标位置/物体）
- 输出：物理仿真中的全身运动（关节力矩→SMPL姿态序列）

---

## Part II：方法与洞察

### 整体设计

TokenHSI的核心设计是将多技能统一问题转化为token序列建模问题：

1. **Proprioception Token（共享）**：人体本体感受（关节位置、速度、接触力等）编码为固定的共享token，所有任务共用
2. **Task Tokens（任务特定）**：
   - 每种任务有独立的task token编码器
   - 坐下任务：椅子形状+目标位置 → task token
   - 搬运任务：物体形状+目标轨迹 → task token
   - 开门任务：门把手形状+铰链位置 → task token
3. **Transformer策略**：proprioception token + task tokens → 自注意力 → 关节力矩
4. **掩码训练**：随机掩码部分task token → 强制模型学习跨技能的共享运动模式（如平衡、步态等）
5. **技能组合**：推理时拼接多个task token → Transformer自然处理可变长度输入 → 零样本组合

### 核心直觉

**什么变了**：从"每个任务一个控制器"到"所有任务共享一个Transformer，通过task token区分"。

**哪些分布/约束/信息瓶颈变了**：
- 共享proprioception token使不同任务的基础运动能力（平衡、步态、姿态调整）可以互相增强 → 跨技能知识共享
- 可变长度task token输入使技能组合变为简单的token拼接 → 无需为组合场景重新训练
- 掩码训练强制模型在部分任务信息缺失时仍能维持基础运动 → 提高了鲁棒性和泛化能力

**为什么有效**：不同HSI技能共享大量底层运动能力（平衡、步态、手臂协调等），区别仅在于任务特定的目标和约束。Token化设计精确地分离了"共享能力"（proprioception token）和"任务特定目标"（task tokens），使Transformer可以高效地学习两者。

**权衡**：Transformer策略的推理延迟高于简单MLP；掩码训练增加了训练复杂度；当前技能数量有限（~5种），更多技能的扩展性待验证。

---

## Part III：证据与局限

### 关键实验信号

- **多技能统一**：单一策略在坐下、搬运、跟随、开门等任务上均达到或接近各任务专用控制器的成功率
- **零样本组合**：搬运+坐下成功率72.1%，搬运+跟随成功率91.1%——无需额外训练
- **形状泛化**：开门任务在未见门把手形状上成功率>90%；坐下任务在未见椅子形状上成功率>85%
- **长时序任务**：连续完成"走到椅子→坐下→站起→走到门→开门"的长时序任务
- **新技能扩展**：通过添加新task token编码器+少量RL训练，快速学习推箱子（100%）和举箱子（80.6%）

### 局限与可复用组件

- **局限**：当前技能种类有限；Transformer推理延迟较高；掩码训练的最优掩码率需要调参；复杂组合（3+技能同时）未充分验证
- **可复用**：任务token化的设计模式可迁移到任何多任务RL场景（机器人操作、游戏NPC等）；掩码训练促进跨技能知识共享的策略适用于任何多技能统一学习；可变长度token输入实现零样本技能组合的思路可用于任何需要灵活组合的控制系统

---

## 本地 PDF 引用

![[paperPDFs/Human_Scene_Interaction/CVPR_2025/2025_TokenHSI_Unified_Synthesis_of_Physical_Human_Scene_Interactions_through_Task_Tokenization.pdf]]
