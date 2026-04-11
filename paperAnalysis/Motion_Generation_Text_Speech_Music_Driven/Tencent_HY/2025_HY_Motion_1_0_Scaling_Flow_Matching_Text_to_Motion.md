---
title: "HY-Motion 1.0: Scaling Flow Matching Models for Text-To-Motion Generation"
venue: Technical Report
year: 2025
tags:
  - Motion_Generation_Text_Speech_Music_Driven
  - task/text-to-motion
  - flow-matching
  - dit
  - reinforcement-learning
  - dataset/MotionX
  - dataset/HumanML3D
  - repr/SMPL-X
  - opensource/full
core_operator: 十亿参数级DiT Flow Matching + 三阶段训练范式（大规模预训练→高质量微调→RLHF对齐），首次将大语言模型的Scaling范式完整迁移到运动生成
primary_logic: |
  文本指令 → T5编码器提取语义嵌入
  → DiT Flow Matching（十亿参数Transformer）在连续运动潜空间中去噪
  → 三阶段训练：3000+小时预训练覆盖200+动作类别 → 400小时高质量数据微调 → RLHF/奖励模型对齐
  → 高保真、指令跟随的3D人体运动序列
claims:
  - "HY-Motion 1.0是首个将DiT-based Flow Matching模型扩展到十亿参数规模的运动生成模型，覆盖200+动作类别6大类"
  - "三阶段训练范式（预训练+SFT+RLHF）显著提升指令跟随能力，在文本-运动对齐上超越现有开源基准"
  - "通过严格的数据清洗和标注流水线处理3000+小时运动数据，实现了运动生成领域最大规模的训练数据覆盖"
pdf_ref: paperPDFs/Motion_Generation_Text_Speech_Music_Driven/Tencent_HY/2025_HY_Motion_1_0_Scaling_Flow_Matching_Text_to_Motion.pdf
category: Motion_Generation_Text_Speech_Music_Driven
---

# HY-Motion 1.0: Scaling Flow Matching Models for Text-To-Motion Generation

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [HuggingFace](https://huggingface.co/tencent/HY-Motion-1.0) · [GitHub](https://github.com/Tencent-Hunyuan/HY-Motion-1.0) · [Project](https://hunyuan.tencent.com/motion)
> - **Summary**: HY-Motion 1.0首次将DiT Flow Matching扩展到十亿参数规模，通过"大规模预训练→高质量微调→RLHF对齐"三阶段范式，实现覆盖200+动作类别的高保真文本驱动运动生成，指令跟随能力显著超越现有开源模型。
> - **Key Performance**:
>   - 覆盖6大类200+动作类别，训练数据3000+小时，为运动生成领域最大规模
>   - 指令跟随能力在人类评估和自动指标上均显著优于现有开源基准

---

## Part I：问题与挑战

### 真正的卡点

文本驱动运动生成面临**规模化瓶颈**和**指令对齐缺失**两大核心挑战：

- **数据与模型规模不足**：现有运动生成模型参数量和训练数据量远小于语言/视觉领域，导致动作覆盖范围有限（通常仅几十类），无法处理复杂、组合式文本指令
- **缺乏对齐训练**：运动生成领域尚未引入类似LLM的RLHF/奖励模型对齐流程，模型输出与用户意图之间存在系统性偏差——生成的运动可能语义正确但不符合用户期望的风格、节奏或细节
- **数据质量参差**：大规模运动数据来源多样（MoCap、视频重建等），噪声、标注不一致、动作类别分布不均等问题严重制约模型泛化

### 输入/输出接口

- 输入：自然语言文本指令（支持复杂组合描述）
- 输出：3D人体运动序列（SMPL-X格式）

---

## Part II：方法与洞察

### 整体设计

HY-Motion 1.0将LLM的Scaling范式完整迁移到运动生成：

1. **架构**：DiT（Diffusion Transformer）+ Flow Matching，参数量扩展到十亿级。运动序列在连续潜空间中表示，T5编码器提取文本语义嵌入作为条件
2. **数据流水线**：严格的运动清洗（物理合理性检查、去噪、标准化）+ 自动/人工标注（覆盖200+类别的细粒度文本描述）
3. **三阶段训练**：
   - Stage 1 预训练：3000+小时大规模运动数据，学习广泛的运动先验
   - Stage 2 SFT微调：400小时高质量精选数据，提升生成质量和多样性
   - Stage 3 RLHF对齐：人类反馈+奖励模型，优化指令跟随和用户偏好对齐

### 核心直觉

**什么变了**：将运动生成从"小数据+小模型+单阶段训练"推向"大数据+大模型+三阶段对齐训练"。

**哪些分布/约束/信息瓶颈变了**：
- 数据规模从百小时级→千小时级，动作类别从几十类→200+类，模型从百万参数→十亿参数——覆盖的运动分布从窄域扩展到广域
- RLHF阶段引入人类偏好信号，将"语义正确但不符合期望"的输出分布向用户意图对齐——这是运动生成领域首次引入此类对齐机制

**为什么有效**：Flow Matching在连续空间中的训练效率优于传统DDPM，使十亿参数DiT的训练成为可能；三阶段范式让模型先学广度（预训练）、再学质量（SFT）、最后学对齐（RLHF），逐步收窄输出分布到高质量+高对齐区域。

**权衡**：依赖大规模高质量标注数据和人类反馈，数据获取成本高；十亿参数模型的推理延迟和部署成本显著高于轻量模型。

---

## Part III：证据与局限

### 关键实验信号

- **覆盖广度**：6大类200+动作类别，远超现有开源模型的覆盖范围，在复杂组合指令上表现出明显优势
- **指令跟随**：人类评估中指令跟随准确率显著优于MoMask、MotionGPT等开源基准；自动指标（R-Precision、FID）同样领先
- **RLHF效果**：对齐阶段后，生成运动在风格一致性和细节匹配上有可感知的提升，尤其在模糊/复杂指令场景下

### 局限与可复用组件

- **局限**：技术报告未公开详细的定量对比表格和消融实验；RLHF的奖励模型设计和人类标注细节披露有限；十亿参数模型的推理效率未充分讨论
- **可复用**：三阶段训练范式（预训练→SFT→RLHF）可直接迁移到其他运动/动作生成任务；数据清洗和标注流水线的设计思路适用于任何大规模运动数据集构建；Flow Matching + DiT的架构组合为运动生成的Scaling提供了可行路径

---

## 本地 PDF 引用

![[paperPDFs/Motion_Generation_Text_Speech_Music_Driven/Tencent_HY/2025_HY_Motion_1_0_Scaling_Flow_Matching_Text_to_Motion.pdf]]
