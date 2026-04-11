---
title: "HHOI: Learning to Generate Human-Human-Object Interactions from Textual Descriptions"
venue: NeurIPS
year: 2025
tags:
  - Human_Object_Interaction
  - Human_Human_Interaction
  - task/human-human-object-interaction
  - diffusion
  - dataset/HHOI
  - repr/SMPL
  - opensource/partial
core_operator: 分治式HHOI生成框架：将双人-物体交互分解为HOI（人-物）和HHI（人-人）两个子问题独立训练，推理时通过统一采样过程组合生成完整HHOI
primary_logic: |
  文本描述 → 分解为HOI条件和HHI条件
  → HOI扩散模型：生成单人-物体交互运动（score-based diffusion）
  → HHI扩散模型：生成双人交互运动（score-based diffusion）
  → 统一采样：在去噪过程中融合HOI和HHI的score → 生成完整的双人-物体交互
  → 合成数据增强：利用图像生成模型合成HHOI数据，弥补真实数据稀缺
claims:
  - "HHOI首次提出双人-物体交互生成问题，并构建了专用数据集和合成数据增强流水线"
  - "分治式框架将HHOI分解为HOI+HHI独立训练，推理时统一采样组合，避免了联合训练的数据需求"
  - "方法可扩展到多人（>2人）场景，在未见文本提示上生成合理的多人-物体交互"
pdf_ref: paperPDFs/Human_Object_Interaction/NeurIPS_2025/2025_HHOI_Learning_to_Generate_Human_Human_Object_Interactions_from_Textual_Descriptions.pdf
category: Human_Object_Interaction
---

# HHOI: Learning to Generate Human-Human-Object Interactions from Textual Descriptions

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://tlb-miss.github.io/hhoi/) · [NeurIPS 2025](https://arxiv.org/abs/2409.08706)
> - **Summary**: HHOI首次定义并解决双人-物体交互生成问题，提出分治式框架——将HHOI分解为HOI和HHI两个子问题独立训练扩散模型，推理时通过统一采样融合两个score生成完整交互。配合新采集的HHOI数据集和基于图像生成模型的合成数据增强，实现文本驱动的多人-物体交互生成。
> - **Key Performance**:
>   - 首个HHOI生成框架，在自建数据集上显著优于仅处理单人HOI的基线
>   - 可扩展到多人（>2人）场景，在未见文本提示上生成合理交互

---

## Part I：问题与挑战

### 真正的卡点

双人-物体交互生成面临**三体耦合建模**和**数据极度稀缺**两大核心挑战：

- **三体耦合**：HHOI涉及两个人和一个物体的三方耦合——两人之间的社交距离和空间配置、每人与物体的接触和操作、以及三者的时序协调。现有方法要么只处理单人-物体（HOI），要么只处理双人交互（HHI），无法同时建模三方关系
- **数据稀缺**：真实的双人-物体交互MoCap数据几乎不存在——采集需要两人同时佩戴MoCap设备并操作物体，成本极高且场景受限
- **联合训练不可行**：直接在三体联合空间上训练扩散模型需要大量HHOI数据，而这些数据不可得

### 输入/输出接口

- 输入：文本描述（描述双人与物体的交互场景）
- 输出：两人的3D运动序列（SMPL格式）+ 物体6DoF轨迹

---

## Part II：方法与洞察

### 整体设计

HHOI采用分治策略，将不可解的联合问题分解为两个可解的子问题：

1. **数据准备**：
   - 新采集HHOI数据集（小规模，用于评估和少量训练）
   - 合成数据增强：用图像生成模型合成HHOI图像 → 从图像中提取HOI和HHI的运动先验 → 扩充训练数据
   - 从HHOI数据中分解出HOI子集（单人+物体）和HHI子集（双人无物体）

2. **独立训练**：
   - HOI扩散模型：在HOI数据上训练，学习单人-物体交互的score function
   - HHI扩散模型：在HHI数据上训练，学习双人交互的score function

3. **统一采样**：
   - 推理时，在每个去噪步融合HOI和HHI的score：\(\nabla \log p(\text{HHOI}) \approx \nabla \log p(\text{HOI}) + \nabla \log p(\text{HHI})\)
   - 通过共享的人体表示（第一个人同时出现在HOI和HHI中）实现两个score的对齐
   - 单次采样过程生成完整的双人-物体交互

### 核心直觉

**什么变了**：从"在联合HHOI空间上直接建模"到"分解为HOI+HHI独立建模，推理时组合"。

**哪些分布/约束/信息瓶颈变了**：
- 分解策略将数据需求从"稀缺的HHOI数据"降低为"相对丰富的HOI数据+HHI数据" → 绕过了联合训练的数据瓶颈
- Score融合在数学上近似联合分布的score → 无需联合训练即可生成联合分布的样本
- 共享人体表示作为两个子模型的"桥梁" → 保证了HOI和HHI生成结果的空间一致性

**为什么有效**：HHOI的联合分布可以近似分解为HOI和HHI的条件分布的乘积（在共享人体状态上条件独立假设）。这个假设在多数场景下合理——一个人与物体的操作方式和两人之间的社交距离/配置可以相对独立地建模。

**权衡**：条件独立假设在强耦合场景（如两人同时搬运同一物体）下可能不成立；合成数据与真实数据存在分布差距；score融合的权重需要手动调整。

---

## Part III：证据与局限

### 关键实验信号

- **HHOI生成质量**：在自建数据集上，分治式框架生成的HHOI在接触合理性、空间配置和运动自然度上显著优于仅使用HOI模型的基线
- **多人扩展**：框架可扩展到3+人场景，通过添加更多HHI pair的score实现多人-物体交互
- **未见文本泛化**：在训练集未覆盖的文本提示上仍能生成合理的HHOI，证明分治式框架的泛化能力
- **合成数据有效性**：加入合成数据后，生成质量有可感知的提升，尤其在数据稀缺的交互类型上

### 局限与可复用组件

- **局限**：条件独立假设限制了强耦合交互的建模质量；自建数据集规模小，评估的统计显著性有限；合成数据的质量和多样性仍有提升空间；物体形状多样性有限
- **可复用**：分治式score融合框架可迁移到任何需要组合多个独立训练模型的生成任务；合成数据增强流水线（图像生成→运动提取）适用于任何缺乏真实数据的交互场景；HHOI问题定义和数据集为后续研究提供了基准

---

## 本地 PDF 引用

![[paperPDFs/Human_Object_Interaction/NeurIPS_2025/2025_HHOI_Learning_to_Generate_Human_Human_Object_Interactions_from_Textual_Descriptions.pdf]]
