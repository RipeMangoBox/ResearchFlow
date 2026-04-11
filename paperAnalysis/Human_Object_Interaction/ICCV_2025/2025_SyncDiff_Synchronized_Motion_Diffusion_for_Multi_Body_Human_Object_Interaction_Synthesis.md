---
title: "SyncDiff: Synchronized Motion Diffusion for Multi-Body Human-Object Interaction Synthesis"
venue: ICCV
year: 2025
tags:
  - Human_Object_Interaction
  - task/human-object-interaction
  - diffusion
  - dataset/GRAB
  - dataset/CHAIRS
  - dataset/CORE4D
  - repr/SMPL
  - opensource/full
core_operator: 多体同步扩散（Alignment Scores + 显式同步推理）：通过数学推导的对齐分数训练和动态图模型最大似然采样推理，实现任意数量人-手-物体的同步交互运动生成
primary_logic: |
  文本/动作标签 + 多体配置（人数、手数、物体数）
  → 各body独立扩散去噪 + alignment scores约束同步性（训练时）
  → 推理时：动态图模型上最大似然采样实现显式多体同步
  → 频率分解：高频交互细节在频域显式建模，避免被低频大幅运动淹没
  → 输出：同步的多体交互运动（无穿透、无接触丢失、时序同步）
claims:
  - "SyncDiff首次提出统一框架处理任意数量人-手-物体的多体交互运动生成"
  - "数学推导的alignment scores在训练中显式约束多体同步性，推理时动态图模型最大似然采样进一步保证同步"
  - "频率分解机制将高频交互细节从低频运动中分离，在频域显式建模，显著减少接触丢失和穿透"
pdf_ref: paperPDFs/Human_Object_Interaction/ICCV_2025/2025_SyncDiff_Synchronized_Motion_Diffusion_for_Multi_Body_Human_Object_Interaction_Synthesis.pdf
category: Human_Object_Interaction
---

# SyncDiff: Synchronized Motion Diffusion for Multi-Body Human-Object Interaction Synthesis

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://syncdiff.github.io/) · ICCV 2025
> - **Summary**: SyncDiff提出统一框架处理任意数量人-手-物体的多体交互运动生成，通过数学推导的alignment scores（训练）和动态图模型最大似然采样（推理）实现多体同步，频率分解机制显式建模高频交互细节。
> - **Key Performance**:
>   - 首个统一处理任意数量人/手/物体多体交互的生成框架
>   - 在5个数据集上（GRAB、CHAIRS、CORE4D等）接触精度和动作语义均优于各场景SOTA

---

## Part I：问题与挑战

### 真正的卡点

多体人-物交互运动生成面临**多体同步**和**高频交互细节丢失**两大核心挑战：

- **多体同步困难**：多个人、手、物体之间存在强耦合——一个body的运动变化会影响所有其他body。独立生成各body的运动会导致穿透、接触丢失、时序不同步
- **高频细节被淹没**：物体交互中的关键信号（接触瞬间、抓握力变化等）是高频的，但被人体大幅度低频运动（行走、转身等）淹没，标准扩散模型难以同时建模两者
- **配置多样性**：不同场景涉及不同数量的人、手、物体，现有方法通常为特定配置（如单人单物）设计，无法泛化

### 输入/输出接口

- 输入：动作标签/文本描述 + 多体配置（人数、手数、物体数）
- 输出：所有body的同步运动序列（人体SMPL + 手部MANO + 物体6DoF）

---

## Part II：方法与洞察

### 整体设计

SyncDiff在标准扩散框架上引入两个同步机制：

1. **Alignment Scores（训练时同步）**：
   - 数学推导：在多body联合扩散过程中，推导出各body去噪分数之间的对齐约束
   - 训练时将alignment scores作为额外损失项，约束各body的去噪方向一致
   - 不需要显式的接触标注，从数据中自动学习同步模式

2. **显式同步推理（推理时同步）**：
   - 构建动态图模型：节点=各body，边=交互关系
   - 在每个去噪步，通过图模型上的最大似然采样协调各body的去噪方向
   - 保证推理时的多体同步性

3. **频率分解**：
   - 将运动信号分解为低频（大幅运动）和高频（交互细节）分量
   - 高频分量在频域显式建模，避免被低频淹没
   - 两个分量分别去噪后合并

### 核心直觉

**什么变了**：从"各body独立生成→事后修正"到"训练和推理时都显式约束多体同步"。

**哪些分布/约束/信息瓶颈变了**：
- Alignment scores在训练时将多body的去噪分数耦合 → 模型学到的不是各body的边际分布，而是联合分布中的条件依赖关系
- 动态图模型在推理时提供了全局协调机制 → 各body的去噪不再独立，而是在图结构上联合优化
- 频率分解打破了高频-低频的信息瓶颈 → 高频交互细节获得了独立的建模通道，不再被低频运动的梯度主导

**为什么有效**：多体交互的本质是联合分布，而非各body边际分布的简单组合。Alignment scores和图模型采样从训练和推理两端逼近联合分布。频率分解则解决了不同频率信号的建模竞争问题。

**权衡**：alignment scores的推导依赖特定的扩散过程假设；图模型采样增加了推理计算量；频率分解的截断频率需要手动设定。

---

## Part III：证据与局限

### 关键实验信号

- **多配置统一**：在5个数据集（GRAB单手抓握、CHAIRS双人搬椅、CORE4D多人多物等）上，SyncDiff以统一框架在每个配置上均优于各配置的专用SOTA
- **同步性提升**：穿透率和接触丢失率显著降低，尤其在多人多物的复杂配置中优势更明显
- **频率分解消融**：去掉频率分解后，高频交互细节（如抓握瞬间的手指运动）质量显著下降
- **alignment scores消融**：去掉alignment scores后，多body运动出现明显的时序不同步

### 局限与可复用组件

- **局限**：当前仅处理刚性物体，未扩展到可变形物体；图模型采样的计算开销随body数量增长；频率分解的截断频率对不同交互类型可能需要调整
- **可复用**：alignment scores的数学推导框架可迁移到任何需要多体同步的扩散生成任务（多人舞蹈、多机器人协作等）；频率分解机制适用于任何需要同时建模多尺度信号的生成任务；动态图模型的推理时同步策略可用于任何多agent协调生成

---

## 本地 PDF 引用

![[paperPDFs/Human_Object_Interaction/ICCV_2025/2025_SyncDiff_Synchronized_Motion_Diffusion_for_Multi_Body_Human_Object_Interaction_Synthesis.pdf]]
