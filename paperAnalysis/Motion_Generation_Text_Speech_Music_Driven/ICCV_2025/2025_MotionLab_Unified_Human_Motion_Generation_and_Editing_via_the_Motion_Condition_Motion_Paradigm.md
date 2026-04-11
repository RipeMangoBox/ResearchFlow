---
title: "MotionLab: Unified Human Motion Generation and Editing via the Motion-Condition-Motion Paradigm"
venue: ICCV
year: 2025
tags:
  - Motion_Generation_Text_Speech_Music_Driven
  - task/text-to-motion
  - task/motion-editing
  - diffusion
  - dataset/HumanML3D
  - dataset/100STYLE
  - repr/HumanML3D-263d
  - opensource/full
core_operator: Motion-Condition-Motion统一范式：将所有运动生成与编辑任务统一为"源运动+条件→目标运动"的条件扩散框架，单模型覆盖7+任务
primary_logic: |
  源运动（可为空/噪声）+ 多模态条件（文本/轨迹/风格/时间掩码）
  → 统一条件注入：运动通过cross-attention、条件通过adaptive normalization融合
  → 条件扩散去噪（Transformer backbone）→ 目标运动
  → 单模型统一处理：文本生成、轨迹生成、运动补全、文本编辑、轨迹编辑、风格迁移等7+任务
claims:
  - "MotionLab以单一模型在7个运动任务上达到或超越各任务专用SOTA模型的性能"
  - "Motion-Condition-Motion范式将运动生成和编辑统一为同一条件扩散框架，无需任务特定架构修改"
  - "在文本驱动生成上FID 0.045优于MotionLCM(0.068)，在风格迁移SRA 0.398优于MCM-LDM(0.304)"
pdf_ref: paperPDFs/Motion_Generation_Text_Speech_Music_Driven/ICCV_2025/2025_MotionLab_Unified_Human_Motion_Generation_and_Editing_via_the_Motion_Condition_Motion_Paradigm.pdf
category: Motion_Generation_Text_Speech_Music_Driven
---

# MotionLab: Unified Human Motion Generation and Editing via the Motion-Condition-Motion Paradigm

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://diouo.github.io/motionlab.github.io/) · ICCV 2025
> - **Summary**: MotionLab提出Motion-Condition-Motion统一范式，将运动生成和编辑的7+任务统一为"源运动+条件→目标运动"的条件扩散框架，单模型在所有任务上达到或超越各任务专用SOTA。
> - **Key Performance**:
>   - 文本生成FID **0.045** vs. MotionLCM(0.068)；风格迁移SRA **0.398** vs. MCM-LDM(0.304)
>   - 单模型覆盖：文本生成、轨迹生成、运动补全、文本编辑、轨迹编辑、风格迁移、组合任务

---

## Part I：问题与挑战

### 真正的卡点

运动生成与编辑领域面临**任务碎片化**和**统一建模困难**两大核心挑战：

- **任务孤岛**：文本生成、轨迹控制、运动编辑、风格迁移等任务各自有专用模型和架构，无法共享知识，部署和维护成本高
- **统一建模的表示难题**：不同任务的输入条件形态差异大（文本、轨迹、源运动、风格参考等），如何在单一框架中统一表示和注入这些异构条件是核心技术挑战
- **编辑任务的保真-可控平衡**：运动编辑需要在修改目标属性的同时保持非目标属性不变，这要求模型精确理解"改什么"和"保什么"的边界

### 输入/输出接口

- 输入：源运动（可为空/全噪声）+ 条件（文本/轨迹/风格参考/时间掩码，可组合）
- 输出：目标3D人体运动序列（HumanML3D 263维表示）

---

## Part II：方法与洞察

### 整体设计

MotionLab的核心设计是将所有运动任务抽象为统一的三元组：

1. **Motion-Condition-Motion范式**：任何任务 = f(源运动, 条件) → 目标运动
   - 生成任务：源运动为空/噪声，条件为文本/轨迹
   - 编辑任务：源运动为待编辑运动，条件为编辑指令
   - 风格迁移：源运动为内容运动，条件为风格参考运动
2. **统一条件注入**：
   - 运动信息通过cross-attention注入（源运动和风格参考均编码为运动token序列）
   - 非运动条件（文本、轨迹）通过adaptive normalization注入
   - 时间掩码用于区分"需要生成"和"需要保持"的帧
3. **Transformer扩散backbone**：在HumanML3D 263维连续空间中进行条件去噪，classifier-free guidance按任务类型调整强度

### 核心直觉

**什么变了**：从"每个任务一个专用模型"到"所有任务共享一个条件扩散模型"。

**哪些分布/约束/信息瓶颈变了**：
- 统一范式消除了任务间的架构壁垒 → 不同任务的训练数据可以互相增强（文本生成的数据帮助文本编辑学习语义理解，轨迹生成的数据帮助轨迹编辑学习空间控制）
- cross-attention + adaptive normalization的双通道注入打破了异构条件的表示瓶颈 → 运动类条件和非运动类条件各走最适合的注入路径，避免信息干扰
- 时间掩码机制提供了帧级别的"改/保"控制 → 编辑任务的保真-可控平衡从隐式学习变为显式约束

**为什么有效**：运动生成和编辑本质上都是条件运动分布的采样，区别仅在于条件的来源和约束的强度。统一范式利用了这一本质共性，让模型在更大的数据池上学习更鲁棒的条件-运动映射。

**权衡**：单模型需要在多任务间平衡，极端场景下可能不如深度优化的专用模型；classifier-free guidance的强度需要按任务手动调整。

---

## Part III：证据与局限

### 关键实验信号

- **全面超越专用模型**：在7个任务上，MotionLab单模型均达到或超越各任务SOTA——文本生成FID 0.045 vs. MotionLCM 0.068；轨迹生成avg error 0.0283 vs. OmniControl 0.0371；风格迁移SRA 0.398 vs. MCM-LDM 0.304
- **任务间正迁移**：联合训练比单任务训练在多数任务上有提升，证明统一范式带来的跨任务知识共享
- **编辑保真度**：文本编辑R@1 56.34%，轨迹编辑R@1 44.62%，均显著优于此前方法

### 局限与可复用组件

- **局限**：依赖HumanML3D 263维表示，限制了对更丰富身体细节（手指、面部）的支持；classifier-free guidance强度需要按任务手动调参；未探索更大规模数据和模型的Scaling行为
- **可复用**：Motion-Condition-Motion统一范式可迁移到其他序列生成/编辑任务（语音、手势等）；双通道条件注入（cross-attention + adaptive norm）的设计模式适用于任何多模态条件扩散模型；时间掩码的帧级控制机制可用于任何需要局部编辑的生成任务

---

## 本地 PDF 引用

![[paperPDFs/Motion_Generation_Text_Speech_Music_Driven/ICCV_2025/2025_MotionLab_Unified_Human_Motion_Generation_and_Editing_via_the_Motion_Condition_Motion_Paradigm.pdf]]
