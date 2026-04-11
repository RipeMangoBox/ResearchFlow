---
title: "LaMP: Language-Motion Pretraining for Motion Generation, Retrieval, and Captioning"
venue: ICLR
year: 2025
tags:
  - Motion_Generation_Text_Speech_Music_Driven
  - task/text-to-motion
  - task/motion-retrieval
  - task/motion-captioning
  - vq-vae
  - dataset/HumanML3D
  - dataset/KIT-ML
  - repr/HumanML3D-263d
  - opensource/full
core_operator: 语言-运动预训练对齐（Language-Motion Pretraining）：用运动数据替代图像数据重新对齐文本编码器，生成motion-informative文本嵌入，统一提升生成/检索/描述三任务
primary_logic: |
  运动序列 + 文本描述 → 双塔对比学习（运动Transformer + 文本Transformer）
  → 语言-运动潜空间对齐：替代CLIP的语言-视觉空间，生成motion-informative文本嵌入
  → 生成：LaMP文本嵌入作为条件 + 自回归掩码预测（无rank collapse）→ 运动序列
  → 检索：运动特征与查询token交互检索文本特征（及反向）
  → 描述：motion-informative运动特征微调LLM → 运动描述
claims:
  - "LaMP通过语言-运动预训练替代CLIP，在文本-运动生成上FID从0.045(MoMask+CLIP)降至0.033(MoMask+LaMP)"
  - "统一的语言-运动对齐同时提升生成、检索、描述三个任务，检索R@1从MoMask的51.0%提升至55.4%"
  - "提出LaMP-BertScore指标评估生成运动与文本描述的语义对齐度，比FID更直接反映语义一致性"
pdf_ref: paperPDFs/Motion_Generation_Text_Speech_Music_Driven/ICLR_2025/2025_Language_Motion_Pretraining_for_Motion_Generation_Retrieval_and_Captioning.pdf
category: Motion_Generation_Text_Speech_Music_Driven
---

# LaMP: Language-Motion Pretraining for Motion Generation, Retrieval, and Captioning

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [GitHub](https://github.com/zhelizhe/LaMP) · [ICLR 2025](https://openreview.net/forum?id=4MJkTvJjBb)
> - **Summary**: LaMP提出语言-运动预训练对齐，用运动数据替代图像数据重新训练文本编码器，生成motion-informative文本嵌入，统一提升文本-运动生成、运动-文本检索和运动描述三个任务，并引入LaMP-BertScore指标评估语义对齐。
> - **Key Performance**:
>   - 替换CLIP后，MoMask生成FID从0.045降至**0.033**
>   - 检索R@1从51.0%提升至**55.4%**；提出LaMP-BertScore语义对齐指标

---

## Part I：问题与挑战

### 真正的卡点

运动-语言任务面临**文本嵌入与运动语义不对齐**的核心瓶颈：

- **CLIP嵌入的模态错配**：现有运动生成方法普遍使用CLIP文本嵌入作为条件，但CLIP是在静态图像-文本对上预训练的，其文本嵌入偏向视觉外观描述，缺乏对运动时序动态（速度、节奏、轨迹变化等）的编码能力
- **三任务割裂**：生成、检索、描述三个运动-语言任务各自使用不同的文本表示和训练策略，无法共享语义对齐的收益
- **评估指标缺陷**：FID衡量分布距离但不直接反映单个样本的语义一致性，缺乏细粒度的语义对齐评估工具

### 输入/输出接口

- 生成：文本描述 → 3D运动序列（HumanML3D 263维）
- 检索：文本查询 ↔ 运动库匹配
- 描述：运动序列 → 自然语言描述

---

## Part II：方法与洞察

### 整体设计

LaMP的核心是构建一个motion-informative的语言-运动对齐空间：

1. **双塔对比预训练**：
   - 运动Transformer编码运动序列为运动特征
   - 文本Transformer编码文本描述为文本特征
   - 对比学习（类似CLIP）在运动-文本对上对齐两个空间
   - 关键区别：训练数据是运动-文本对（而非图像-文本对），文本编码器学到的嵌入天然包含运动时序语义

2. **下游任务适配**：
   - 生成：LaMP文本嵌入替代CLIP嵌入作为条件 + 自回归掩码预测（设计了防止rank collapse的机制）
   - 检索：运动特征通过query token与文本特征交互，支持双向检索
   - 描述：motion-informative运动特征作为前缀，微调LLM生成描述

3. **LaMP-BertScore**：基于LaMP对齐空间计算生成运动与文本描述的token级语义相似度，比FID更直接反映语义一致性

### 核心直觉

**什么变了**：从"借用视觉-语言对齐空间（CLIP）做运动条件"到"专门构建语言-运动对齐空间"。

**哪些分布/约束/信息瓶颈变了**：
- CLIP文本嵌入中缺失的运动时序信息被补回 → 文本条件从"描述外观"变为"描述动态"，生成模型接收到的语义信号更精确
- 统一的对齐空间使三个任务共享同一套语义表示 → 对齐质量的提升同时惠及生成、检索和描述
- LaMP-BertScore提供了token级语义对齐度量 → 评估从分布级（FID）细化到样本级，可以发现FID无法捕捉的语义偏差

**为什么有效**：运动-语言对齐的核心是文本嵌入需要编码运动相关的语义（时序、空间轨迹、身体部位协调等），而这些信息在CLIP的图像-文本预训练中几乎不存在。LaMP通过在运动-文本对上直接训练，让文本编码器学到了这些运动特有的语义维度。

**权衡**：需要额外的预训练阶段（运动-文本对比学习）；运动数据规模远小于图像数据，对齐空间的泛化能力可能不如CLIP；LaMP-BertScore的有效性依赖于对齐空间的质量。

---

## Part III：证据与局限

### 关键实验信号

- **生成提升**：将MoMask的文本条件从CLIP替换为LaMP，FID从0.045降至0.033，R-Precision Top-1从0.518提升至0.554——仅替换文本嵌入即获得显著提升
- **检索提升**：运动-文本检索R@1从51.0%提升至55.4%，证明对齐空间质量的提升
- **描述提升**：基于LaMP运动特征微调的LLM在描述准确性上优于基于CLIP特征的基线
- **LaMP-BertScore验证**：与人类评估的相关性高于FID，能捕捉FID遗漏的语义偏差

### 局限与可复用组件

- **局限**：预训练数据规模受限于现有运动数据集（HumanML3D ~15K），对齐空间的泛化能力有待验证；自回归掩码预测的rank collapse防止机制增加了训练复杂度；LaMP-BertScore尚未被社区广泛采用验证
- **可复用**：语言-运动对比预训练范式可直接替换任何使用CLIP文本嵌入的运动生成方法；LaMP-BertScore可作为通用的运动-文本语义对齐评估工具；双塔对比学习+下游任务适配的框架可迁移到其他模态对齐任务（语音-文本、手势-文本等）

---

## 本地 PDF 引用

![[paperPDFs/Motion_Generation_Text_Speech_Music_Driven/ICLR_2025/2025_Language_Motion_Pretraining_for_Motion_Generation_Retrieval_and_Captioning.pdf]]
