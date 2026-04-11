---
title: "MotionGPT: Human Motion as a Foreign Language"
venue: NeurIPS
year: 2023
tags:
  - Motion_Generation_Text_Speech_Music_Driven
  - task/text-to-motion
  - task/motion-captioning
  - vq-vae
  - autoregressive
  - dataset/HumanML3D
  - dataset/KIT-ML
  - repr/HumanML3D-263d
  - opensource/full
core_operator: 运动-语言统一建模：将运动VQ离散化为"运动词汇"，与文本token在同一T5架构中联合建模，通过prompt学习统一处理生成/描述/预测/补全多任务
primary_logic: |
  运动序列 → VQ-VAE离散化为运动token（"运动词汇"）
  → 运动token + 文本token混合输入T5 Transformer
  → Prompt-based多任务训练：不同任务通过不同prompt模板区分
  → 统一模型输出：文本token（描述）或运动token（生成/预测/补全）→ VQ解码
claims:
  - "MotionGPT首次将运动视为'外语'，在统一T5框架中联合处理生成、描述、预测、补全四个任务"
  - "在HumanML3D上文本-运动生成FID 0.232，运动描述CIDEr显著优于此前专用模型"
  - "Prompt-based多任务训练使单一模型在四个任务上均达到或接近各任务SOTA"
pdf_ref: paperPDFs/Motion_Generation_Text_Speech_Music_Driven/NeurIPS_2023/2023_MotionGPT_Human_Motion_as_Foreign_Language.pdf
category: Motion_Generation_Text_Speech_Music_Driven
---

# MotionGPT: Human Motion as a Foreign Language

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [GitHub](https://github.com/OpenMotionLab/MotionGPT) · [NeurIPS 2023](https://arxiv.org/abs/2306.14795)
> - **Summary**: MotionGPT首次将人体运动视为"外语"，通过VQ离散化构建"运动词汇"，在统一T5架构中与文本联合建模，以prompt学习范式统一处理生成、描述、预测、补全四个运动-语言任务。
> - **Key Performance**:
>   - 单模型统一四任务：文本-运动生成FID 0.232，运动描述CIDEr优于专用模型
>   - Prompt-based训练使各任务间正迁移，无需任务特定架构

---

## Part I：问题与挑战

### 真正的卡点

运动-语言建模面临**多任务统一**和**运动-语言对齐**两大核心挑战：

- **任务碎片化**：生成（文本→运动）、描述（运动→文本）、预测（历史→未来）、补全（首尾→中间）各自使用独立模型和训练策略，无法共享跨任务知识
- **运动-语言语义鸿沟**：运动是连续时序信号，语言是离散符号序列，两者的表示空间和生成范式差异巨大，难以在统一框架中处理
- **预训练模型复用困难**：大语言模型在文本上积累了丰富的语义知识，但无法直接处理运动数据

### 输入/输出接口

- 输入：文本prompt + 运动token（按任务不同组合）
- 输出：运动token序列（生成/预测/补全）或文本token序列（描述）

---

## Part II：方法与洞察

### 整体设计

MotionGPT的核心思路是将运动"翻译"为语言模型可理解的token序列：

1. **运动词汇构建**：VQ-VAE将连续运动序列编码为离散运动token，构成"运动词汇表"，与文本词汇表合并
2. **统一T5架构**：运动token和文本token在同一encoder-decoder Transformer中处理，模型不区分模态——运动就是一种"外语"
3. **Prompt-based多任务**：
   - 生成："Generate motion: [text description]" → [motion tokens]
   - 描述："Describe motion: [motion tokens]" → [text description]
   - 预测："Predict motion: [history tokens]" → [future tokens]
   - 补全："Fill motion: [start tokens] [end tokens]" → [middle tokens]
4. **混合预训练+微调**：先在运动-文本混合数据上预训练，再在各任务的prompt模板上微调

### 核心直觉

**什么变了**：从"每个运动任务一个专用模型"到"运动=外语，所有任务=翻译/生成"。

**哪些分布/约束/信息瓶颈变了**：
- VQ离散化将运动从连续空间映射到离散token空间 → 消除了运动-语言的表示形式差异，两者可以在同一序列空间中处理
- T5的encoder-decoder架构天然支持序列到序列的映射 → 生成、描述、预测、补全都可以统一为seq2seq任务
- Prompt模板提供了任务区分信号 → 模型通过prompt理解当前任务类型，共享的backbone在多任务间传递知识

**为什么有效**：人体运动确实具有类似语言的结构——有"词汇"（基本动作单元）、"语法"（动作组合规则）、"语义"（动作含义）。VQ离散化恰好捕捉了这种结构，使得语言模型的建模能力可以直接迁移。

**权衡**：VQ离散化引入量化误差，限制了运动细节的保真度；T5架构的参数效率不如专用架构；运动数据规模远小于文本数据，预训练的收益有限。

---

## Part III：证据与局限

### 关键实验信号

- **多任务统一**：单一MotionGPT在四个任务上均达到或接近各任务SOTA，证明统一建模的可行性
- **跨任务正迁移**：联合训练比单任务训练在多数任务上有提升，尤其描述任务受益于生成任务的运动理解能力
- **生成质量**：HumanML3D上FID 0.232，虽不及后续扩散方法，但在当时与T2M-GPT等自回归方法可比
- **模型规模探索**：实验了不同规模的MotionGPT（small/base/large），base效果最优——受限于运动数据规模，大模型反而过拟合

### 局限与可复用组件

- **局限**：VQ量化精度限制了运动细节；运动数据规模不足以支撑大模型；生成质量在后续扩散方法面前已不具优势；prompt设计较为简单，未充分利用LLM的推理能力
- **可复用**：运动-语言统一建模的范式为后续Motion-Agent、MotionLLM等工作奠定了基础；prompt-based多任务训练策略可迁移到其他多模态统一模型；"运动=外语"的类比为运动表示学习提供了有价值的视角

---

## 本地 PDF 引用

![[paperPDFs/Motion_Generation_Text_Speech_Music_Driven/NeurIPS_2023/2023_MotionGPT_Human_Motion_as_Foreign_Language.pdf]]
