---
title: "Motion-Agent: A Conversational Framework for Human Motion Generation with LLMs"
venue: ICLR
year: 2025
tags:
  - Motion_Generation_Text_Speech_Music_Driven
  - task/text-to-motion
  - task/motion-editing
  - vq-vae
  - autoregressive
  - dataset/HumanML3D
  - dataset/KIT-ML
  - opensource/full
core_operator: MotionLLM + GPT-4 Agent框架：将运动离散化为VQ token对齐LLM词表，仅微调1-3%参数即达到扩散模型水平，再通过GPT-4多轮对话实现复杂运动组合生成
primary_logic: |
  用户多轮对话指令 → GPT-4解析为结构化运动子任务序列
  → MotionLLM（开源LLM + LoRA adapter）：运动VQ token与语言token共享词表空间
  → 自回归生成运动token → VQ解码器重建运动序列
  → 多轮对话支持：生成、编辑、理解、组合等任务通过对话链式完成
claims:
  - "MotionLLM仅微调1-3%参数（LoRA），在文本-运动生成上达到与从头训练的扩散模型和Transformer方法可比的性能"
  - "Motion-Agent通过GPT-4多轮对话实现复杂运动序列的组合生成，支持生成、编辑、理解等多任务"
  - "运动VQ token与LLM词表的对齐策略使得运动生成可以直接复用预训练语言模型的语义理解能力"
pdf_ref: paperPDFs/Motion_Generation_Text_Speech_Music_Driven/ICLR_2025/2025_Motion_Agent_A_Conversational_Framework_for_Human_Motion_Generation_with_LLMs.pdf
category: Motion_Generation_Text_Speech_Music_Driven
---

# Motion-Agent: A Conversational Framework for Human Motion Generation with LLMs

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://knoxzhao.github.io/Motion-Agent) · [ICLR 2025](https://openreview.net/forum?id=yphKDLPaFR)
> - **Summary**: Motion-Agent提出MotionLLM（开源LLM + 运动VQ token对齐 + LoRA微调1-3%参数），达到扩散模型级别的生成性能，再通过GPT-4多轮对话框架实现复杂运动序列的组合生成、编辑和理解。
> - **Key Performance**:
>   - MotionLLM仅微调1-3%参数，文本-运动生成性能与从头训练的扩散/Transformer方法可比
>   - 首次通过多轮对话实现复杂运动序列的组合生成和交互式编辑

---

## Part I：问题与挑战

### 真正的卡点

现有运动生成方法面临**训练成本高**和**任务泛化受限**两大核心挑战：

- **从头训练成本高**：扩散模型和Transformer方法需要从头训练全部参数，数据效率低，且每个任务（生成、编辑、理解）通常需要独立模型
- **复杂运动组合困难**：单次文本输入难以描述复杂的多阶段运动序列（如"先走到门口，然后开门，再坐下"），现有方法缺乏多轮交互能力
- **语义理解瓶颈**：运动生成模型的文本理解能力受限于训练数据规模，无法利用LLM已有的丰富语义知识

### 输入/输出接口

- 输入：用户自然语言对话（支持多轮）
- 输出：3D人体运动序列（HumanML3D格式）+ 运动描述文本

---

## Part II：方法与洞察

### 整体设计

Motion-Agent采用两层架构：

1. **MotionLLM（底层执行器）**：
   - 运动VQ-VAE将运动序列编码为离散token（codebook size 512）
   - 运动token直接扩展到开源LLM的词表空间（总词表~256K）
   - LoRA adapter微调1-3%参数，使LLM学会在运动token空间中自回归生成
   - 同一模型支持生成（文本→运动token）和描述（运动token→文本）

2. **GPT-4 Agent（上层规划器）**：
   - 解析用户多轮对话意图，分解为结构化子任务
   - 调用MotionLLM执行各子任务，组合结果
   - 支持交互式编辑：用户可以在对话中逐步修改生成结果

### 核心直觉

**什么变了**：从"为运动任务从头训练专用模型"到"复用预训练LLM的语义能力，仅微调极少参数适配运动模态"。

**哪些分布/约束/信息瓶颈变了**：
- 运动VQ token与LLM词表的对齐打破了运动-语言的模态壁垒 → LLM预训练获得的语义理解能力（同义词、上下位关系、组合语义等）可以直接迁移到运动生成
- LoRA微调仅调整1-3%参数 → 保留了LLM的语言能力，同时学会运动token的生成分布，训练成本大幅降低
- GPT-4作为上层规划器引入了多轮推理能力 → 复杂运动序列的组合从"一次性生成"变为"对话式逐步构建"

**为什么有效**：运动生成的核心难点之一是语义理解，而LLM已经在海量文本上学到了丰富的语义知识。通过token对齐，这些知识可以零成本迁移。GPT-4的推理能力则解决了复杂任务分解问题。

**权衡**：依赖GPT-4 API增加了推理成本和延迟；VQ离散化引入量化误差；LoRA微调的表达能力上限低于全参数训练。

---

## Part III：证据与局限

### 关键实验信号

- **参数效率**：仅微调1-3%参数，在HumanML3D上FID和R-Precision与T2M-GPT、MoMask等从头训练方法可比
- **多轮对话**：Motion-Agent成功生成了需要3-5轮对话才能完整描述的复杂运动序列，这是此前方法无法实现的
- **多任务统一**：同一MotionLLM支持生成、描述、编辑，无需切换模型

### 局限与可复用组件

- **局限**：依赖GPT-4 API，成本和延迟不可控；VQ量化精度限制了细粒度运动质量；多轮对话的运动拼接可能产生不连续；未在大规模数据上验证Scaling行为
- **可复用**：运动VQ token与LLM词表对齐的策略可迁移到其他时序信号（手势、面部表情等）；LoRA微调范式为任何新模态接入LLM提供了低成本路径；GPT-4 Agent的多轮规划框架可用于任何需要复杂任务分解的生成场景

---

## 本地 PDF 引用

![[paperPDFs/Motion_Generation_Text_Speech_Music_Driven/ICLR_2025/2025_Motion_Agent_A_Conversational_Framework_for_Human_Motion_Generation_with_LLMs.pdf]]
