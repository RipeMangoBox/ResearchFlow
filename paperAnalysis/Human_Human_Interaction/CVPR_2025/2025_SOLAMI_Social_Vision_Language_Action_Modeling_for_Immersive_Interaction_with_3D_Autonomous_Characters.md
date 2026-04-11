---
title: "SOLAMI: Social Vision-Language-Action Modeling for Immersive Interaction with 3D Autonomous Characters"
venue: CVPR
year: 2025
tags:
  - Human_Human_Interaction
  - task/social-interaction
  - task/motion-generation
  - diffusion
  - dataset/SynMSI
  - repr/SMPL-X
  - opensource/partial
core_operator: 端到端社交视觉-语言-动作模型（Social VLA）：将用户语音+身体语言感知、语言理解、3D角色动作生成统一为单一多模态模型，实现沉浸式VR社交交互
primary_logic: |
  用户语音 + 用户身体运动（VR感知）
  → 多模态编码器（语音→语义token + 运动→运动token）
  → 统一Transformer解码：联合推理语言回复 + 角色反应运动
  → 合成多模态数据集SynMSI提供训练监督（语音-运动-语言三模态对齐）
  → 3D自主角色在VR中实时感知、理解、交互
claims:
  - "SOLAMI是首个将语音感知、语言理解和3D运动生成统一为端到端Social VLA模型的框架，实现沉浸式VR社交交互"
  - "合成数据集SynMSI通过多阶段流水线生成高质量语音-运动-语言三模态对齐数据，有效缓解真实社交交互数据稀缺问题"
  - "在VR交互场景中，SOLAMI生成的角色反应在语义一致性和运动自然度上显著优于pipeline拼接基线"
pdf_ref: paperPDFs/Motion_Generation_Text_Speech_Music_Driven/CVPR_2025/2025_SOLAMI_Social_Vision_Language_Action_Modeling_for_Immersive_Interaction_with_3D_Autonomous_Characters.pdf
category: Human_Human_Interaction
---

# SOLAMI: Social Vision-Language-Action Modeling for Immersive Interaction with 3D Autonomous Characters

> [!abstract] **Quick Links & TL;DR**
>
> - **Links**: [Project Page](https://solami-ai.github.io/) · [CVPR 2025](https://arxiv.org/abs/2412.00174)
> - **Summary**: SOLAMI提出端到端Social VLA模型，将语音感知、语言理解和3D角色运动生成统一为单一多模态框架，配合合成数据集SynMSI，首次实现用户在VR中通过语音和身体语言与3D自主角色的沉浸式社交交互。
> - **Key Performance**:
>   - 首个端到端Social VLA框架，统一语音-语言-运动三模态
>   - 在VR交互场景中，角色反应的语义一致性和运动自然度显著优于pipeline拼接基线

---

## Part I：问题与挑战

### 真正的卡点

赋予3D自主角色社交智能面临**多模态统一建模**和**社交交互数据稀缺**两大核心挑战：

- **模态割裂**：现有方法将语音理解、语言生成、运动生成作为独立模块串联，模块间信息损失导致角色反应缺乏语义-动作一致性（如说"你好"时身体没有相应的招手动作）
- **社交交互数据极度稀缺**：真实的语音-身体运动-语言三模态对齐的社交交互数据几乎不存在，MoCap采集成本极高且场景受限
- **实时性要求**：VR沉浸式交互要求角色在感知用户输入后快速响应，端到端模型需要在推理延迟和生成质量间取得平衡

### 输入/输出接口

- 输入：用户语音（麦克风）+ 用户身体运动（VR追踪）
- 输出：角色语音回复 + 角色3D身体运动反应（SMPL-X格式）

---

## Part II：方法与洞察

### 整体设计

SOLAMI的核心是将社交交互建模为端到端的多模态序列生成问题：

1. **多模态token化**：语音通过语音编码器转为语义token，身体运动通过运动VQ-VAE转为离散运动token，语言文本直接使用LLM tokenizer
2. **统一Transformer解码**：所有模态token在同一序列空间中，Transformer联合推理生成角色的语言回复token和运动反应token
3. **SynMSI合成数据集**：多阶段流水线生成训练数据——先用LLM生成对话脚本，再用TTS合成语音，最后用运动生成模型合成对应身体动作，形成三模态对齐的社交交互数据

### 核心直觉

**什么变了**：从"语音理解→语言生成→运动生成"的串联pipeline到"感知-理解-行动"的端到端统一模型。

**哪些分布/约束/信息瓶颈变了**：
- 端到端建模消除了模块间的信息瓶颈 → 语义理解可以直接影响运动生成（"我很开心"→ 同时生成积极语言回复和欢快身体动作），而非通过中间文本传递
- SynMSI合成数据打破了真实社交交互数据的稀缺瓶颈 → 模型可以在大规模合成数据上学习语音-运动-语言的联合分布
- 统一token空间使得跨模态注意力自然发生 → 角色的语言和动作在生成时就是协调的，而非事后对齐

**为什么有效**：社交交互的本质是多模态信号的同步协调，端到端模型天然适合这种联合生成需求。合成数据虽然不如真实数据，但提供了足够的分布覆盖让模型学到跨模态关联的基本模式。

**权衡**：合成数据与真实交互存在分布差距；端到端模型的可控性不如pipeline（难以单独调整某个模态）；VR部署的计算资源需求较高。

---

## Part III：证据与局限

### 关键实验信号

- **端到端优势**：与pipeline基线（ASR→LLM→TTS+Motion Gen）相比，SOLAMI在语义一致性（语言回复与运动反应的匹配度）上显著提升
- **运动自然度**：生成的角色运动在用户研究中被评为更自然、更符合社交情境
- **SynMSI有效性**：在合成数据上训练的模型可以泛化到真实VR交互场景，证明合成流水线的有效性

### 局限与可复用组件

- **局限**：合成数据的多样性和真实性仍有提升空间；当前仅支持双人交互，多人场景未探索；VR部署的延迟和计算成本较高；运动生成质量受限于底层运动VQ-VAE的表示能力
- **可复用**：Social VLA的端到端多模态框架可迁移到其他人机交互场景（机器人、虚拟助手等）；SynMSI的合成数据流水线可用于任何缺乏真实多模态交互数据的领域；多模态token统一的设计模式适用于任何需要跨模态联合生成的任务

---

## 本地 PDF 引用

![[paperPDFs/Motion_Generation_Text_Speech_Music_Driven/CVPR_2025/2025_SOLAMI_Social_Vision_Language_Action_Modeling_for_Immersive_Interaction_with_3D_Autonomous_Characters.pdf]]
