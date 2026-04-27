---
title: "RoboBrain 2.0 Technical Report"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/video-understanding
  - task/visual-question-answering
  - reinforcement-learning
  - chain-of-thought
  - dataset/EgoPlan2
  - dataset/Where2Place
  - opensource/full
core_operator: 通过异构视觉编码器与语言解码器、具身时空数据构造和三阶段训练，把感知、推理、规划统一到同一序列解码接口中
primary_logic: |
  多视角图像/视频/场景图/语言指令 → 统一token化与视觉投影、分阶段具身时空训练及CoT+RL后训练 → 输出文本计划、空间坐标与交互决策
claims:
  - "RoboBrain-2.0-32B在RoboSpatial(72.43)、RefSpatial-Bench(54.00)、Where2Place(73.59)、SAT(86.67)和EgoPlan2(57.23)上取得报告中最高分，而7B版本在BLINK(83.95)、CV-Bench(85.75)、Multi-Robot Planning(81.50)和RoboBench Planning(72.16)上排名第一 [evidence: comparison]"
  - "该系统把多图像、长视频、scene graph与自然语言统一为单一解码序列，并可直接生成自由文本、点/框/轨迹坐标及可选推理链 [evidence: analysis]"
  - "报告中的混合比特推理方案保持视觉编码器全精度、将语言模块权重量化为INT8，使端到端具身任务推理延迟约降低30%，且报告称精度影响可忽略 [evidence: analysis]"
related_work_position:
  extends: "RoboBrain (Ji et al. 2025)"
  competes_with: "VeBrain-8B; Magma"
  complementary_to: "RoboOS (Tan et al. 2025)"
evidence_strength: moderate
pdf_ref: paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_RoboBrain_2_0_Technical_Report.pdf
category: Embodied_AI
---

# RoboBrain 2.0 Technical Report

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2507.02029), [Project / Code / Checkpoints / Benchmark](https://superrobobrain.github.io)
> - **Summary**: 这是一份具身VLM系统报告：作者用异构视觉-语言架构、空间/时间数据合成和三阶段训练，把机器人所需的空间定位、长期规划与交互式推理统一进一个可输出文本与坐标的模型接口。
> - **Key Performance**: 32B在Where2Place达到73.59、EgoPlan2达到57.23；7B在RoboBench Planning达到72.16

> [!info] **Agent Summary**
> - **task_path**: 多视角图像/视频 + 语言指令 + scene graph -> 文本计划 / 空间坐标 / 推理链
> - **bottleneck**: 现有通用VLM缺少可执行的空间grounding、跨步骤时序建模和具身任务所需的因果推理链
> - **mechanism_delta**: 在Qwen2.5-VL初始化骨架上加入scene graph输入、具身空间/时间合成数据，以及SFT→CoT-SFT→RFT的三阶段训练
> - **evidence_signal**: 跨空间与时间基准的系统性对比显示其在机器人导向的空间指向、放置、轨迹与长期规划任务上明显优于多数开源/闭源基线
> - **reusable_ops**: [scene-graph-serialization, coordinate-decoding, staged-spatiotemporal-curriculum]
> - **failure_modes**: [7B在EgoPlan2这类长时日常计划任务上明显掉点, 32B在VSI-Bench与EmbSpatial上并非最佳]
> - **open_questions**: [如何把高层规划稳定接到低层VLA控制, 合成与教师生成CoT数据在真实机器人分布外是否稳健]

## Part I：问题与挑战

这篇报告要解决的，不是“再做一个更强的图像问答模型”，而是把通用多模态模型变成**能服务真实具身任务的高层认知模块**。

### 1. 真正的问题是什么
作者明确指出三类核心瓶颈：

1. **空间理解不够可执行**  
   传统VLM常能“描述关系”，但机器人需要的是更强的空间输出：点、框、轨迹、可放置区域、目标相对位置等。  
   也就是说，问题从“看懂”变成了“给出可以落到控制器上的空间表示”。

2. **时间建模不够长程**  
   真实任务不是单轮问答，而是带历史、带反馈、甚至带多智能体协作的长期过程。  
   瓶颈不在于识别单帧，而在于能否把“过去观察—当前状态—下一步决策”串起来。

3. **推理链不够贴合具身约束**  
   机器人指令往往包含目标、约束、空间关系、动作顺序和失败恢复。  
   通用VLM即使会CoT，也未必会把这些约束变成具身任务中的因果链。

### 2. 为什么是现在
因为几个条件同时成熟了：

- 开源/闭源VLM骨干已经足够强，可以继承通用视觉语言能力；
- 具身评测开始细分到**spatial referring、affordance、trajectory、multi-robot planning**；
- 可以借助外部模型、模拟器与3D工具链，批量合成稀缺的空间/时间监督数据。

### 3. 输入/输出接口与边界
**输入**：
- 多视角图像
- 长视频帧
- 自然语言指令
- scene graph（结构化环境状态）

**输出**：
- 自由文本计划/解释
- 点、框、轨迹等空间坐标
- 可选的推理链

**边界条件**：
- 它更像机器人系统里的“脑”和“规划层”，不是直接输出底层电机控制的VLA policy。
- 它依赖下游控制器、机器人OS、甚至外部scene graph维护模块来真正执行。

---

## Part II：方法与洞察

### 1. 系统栈：把具身任务改写成统一序列建模问题

RoboBrain 2.0采用一个典型但面向具身任务强化过的异构架构：

- **视觉侧**：约689M参数的vision encoder，支持高分辨率、多图像、视频、动态分辨率。
- **语言侧**：7B/32B decoder-only LLM，初始化自Qwen2.5-VL。
- **跨模态桥接**：MLP projector把视觉特征投到语言token空间。
- **时空标记**：视频用timestamp token，多视角输入用视角标识。
- **结构化状态输入**：scene graph直接串成token流，与视觉和语言一起送入decoder。

这一步的意义在于：  
作者没有把“感知、规划、空间输出”拆成多个专家模块，而是尽量把它们压到**同一个可生成接口**里。

### 2. 数据与训练配方：重点不是架构新，而是监督分布变了

报告里的真正大改动更像是**数据分布重构**：

- **通用MLLM数据**：保留一般VQA/对话/OCR等基础能力；
- **空间数据**：visual grounding、object pointing、affordance、3D spatial understanding、spatial referring；
- **时间数据**：egocentric planning、ShareRobot、AgiBot planning、multi-robot planning、close-loop interaction。

尤其关键的是两类增强：

- **空间监督从“描述”升级为“坐标/落点/轨迹”**
- **时间监督从“看视频回答”升级为“带历史、带反馈、带多智能体协作”的长期决策**

训练采用三阶段：

1. **Stage 1: Foundational Spatiotemporal Learning**  
   先保留基础视觉语言理解，并获得基础空间/时间感知。

2. **Stage 2: Embodied Spatiotemporal Enhancement**  
   再灌入高分辨率、多视角、egocentric、scene-graph、multi-agent等具身样本，使模型更像“机器人场景模型”。

3. **Stage 3: Chain-of-Thought Reasoning in Embodied Contexts**  
   先做CoT-SFT，再做基于GRPO的RFT，奖励信号主要针对**答案正确性与格式正确性**。

### 3. 工程侧：这不是只追指标的模型，而是带部署意识的系统报告

报告在工程实现上给了很多系统细节，这也是它作为 technical report 的价值：

- **FlagScale训练框架**
  - 非均匀pipeline并行
  - 仅ViT侧重计算重算
  - 预分配显存缓解碎片化
  - 分布式数据加载与故障恢复

- **预处理优化**
  - 只预处理JSON、不重编码图片
  - 报告称预处理时间可下降约90%

- **RL后训练基础设施**
  - 使用VeRL执行RLVR/GRPO流程

- **推理部署**
  - 视觉编码器保留全精度
  - 语言模块做INT8权重量化
  - 报告称端到端延迟下降约30%

### 核心直觉

**这篇报告真正拧动的旋钮，不是“换了一个更神奇的模型结构”，而是：把机器人任务的输入状态和输出形式都改造成LLM真正能学到的形式。**

更具体地说：

- **What changed**  
  从“图像+问题→文本答案”，变成“多视角/视频/scene graph/指令→文本计划+空间坐标+推理链”。

- **Which bottleneck changed**  
  改变的是监督分布和接口约束：  
  过去模型主要学静态VQA；现在模型被迫学习**空间落点、跨时序依赖、闭环反馈、多agent协作**。

- **What capability changed**  
  能力从“会说”更接近“会为机器人给出可执行高层决策”。

**为什么这套设计有效：**

1. **scene graph降低状态压缩损失**  
   纯视觉输入对长期任务不友好，结构化环境状态让模型不用全靠隐式记忆去维护世界模型。

2. **坐标输出让监督直接对齐执行接口**  
   相比只输出文字，“点/框/轨迹”更接近机器人下游控制器需要的表示。

3. **课程式训练降低能力冲突**  
   先学通用感知，再学具身时空，再学CoT/RL，避免一开始就让模型在高难度具身分布上不稳定。

4. **CoT + RFT让格式和推理更稳定**  
   对具身任务来说，答案不仅要“对”，还要“能被解析与执行”。

### 战略取舍

| 设计选择 | 带来的能力 | 代价 / 风险 |
| --- | --- | --- |
| 异构视觉编码器 + LLM解码器 | 能复用强语言推理能力，并支持多图像/长视频输入 | 视觉-语言对齐主要经MLP projector，跨模态瓶颈仍可能存在 |
| scene graph作为显式输入 | 降低长期任务中的状态遗忘，利于规划与多轮交互 | 依赖外部环境建模质量，现实部署中scene graph维护不免费 |
| 大量合成空间/时间数据 | 快速补齐稀缺具身监督，尤其是坐标/放置/多机器人协作 | 会引入教师偏差、模拟器偏差与模板偏差 |
| CoT-SFT + RFT | 提升复杂任务下的推理链与输出格式稳定性 | 训练流程更复杂，且增益来源难与数据构造解耦 |
| 7B/32B双版本 | 兼顾部署与性能 | 大模型并非处处更强，说明规模不是唯一决定因素 |
| 混合比特推理 | 降低延迟，面向部署更实际 | 精度敏感模块需谨慎保留高精度，量化收益与硬件强相关 |

---

## Part III：证据与局限

### 1. 关键证据：能力跃迁发生在哪

**信号A：机器人导向的空间任务提升最明显。**  
最强证据不是一般VQA，而是那些输出必须“可执行”的空间基准：

- **RefSpatial-Bench**：32B达到 **54.00**，显著高于表中多数通用VLM；
- **Where2Place**：32B达到 **73.59**，比通用基线高出一大截；
- **RoboSpatial**：32B达到 **72.43**，也明显领先。

**结论**：  
这说明它最显著的能力跳跃，不是“更会看图说话”，而是**更会给机器人做空间落点与放置决策**。

---

**信号B：时间能力提升主要体现在“具身规划”，而不是所有时序任务都同步提升。**

- **Multi-Robot Planning**：7B达到 **81.50**、32B达到 **80.33**，都强于多数对比模型；
- **EgoPlan2**：32B达到 **57.23**，优于表中基线，但7B只有 **33.23**；
- **RoboBench Planning**：7B达到 **72.16**，反而优于32B。

**结论**：  
它对“机器人规划分布”很有效，但**规模扩大不自动等于时序能力全面提升**。  
说明模型收益更可能来自训练分布和任务对齐，而非单纯参数量。

---

**信号C：系统报告不仅报能力，还报部署收益。**

- 混合比特量化带来约 **30%** 推理延迟下降；
- 训练与预处理管线都围绕吞吐、显存和故障恢复做了专门设计。

**结论**：  
这是一个带明显工程落地取向的具身系统报告，而不是只在benchmark上堆指标。

### 2. 证据强度为什么只是 moderate
虽然报告覆盖了很多基准，但证据仍然偏**comparison-heavy**：

- 有大量对比表；
- 但缺少系统性ablation，无法清楚拆开：
  - 架构贡献有多少，
  - 数据构造贡献有多少，
  - CoT/RFT贡献有多少，
  - scene graph输入到底贡献多大。

所以它证明了“系统整体有效”，但还没有充分证明“每个关键设计的因果贡献”。

### 3. 局限性

- **Fails when**: 任务更依赖通用人类日常活动先验或抽象视觉-空间整合、而非机器人导向的坐标/规划输出时，优势不稳定；例如7B在EgoPlan2明显落后，32B在VSI-Bench与EmbSpatial上也不是最强。
- **Assumes**: 依赖大规模合成与教师生成数据，以及多个外部模型/工具链（如GPT-4o、DeepSeek-V3、GroundingDINO、SAM 2.1、UniDepth V2、Qwen2.5-VL）；同时依赖长上下文和大规模GPU训练基础设施，这些都会影响严格复现。
- **Not designed for**: 直接低层动作控制、形式化安全保证、或无外部控制器/机器人OS支持的端到端实时自治执行；此外，报告没有给出系统级安全边界与风险缓解的完整披露。

补充一点：作者在Where2Place上提到对测试集错误样本做了人工筛除，这意味着**复现实验时必须严格对齐评测协议**。

### 4. 可复用组件

这篇报告里最值得迁移的，不一定是整模，而是这些“操作符”：

- **scene graph 序列化输入**：把环境状态显式送进LM，而不是全靠视觉记忆。
- **坐标化输出接口**：把point / bbox / trajectory当成统一生成目标。
- **具身课程训练**：通用VLM能力 → 具身时空能力 → CoT/RL后训练。
- **工程技巧**：ViT侧重算、显存预分配、JSON-only预处理、混合比特推理。

---

## Local PDF reference

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_RoboBrain_2_0_Technical_Report.pdf]]