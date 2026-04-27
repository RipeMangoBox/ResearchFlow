---
title: "GraspMolmo: Generalizable Task-Oriented Grasping via Large-Scale Synthetic Data Generation"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/task-oriented-grasping
  - synthetic-data-generation
  - point-prediction
  - grasp-proposal-matching
  - dataset/PRISM
  - dataset/TaskGrasp-Image
  - opensource/no
core_operator: 用“抓取描述”把任务语义与候选抓取桥接起来，在大规模合成场景上微调 Molmo 预测抓取点，再回配到稳定 6-DoF 抓取。
primary_logic: |
  自然语言任务指令 + 单帧 RGB-D → 用 PRISM 学习“任务语义-物体部位-抓取方式”的对应关系，并由 VLM 输出图像抓取点/抓取描述 → 在候选稳定抓取集合中做像素空间最近匹配 → 输出符合任务语义的 6-DoF 稳定抓取
claims:
  - "GraspMolmo 在 TaskGrasp-Image 上达到 76.7% top-1 抓取预测准确率，高于 GraspGPT 的 72.3% [evidence: comparison]"
  - "在更复杂的 PRISM-Test 上，GraspMolmo 达到 62.5%，明显高于 Molmo 的 49.8% 和 GraspGPT 的 40.0% [evidence: comparison]"
  - "在真实机器人 PRISM-Real 上，GraspMolmo 达到 70.4% 抓取预测成功率和 61.1% 整体执行成功率，且相对 Molmo/GraspGPT 的提升经配对 t 检验达到 p < 0.05 [evidence: comparison]"
related_work_position:
  extends: "Molmo (Deitke et al. 2024)"
  competes_with: "GraspGPT (Tang et al. 2023); FoundationGrasp (Tang et al. 2024)"
  complementary_to: "M2T2 (Yuan et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_GraspMolmo_Generalizable_Task_Oriented_Grasping_via_Large_Scale_Synthetic_Data_Generation.pdf
category: Embodied_AI
---

# GraspMolmo: Generalizable Task-Oriented Grasping via Large-Scale Synthetic Data Generation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2505.13441)
> - **Summary**: 论文通过构建大规模合成任务导向抓取数据集 PRISM，并将 Molmo 微调为“看单帧 RGB-D + 听自然语言指令就能指出该抓哪里”的模型，显著提升了开放词汇任务导向抓取在复杂场景和真实机器人上的泛化能力。
> - **Key Performance**: PRISM-Real 上抓取预测成功率 70.4%、整体执行成功率 61.1%；PRISM-Test 上 62.5%，明显高于 GraspGPT 的 40.0%

> [!info] **Agent Summary**
> - **task_path**: 单帧 RGB-D + 开放词汇任务指令 -> 任务语义正确的 6-DoF 抓取
> - **bottleneck**: 缺少大规模、复杂场景、自然语言驱动的任务-抓取对应数据，导致模型只会“稳定抓住”，不会“按任务抓对部位”
> - **mechanism_delta**: 用“抓取描述”把任务与抓取解耦生成可扩展标注，再把 Molmo 微调为抓取点预测器并回配到稳定抓取候选
> - **evidence_signal**: 真实机器人 PRISM-Real 上 70.4% prediction / 61.1% overall，较 Molmo/GraspGPT 显著更高
> - **reusable_ops**: [抓取描述桥接标注, 像素空间候选抓取匹配]
> - **failure_modes**: [抓对物体但抓错部位, 指向错误物体, 点到正确区域但匹配到错误候选抓取]
> - **open_questions**: [能否去掉外部抓取提议器直接输出稳定 6-DoF 抓取, 能否把零样本双臂提示扩展成可系统评测的双臂抓取能力]

## Part I：问题与挑战

这篇论文解决的不是传统“能不能抓稳”问题，而是更难的 **任务导向抓取（Task-Oriented Grasping, TOG）**：  
同一个物体，因为任务不同，正确抓法会完全不同。比如茶壶“倒茶”应抓手柄，而不是壶身；刀具“安全递给别人”和“切菜”需要抓不同部位。

### 真正的瓶颈是什么？

核心瓶颈是：**语义监督稀缺，而不是抓取几何本身不够成熟。**

现有 6-DoF 抓取器已经能给出大量“稳定抓取”，但它们通常不知道：
- 哪个部位最符合任务语义；
- 自然语言中的细粒度意图如何映射到抓取 affordance；
- 在遮挡、杂乱、多物体场景里，应该抓哪个物体、哪个部位。

作者认为，先前 TOG 方法的主要限制在于数据：
1. **数据规模小**：难以覆盖“任务 × 物体 × 抓法”的组合爆炸；
2. **语言过于模板化**：像 “grasp the mug to pour” 这种短模板，离真实家庭指令很远；
3. **场景过于简单**：少遮挡、少干扰物，难以迁移到真实家居；
4. **部署假设过强**：一些方法依赖预分割点云、多视角输入、顶视相机，真实应用不友好。

### 输入/输出接口

- **输入**：单帧 RGB-D + 自然语言任务指令
- **输出**：一个与任务语义匹配的 6-DoF 抓取（实际流程里先输出图像抓取点，再映射到候选稳定抓取）

### 为什么现在值得做？

因为两个条件同时成熟了：
- **VLM 的空间指向能力成熟**：Molmo 这类模型已经擅长 point grounding 和空间理解；
- **大规模合成数据生成变得可行**：程序化场景生成 + LLM 语义生成 + 人工校验，使复杂任务语义标注终于能扩到足够规模。

换句话说，这篇论文抓住的是一个时间点：**几何抓取能力已经够强，但语义抓取监督还没跟上。**

---

## Part II：方法与洞察

作者的方法可以概括成一句话：

> 不让模型从头学“完整抓取控制”，而是把问题拆成  
> **“语义上该抓哪里”** 和 **“几何上怎样稳定抓”** 两部分分别解决。

### 方法骨架

#### 1. 构建 PRISM：大规模任务导向抓取合成数据

PRISM 的规模：
- 10,000 个场景
- 2,356 个物体实例
- 91 个物体类别
- 378,844 个样本
- 场景来自程序化组合，带复杂纹理、遮挡、视角和光照变化

但真正关键的不只是“合成”，而是它的 **标注分解方式**。

#### 2. 用“抓取描述”作为任务与抓取之间的桥

直接给每个 `任务-物体-抓取` 三元组做标注，成本会爆炸。  
作者的关键技巧是把它拆成两端：

- 对每个 `物体-抓取`，生成一个**抓取描述**  
  例如：“抓住茶壶把手，夹爪从两侧夹持”
- 对每个任务，生成一个“为了完成这个任务应如何抓”的**任务侧抓取描述**
- 然后用文本匹配把二者连接起来

这相当于把原本耦合的三元组标注，变成：
- 一边标“抓法是什么”
- 一边标“任务需要什么抓法”
- 最后再做匹配

这一步是整个工作最核心的扩展性来源。

#### 3. LLM 生成 + 人工校验，保证数据规模与质量平衡

PRISM 的生成流程大致是：
- 用 ShapeNet-Sem + ACRONYM 生成物体与候选抓取
- 用 GPT-4o 看多视角抓取图，生成抓取描述
- 用人工在 Prolific 上校验/修正这些抓取描述
- 用 GPT-4.1 为每类物体生成多样任务
- 再用 GPT-4o 做任务描述与抓取描述的匹配

一个很重要的信号是：  
**只有 45% 的抓取描述初始可直接通过，55% 需要人工编辑。**

这说明作者并没有把 LLM 标注当成“天然干净数据”，而是明确承认闭源模型会出错，并用人工过滤补上质量缺口。

#### 4. 把 Molmo 微调成任务条件抓取点预测器

GraspMolmo 不是直接回归完整 6-DoF 抓取姿态。  
它做的是：
- 输入：RGB-D + 任务文本
- 输出：图像上的一个抓取点，以及对应的抓取描述

作者还把自然语言抓取描述当成一种中间 reasoning scaffold，让模型不仅“点出来”，还要“说出来”。

#### 5. 从图像点回配到稳定抓取候选

由于真实执行仍需要稳定的 6-DoF 抓取，作者没有让 VLM 直接负责所有几何细节，而是：
- 先由稳定抓取生成器给出候选抓取；
- 再把每个候选抓取映射到图像平面；
- 选择与 VLM 预测点最近的候选抓取。

而且作者专门强调：  
他们不是简单做深度回投影，而是 **在像素空间里匹配**，因为：
- 深度图可能缺失；
- 在物体边界，像素小误差会放大成 3D 大误差。

这是一种很实用的系统设计：**把语义 grounding 留给 VLM，把抓稳这件事留给 grasp proposer。**

### 核心直觉

#### 直觉 1：先解决“语义监督分布”问题，模型才可能学会任务导向抓取

**改变了什么**：  
从小规模、模板化、简单场景的 TOG 数据，变成大规模、开放词汇、复杂杂乱场景的任务-抓取对应数据。

**改变了哪个瓶颈**：  
把原来的监督瓶颈从“几乎学不到任务语义差异”，变成“可以反复看到同一类任务意图如何对应不同物体部位”。

**带来了什么能力变化**：  
模型不再只会找“可抓区域”，而开始会找“为这个任务应抓的区域”。

#### 直觉 2：把“抓哪儿”与“怎么稳定抓”解耦，能更好利用基础模型能力

**改变了什么**：  
输出空间从完整 6-DoF 抓取姿态，改成图像点定位 + 候选抓取重排序。

**改变了哪个瓶颈**：  
把高维几何控制难题，变成 VLM 更擅长的语言条件空间指向问题。

**带来了什么能力变化**：  
模型更容易从自然语言任务学到部位级 affordance，同时仍保留稳定抓取执行能力。

#### 直觉 3：合成数据不是单独使用，而是与真实图像共训

**改变了什么**：  
PRISM 不是纯模拟闭环训练；作者将 PRISM 与 PixMo、TaskGrasp-Image 混合训练。

**改变了哪个瓶颈**：  
减轻“语义学到了、视觉域却过拟合模拟渲染”的 sim-to-real 问题。

**带来了什么能力变化**：  
在未见物体、真实家庭场景上的迁移能力更强。

### 为什么这个设计有效？

因为它把不同模块放在各自最擅长的位置：
- **LLM**：生成任务与抓取描述，提供常识语义扩展
- **VLM**：从图像和语言中找出“该抓哪里”
- **grasp proposer**：保证候选抓取在物理上稳定
- **真实图像共训**：补视觉域差

这不是“一个模型做完全部事情”，而是一个典型的 **能力解耦系统**。

### 战略权衡

| 设计选择 | 解决的瓶颈 | 获得的能力 | 代价/风险 |
|---|---|---|---|
| 抓取描述桥接任务与抓取 | 三元组标注组合爆炸 | 能扩展到大规模开放词汇任务数据 | 依赖 GPT-4o / GPT-4.1 与人工校验，存在标注噪声与成本 |
| 图像点预测 + 候选抓取匹配 | 直接学 6-DoF 太难、太依赖 embodiment | 单帧 RGB-D 即可做任务语义抓取 | 仍依赖外部 grasp proposer，旋转细节受限 |
| 像素空间回配而非 3D 回投影 | 深度缺失与边界误差 | 更稳健地把语义点映射到执行抓取 | 若候选抓取覆盖不足，语义再准也无法执行 |
| PRISM 与真实图像共训 | 纯模拟训练迁移差 | 零样本 real transfer 更可行 | 训练 recipe 更复杂，计算开销更高 |

---

## Part III：证据与局限

### 关键证据链

#### 1. 在简单 benchmark 上只是小幅领先，但在复杂场景上优势显著拉大

- **TaskGrasp-Image**：76.7%，略高于 GraspGPT 的 72.3%
- **PRISM-Test**：62.5%，明显高于 Molmo 的 49.8% 与 GraspGPT 的 40.0%

这说明它的主要增益，不是“在老 benchmark 上多挤出几个点”，而是 **在复杂任务语义、复杂视觉场景下更稳**。

#### 2. 真实机器人零样本迁移是最强证据

在 PRISM-Real 上：
- **prediction success**：70.4%
- **overall success**：61.1%

相比之下：
- GraspGPT：35.2% / 24.1%
- Molmo：35.2% / 33.3%

而且论文报告相对 Molmo 和 GraspGPT 的提升具有统计显著性（paired t-test, p < 0.05）。  
这直接支撑了论文最重要的 claim：**合成任务语义数据确实转化成了真实场景能力**。

#### 3. PRISM-Test 比传统 benchmark 更能预测真实部署效果

作者给出一个很有价值的分析：
- TaskGrasp-Image 与真实效果的相关性：R² = 0.72
- PRISM-Test 与真实效果的相关性：R² = 0.96

这意味着：  
**如果你的目标是部署到真实复杂环境，那么旧式简单 benchmark 不够“诊断型”。**

#### 4. 双臂能力目前只是可行性展示，不是已证实结论

论文展示了零样本双臂语义抓取的定性案例，但没有系统 benchmark。  
所以更合理的理解是：**架构有扩展潜力**，而不是“双臂任务已被充分验证”。

### 1-2 个最值得记的指标

- **PRISM-Real**：70.4% prediction / 61.1% overall
- **PRISM-Test**：62.5%，较 GraspGPT 的 40.0% 有明显优势

### 局限性

- **Fails when**: 场景中存在多个相似物体或多个相似功能部位、需要很细粒度的旋转调整、或候选抓取集合覆盖不足时会失败；作者在真实场景中统计到的主要失败是抓对物体但抓错部位（62%）、选错物体（24%）、点对但回配到错误候选抓取（5%）。
- **Assumes**: 需要外部稳定抓取提议器（文中真实实验用 M2T2）、后续运动规划与控制器、单帧 RGB-D 感知；训练数据构建依赖 GPT-4o / GPT-4.1 与人工校验，训练本身使用 64×H100 跑 10k steps，复现成本不低。
- **Not designed for**: 端到端机器人控制、长时序操作规划、无候选集条件下直接生成稳定 6-DoF 抓取、严谨评测的双臂操作系统；双臂部分目前只是 prompt-based 的零样本展示。

### 复现与扩展时需要特别注意的依赖

1. **数据生成依赖闭源 API**：PRISM 的关键步骤用到 GPT-4o / GPT-4.1。  
2. **数据质量依赖人工过滤**：55% 抓取描述需要人工改写，说明“纯自动生成”大概率不够干净。  
3. **推理仍依赖 grasp proposer**：论文确实去掉了 GraspGPT 那类额外的 SAM2、GroundingDINO、GPT-4o 推理链路，但没有去掉稳定抓取候选生成器。  
4. **开源状态无法由文内链接直接核验**：论文声称将发布数据、模型、代码和 benchmark，但提供文本中未包含可核验链接，因此这里保守记为 `opensource/no`。

### 可复用组件

- **抓取描述桥接标注范式**：适合所有“任务语义 × 物体 affordance”组合爆炸的问题
- **PRISM 数据生成管线**：可迁移到 task-semantic manipulation 的其他子任务
- **像素空间候选抓取匹配**：适合把 VLM 指点能力接到几何执行模块上
- **TaskGrasp-Image**：把点云式 benchmark 转成图像输入 benchmark 的思路很有实用价值

## Local PDF reference

![[paperPDFs/Vision_Action_VA_Models_2025/arXiv_2025/2025_GraspMolmo_Generalizable_Task_Oriented_Grasping_via_Large_Scale_Synthetic_Data_Generation.pdf]]