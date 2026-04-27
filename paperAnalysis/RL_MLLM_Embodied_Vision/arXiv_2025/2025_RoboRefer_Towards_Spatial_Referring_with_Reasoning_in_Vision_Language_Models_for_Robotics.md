---
title: "RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics"
venue: NeurIPS
year: 2025
tags:
  - Embodied_AI
  - task/spatial-referring
  - reinforcement-learning
  - depth-encoder
  - process-reward
  - dataset/RefSpatial
  - dataset/RefSpatial-Bench
  - opensource/no
core_operator: 通过独立深度编码器增强3D空间感知，并用带度量敏感过程奖励的强化微调把复杂空间指令分解为逐步点定位推理。
primary_logic: |
  RGB/RGB-D观测 + 空间约束文本指令 → 先以独立深度分支和大规模空间数据完成单步空间理解SFT，再用带关键步骤奖励的RFT进行多步显式推理 → 输出满足约束的2D目标点并映射为机器人执行锚点
claims:
  - "RoboRefer-8B-SFT在CV-Bench、BLINK、RoboSpatial、SAT和EmbSpatial等单步空间理解基准上达到89.6%的平均成功率，并整体超过Gemini-2.5-Pro约5个点 [evidence: comparison]"
  - "RoboRefer-2B-RFT在RefSpatial-Bench上的平均准确率比Gemini-2.5-Pro高17.4个点，并在未见空间关系组合上比2B-SFT高9.1个点 [evidence: comparison]"
  - "在2B-RFT中加入度量敏感过程奖励可将RefSpatial-Bench成绩从48.0%提升到53.0%，说明中间步骤监督能提升多步空间指代精度 [evidence: ablation]"
related_work_position:
  extends: "NVILA (Liu et al. 2024)"
  competes_with: "SpatialRGPT (Cheng et al. 2024); RoboPoint (Yuan et al. 2024)"
  complementary_to: "SoFar (Qi et al. 2025); OpenVLA (Kim et al. 2024)"
evidence_strength: strong
pdf_ref: paperPDFs/RL_MLLM_Embodied_Vision/arXiv_2025/2025_RoboRefer_Towards_Spatial_Referring_with_Reasoning_in_Vision_Language_Models_for_Robotics.pdf
category: Embodied_AI
---

# RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2506.04308), [Project](https://zhoues.github.io/RoboRefer)
> - **Summary**: 该工作把机器人空间指代拆成“先看懂3D空间、再逐步推理约束”的两阶段问题，用独立深度分支 + 过程奖励RFT，让VLM能输出可直接用于抓取/放置/导航的目标点。
> - **Key Performance**: 单步空间理解平均成功率 **89.6%**；在 RefSpatial-Bench 上比 Gemini-2.5-Pro 平均高 **17.4 个百分点**。

> [!info] **Agent Summary**
> - **task_path**: RGB/RGB-D观测 + 空间约束指令 -> 2D目标点 -> 抓取/放置/导航执行
> - **bottleneck**: 现有VLM既缺稳定3D空间感知，又缺可泛化的多步空间推理；仅靠SFT容易记忆答案而非学会逐步解析约束
> - **mechanism_delta**: 在NVILA上加入独立深度分支做空间理解冷启动，再用带结果奖励与关键步骤过程奖励的GRPO强化多步空间指代推理
> - **evidence_signal**: 2B-RFT在RefSpatial-Bench上较Gemini-2.5-Pro高17.4个点，且在未见组合上较2B-SFT高9.1个点
> - **reusable_ops**: [dedicated-depth-branch, metric-sensitive-process-reward]
> - **failure_modes**: [depth-to-3d映射噪声会传导到执行, 面对隐含人类意图或高度歧义短指令时能力有限]
> - **open_questions**: [能否直接输出3D点或轨迹而非2D点, 如何把隐式人类意图纳入空间指代推理]

## Part I：问题与挑战

### 1）这篇论文到底在解什么问题？
论文研究的是 **spatial referring for robotics**：给定机器人观测到的 RGB 或 RGB-D 场景，以及一句带空间约束的语言指令，模型要输出一个**精确的2D点**，作为抓取点、放置点或导航目标。

这和普通 VQA / grounding 的区别在于：
- 不是回答“哪个物体更近”，而是要给出**可执行坐标**；
- 不是单一属性匹配，而是要处理**组合空间约束**；
- 不是2D图像里的“左上角框”，而是面向真实机器人，后续要通过深度映射到3D执行。

### 2）真正的瓶颈是什么？
作者把难点拆成两层，而且这两层缺一不可：

1. **单步空间理解不足**  
   VLM能看懂语义，但对距离、前后、朝向、相对远近这类3D关系并不稳定。  
   现有改法要么依赖昂贵的多视角3D重建，要么把深度当作RGB一样喂给共享编码器，容易出现**模态干扰**，反而伤到原始图像理解能力。

2. **多步空间推理缺失**  
   真正的机器人指令经常是组合式的：  
   “拿离相机最近的杯子前方的苹果，再放到离我最近的盘子和酱油碟之间。”  
   这要求模型先定位参照物，再逐步组合约束，最后输出目标点。现有数据集和评测大多只覆盖单步或最多两步，导致模型几乎没被系统训练过这类能力。

3. **SFT容易学成“答案记忆”，而不是“过程泛化”**  
   即便训练数据里带有推理链，纯SFT也可能只是把常见模式背下来；一旦空间关系换组合、推理链更长，泛化就会掉。

### 3）为什么现在值得做？
因为机器人系统已经越来越能执行动作，但前端的**空间指代解析**仍是短板。  
如果能把语言约束统一映射成目标点，很多下游系统都能复用：抓取、放置、导航、长时序操作都能共享这个接口。

### 4）输入/输出接口与边界条件
- **输入**：RGB 或 RGB-D 图像 + 空间约束语言指令
- **输出**：图像平面中的单个2D点
- **优势**：比bbox更适合机器人，遮挡时也能只指向可见且可执行的局部
- **边界**：最终落地执行仍依赖深度与相机标定，把2D点映射到3D空间；论文主要覆盖最多5步的空间推理

---

## Part II：方法与洞察

### 方法总览
RoboRefer的核心不是单一结构创新，而是把**表示、训练目标、数据分布**一起改了：

1. **结构层**：给VLM加一个独立的深度编码器  
   - RGB分支继续负责语义与图像理解  
   - Depth分支专门补3D空间线索  
   - 两者通过projector对齐到LLM  
   - 关键点：**不共享编码器**，避免深度输入污染原RGB表征

2. **训练层**：先SFT学“空间感知”，再RFT学“逐步推理”  
   - **SFT阶段**  
     - 先做 depth alignment，只训练深度projector  
     - 再全模型微调，学习单步空间理解 + 带显式推理过程的数据
   - **RFT阶段**  
     - 基于GRPO做强化微调  
     - 不只奖励最终答案是否命中，还奖励中间关键步骤的格式与精度  
     - 目标是让模型真正学会“先找谁、再算谁、最后落点”

3. **数据层**：构建一个从浅到深的空间学习配方  
   - **2D Web Images**：学基础空间概念与跨场景深度先验
   - **3D Embodied Videos**：学室内场景下更精细的度量空间关系
   - **Simulation Data**：学多步空间推理过程和关键中间步骤

其中，RefSpatial 共 **2.5M samples / 20M QA pairs**，覆盖 **31种空间关系**、最多 **5步推理**；RefSpatial-Bench 则专门评测复杂多步空间指代。

### 核心直觉

#### 直觉1：把“深度”从附属图像变成独立信息通道
**改变了什么**：从“RGB/Depth共享同一个视觉编码器”改成“独立深度编码器”。  
**改变了哪种瓶颈**：减少模态竞争，避免深度扰乱RGB已有的语义表征。  
**带来什么能力变化**：模型更稳定地获得 near/far、朝向、透视尺度等3D cues，同时不明显牺牲通用VQA能力。

#### 直觉2：把“直接答点”改成“逐步感知 + 逐步校正”
**改变了什么**：RFT不只看最终点是否命中，还对中间关键步骤施加度量敏感奖励。  
**改变了哪种瓶颈**：训练信号从单一终局监督，变成对推理链中关键感知节点的局部校正。  
**带来什么能力变化**：对未见组合关系、长链空间约束更稳，因为模型被鼓励学“过程正确”，而非只背终答案。

#### 直觉3：多步推理能力不是凭空涌现，而是要有匹配的数据分布
**改变了什么**：训练数据从普通视觉指令数据，换成覆盖 2D/3D/sim 的空间数据配方。  
**改变了哪种瓶颈**：补齐了从基础空间概念、室内度量关系到多步推理轨迹的分布缺口。  
**带来什么能力变化**：SFT阶段就能形成“空间冷启动”，RFT只需要继续把这种能力往显式推理和泛化上推进。

### 关键机制拆解

#### A. 点式表述：把空间指代统一成机器人可执行接口
作者明确不走 region/bbox 输出，而是直接预测**2D点**。  
这一步很重要，因为它把：
- 操作目标定位
- 放置区域定位
- 导航路标定位

统一成了同一种接口。  
这也是为什么 RoboRefer 更像机器人前端的“空间解析器”，而不是传统REC模型。

#### B. 两步SFT：先把深度接上，再把空间能力训起来
SFT分为两段：
- **Depth alignment**：只训练深度projector，让深度特征能进入语言空间
- **Spatial understanding enhancement**：联合RGB/RGB-D继续微调，让模型学会更强的空间理解，并从显式推理样本中得到初步的多步能力

这个设计的价值在于：  
不是粗暴把深度扔进去，而是先做对齐，再做联合学习。

#### C. RFT：过程奖励而不是只看最后坐标
RFT用了四类奖励，概念上可分为两组：
- **结果奖励**：格式正确、最终点接近GT
- **过程奖励**：中间步骤格式正确、关键感知结果准确

这里最有价值的是**metric-sensitive process reward**：  
不同感知类型用不同度量，例如位置看点误差、朝向看方向误差。  
而且它是**order-invariant** 的，不强制模型必须按某个固定顺序推理，这比死板的teacher-forcing链条更利于泛化。

### 战略权衡表

| 设计选择 | 解决的核心问题 | 能力收益 | 代价/风险 |
|---|---|---|---|
| 独立深度编码器 | 共享编码器导致RGB/Depth互相干扰 | 更强3D空间感知，同时保住通用图像理解 | 额外参数与对齐训练成本 |
| 点级输出而非bbox | 机器人执行需要精确锚点 | 更适合抓取/放置/导航，遮挡下也更实用 | 需要可靠深度与相机标定完成3D落地 |
| SFT后接RFT | 纯SFT容易背答案、泛化差 | 未见组合与长链推理更强 | RFT训练复杂，依赖较好初始化 |
| 结果奖励 + 过程奖励 | 只看终局会忽略中间错误 | 提升中间感知精度，减少链式误差累积 | 需要关键步骤标注，数据构建成本高 |
| 2D + 3D + Sim数据配方 | 单一数据源覆盖不足 | 同时获得跨场景深度先验、室内几何、显式推理能力 | 数据流水线重、再现门槛高 |

---

## Part III：证据与局限

### 关键实验信号

#### 1）比较信号：单步空间理解确实被“独立深度分支 + 空间数据”拉起来了
在 CV-Bench、BLINK、RoboSpatial、SAT、EmbSpatial 等公开基准上，RoboRefer-SFT 达到 **89.6% 平均成功率**。  
这说明它不是只会自己新 benchmark 上的题，而是在现有空间理解基准上也提升明显。

更关键的是：
- 2B-SFT 相比 NVILA-2B 有 **21.7 个点绝对提升**
- RGB-D 推理相对 RGB-only 在3D类 benchmark 上收益更明显

这直接支撑了论文第一层主张：**独立深度通道确实补上了3D空间感知短板**。

#### 2）比较信号：RFT带来的不是小修小补，而是多步推理能力跃迁
在作者提出的 RefSpatial-Bench 上，2B-RFT 比 Gemini-2.5-Pro **高17.4个点**。  
更有说服力的是 unseen 组合测试：
- 2B-SFT：33.77
- 2B-RFT：41.56

也就是 **+9.1个点**。  
这说明RFT学到的不是“更会做训练分布里的题”，而是对新组合有更强泛化。

#### 3）消融信号：过程奖励真的在修“中间步骤”
RFT消融里，去掉过程奖励后：
- RefSpatial-Bench 从 **53.0 降到 48.0**

这条证据很关键，因为它直接回答“为什么RFT有效”：  
不是因为多训了几轮，而是因为**关键步骤上的局部奖励**改善了链式推理的中间精度。

#### 4）消融信号：独立深度编码器不是装饰，而是在“增益 + 保真”之间更平衡
和共享编码器相比，独立深度编码器：
- 在 BLINK 上从 **80.02 提升到 85.26**
- 在 MME / OK-VQA / POPE 上保持相当或略优

所以作者的设计不是单纯追求空间性能，而是尽量避免“为了空间理解牺牲通用视觉能力”。

#### 5）系统信号：它可以被当作机器人系统里的空间前端
在 Open6DOR V2 position track 集成实验中：
- 相比 SoFar，成功率 **+6.8 个点**
- 执行时间 **下降27.5%**

在真实机器人实验里，模型还能在动态环境下以 **2.5 Hz** 更新目标点。  
这说明 RoboRefer 不是只在离线 benchmark 上得分高，而是真能作为机器人系统里的**定位/放置解析模块**使用。

### 能力跃迁到底体现在哪？
相较于 prior work，RoboRefer 的跳跃不只是“更懂空间”：

- **从2D grounding走向3D-aware point grounding**
- **从单步识别走向多步组合空间推理**
- **从SFT记答案走向RFT优化过程**
- **从评测分数走向真实机器人可用接口**

简言之，前人多是在提升“能不能看懂一个空间关系”；这篇论文要解决的是“能不能把多个空间约束一步步落到一个可执行点上”。

### 局限性
- **Fails when**: 深度估计噪声较大、遮挡严重或2D点难稳定映射到3D执行位姿时；指令依赖人类隐含意图而不是显式空间关系时；超出其主要训练/评测范围的更长推理链或精确量化几何任务时。
- **Assumes**: 可获得RGB-D输入，或能用 DepthAnything V2 / FoundationStereo 等外部模型估深；相机内外参与2D点到3D坐标映射可靠；存在大规模RefSpatial数据和关键步骤标注支撑SFT/RFT；受算力限制，文中RFT只训练了2B模型。
- **Not designed for**: 直接输出3D点、操作轨迹或抓取姿态；不可见目标的完整场景重建；纯低层闭环控制；显式建模人类意图歧义的语言交互。

### 复用价值高的组件
- **独立深度分支**：适合迁移到其他需要3D感知但不想破坏RGB语义能力的VLM
- **过程奖励设计**：适合任何“最终答案不足以约束中间步骤”的多步视觉推理任务
- **点式统一接口**：适合和 grasp planner、navigation policy、placement policy 做模块化拼接
- **2D/3D/Sim 三段式数据配方**：适合构建从空间概念到显式推理的课程式训练分布

### 一句话总结
RoboRefer最有价值的地方，不是“又做了一个空间VLM”，而是证明了：**机器人空间指代要同时解决3D感知与多步推理，而最有效的旋钮是“独立深度表征 + 过程监督型RFT”**。

## Local PDF reference
![[paperPDFs/RL_MLLM_Embodied_Vision/arXiv_2025/2025_RoboRefer_Towards_Spatial_Referring_with_Reasoning_in_Vision_Language_Models_for_Robotics.pdf]]