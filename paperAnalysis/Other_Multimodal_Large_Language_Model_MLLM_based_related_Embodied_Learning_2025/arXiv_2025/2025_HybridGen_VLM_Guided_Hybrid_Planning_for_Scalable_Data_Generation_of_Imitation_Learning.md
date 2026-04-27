---
title: "HybridGen: VLM-Guided Hybrid Planning for Scalable Data Generation of Imitation Learning"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/imitation-learning
  - task/robot-manipulation
  - hybrid-planning
  - keypoint-constraints
  - pose-transformation
  - dataset/Robomimic
  - opensource/no
core_operator: 用VLM把示范拆成“专家精细段”和“可规划段”，对前者做对象中心位姿迁移、对后者做语义约束重规划，再用抓取物相对目标物的位姿关系做二次扩增
primary_logic: |
  少量人类示范视频/轨迹 + 任务描述 + 场景RGB
  → VLM分解子任务并识别专家依赖段，CLIP提取关键点、Gemini生成时空约束
  → 专家段做位姿适配，可规划段做带语义/碰撞/IK约束的路径重规划
  → 基于“抓取物相对目标物”的选择与变换进行第二阶段大规模扩增
  → 输出与人类示范同格式、可供多种IL算法训练的大规模示范数据
claims:
  - "Claim 1: 在 7 个操作任务、18 个变体上，用 HybridGen 数据训练的 BC-RNN 平均成功率为 76.2%，高于 MimicGen 的 71.2%；各任务最难变体平均为 59.7%，高于 49.5% [evidence: comparison]"
  - "Claim 2: 在 Square D0-D2 上，使用 HybridGen 数据训练的 BC-Transformer 与 Diffusion Policy 均优于使用 MimicGen 数据训练的对应模型，表明该数据对不同 IL 架构具有通用性 [evidence: comparison]"
  - "Claim 3: 去掉 VLM 重规划或去掉 GRT 子任务选择都会让 Square D2 / Threading D2 从完整系统的 58.7/46.7 下降到 51.3/44.7 或 54.7/42.0，说明两部分都对性能有因果贡献 [evidence: ablation]"
related_work_position:
  extends: "MimicGen (Mandlekar et al. 2023)"
  competes_with: "MimicGen (Mandlekar et al. 2023); SkillMimicGen (Garrett et al. 2024)"
  complementary_to: "Diffusion Policy (Chi et al. 2024); BC-Transformer (Robomimic)"
evidence_strength: moderate
pdf_ref: paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_HybridGen_VLM_Guided_Hybrid_Planning_for_Scalable_Data_Generation_of_Imitation_Learning.pdf
category: Embodied_AI
---

# HybridGen: VLM-Guided Hybrid Planning for Scalable Data Generation of Imitation Learning

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.13171)
> - **Summary**: HybridGen 的核心不是让 VLM 直接生成机器人动作，而是让它负责“理解视频并提出语义约束”，再把复杂精细段交给专家示范、把可规划段交给约束规划器，从而把少量示范放大成可供多种模仿学习算法使用的大规模训练数据。
> - **Key Performance**: 7 个任务 18 个变体上平均成功率 76.2% vs. 71.2%；各任务最难变体平均 59.7% vs. 49.5%。

> [!info] **Agent Summary**
> - **task_path**: 少量人类示范视频/轨迹 + 任务描述 + 场景RGB → 大规模、格式一致的机器人示范数据 → 多种模仿学习策略
> - **bottleneck**: 直接用 VLM 生成轨迹缺少运动学/碰撞可行性，而传统 pose augmentation 又只能做几何搬运，缺少语义层面的轨迹多样性
> - **mechanism_delta**: 先用 VLM 将轨迹切成“专家依赖段/可重规划段”，再分别做对象中心位姿迁移与语义约束重规划，最后用抓取物-目标物相对位姿做二次扩增
> - **evidence_signal**: 跨 7 个任务的主比较实验显示平均 +5pt，且去掉 VLM 或 GRT 都会退化
> - **reusable_ops**: [video-based phase decomposition, vlm constraint mining, object-centric pose transfer]
> - **failure_modes**: [高精度接触段被误分为可规划段, 关键点或约束提取误差传导到路径规划]
> - **open_questions**: [真实机器人上的成本与稳定性是否成立, 对可变形物体或更强接触任务是否仍能稳定分段]

## Part I：问题与挑战

这篇工作的真实问题，不是“怎么再做一个更强的 IL policy”，而是**怎么低成本制造足够多、足够 diverse、同时又可执行的示范数据**。

### 1. 真正的瓶颈是什么
模仿学习在机器人上早就知道一件事：**数据量和多样性直接影响泛化**。但复杂操作任务的数据采集特别贵，尤其是：
- 穿孔、装配、关盖、插入这类**高精度接触**阶段；
- 多阶段长时序任务中的**中间搬运/对齐**阶段；
- 需要覆盖不同初始位姿、不同物体朝向的**分布外初始化**。

现有两条路都不够：
1. **直接让 VLM 控轨迹**：它懂场景和任务语义，但不懂机器人运动学、碰撞、IK 可行性，往往只能给出“应该怎么做”的粗描述。
2. **只做几何式数据增强**：像 MimicGen 一类方法可以把已有轨迹搬到新场景，但生成的还是“同一类运动的几何变体”，难以显著改变轨迹语义和中间路径结构。

所以真正的 bottleneck 是：  
**如何把“语义多样性”和“机械可执行性”拆开处理。**

### 2. 为什么是现在
这件事现在值得做，有两个现实原因：
- **VLM 已经足够擅长看视频、看图、读任务描述**，可以帮助识别任务阶段与关键约束；
- **IL 的数据扩展价值越来越明确**，但真实机器人上大规模人工采集仍然昂贵，因此“自动生成高质量示范”成为更现实的突破口。

### 3. 输入/输出接口与边界条件
**输入：**
- 少量成功的人类示范；
- 任务文本描述；
- 场景 RGB 图像；
- 示范回放视频。

**输出：**
- 与原始人类示范**同格式**的大规模轨迹数据；
- 可直接喂给 BC-RNN、BC-Transformer、Diffusion Policy 等 IL 算法。

**边界条件：**
- 任务主要是**抓取物体 A 与目标物体 B 的对象中心操作**；
- 主要评估在仿真任务上完成；
- 生成后的轨迹需要在仿真中执行，**只保留成功轨迹**。

这说明它不是一个“纯生成模型”，而是一个**带执行筛选环节的数据工厂**。

## Part II：方法与洞察

### 方法主线

HybridGen 的方法可概括为两级分解、两阶段扩增。

#### 1. 双粒度任务分解：先决定哪些部分该“保留专家”，哪些部分可“自动生成”
作者先把每条示范分成两层结构：

- **子任务级分解**：把整条轨迹切成 object-centric subtasks；
- **位姿级分解**：把每个 pose 标成两类：
  - **data-dependent pose**：需要高精度、强接触、强技巧的专家依赖段；
  - **replanning pose**：可由规划器重新生成的平移、转向、靠近、抓取等段。

这里的关键是：  
**VLM 不再直接生成动作，而是先帮你决定“哪里必须信专家，哪里可以自动化”。**

作者用 Gemini 分析示范视频，输出哪些时间段属于 expert-dependent。为缓解时序定位精度不足，他们还对视频帧做了更细粒度插值再映射回轨迹。

#### 2. 第一阶段扩增：专家段做位姿适配，可规划段做 VLM 约束重规划
这一步是 HybridGen 最有信息增益的部分。

- 对 **expert-dependent segments**：  
  直接做 pose adaptation，把源示范迁移到新场景，但保持“目标物体与关键姿态”的相对关系不变。  
  目的：保住高精度接触技能，不让规划误差破坏细节。

- 对 **replanning segments**：  
  不是简单插值，而是先让系统提取约束，再做规划。
  - 用 **CLIP** 根据任务文本在图像上找 task-relevant keypoints；
  - 用 **Gemini** 看：
    1. 任务文本，
    2. 标了 keypoints 的场景图，
    3. 示范回放视频，
    
    输出每个阶段的 path/sub-goal constraints；
  - 再由规划器在这些约束下生成轨迹，同时考虑：
    - 语义约束，
    - 碰撞，
    - 平滑性，
    - IK 可达性。

所以第一阶段不是“把轨迹抄一遍”，而是**让 VLM 给出语义护栏，让经典规划负责可执行性**。

#### 3. 第二阶段扩增：用对象相对位姿重参数化，放大可组合性
如果第一阶段主要解决“多样性”，第二阶段主要解决“规模”。

作者提出的关键改动是：  
以往方法常把选择/迁移锚定在**末端执行器相对目标物体**的关系上，这会把子任务选择绑死在原始 grasp pose 附近。  
HybridGen 改成锚定在**抓取物体 A 相对目标物体 B**的位姿关系上。

对应效果是：
- 可用 GRT（Nearest Grasp Object Relative to Target Object）在源示范中找 top-k 最近候选；
- 子任务不再强依赖源数据的绝对抓取姿态；
- 第一阶段已经生成出的新轨迹，也能在第二阶段继续被自由重组和扩增。

这一步本质上是在做一个更合适的**对象中心坐标重参数化**。

#### 4. 输出为何能跨算法复用
HybridGen 一直强调一点：  
它生成的数据格式与人类示范一致，不绑定某种特定 policy learner。

这意味着它不是“给某个模型做专用数据合成器”，而是一个**上游数据层方法**，理论上可插到多种 IL pipeline 前面。

### 核心直觉

HybridGen 最关键的变化，不是“让 VLM 更会控制机器人”，而是：

> **把 VLM 从低层动作生成器，降级成高层语义约束提取器。**

这一步改变了三个层面的瓶颈：

1. **what changed**  
   从“VLM 直接出动作/轨迹”  
   变成“VLM 只负责分段 + 约束提取”。

2. **which distribution / constraint bottleneck changed**  
   原来一个模型必须同时解决：
   - 任务语义理解，
   - 时序阶段划分，
   - 几何/碰撞/IK 可行性，
   - 高精度接触细节。  
   
   现在被拆成三块：
   - **专家示范**保住高精度接触；
   - **VLM**提供语义约束；
   - **经典规划器**保证物理和运动学可行。

3. **what capability changed**  
   数据增强不再只是“同一轨迹的空间平移/旋转”，而是能在可规划阶段产生**新的中间路径分布**；  
   同时，对象相对位姿的重参数化又让不同示范片段可以更自由拼接，提升跨场景泛化。

一句话概括其因果链：

**分段解耦 → 语义与力学不再互相拖累 → 可规划段获得真正的多样性，精细段仍保持可靠性。**

### 策略取舍表

| 设计选择 | 解决的瓶颈 | 带来的能力 | 代价 / 风险 |
|---|---|---|---|
| 保留 expert-dependent 段 | 高精度接触容易被规划误差破坏 | 精细操作成功率更稳 | 仍依赖少量高质量人类示范 |
| VLM 只生成约束，不直接控动作 | 语义理解与机械可行性耦合过强 | 中间路径更合理、更有语义差异 | 约束质量受 Gemini/关键点质量影响 |
| CLIP 提取 task-conditioned keypoints | 静态图像关键点可能与任务无关 | 约束更贴任务而非仅贴外观 | 关键点定位误差会放大到规划 |
| 以“抓取物相对目标物”做检索与变换 | 子任务选择受绝对 grasp pose 约束 | 可自由重组子任务，二阶段扩增更有效 | 假设任务可由二物体相对位姿较好描述 |
| 先执行再保留成功轨迹 | 纯生成结果质量不可控 | 数据净度更高 | 需要仿真执行器与额外计算成本 |

## Part III：证据与局限

### 关键实验信号

#### 1. 主比较信号：收益主要来自更难的分布
最核心结果是对 MimicGen 的比较。

- **总体平均**：76.2% vs. 71.2%，约 +5pt。
- **各任务最难变体平均**：59.7% vs. 49.5%，提升更明显。

这很重要，因为它说明 HybridGen 的优势不是只在简单场景“刷高分”，而是在**分布更宽、初始化更复杂、路径更难泛化**的设置里体现出来。

同时，收益并非完全单调：
- 明显提升的例子：Threading D2、Hammer Cleanup D1、Mug Cleanup D1、Coffee Prep D1；
- 也存在回退：Coffee D2、Piece Assembly D1、Coffee Prep D0。

这说明该方法**不是“对所有任务一律更好”**，而是更偏向在“需要更多路径多样性”的任务上有效。

#### 2. 跨算法信号：数据层方法确实能迁移到不同 policy learner
作者用 BC-Transformer 和 Diffusion Policy 在 Square D0-D2 上做了验证，结果都优于用 MimicGen 数据训练的对应模型。

这个信号支持了作者一个重要主张：  
**HybridGen 改善的是训练数据分布，而不只是某个特定 BC-RNN 实现。**

但也要注意，这部分证据只覆盖了 **Square** 任务，而不是全部 7 个任务，所以“算法通用性”的证据是存在的，但覆盖面还不算很广。

#### 3. 消融信号：VLM 和 GRT 都是实质贡献，不是装饰模块
在 Square D2 / Threading D2 上：
- 完整系统：58.7 / 46.7
- 去掉 VLM：51.3 / 44.7
- 去掉 GRT：54.7 / 42.0
- 两者都去掉：49.3 / 38.0

结论很清楚：
- **VLM 重规划**确实提供了超越线性插值的轨迹质量；
- **GRT 子任务选择**确实让示范片段的复用更合理；
- 两者结合最好，说明这是个**协同结构**，不是单一模块在起作用。

### 这篇方法真正“跳”的能力在哪里
与 MimicGen 这类主要做 pose-level 迁移的方法相比，HybridGen 的能力跃迁点在于：

1. **不只改起点/终点，也改中间路径语义**  
   这让数据分布更宽，而不是仅有空间重映射。

2. **让 VLM 发挥在它擅长的层级**  
   它不再承担低层控制，而只负责“看懂任务并给出路径条件”。

3. **把第一阶段生成的数据再变成第二阶段的素材**  
   于是规模不是线性扩张，而是带组合性的放大。

### 局限性

- **Fails when**: 高精度接触段被错误划为可规划段时，重规划轨迹容易失真；关键点或约束抽取不准时，路径会被错误语义牵引；从 Coffee D2、Coffee Prep D0 的回退也能看出，额外重规划并不总是收益。
- **Assumes**: 任务可以用“抓取物体 A 与目标物体 B 的相对位姿”较好建模；有少量成功人类示范可供起步；有仿真环境执行并筛除失败轨迹；依赖闭源 Gemini、CLIP、IKFast、碰撞/SDF 建模，以及双 RTX 4090 级别算力。
- **Not designed for**: 直接在真实机器人上无筛选地大规模生成数据；可变形物体、复杂力控接触、多手协作等不易用刚体 keypoint 约束表达的任务；让 VLM 直接输出闭环低层控制。

### 可复用组件
如果把这篇论文当“方法积木”，最值得复用的是：

- **VLM-based trajectory decomposition**：把示范分成专家依赖段与可规划段；
- **task-conditioned keypoint extraction**：用文本引导关键点抽取，而不是盲目从图像取显著点；
- **VLM-to-constraint interface**：让多模态模型输出约束，而不是输出动作；
- **object-centric subtask retrieval / transfer**：用抓取物相对目标物的位姿做检索与变换；
- **success-filtered data factory**：生成 → 执行 → 保留成功样本的流水线。

整体上，这是一篇很典型的“**把基础模型放到正确层级**”的 Embodied 数据工程论文：  
它并不试图让 VLM 端到端取代机器人控制，而是让 VLM 负责理解，让规划器负责可行，让专家数据负责精度。这个分工，是它比“纯 VLM 控制”或“纯几何增强”都更合理的地方。

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_HybridGen_VLM_Guided_Hybrid_Planning_for_Scalable_Data_Generation_of_Imitation_Learning.pdf]]