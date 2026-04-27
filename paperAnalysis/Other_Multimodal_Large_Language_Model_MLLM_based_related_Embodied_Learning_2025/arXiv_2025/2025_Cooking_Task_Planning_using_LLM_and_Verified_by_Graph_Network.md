---
title: "Cooking Task Planning using LLM and Verified by Graph Network"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-task-planning
  - task/video-understanding
  - graph-verification
  - state-tracking
  - chain-of-thought
  - dataset/YouTube-Shorts-Cooking-Videos
  - opensource/no
core_operator: 让多模态LLM先从带字幕烹饪视频估计每个片段的目标对象状态，再用FOON功能单元逐步校验动作前置条件，并把失败原因反馈给LLM重规划。
primary_logic: |
  带字幕烹饪视频 + 本地厨房环境状态 + 允许动作库
  → LLM分段理解视频并生成目标对象状态与动作序列，FOON按环境对象节点/目标对象节点做逐步可行性校验并反馈纠错
  → 面向双臂机器人的可执行烹饪任务图与动作计划
claims:
  - "在5个食谱的完整任务图生成上，本文方法成功生成4/5个可执行任务序列，而few-shot LLM-only基线仅为1/5 [evidence: comparison]"
  - "在10个YouTube烹饪视频的276张图像上，系统对无关片段移除、可解析图输出和目标对象状态估计分别达到94%、98%和86%的成功率 [evidence: analysis]"
  - "在真实双臂UR3e牛丼实验中，系统经过3轮重规划后补全了视频中省略的切洋葱步骤并完成整道流程 [evidence: case-study]"
related_work_position:
  extends: "Efficient Task/Motion Planning for a Dual-arm Robot from Language Instructions and Cooking Images (Takada et al. 2022)"
  competes_with: "PROGPROMPT (Singh et al. 2023); From Cooking Recipes to Robot Task Trees—Improving Planning Correctness and Task Efficiency by Leveraging LLMs with a Knowledge Network (Sakib & Sun 2024)"
  complementary_to: "LLM+P (Liu et al. 2023); Consolidating Trees of Robotic Plans Generated Using Large Language Models to Improve Reliability (Sakib & Sun 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_Cooking_Task_Planning_using_LLM_and_Verified_by_Graph_Network.pdf
category: Embodied_AI
---

# Cooking Task Planning using LLM and Verified by Graph Network

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.21564)
> - **Summary**: 本文把“从带字幕烹饪视频直接生成机器人长程计划”拆成“目标状态估计 + FOON逐步验证/重规划”，用显式状态图约束LLM，降低烹饪任务中的幻觉、漏步和环境错配问题。
> - **Key Performance**: 完整任务图生成成功率 **4/5 vs 1/5**（相对few-shot LLM-only）；目标对象节点状态估计成功率 **86%**（10个视频、276张图像）。

> [!info] **Agent Summary**
> - **task_path**: 带字幕烹饪视频 + 本地厨房布局 -> 双臂机器人可执行烹饪任务图/动作序列
> - **bottleneck**: 视频里存在省略步骤、镜头/器具差异和“视频环境—机器人环境”错配，直接让LLM输出长程计划容易幻觉且无法保证动作前置条件成立
> - **mechanism_delta**: 用“目标对象节点/环境对象节点”把视频目标状态与本地真实状态显式分离，并用FOON功能单元逐步校验、失败即反馈重规划
> - **evidence_signal**: 5个recipe上的完整任务图成功率从1/5提升到4/5，且真实双臂实验能补全被视频省略的Cut步骤
> - **reusable_ops**: [target-vs-environment-state-split, state-mismatch-feedback-replanning]
> - **failure_modes**: [sequential-validation-cannot-revise-earlier-actions, out-of-vocabulary-actions-and-object-paraphrases]
> - **open_questions**: [how-to-enable-global-backtracking-over-whole-plan, how-to-couple-online-perception-with-dynamic-replanning]

## Part I：问题与挑战

这篇论文真正要解决的，不是“看懂一个做饭视频”本身，而是把**噪声很大的烹饪视频**转成**在特定机器人厨房里真的能执行的长程操作计划**。

### 1. 真正难点是什么
烹饪视频到机器人计划，中间至少有三层鸿沟：

1. **视频语义不完整**
   - 镜头角度变化大；
   - 器具和食材名称不标准；
   - 视频常省略中间步骤，比如先切好再直接出现下锅。

2. **视频环境与机器人环境不一致**
   - 视频里的锅、碗、砧板摆放与机器人工作台不同；
   - 机器人有固定可达范围、抓取姿态和双臂协作约束。

3. **LLM长程规划容易幻觉**
   - 能生成“看起来合理”的步骤；
   - 但未必满足当前状态前置条件，比如手已经拿着东西却继续执行 pick。

所以瓶颈不是单一感知误差，而是：**开放世界视频理解 + 符号状态一致性 + 本地执行约束**同时存在，而纯LLM通常只擅长前两者中的“语言推断”，不擅长最后的严格一致性检查。

### 2. 为什么现在值得做
因为网络上有大量烹饪视频，天然是廉价任务知识源；同时，多模态LLM已经能从图像+字幕中提取较强语义。但如果没有一个显式验证层，这些能力无法可靠落到机器人执行上。  
这篇工作的价值就在于：**把LLM的泛化理解能力，接到一个可验证的任务图上**，让“会说”更接近“能做”。

### 3. 输入 / 输出接口
- **输入**：
  - YouTube烹饪视频及字幕；
  - 机器人本地环境布局；
  - 允许动作集合；
  - 预定义对象属性与动作模板。
- **输出**：
  - 适配本地环境的FOON任务图；
  - 可供双臂机器人执行的动作序列与后续运动规划输入。

### 4. 边界条件
这套系统并不是开放式通用厨房智能体，而是在以下条件下工作：
- 动作类型是**预定义有限集**；
- 目标环境是**固定区域划分**的厨房台面；
- 运动可达性通过**预先离散化工作空间**来保证；
- 文中真实执行只展示了**单个牛丼食谱**；
- 环境对象节点由人工构建，不是完全自动建模。

---

## Part II：方法与洞察

整体上，方法是一个“**LLM负责猜目标，图结构负责验逻辑**”的两阶段闭环。

### 方法主线

#### Step 1：视频预处理
作者先对带字幕视频做OCR，按字幕相似性分段，再把每段视频提取成关键帧，并拼成 **3×3 grid** 图像，作为LLM输入。

这一步的作用不是提升精度本身，而是把长视频压成**若干离散任务片段**，减少LLM需要一次性处理的时序长度。

#### Step 2：估计 Target Object Node
第一个LLM代理根据：
- 场景关键帧；
- 字幕；
- 允许动作；
- 对象属性定义；

去预测每段结束时应达到的**目标对象状态**。  
例如，不直接说“现在做什么动作”，而是先说“洋葱应该变成切碎状态、在锅里/碗里/砧板上”。

这是关键的状态化改写：先定义“要达到什么”，再问“怎么做到”。

#### Step 3：从 Environment Node 到 Target Node 的动作规划
第二个LLM代理比较：
- 当前本地环境对象状态（Environment Object Node）
- 视频推断出的目标状态（Target Object Node）

然后输出要执行的动作序列，以及动作变量，如：
- 用哪只手；
- 从哪个位置取；
- 操作哪个对象。

#### Step 4：FOON功能单元实例化与逐步验证
系统把LLM输出动作映射到**带变量的FOON functional unit**。  
然后检查该动作的输入前置条件是否与当前环境节点一致。

如果一致：
- 动作可执行；
- 更新环境对象状态；
- 把该功能单元接入任务图。

如果不一致：
- 找出具体冲突属性；
- 生成错误反馈prompt；
- 让LLM重规划。

#### Step 5：运动规划落地
任务图通过逻辑验证后，再交给运动规划模块。这里作者没有做完整几何推理，而是：
- 预先离散化可达空间；
- 手工选取可执行初始摆放；
- 用RRT-Connect和线性轨迹生成各动作。

也就是说，本文重点不是几何层最优，而是**任务层计划的正确性**。

### 核心直觉

以前的做法更像是：  
**视频/字幕 → LLM直接生成长动作序列**。  
这会把多个困难混在一起：
- 视频理解；
- 状态跟踪；
- 前置条件判断；
- 环境适配；
- 缺失步骤补全。

本文的变化是：  
**先把视频压成“每一段结束时的目标状态”，再用FOON把动作生成变成“受约束的状态转移问题”**。

也就是：

- **what changed**：从“一步到位的自由文本动作生成”改成“目标状态估计 + 图结构逐步验证”；
- **which bottleneck changed**：把开放式长序列生成的搜索空间，压缩成显式对象属性上的局部状态转移，并把错误暴露为可解释的属性不匹配；
- **what capability changed**：系统更容易发现LLM幻觉、补全视频省略动作、并适配本地厨房环境。

更因果地说，FOON在这里不是知识库装饰，而是一个**外部一致性约束器**：
- LLM负责提出候选动作；
- FOON负责判断“当前状态下这步是否成立”；
- 一旦失败，反馈不再是模糊的“请重试”，而是具体到“手非空”“对象位置不对”“状态不匹配”。

这使重规划从“重新瞎猜”变成“带着结构化错误信号修正”。

### 战略取舍

| 设计选择 | 带来的能力 | 代价/限制 |
|---|---|---|
| 目标对象节点 / 环境对象节点分离 | 把视频目标状态与本地真实状态解耦，支持环境适配 | 需要手工定义对象属性与状态空间 |
| FOON逐步验证 | 明确检查每步前置条件，抑制LLM幻觉 | 只做局部验证，缺少全局回溯 |
| 带变量功能单元 | 把动作模板标准化，便于从文本转图 | 受限于动作库覆盖率 |
| 字幕+关键帧联合输入 | 用廉价公开视频驱动任务学习 | OCR噪声、视频剪辑和字幕缺失会传导误差 |
| 离散化工作空间 | 便于真实机器人执行演示 | 几何泛化弱，对新布局适应有限 |

---

## Part III：证据与局限

### 关键证据信号

1. **比较信号：长程计划可靠性显著提升**  
   在5个食谱的完整任务图生成上，本文方法成功 **4/5**，而few-shot LLM-only仅 **1/5**。  
   这直接支持论文最核心的主张：**图验证层确实能减少长程烹饪规划中的逻辑错误**。

2. **分析信号：视频理解到结构化状态输出基本可用**  
   在10个视频、276张图像上：
   - 无关片段移除成功率 **94%**
   - LLM输出可成功转成图结构 **98%**
   - 目标对象节点状态估计成功率 **86%**

   这说明前端“视频 → 状态”的链路不是完美，但已足够支撑后端计划生成。  
   同时也说明最大误差不是JSON格式，而是**动作覆盖不足和命名变化**。

3. **案例信号：能补全视频中省略的步骤**  
   在真实双臂UR3e牛丼实验中，视频里省略了切洋葱步骤，系统仍能根据目标状态差异自动补出 **Cut**。  
   这说明方法不是机械模仿视频，而是做了某种**状态驱动的任务补全**。

4. **鲁棒性侧信号：OCR有噪声，但问题可被后续过滤**  
   OCR提取字幕总词数比标注多 **17%**，但作者认为这些主要是冗余信息，可被LLM过滤。  
   这不是强证据，但说明系统对轻度文本噪声有一定容忍度。

### 1-2个最关键指标
- **完整任务图成功率：4/5 vs 1/5**
- **目标对象状态估计成功率：86%**

### 局限性
- **Fails when**: 早先动作虽然局部可行、但会破坏未来步骤时；视频里出现未包含在预定义动作集中的操作时；对象名称发生较大paraphrase导致节点类型/属性映射错误时。
- **Assumes**: 依赖闭源ChatGPT-4o；依赖OCR字幕和手工设计的对象属性；环境对象节点由人工创建；动作模板库是预定义的；完整食谱评测中还对3个目标节点做了轻微人工修正；真实执行依赖离散化工作空间与预先选好的可达摆放。
- **Not designed for**: 在线执行失败恢复（抓取失败、位姿估计误差、碰撞后状态变化）；全局几何可行性联合优化；开放厨房环境中的任意新器具/新布局泛化。

### 可复用组件
- **环境状态 vs 目标状态双节点表示**：适合任何“演示域 ≠ 执行域”的任务迁移。
- **基于状态不匹配的反馈重规划**：比自然语言自反思更可控，可迁移到装配、整理等长程任务。
- **带变量的动作模板库**：适合把LLM文本输出转成结构化任务图。
- **字幕驱动的视频分段**：适合短视频流程理解的低成本前处理。

### 一句话总结
这篇论文的能力跃迁不在于“LLM更懂做饭”，而在于它给LLM加了一个**显式、逐步、可解释的状态验证回路**。因此，系统从“能说步骤”更接近“能在真实厨房里按条件执行步骤”。

## Local PDF reference

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_Cooking_Task_Planning_using_LLM_and_Verified_by_Graph_Network.pdf]]