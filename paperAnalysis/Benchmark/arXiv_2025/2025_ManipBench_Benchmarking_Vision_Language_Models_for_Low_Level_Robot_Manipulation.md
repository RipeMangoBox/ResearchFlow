---
title: "ManipBench: Benchmarking Vision-Language Models for Low-Level Robot Manipulation"
venue: CoRL
year: 2025
tags:
  - Survey_Benchmark
  - task/MLLM-evaluation
  - task/visual-question-answering
  - multiple-choice-evaluation
  - mark-based-visual-prompting
  - keypoint-affordance
  - dataset/ManipBench
  - dataset/DROID
  - dataset/Bridge
  - opensource/promised
core_operator: 将低层机器人操控理解转写为带关键点与网格标注的多项选择题，以统一评测VLM对动作后果的判断能力
primary_logic: |
  低层操控评测目标 → 从真实机器人数据、自建布料场景与仿真中抽取关键点/轨迹并构造成MCQ → 用准确率、人类对照与真实机器人相关性统计评分 → 揭示VLM在精细操控上的能力边界
claims:
  - "ManipBench包含12617道多项选择题，覆盖抓取放置、关节体、可变形体、工具和动态操控，并将布料操控拆成10个可诊断维度 [evidence: analysis]"
  - "ManipBench得分与7个未见真实机器人任务上的成功率显著正相关，Pearson=0.889（p=0.003） [evidence: analysis]"
  - "当前最强VLM虽明显高于随机，但仍存在显著人类差距，例如Bridge Q2最佳模型为0.541，而人类为0.825 [evidence: comparison]"
related_work_position:
  extends: "MOKA (Fang et al. 2024)"
  competes_with: "PhysBench (Chow et al. 2025); VLABench (Zhang et al. 2024)"
  complementary_to: "OpenVLA (Kim et al. 2024); RT-2 (Brohan et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Benchmark/arXiv_2025/2025_ManipBench_Benchmarking_Vision_Language_Models_for_Low_Level_Robot_Manipulation.pdf
category: Survey_Benchmark
---

# ManipBench: Benchmarking Vision-Language Models for Low-Level Robot Manipulation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2505.09698), [Project](https://manipbench.github.io/)
> - **Summary**: 该工作提出 ManipBench，把机器人低层操控中的“抓哪里、放哪里、往哪推”统一改写成带关键点/轨迹候选的多项选择题，从而低成本、可分解地评测 VLM 的物理动作理解能力。
> - **Key Performance**: ManipBench 分数与 7 个未见真实机器人任务成功率的 Pearson 相关系数为 **0.889**（p=0.003）；**Gemini-2.5-pro** 在真实机器人实验中达到 **18/21** 成功。

> [!info] **Agent Summary**
> - **task_path**: 单张场景图像+语言指令+标记关键点/网格 → 选择抓取点/释放格/运动方向/轨迹候选 → 低层操控推理得分
> - **bottleneck**: 缺少既可扩展又与真实机器人表现相关的低层操控评测，导致难以比较 VLM 作为机器人 agent 的真实价值
> - **mechanism_delta**: 用关键点/轨迹候选的 MCQ 替代真实 rollout，把评测焦点从控制执行噪声转到动作后果判断
> - **evidence_signal**: 33 个 VLM 在 12617 题上的系统比较，并与 7 个真实机器人任务成功率呈显著正相关
> - **reusable_ops**: [MOKA式标点预处理, 关键点到MCQ转写]
> - **failure_modes**: [放置终点预测明显弱于抓取点预测, 关节体与动态操控显著难于pick-and-place]
> - **open_questions**: [MCQ得分能否预测自由生成动作的真实执行效果, 如何覆盖袋类等更复杂可变形物体与长时序操作]

## Part I：问题与挑战

这篇论文要解决的，不是“再做一个更强的 VLM”，而是一个更基础的问题：**我们到底该如何可靠地测 VLM 是否真的理解低层机器人操控？**

### 真实问题是什么
现有很多 VLM/LLM for Robotics 的工作，评测重点常在：
- 高层任务规划；
- 一般物理常识或对象属性判断；
- 空间关系理解；
- 长时序 embodied planning。

但机器人低层操控真正关键的是更细粒度的问题：
- 机械臂应该**接触哪里**；
- 抓起后应该**放到哪里**；
- 保持接触时应该**往哪个方向移动**；
- 面对布料、抽屉、工具、动态物体时，动作后果会怎样变化。

这类能力如果不单独测，VLM 作为机器人 agent 的效果就很难判断。

### 真正瓶颈在哪里
作者指出的核心瓶颈有两个：

1. **评测成本高**  
   如果靠真实 rollout 或完整策略执行来测，每个模型、每个任务的评估都很贵，也难以大规模覆盖。

2. **评测信号不纯**  
   连续轨迹输出会混入控制器质量、策略多模态、仿真器差异等噪声。最后很难分辨：模型到底是“不懂动作后果”，还是“输出格式/控制链路不匹配”。

### 为什么现在要解决
因为 VLM 已经不只是高层 planner，近年的工作开始直接让它们参与：
- 关键点选择；
- 低层轨迹生成；
- affordance 推断；
- 开放世界 manipulation。

这意味着社区现在急需一个问题导向很明确的 benchmark，来回答：**哪个 VLM 真适合做低层操控决策？**

### 输入/输出接口与边界
ManipBench 的统一接口很清晰：

- **输入**：场景图像 + 语言任务描述 + 标注好的关键点/网格/候选轨迹；
- **输出**：从 4 个候选项中选择正确动作；
- **动作抽象**：抓取点、释放格、接触后运动方向、整条候选轨迹。

它的边界也很明确：
- 主要是**单步或短程低层决策**；
- 主要评测**动作后果理解**，不是控制稳定性；
- 多数题目是**给定候选项再选择**，不是自由生成连续动作；
- 不以高层规划为重点。

---

## Part II：方法与洞察

ManipBench 的核心贡献，是把原本难以规模化评测的低层操控理解，重写成一个统一、可批量跑的 MCQ benchmark。

### 评测设计怎么搭起来的

#### 1. 三类数据源
作者从三个来源构建题目：

- **公共真实机器人数据**：DROID、Bridge  
  用真实 demonstration 生成题目，覆盖 pick-and-place 与 articulated manipulation。
- **自建布料操控场景**：  
  专门针对 deformable object，手工设计 10 个能力维度，如状态理解、逆动力学、时序、fabric-object / fabric-fabric interaction 等。
- **仿真任务**：  
  从 SimplerEnv、RLBench、SoftGym 和 IsaacSim 中构建 pick-and-place、抽屉关闭、绳子拉直、工具操作、ball shooting 等任务。

总计 **12617** 道题。

#### 2. 动作被离散成“可选的物理决策”
大部分题目都围绕三类低层动作原语：
- **contact-initiation**：从哪里接触/抓取；
- **contact-release**：在哪里释放/放置；
- **post-contact motion**：接触后往哪移动。

这相当于把“连续动作空间”压缩成“可诊断的离散决策空间”。

#### 3. MOKA 风格的标记式视觉提示
对真实图像，作者使用 MOKA-style 流程做预处理：
- 先用 GPT-4o 识别任务中的关键对象；
- 用 Grounded SAM 做分割；
- 从 mask 中采样中心点和轮廓点；
- 再叠加网格与标记点，形成 VLM 输入图像。

这样做的好处是：把动作选择显式绑定到图像上的点或格子，减少语言-动作对齐时的歧义。

#### 4. 题型不是一刀切，而是带诊断结构
公共机器人数据的题型尤其有意思：

- **Q1**：直接选完整轨迹候选；
- **Q2**：先选抓取点，再选终点格。

Q2 的作用不是“再出一道题”，而是把低层操控分解成：
- 物体/接触点识别；
- 放置后果推理。

这让 benchmark 能区分“会抓不会放”这类失败模式。

### 核心直觉

**这篇论文真正改变的，不是模型，而是“测量瓶颈”。**

- **什么变了**：  
  从“让模型直接生成连续动作并 rollout 验证”，改成“让模型在显式候选关键点/轨迹中做判别”。

- **哪个约束变了**：  
  原来评测受限于控制器差异、轨迹多模态、输出格式不统一；现在这些噪声被大幅压缩，信息瓶颈集中到**对动作后果的理解**。

- **能力上发生了什么变化**：  
  评测终于可以高通量、跨任务、跨模型地比较 VLM 在低层 manipulation 上的真实差异，并且能进一步定位差异出在：
  - 抓点选择；
  - 终点选择；
  - 时序理解；
  - 关节体/布料/动态场景理解。

**为什么这套设计有效？**  
因为低层 manipulation 虽然最终落在连续控制上，但很多关键失败先发生在更上游的“动作语义选择”阶段：接触点不对、方向错、释放位置不对。ManipBench 保留了这层最关键的决策结构，同时避开了 rollout 带来的高成本与混杂因素。

### 战略取舍

| 设计选择 | 解决的测量瓶颈 | 收益 | 代价/偏差 |
|---|---|---|---|
| 用 MCQ 代替 rollout | 控制器、执行器、仿真差异带来的噪声 | 可扩展到 33 个模型、大规模题库 | 不能直接测自由生成动作 |
| 用标记点/网格提示动作 | 文本到动作的对齐歧义 | 输出统一、易比较、可定位错误 | 依赖分割与关键点标注质量 |
| 混合真实数据、自建布料、仿真 | 单一数据源覆盖窄 | 同时覆盖 rigid / deformable / dynamic | 分布异质，构造成本高 |
| 将布料任务拆成 10 维 | 只看总分难以定位盲点 | 可诊断具体物理推理短板 | 维度设计本身带有任务建模假设 |
| 加入真实机器人相关性验证 | benchmark 可能“只会考试” | 证明其外部有效性 | 真实实验仍然是“从候选动作中选” |

---

## Part III：证据与局限

### 关键证据信号

- **信号 1｜跨 33 个 VLM 的横向比较**  
  Benchmark 能明显拉开模型差距。闭源模型整体领先，其中 **Gemini-2.5-pro** 总体最强；开源模型中 **InternVL2.5-38B** 在公共数据子集上表现突出。  
  **结论**：ManipBench 不是“大家都差不多”的饱和测试，而是真能区分低层操控推理能力。

- **信号 2｜错误模式分解**  
  在公共数据的 Q2 中，**终点格/放置位置预测显著难于抓取点预测**。  
  **结论**：当前 VLM 的主要短板不只是“看不懂目标物体”，而是“推不准动作后果”。

- **信号 3｜任务类别边界清晰**  
  聚合后，pick-and-place 相对容易，而 articulated object 与 dynamic manipulation 更难。  
  **结论**：现有 VLM 对受约束接触、非刚体变化、动态后果的理解仍明显不足。

- **信号 4｜外部有效性**  
  Benchmark 得分与 7 个未见真实机器人任务成功率显著正相关，**Pearson=0.889（p=0.003）**；真实机器人上 **Gemini-2.5-pro 达到 18/21**。  
  **结论**：ManipBench 不只是“图文考试”，而是对真实 agent 选动作能力有预测力。

- **信号 5｜人类差距仍大**  
  人类在多个子集上接近满分，而最强模型仍显著落后；例如 Bridge Q2 最佳模型 **0.541**，人类 **0.825**。  
  **结论**：低层 manipulation reasoning 远未接近饱和。

### 局限性

- Fails when: 需要自由生成连续动作、长时序闭环纠错、复杂恢复策略，或涉及袋类等未覆盖的复杂可变形物体时，这个 benchmark 不能充分代表真实性能。
- Assumes: 动作可以被离散成候选关键点/轨迹；预处理依赖 GPT-4o、Grounded SAM、GroundingDINO 与部分人工标注；真实世界验证与 benchmark 共享“从给定选项中选动作”的交互形式。
- Not designed for: 高层任务规划、控制器鲁棒性、完整 VLA policy 的端到端评测、多步技能组合与执行代价分析。

此外，复现还依赖一定资源与工具链：
- 评测使用了 **2×RTX 4090**，更大模型还用了 **5×RTX 6000 Ada**；
- 预处理链路含闭源组件；
- 布料维度和部分选项需要人工设计。  
这些因素都会影响基准的完全可复现性与扩展成本。另一个保守点是：论文称其为 open-source benchmark，但正文也提到部分网站/基准内容会在后续公开，因此当前更像“已公开项目页、完整开放待落地”。

### 可复用组件

- **MOKA式关键点/网格标注预处理**：适合把 manipulation 场景统一转成 VLM 可回答的视觉提示。
- **Q1/Q2 分解式题型**：可直接复用于区分“接触点选择”与“动作后果推理”。
- **布料 10 维诊断框架**：适合扩展到更多 deformable benchmark。
- **Benchmark-Real Transfer 协议**：先做离线题库，再用少量真实机器人任务验证外部有效性。

## Local PDF reference
![[paperPDFs/Benchmark/arXiv_2025/2025_ManipBench_Benchmarking_Vision_Language_Models_for_Low_Level_Robot_Manipulation.pdf]]