---
title: "Real-Time Verification of Embodied Reasoning for Generative Skill Acquisition"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/embodied-skill-acquisition
  - task/robot-manipulation
  - process-reward-model
  - mcts
  - dynamic-exemplar-retrieval
  - dataset/PartNetMobility
  - opensource/promised
core_operator: 用动态示例检索与MCTS自动密集标注训练过程奖励模型，在技能生成时实时验证并重排场景配置和子任务监督。
primary_logic: |
  任务规格（名称/描述/部件树/初始配置）与相似成功任务 → LLM生成场景配置、子任务监督和成功指标 → 通过MCTS补全并执行候选轨迹以自动标注子任务贡献 → 训练PRM对轨迹前缀打分并筛选更可学的技能方案
claims:
  - "使用动态 exemplar task pool 可将 BaseModel 的 ATSR 从 0.53 提升到 0.74，并将平均子任务数从 17.8 降到 9.0 [evidence: comparison]"
  - "在 task-based split 上，PRM-last 将 ATSR/ASSR 从 0.67/0.67 提升到 0.91/0.91；在 solution-based split 上提升到 0.92/0.97 [evidence: comparison]"
  - "基于执行反馈训练的 PRM 在文中报告设置下优于 GPT-4o、Claude-3.5-Sonnet、Gemini-2.0-Flash、DeepSeek-V3 和 Qwen2.5-72B-Instruct 的 LLM-as-a-Judge 验证基线 [evidence: comparison]"
related_work_position:
  extends: "RoboGen (Wang et al. 2024)"
  competes_with: "LLM-as-a-Judge; RoboGen (Wang et al. 2024)"
  complementary_to: "Eureka (Ma et al. 2023); Domain Randomization (Tobin et al. 2017)"
evidence_strength: moderate
pdf_ref: paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_Real_Time_Verification_of_Embodied_Reasoning_for_Generative_Skill_Acquisition.pdf
category: Embodied_AI
---

# Real-Time Verification of Embodied Reasoning for Generative Skill Acquisition

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2505.11175)
> - **Summary**: 这篇工作把数学推理中的过程验证器迁移到具身技能习得中，用“相似任务检索 + 子任务成功指标 + MCTS 自动密集标注的 PRM”来实时筛选更值得训练的场景配置与子任务监督。
> - **Key Performance**: 引入 exemplar task pool 后 BaseModel 的 ATSR 由 0.53 升至 0.74；PRM-last 在 novel-task split 上将 ATSR/ASSR 从 0.67/0.67 提升到 0.91/0.91。

> [!info] **Agent Summary**
> - **task_path**: 任务描述/部件树/初始配置 -> 场景配置与子任务监督/成功指标 -> 可执行的具身技能学习方案
> - **bottleneck**: 候选 supervision 是否“值得训练”只能靠昂贵仿真与子任务训练后验判断，新任务又缺少结构化参考
> - **mechanism_delta**: 用动态相似任务检索补足结构先验，再用 MCTS 自动标注训练出的 PRM 对推理前缀做实时验证与重排
> - **evidence_signal**: PRM-last 将 task-split 的 ATSR/ASSR 从 0.67/0.67 提升到 0.91/0.91
> - **reusable_ops**: [dynamic-exemplar-retrieval, MCTS-prefix-reward-labeling]
> - **failure_modes**: [per-subtask-policy-training-cost, mis-specified-success-metrics]
> - **open_questions**: [how-to-reduce-rollout-and-training-cost, how-to-close-sim2real-gap]

## Part I：问题与挑战

**What/Why**：这篇文章真正要解决的，不是“LLM 会不会分解任务”，而是**LLM 生成出来的场景配置和子任务监督，到底哪一条值得投入昂贵的技能训练**。

在生成式具身技能习得里，系统需要同时生成：

- **场景配置**：机器人、物体、关节、空间关系；
- **子任务分解**：把长任务拆成可训练的短步骤；
- **训练监督**：每个子任务的 reward function 或 execution primitive；
- **成功指标**：何时算子任务完成、何时算全任务完成。

### 真正瓶颈

1. **具身任务比数学题更欠结构化**  
   数学题通常“题目给定、答案可验”；这里连 scene、subtask、reward 都要一起生成。

2. **正确性无法直接验算**  
   子任务 supervision 看起来合理，不代表真的可学、可执行。必须经过仿真训练/执行才能知道。

3. **穷举验证太贵**  
   每个子任务可能要跑 motion planning 或 RL；论文实现里 RL 子任务可达 **1M environment steps**，再叠加 MCTS 补全与多次仿真，成本非常高。

### 输入 / 输出接口

- **输入**：任务名与描述、PartNetMobility 的 articulation tree、初始配置、仿真 API。
- **输出**：场景配置 + 一串子任务；每个子任务附带训练监督与 success metric。

### 为什么现在值得做

RoboGen 一类框架已经证明：LLM 可以大规模“生成任务和 supervision”。  
但一旦生成能力扩张，系统的主瓶颈就从“想不出候选方案”，转成“**怎样快速辨别哪条方案真的能学成**”。如果没有 verifier，规模化只会更快地产生无效 supervision。

### 边界条件

这篇论文的结论主要成立于：

- articulated object 的**仿真操作任务**；
- 可访问底层 API、关节状态、link 状态；
- 每个子任务能被显式写成 reward-based 或 primitive-based supervision；
- 使用 RoboGen 风格的 motion planning / RL 训练栈。

---

## Part II：方法与洞察

VERGSA 的核心不是重新发明低层控制器，而是把“技能生成”变成一个**可被实时验证的推理过程**。

### 方法主线

1. **动态 exemplar task pool**
   - 维护一个成功任务池；
   - 用任务名+描述做 embedding；
   - 对新任务检索 top-2 相似成功任务；
   - 新任务一旦执行成功，再加入池中。
   
   作用：给新任务补充结构先验，减少 LLM 在 scene 和 subtask 生成上的发散。

2. **把 success metric 也一并生成**
   - policy model 不只生成“做什么”，还生成“怎么算成功”。
   - 这是从数学验证迁移到具身任务的关键：先把“可验证标准”造出来，后面才有 verifier 可学。

3. **ARLET-MCTS 自动奖励标注**
   - 先生成若干 base solutions；
   - 从某个中间 subtask 出发，用 completer 补全后续步骤；
   - 在仿真中执行这些 completed solutions；
   - 若某个 subtask 自身成功，且它所属 completed solution 最终也成功，则标正例，否则标负例。

   作用：把昂贵而稀疏的 whole-task 成败，转成对子任务前缀更密集的训练信号。

4. **PRM 作为 critic，对前缀打分**
   - 输入不是整条最终答案，而是**scene + 连续子任务前缀**；
   - 输出该前缀通向成功技能习得的概率；
   - 推理时用它给多个候选解排序，选出更值得训练的那条。

### 核心直觉

**把“完整技能链执行后才知道好坏”的后验判断，改成“前缀是否位于成功轨迹流形上”的前馈验证问题。**

#### 什么变了

- 从**固定 exemplar 提示** → **动态相似任务检索**
- 从**只看全任务成败** → **显式生成子任务 success metric**
- 从**终局稀疏反馈** → **MCTS 产生的前缀级密集标注**
- 从**LLM 直接拍脑袋选方案** → **PRM 基于执行反馈重排方案**

#### 变的是哪种瓶颈

- **信息瓶颈**：相似任务检索降低 prompt 空间熵，减少无根据分解；
- **监督瓶颈**：success metric 让“是否成功”从隐变量变成可检查变量；
- **搜索瓶颈**：PRM 学的是“这段前缀值不值得继续”，所以可以更早筛掉坏轨迹；
- **成本瓶颈**：虽然标注仍贵，但比对整条候选链全部穷举训练更可扩展。

#### 为什么这套设计有效

因为具身技能 supervision 的问题，不在于“代码是否合法”，而在于“**这段 supervision 训练出来的 policy 能否把任务做成**”。

- **LLM-as-a-Judge** 更像静态审稿人：看语法、API、逻辑是否像样；
- **PRM** 更像执行导向的 critic：学的是“这条前缀在真实执行反馈下是否会通向成功”。

这就是它优于纯 judge 式验证的因果原因。

### 战略性 trade-off

| 设计选择 | 改变了什么约束 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 动态 exemplar 检索 | 给新任务注入相似成功结构 | 更少无意义子任务，更稳的 scene/supervision 生成 | 依赖任务池质量，可能引入检索偏置 |
| 子任务 success metric 显式生成 | 让具身过程可验证 | 可以对子任务早停、局部判断 | metric 本身若写错，会把 verifier 带偏 |
| MCTS 自动密集标注 | 把终局稀疏回报展开到中间步骤 | 无需手工 reward labeling 也能训练 PRM | 需要大量 rollouts 与仿真执行 |
| PRM 前缀验证 | 从“生成后盲试”改为“生成时重排” | 对新任务和见过任务都能提升成功率 | 聚合策略敏感，仍依赖训练分布 |

---

## Part III：证据与局限

**So what**：这篇论文的能力跃迁不在低层控制算法本身，而在于它让系统能更早、更便宜地判断“哪条技能生成轨迹值得训练”。这把 generative skill acquisition 从**盲目试错**推进到**验证驱动的搜索**。

### 关键证据信号

1. **Signal 1：示例池确实减少了“瞎分解”**
   - BaseModel 加入 exemplar task pool 后：
     - **ATSR：0.53 → 0.74**
     - **ASSR：0.55 → 0.80**
     - **平均子任务数：17.8 → 9.0**
   - GPT-4o 上也有一致增益（ATSR 0.81 → 0.86）。

   **结论**：相似任务检索不是装饰，它在实质上降低了无效步骤和无效 supervision。

2. **Signal 2：PRM 在新任务和已见任务上都有效**
   - **task-based split（看泛化）**：BaseModel 0.67/0.67 → PRM-last 0.91/0.91
   - **solution-based split（看同任务 refinement）**：BaseModel 0.56/0.62 → PRM-last 0.92/0.97

   **结论**：PRM 学到的不是简单记忆，而是对“成功前缀”的判别能力。

3. **Signal 3：PRM 明显强于 LLM-as-a-Judge**
   - 最好的 judge 基线在 task split 上 ATSR 大约 **0.78**；
   - solution split 上 ATSR 大约 **0.65**；
   - 都低于 PRM-last。

   **结论**：静态检查“代码是否看起来合理”，不如基于执行结果学习“它是否真能学成”。

4. **Signal 4：Last 聚合最好**
   - 多种聚合策略里，**last** 最优。

   **解释**：在这类串行具身任务里，最终子任务能否成功，往往隐含前面关键步骤已被正确铺垫；但这也意味着模型可能更依赖链尾信号。

### 关键指标

- **ATSR**：Average Task Success Rate  
- **ASSR**：Average Subtask Success Rate

这两个指标足够抓住论文核心：它到底有没有把“整体任务更容易成功”和“局部 supervision 更容易成功”同时做上去。

### 证据的边界

证据是积极的，但还算不上特别强：

- 主要建立在**单一仿真栈**上；
- PRM 数据规模为 **30 tasks / 150 solutions / 287 subtasks**；
- 缺少跨模拟器、跨机器人、跨感知输入形式的广泛验证；
- 主要对比是 internal base model 与 judge baseline，而不是大量外部 embodied pipeline。

因此把它评为 **moderate** 更合适。

### 局限性

- **Fails when**: 任务特别长、接触丰富且多分支，或 LLM 生成的 success metric 本身有偏差时，验证链条会失真；表 10 中已有多个任务仍是 0% 成功率，如 Open Door、Open Lighter Lid、Press Button to Access Menu、Adjust Fan Speed。
- **Assumes**: 依赖 PartNetMobility 式部件结构、RoboGen 风格 API、可执行仿真环境、每个子任务可单独训练的 RL/规划器；实验还依赖较高算力（8×RTX 4090）、多次 MCTS rollouts，以及对 246 个 task initiatives 的人工筛选；论文只承诺数据集公开，未在文中给出已发布代码/项目。
- **Not designed for**: 直接真机部署、端到端视觉闭环控制、没有显式子任务分解的策略学习，以及缺乏可程序化 success check 的开放环境。

### 可复用组件

- **动态相似任务检索**：适合任何“prompt 生成 supervision”的具身任务管线。
- **子任务 success metric 生成**：把不可直接验证的问题变成可执行判断。
- **MCTS 前缀级自动标注**：适合从稀疏执行结果合成 dense verifier 数据。
- **小型 PRM 重排器**：可与更强 policy model、reward generator 或 sim2real 模块组合。

## Local PDF reference

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_Real_Time_Verification_of_Embodied_Reasoning_for_Generative_Skill_Acquisition.pdf]]