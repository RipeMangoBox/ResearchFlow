---
title: "Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use"
venue: arXiv
year: 2025
tags:
  - Others
  - task/multi-step-reasoning
  - task/tool-use
  - reinforcement-learning
  - offline-rl
  - synthetic-data
  - dataset/HotPotQA
  - dataset/GSM8K
  - dataset/CofCA
  - dataset/MuSiQue
  - dataset/BeerQA
  - opensource/no
core_operator: "将多步工具使用轨迹按动作切成子轨迹，用上下文感知的生成式奖励模型逐步打分并进行离线强化学习。"
primary_logic: |
  问题文本 + 可调用工具（搜索/计算器） + 模型生成的离线多步轨迹
  → 将每条轨迹按动作切分为多个子轨迹，并按步骤合理性进行过程过滤
  → 用生成式奖励模型对每一步动作在当前上下文中的合理性打分，执行离线策略优化
  → 提升多步推理、工具调用规划、停止决策与跨任务泛化
claims:
  - "SWiRL 相对基线在 GSM8K、HotPotQA、CofCA、MuSiQue、BeerQA 上分别带来 21.5%、12.3%、14.8%、11.1%、15.3% 的相对准确率提升 [evidence: comparison]"
  - "仅用 HotPotQA 训练的 SWiRL 仍可让 GSM8K 零样本表现相对提升 16.9%，反向仅用 GSM8K 训练也可让 HotPotQA 提升 9.2% [evidence: comparison]"
  - "对 SWiRL 而言，按步骤进行过程过滤但不过滤最终答案的训练数据优于结果过滤，说明模型能从最终答案错误但步骤局部合理的轨迹中学习 [evidence: ablation]"
related_work_position:
  extends: "OREO (Wang et al. 2024)"
  competes_with: "OREO (Wang et al. 2024); DQO (Liu et al. 2024)"
  complementary_to: "RLEF (Gehring et al. 2025)"
evidence_strength: strong
pdf_ref: "paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Collective_multi_agent_reasoning/arXiv_2025/2025_Synthetic_Data_Generation_&_Multi_Step_RL_for_Reasoning_&_Tool_Use.pdf"
category: Others
---

# Synthetic Data Generation & Multi-Step RL for Reasoning & Tool Use

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2504.04736)
> - **Summary**: SWiRL 通过“多步轨迹切分 + 逐步过程奖励 + 离线 RL”，把原本只在最终答案上给反馈的训练，改成对每一步工具调用与推理动作给反馈，从而提升多步 reasoning 与 tool use。
> - **Key Performance**: GSM8K 相对提升 21.5%；仅用 HotPotQA 训练时，GSM8K 零样本相对提升 16.9%

> [!info] **Agent Summary**
> - **task_path**: 问题文本 + 搜索/计算器工具 -> 多步查询/推理/停止决策 -> 最终答案
> - **bottleneck**: 终局单奖励无法为中间查询、工具调用时机和停止条件做有效信用分配
> - **mechanism_delta**: 把每条多步轨迹按动作切成子轨迹，并用上下文感知奖励模型逐步打分做离线 RL
> - **evidence_signal**: 5 个基准稳定增益 + 过程过滤优于结果过滤 + 过程标签准确率提升
> - **reusable_ops**: [轨迹切分为子轨迹, 过程过滤加步骤级奖励]
> - **failure_modes**: [超出固定步数预算的长链任务, 数值/单位判断或工具解析出错]
> - **open_questions**: [能否用开源 judge 复现效果, 能否扩展到真实在线环境与更长 horizon]

## Part I：问题与挑战

这篇论文真正要解决的，不是“模型最后答得对不对”，而是**多步 agent 任务里的信用分配失配**。

传统 RLHF / RLAIF / RLEF 大多把整段回答视为一次动作，只在结尾给奖励；但在多跳问答、数学推理、工具调用里，模型实际在做的是一串决策：

1. 先判断要不要调用工具  
2. 再决定查什么/算什么  
3. 读回结果后判断是否继续  
4. 最后再综合成答案

这里的瓶颈是：**一旦前面某一步 query 或计算方向错了，后面再努力也常常救不回来；而终局奖励又太稀疏，看不出究竟是哪一步出了问题。**

为什么现在值得做：
- LLM 应用正在从单轮对话转向 agentic reasoning + tool use。
- 在线 RL 直接接真实工具，训练会被环境延迟拖慢，也更难复现。
- 只保留“最终答对”的轨迹，会丢掉很多“前几步其实合理、只是最后综合失败”的可学习信号。

**输入 / 输出接口**很明确：
- 输入：问题 + 历史上下文 + 工具反馈
- 每一步动作：`<search_query>` / `<math_exp>` / `<answer>`
- 输出：最终简短答案
- 约束：问答任务最多 5 步，数学任务最多 10 步

边界上，它验证的是**可解析、可执行的工具接口**（检索 / 计算器），不是开放网页浏览或持续变化的复杂环境。

## Part II：方法与洞察

SWiRL 可以概括成一句话：**先合成多步轨迹，再把轨迹切细，然后对“下一步动作”做离线 RL。**

整体分两阶段：

1. **Stage 1：合成多步数据**
   - 用 Gemma 2 配合搜索或计算器生成多步轨迹
   - HotPotQA 上生成 5 万条轨迹，GSM8K 上生成 3.75 万条轨迹
   - 每条轨迹都包含若干步：推理文本 + 工具调用或最终答案

2. **Stage 2：Step-Wise RL**
   - 如果一条轨迹有 \(k\) 个动作，就把它拆成 \(k\) 个子轨迹
   - 第 \(i\) 个子轨迹只看到从起点到当前步的上下文
   - 奖励模型只评估“当前这一步在当前上下文里是否合理”
   - 再据此做离线策略优化

推理时也保持同样范式：模型一步步决定继续搜/算，还是直接回答。

### 核心直觉

**真正的改动**不是“多加一点 RL”，而是把优化目标从“最后答对”改成“当前上下文下，下一步该怎么做”。  

这带来了三个关键变化：

- **奖励密度变了**：从整条轨迹一个终局奖励，变成每一步一个上下文奖励
- **训练分布变了**：一条长轨迹被展开成多个局部决策状态
- **监督信号变了**：不仅看最后对错，还看每一步是否合理

因此改变的不是表面 loss，而是**信息瓶颈**：
- 以前模型只知道“这题最后错了”
- 现在模型知道“这一步 query 不合理”“这时还不该停”“这时已经足够回答了”

这为什么有效：
1. **缓解长链 credit assignment**：模型学的是下一步动作质量，而不是赌最终侥幸答对。
2. **保留有价值的负例**：最终答案错的轨迹里，前几步可能仍然是合理的；SWiRL 能从这些局部正确动作里学习。
3. **过程过滤比结果过滤更契合 RL**：只要每一步局部合理，RL 就能优化策略；不需要把学习限制在“最终全对”的窄分布上。

### 战略权衡

| 设计选择 | 改变了什么 | 能力收益 | 代价/风险 |
|---|---|---|---|
| 轨迹切分为子轨迹 | 终局稀疏奖励 → 步骤级稠密奖励 | 更好学习 query、停止、综合 | 依赖步骤级评分质量 |
| 过程过滤而非结果过滤 | correct-only 数据 → 局部合理的混合结果数据 | 泛化更强，不依赖金标签 | 可能引入最终错误轨迹噪声 |
| 离线工具轨迹 | 在线环境慢且不稳定 → 固定训练集 | 可并行、可复现 | 不能覆盖真实在线分布漂移 |
| 生成式奖励模型 | 手工规则难覆盖多样动作 | 可统一评估搜索、计算、回答 | 依赖闭源 judge，存在偏差 |

一句话总结 Part II：**SWiRL 学的不是“答案模板”，而是“在多步工具环境里做下一步决策的策略”。**

## Part III：证据与局限

### 关键证据信号

- **多数据集比较信号**：在 GSM8K、HotPotQA、CofCA、MuSiQue、BeerQA 五个基准上都优于基线，且提升不是只出现在单一任务。
- **跨任务迁移信号**：只用 HotPotQA 训练，GSM8K 仍提升 16.9%；只用 GSM8K 训练，HotPotQA 也提升 9.2%。这比单纯“数据集蒸馏”更像是在学通用多步推理/工具使用策略。
- **机制 ablation 信号**：SWiRL 最好的不是 outcome filtering，而是 **process-only filtering**。这直接支持论文核心论点：RL 可以从“局部合理但结局错误”的轨迹中获益。
- **对比 SFT 信号**：同样的合成轨迹下，SFT 往往不如 SWiRL，甚至会比 base model 更差；说明这里的收益不是“多看点轨迹”本身，而是**逐步奖励优化**。
- **过程质量信号**：平均 process label accuracy 从 HotPotQA 上的 82.5% 提升到 91.0%，在分布外 GSM8K 上也从 87.5% 提升到 91.6%。这说明提升不只是答案格式层面，而是中间决策质量真的更好了。
- **非单纯蒸馏信号**：SWiRL 后的模型在部分 OOD benchmark 上还能超过作为 reward model 的 Gemini 1.5 Pro，说明它不是简单模仿 judge。

两个最关键的数字可以记住：
- **GSM8K 相对提升 21.5%**
- **HotPotQA 上平均过程正确率 82.5% → 91.0%**

### 局限性

- **Fails when**: 任务需要超过 5/10 步的更长 horizon；工具不是这种可解析的检索/计算接口；数值与单位判断受 LLM judge 噪声影响时；小模型（2B/9B）做跨域泛化时效果不稳定。
- **Assumes**: 有一个足够强的步骤级 reward model / judge（文中是闭源 Gemini 1.5 Pro Thinking）；能离线批量生成带工具反馈的轨迹；推理基础设施包含 Gecko 检索与 SymPy 计算；评测接受 LLM-as-a-judge。
- **Not designed for**: 真正在线、状态持续变化的交互环境；开放式网页导航/软件操作；安全关键场景下需要严格可验证奖励的部署。

额外要注意的复现边界：
- 训练和过滤强依赖**闭源 Gemini 1.5 Pro**；
- 评测也大量依赖 **LLM judge**，虽然作者做了小规模人工核查，但绝对分数仍可能受 judge 偏差影响；
- 因此这篇论文的方法思想很可复用，但**完整实验链路不是纯开源可复现**。

### 可复用部件

- **轨迹 → 子轨迹** 的数据重构方式  
- **过程过滤优先于结果过滤** 的筛选思路  
- **上下文感知的步骤级生成式奖励**  
- **固定离线工具执行日志** 的可复现 RL 管线  

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Collective_multi_agent_reasoning/arXiv_2025/2025_Synthetic_Data_Generation_&_Multi_Step_RL_for_Reasoning_&_Tool_Use.pdf]]