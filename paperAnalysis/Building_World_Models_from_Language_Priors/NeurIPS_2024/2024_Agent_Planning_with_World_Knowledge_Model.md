---
title: "Agent Planning with World Knowledge Model"
venue: NeurIPS
year: 2024
tags:
  - Embodied_AI
  - task/agent-planning
  - trajectory-contrast
  - knn-retrieval
  - dual-model-architecture
  - dataset/ALFWorld
  - dataset/WebShop
  - dataset/ScienceWorld
  - opensource/full
core_operator: "用专家/探索轨迹对比自蒸馏出任务知识与状态知识，训练独立 WKM 在推理时分别提供全局先验和局部动作约束。"
primary_logic: |
  任务指令 + 部分观测交互历史
  → 从专家轨迹与经验探索轨迹合成实例级任务知识，并从专家轨迹按步总结状态知识、构建 (action, state, action) 检索库
  → 训练共享骨干的 agent LoRA 与 WKM LoRA，推理时由 WKM 生成任务知识与状态知识，再与 agent 动作分布加权
  → 输出更少试错、更少幻觉动作的下一步规划
claims:
  - "在 Mistral-7B、Gemma-7B、Llama-3-8B 上，WKM 在 ALFWorld、WebShop、ScienceWorld 的全部测试划分中都优于 NAT、ETO 和 KnowAgent；例如 Mistral-7B 在 ALFWorld unseen 达到 76.87 [evidence: comparison]"
  - "在 ALFWorld 的分析实验中，WKM 的平均规划步数低于 NAT、ETO、KnowAgent，且含无效动作的轨迹比例降到 32.86%（seen）/29.85%（unseen） [evidence: analysis]"
  - "消融表明任务知识和状态知识都带来增益，而把状态知识显式拼接进上下文反而明显差于检索式隐式约束；去掉 rejected trajectories 也会降级 [evidence: ablation]"
related_work_position:
  extends: "KnowAgent (Zhu et al. 2024)"
  competes_with: "ETO (Song et al. 2024); KnowAgent (Zhu et al. 2024)"
  complementary_to: "Reflexion (Shinn et al. 2023); AgentTuning (Zeng et al. 2023)"
evidence_strength: strong
pdf_ref: "paperPDFs/Building_World_Models_from_Language_Priors/NeurIPS_2024/2024_Agent_Planning_with_World_Knowledge_Model.pdf"
category: Embodied_AI
---

# Agent Planning with World Knowledge Model

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2405.14205) · [Code](https://github.com/zjunlp/WKM)
> - **Summary**: 这篇论文把 LLM agent 规划中的“世界知识”从手工规则或隐式记忆，升级为一个可训练的独立知识模型：开局提供任务级先验，过程中提供状态级约束，从而减少盲目试错和幻觉动作。
> - **Key Performance**: Mistral-7B 上 ALFWorld seen/unseen 达到 73.57/76.87（较最佳基线 +3.13/+5.44）；ALFWorld 含无效动作的轨迹比例降至 32.86%/29.85%。

> [!info] **Agent Summary**
> - **task_path**: 文本任务指令 + 部分可观测动作/观察历史 -> 下一步环境动作
> - **bottleneck**: 单一 agent LM 同时承担“任务先验”和“当前状态判断”，导致前期盲搜、后期状态混乱与幻觉动作
> - **mechanism_delta**: 新增独立 WKM，先生成实例级 task knowledge 作为全局先验，再用 state knowledge 检索出的动作分布对 agent 决策做隐式约束
> - **evidence_signal**: 3 个 benchmark × 3 个开源骨干的一致领先，外加 task/state 消融和步数/无效动作率分析
> - **reusable_ops**: [trajectory-pair knowledge synthesis, state-to-action retrieval reranking]
> - **failure_modes**: [state-knowledge-base 分布外检索失真, 显式 state knowledge 拼接导致上下文污染]
> - **open_questions**: [如何在线更新世界知识, 如何扩展到多模态或可搜索的 world model]

## Part I：问题与挑战

**What/Why：真正的瓶颈不是“LLM 不会输出动作”，而是它没有一个可调用的世界知识层。**

这篇论文研究的是**部分可观测、文本交互式的 agent planning**。输入是任务指令和历史轨迹（动作、观察、可选 reasoning），输出是下一步动作。作者认为，现有 LLM agent 的错误主要来自两类知识缺失：

1. **全局规划缺先验**：任务刚开始时环境信息很少，模型不知道目标物大概率在哪、通常工作流是什么，于是只能盲目试错。
2. **局部规划缺状态约束**：随着轨迹变长，模型容易混淆“我现在已经做过什么、当前环境允许什么”，从而产生 hallucinatory actions。

因此，问题不只是 prompt 不够好，也不只是负样本训练不够多；更深层的瓶颈是：**知识建模与动作生成被硬塞进同一个自回归策略模型里**。模型既要记“任务常识”，又要跟踪“当前状态”，负担过重。

**为什么现在值得做：**
- 开源 7B–8B LLM 已足以作为 agent backbone，但纯 REACT/Reflexion 在复杂交互环境中仍明显不稳。
- 现成 benchmark 已提供 expert trajectories、探索轨迹和标准评测，可支持“从轨迹里学知识”。
- 这使得 world knowledge 可以从**手工规则**转向**参数化、可复用的模块**。

**边界条件：**
- 只处理文本化模拟环境，不是视觉输入或真实机器人闭环。
- 动作集合由环境给定，属于离散候选动作。
- WKM 是 **world knowledge model**，不是能显式滚动预测环境转移、配合搜索的完整 world model。

## Part II：方法与洞察

作者的核心设计不是简单“给 agent 多喂一些知识文本”，而是把系统拆成两个角色：

- **Agent model**：负责输出动作。
- **World Knowledge Model (WKM)**：负责生成两类知识  
  - **Task knowledge**：任务开始前的全局先验  
  - **State knowledge**：每一步对当前状态的概括

然后用 task knowledge 改变全局搜索方向，用 state knowledge 的检索结果约束局部动作选择。

### 核心直觉

**改变了什么：** 从“单模型直接拟合动作轨迹”，变成“知识模型先塑形动作分布，再由 agent 执行动作”。

**改变了哪个瓶颈：**
- task knowledge 把“缺少任务先验”转成了**实例级工作流知识生成**问题；
- state knowledge 把“长上下文下的状态混乱”转成了**相似状态下下一动作约束**问题。

**能力上发生了什么变化：**
- 前期更快进入正确搜索子空间，减少 blind trial-and-error；
- 后期更少输出与环境状态不匹配的非法动作；
- 相比固定数据集级规则，实例级知识对 unseen tasks 更稳。

**为什么这套设计有效：**
1. **任务知识来自成功/失败轨迹对比，而非手工常识。**  
   它更接近“导致成功的关键因素总结”。
2. **状态知识不直接拼进 agent 上下文，而是用于检索。**  
   这样把冗长自然语言压缩成一个动作分布约束，避免上下文污染。
3. **最终动作由 agent 概率与 knowledge 概率加权。**  
   state knowledge 充当约束器，而不是完全替代 planner。

### 方法流程

1. **经验代理探索**  
   先用 expert trajectories 训练一个 experienced agent，再让它回到训练任务中探索，得到 sampled/rejected trajectories。  
   目的不是拿最差失败样本，而是拿“有一定能力但还会犯错”的轨迹，方便抽取更有针对性的知识。

2. **任务知识合成**  
   把 expert trajectory 当作 chosen、探索轨迹当作 rejected，提示模型自己总结：  
   “这类任务应该怎么做、物体可能在哪、动作 workflow 是什么。”  
   得到的是**实例级 task knowledge**。

3. **状态知识总结与知识库构建**  
   对 expert trajectory 的每一步，总结当前状态知识；再构建 `(a_t, s_t, a_{t+1})` 三元组库。  
   推理时，当前 state knowledge 会检索这个库，统计“相似状态下常见的下一动作”。

4. **分开训练两个 LoRA**  
   - **Agent LoRA**：学习在 task knowledge 条件下生成动作；
   - **WKM LoRA**：学习生成 task knowledge 与 state knowledge。  
   二者共享同一个 backbone，但职责分离。

5. **推理时联合决策**  
   - 开始时：WKM 先生成 task knowledge，作为全局先验拼到任务描述后；
   - 每一步：WKM 生成当前 state knowledge，检索知识库得到下一动作分布；
   - 最终：把检索分布和 agent 自身动作分布做加权，选出下一步动作。  
   关键点是：**state knowledge 本身不进入 agent 的上下文**。

### 战略取舍

| 设计选择 | 主要收益 | 代价/风险 |
|---|---|---|
| 用 chosen vs rejected 轨迹对比生成 task knowledge | 把成功经验提炼成实例级工作流先验，提升全局规划与 unseen 泛化 | 依赖 expert/探索轨迹质量；知识总结错误会污染先验 |
| 用 state knowledge 检索约束，而不是显式把 state text 拼进 prompt | 降低 hallucination，避免长文本反馈挤占上下文 | 知识库来自训练集，分布外状态时泛化有限 |
| 独立 WKM + agent，共享 backbone 但分开 LoRA | 模块化、可 weak-guide-strong、可多任务复用 | 推理需额外生成与检索，时间开销约 2.5:1 |

## Part III：证据与局限

**So what：这篇论文的能力跃迁，不在于“更会拟合轨迹”，而在于“显式用知识重塑动作分布”。**

### 关键证据信号

- **比较信号：跨数据集、跨骨干的一致增益**  
  在 Mistral-7B、Gemma-7B、Llama-3-8B 上，WKM 都稳定超过 NAT、ETO、KnowAgent。  
  最醒目的例子是 Mistral-7B 在 ALFWorld unseen 上达到 **76.87**，高于最佳基线 **71.43**。这说明收益不是某个 backbone 的偶然适配。

- **分析信号：更少盲搜，更少幻觉动作**  
  在 ALFWorld 的分析实验中，WKM 的平均规划步数低于 NAT/ETO/KnowAgent；同时，含无效动作的轨迹比例降到 **32.86% / 29.85%**（seen / unseen）。  
  这正对应了论文最初提出的两个瓶颈：全局试错过多、局部动作幻觉过多。

- **消融信号：task knowledge 是主增益，state knowledge 是约束增益**  
  去掉 task knowledge 或 state knowledge 都会掉点，但 task knowledge 的贡献更大、更稳；说明把“搜索方向”先纠正，比后续局部修补更重要。  
  同时，**显式把 state knowledge 拼进上下文反而更差**，说明作者抓住的真实因果点是“长上下文噪声”，而不是简单“知识越多越好”。

- **迁移信号：知识层有复用潜力**  
  弱知识模型指导强 agent（Mistral-WKM -> GPT-4/ChatGPT）仍有提升；多任务统一 WKM 也未明显退化。  
  这表明 WKM 更像一个外挂的 knowledge layer，而不只是单任务 trick。

### 局限性

- **Fails when**: 当前状态明显偏离训练期 state knowledge base、环境需要在线吸收新反馈、或把知识库权重设得过高时；论文中 `γ=0`（完全相信知识库）会明显崩塌，说明检索知识只能做约束，不能单独承担规划。
- **Assumes**: 有高质量 expert trajectories 与探索轨迹；知识可以被文本化表达；环境给出离散候选动作；训练数据中部分 rationales 与 WebShop 轨迹依赖 GPT-4/启发式数据构造；推理需额外生成与检索，时间开销约为纯 agent 的 **2.5x**；实验资源为 8×V100 32G、约 12 小时。
- **Not designed for**: 真正可搜索的 world model（如结合 MCTS 的环境预测）、多模态世界知识、开放动作空间的端到端规划、或会随环境持续在线更新的知识系统。

### 可复用组件

- **trajectory-pair -> task knowledge**：把 chosen/rejected 轨迹差异蒸馏成可读的工作流知识。
- **state knowledge -> action prior**：用状态文本做检索键，把局部知识转成动作分布约束。
- **外挂式 knowledge LoRA**：知识模块可与更强或不同的 agent backbone 组合，支持 weak-guide-strong。

## Local PDF reference

![[paperPDFs/Building_World_Models_from_Language_Priors/NeurIPS_2024/2024_Agent_Planning_with_World_Knowledge_Model.pdf]]