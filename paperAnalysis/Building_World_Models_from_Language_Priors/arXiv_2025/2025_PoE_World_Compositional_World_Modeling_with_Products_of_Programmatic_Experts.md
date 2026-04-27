---
title: "PoE-World: Compositional World Modeling with Products of Programmatic Experts"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/world-model-learning
  - task/model-based-planning
  - program-synthesis
  - product-of-experts
  - hierarchical-planning
  - dataset/Atari
  - opensource/full
core_operator: 将复杂世界动力学拆成由LLM合成的大量小程序专家，并用指数加权乘积与硬约束组合成可规划的概率世界模型
primary_logic: |
  少量对象级演示与交互历史 → 按对象/交互合成程序专家、把确定性代码解释为概率分布并优化/剪枝专家权重，再叠加硬约束 → 得到可在新关卡上泛化并支持分层规划/仿真预训练的组合式世界模型
claims:
  - "PoE-World + Planner 在 Montezuma’s Revenge 与其重组关卡 MR-Alt 上都得到 100 分，而表中的 PPO、ReAct 与 WorldCoder 均为 0 分；其真实环境训练预算不超过 3k steps [evidence: comparison]"
  - "相较 WorldCoder，PoE-World 在 Montezuma’s Revenge 的下一观测预测准确率从 0.36 提升到 0.75，在 1000 个随机测试帧上的准确率从 0.10 提升到 0.31 [evidence: comparison]"
  - "去掉 hard constraints 后，Montezuma’s Revenge 的规划成功次数从 5/9 降到 2/9，而一步预测准确率几乎不变（0.31→0.30），说明约束主要改善长时规划可用性 [evidence: ablation]"
related_work_position:
  extends: "WorldCoder (Tang et al. 2024)"
  competes_with: "WorldCoder (Tang et al. 2024); CodeWorldModels (Dainese et al. 2024)"
  complementary_to: "VisualPredicator (Liang et al. 2024); OCAtari (Delfosse et al. 2023)"
evidence_strength: moderate
pdf_ref: paperPDFs/Building_World_Models_from_Language_Priors/arXiv_2025/2025_PoE_World_Compositional_World_Modeling_with_Products_of_Programmatic_Experts.pdf
category: Embodied_AI
---

# PoE-World: Compositional World Modeling with Products of Programmatic Experts

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2505.10819), [Project](https://topwasu.github.io/poe-world), [Code](https://github.com/topwasu/poe-world)
> - **Summary**: 论文把“学一个巨大的世界程序”改成“学很多个局部因果程序再做乘积组合”，从而在极少对象级演示下学出可规划、可组合泛化的世界模型。
> - **Key Performance**: 在 MR / MR-Alt 上分别取得 **100 / 100**，而 PPO、ReAct、WorldCoder 均为 **0 / 0**；在 MR 的 1000 个随机测试帧上，下一观测预测准确率 **0.31 vs 0.10（WorldCoder）**。

> [!info] **Agent Summary**
> - **task_path**: 少量对象级演示 + 历史观测/动作 -> 下一时刻对象状态分布 -> 分层规划/仿真预训练
> - **bottleneck**: 单一大程序的世界模型搜索与修补在复杂、随机、部分可观测环境中不可扩展
> - **mechanism_delta**: 把单体世界程序替换为按对象/交互分解的概率程序专家集合，并用 hard constraints 收紧可行状态支撑集
> - **evidence_signal**: 在短演示和 <3k 真实环境步的低数据设定下，只有 PoE-World 能在 MR 与 MR-Alt 上拿到 100 分
> - **reusable_ops**: [对象变化到程序专家的两阶段LLM合成, 专家权重优化与低权重剪枝]
> - **failure_modes**: [依赖高质量对象检测与手工修补的OCAtari前端, 长时规划对约束质量和搜索过程较敏感]
> - **open_questions**: [如何从像素端自动得到可编程对象表征, 如何把程序世界模型进一步用于主动探索与奖励学习]

## Part I：问题与挑战

这篇论文要解决的，不是“如何把 Atari 打到最高分”，而是更基础的一步：**能否在只有极少观测时，快速学出一个可用的世界模型**，并让它支持后续规划与泛化。

### 1. 真正的问题是什么
作者针对的是复杂、非 gridworld 的环境动力学建模，目标输入/输出接口是：

- **输入**：短演示轨迹（<1000 帧）、对象级观测历史、动作历史
- **输出**：下一时刻对象状态的概率分布
- **用途**：给规划器做 imagined rollout，或当作模拟器给 RL 预训练

### 2. 真正的瓶颈是什么
作者抓得很准：现有两类路线各卡在不同地方。

- **神经世界模型**：足够灵活，但样本效率差，且会出现“穿墙、瞬移”这类违反因果/物理的 rollout。
- **程序世界模型**：样本效率高，但如果要求 LLM 一次性写出一个完整大程序，搜索空间会爆炸；一处 bug 还会污染整套世界知识。

所以核心瓶颈并不是“LLM 不会写代码”，而是：

> **如何把复杂世界动力学分解成可独立发现、独立调权、可组合复用的小规律。**

### 3. 为什么现在值得做
因为 LLM 的代码合成能力，已经足以稳定生成很多**局部规则**；但此前程序世界模型主要还停留在文本/网格世界。PoE-World 的价值在于：它把这个方向首次推到更复杂的 Atari 对象世界，并测试了**少样本、随机性、部分可观测、长时规划**这几个关键难点。

### 4. 边界条件
这篇论文的边界也很明确：

- **不是**从像素端到端学世界模型，而是依赖对象检测后的符号输入
- **不是**解决 exploration
- **不是**解决 reward learning
- **也不是**把 planning 本身做到最优，只是证明学到的世界模型“足够可用”

---

## Part II：方法与洞察

### 方法骨架

PoE-World 的核心表示是：**世界模型 = 多个程序专家的乘积**。

#### 1. 表示：一个世界，不再由一个程序负责
每个 expert 都是一个小 Python 程序，只表达一个局部因果律，例如：

- 玩家碰到 skull 会死亡
- 玩家站在 platform 上按 RIGHT 会向右移动
- 玩家接触 ladder 时按 DOWN 会改变纵向速度

这些程序不是直接输出单一点预测，而是被解释成**概率分布**：

- 程序显式设置的对象属性：给一个尖峰分布
- 未设置的属性：给均匀分布
- 多个 expert 再做加权乘积，形成整体下一时刻分布

这一步很关键：它把“符号规则”变成了“可组合的概率模型”。

#### 2. 可计算性：按对象属性分解
为了让乘积模型可归一化，作者假设对象各属性在给定历史下条件独立，因此可以按 feature 分别归一化。代价是独立性假设更强，但换来 tractable inference。

#### 3. 学习：LLM 负责提案，优化器负责筛选
学习流程是一个循环：

1. 从演示/交互轨迹中取一小段 transition
2. 把对象变化转成自然语言描述
3. 用多个 LLM 合成模块生成候选 causal explanations，再转成 Python experts
4. 用最大似然去优化每个 expert 的标量权重
5. 剪掉低权重 expert
6. 有新轨迹后继续“debug”世界模型

一个很实用的设计是：**LLM 只负责生成很多局部假说，不负责决定谁最终可信**。可信度由后续权重学习决定。

#### 4. Hard constraints：给 rollout 加“物理可行性门”
PoE 乘积模型会产生“模糊但大致对”的未来分布，但长时规划时，这种模糊会累积成灾。作者于是额外学习一组 hard constraints，例如：

- 玩家在梯子上时身体中心要对齐梯子
- 玩家站在平台上时脚底要对齐平台顶面

这些约束不是为了提高平均一步预测分数，而是为了**阻止 rollout 落入明显不可能的状态**。

#### 5. 使用：分层规划，而不是直接暴力搜动作
在 Montezuma’s Revenge 里，拿到第一个正奖励可能需要 100+ 步，直接在 8^100 的动作树上搜不现实。于是作者用了一个接近 TAMP 的分层规划：

- 高层：基于对象接触关系构图，找子目标序列
- 低层：再用世界模型在动作空间里完成子目标

此外，他们还展示了一个附加用途：**在世界模型里先预训练 PPO，再回真实环境微调**。

### 核心直觉

这篇论文最重要的“因果旋钮”有三个：

1. **从单体程序改成多专家组合**  
   变化前：要一次性搜索一个巨大 simulator。  
   变化后：只需生成很多局部规则，再让数据调权和删错。  
   **结果**：结构搜索的难点从“全局组合爆炸”变成“局部提案 + 连续选择”。

2. **从点式/模糊预测改成乘积分布 + 约束裁剪**  
   变化前：一步预测即便大致正确，长时 rollout 也会慢慢漂到不可能状态。  
   变化后：PoE 用多个规则共同收缩分布，hard constraints 再砍掉非法支撑集。  
   **结果**：世界模型更适合长时规划，不只是更适合做 one-step fitting。

3. **从潜状态压缩改成全历史条件化**  
   变化前：若所有 expert 都依赖一个共享 latent state，新增一个机制会改坏所有 expert 的接口。  
   变化后：expert 直接看 history，新增局部机制时不必全球重写。  
   **结果**：部分可观测下仍能保持模块化增量更新。

一句话概括为什么它有效：

> **LLM 更擅长写“局部规律”，PoE 更擅长把这些局部规律拼成整体，约束则负责把“差不多对”修成“能拿来规划”。**

### 策略权衡

| 设计选择 | 改变了什么瓶颈 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 单一世界程序 → 多个程序专家 | 降低全局程序搜索难度 | 更可扩展、可局部更新、可组合泛化 | 需要额外做权重学习与剪枝 |
| 确定性代码 → 概率解释 | 让程序能表达随机性/不确定性 | 可建模随机和部分可观测环境 | 预测会变“模糊”，需进一步约束 |
| 乘积模型 + hard constraints | 收紧未来状态支撑集 | 长时 rollout 更稳，更适合规划 | 约束质量差会误杀可行状态 |
| 全历史条件化 | 避免 latent 接口耦合 | 新机制可局部加入，不必全局重写 | 历史变长，状态不够紧凑 |
| 符号对象输入 | 避开像素建模难题 | 少数据、高解释性 | 强依赖检测器与手工修补 |
| 分层规划 | 降低长时动作搜索难度 | 能处理 MR 这种稀疏奖励长时任务 | 规划耗时大，且并非本文主要创新点 |

---

## Part III：证据与局限

### 关键信号

- **比较信号：低数据下的任务成功**
  - 在 **Montezuma’s Revenge / MR-Alt** 上，PoE-World + Planner 分数都是 **100**，而 PPO（100k 与 20M steps）、ReAct、WorldCoder 都是 **0**。
  - 这说明它的提升点不是“最终上限更高”，而是**在极少数据下快速学到足够正确的因果结构**。

- **泛化信号：对重组关卡的 zero-shot 迁移**
  - Alt 环境没有给新演示，只是重排已有对象和关系。
  - PoE-World 仍能在 MR-Alt 成功，说明它学到的更像**可重组规则**，而不是对单一路线的轨迹记忆。

- **模型质量信号：一步预测优于 WorldCoder**
  - 在 MR 上，下一观测预测准确率 **0.75 vs 0.36**（train），随机测试帧上 **0.31 vs 0.10**。
  - 对象属性预测也更高：MR 随机测试帧 **0.76 vs 0.58**。
  - 这支持论文的主张：**模块化程序专家比单体程序更容易学对关键动力学。**

- **因果消融信号：hard constraints 主要帮助 planning，而非 one-step score**
  - 去掉约束后，MR 规划成功从 **5/9** 掉到 **2/9**；
  - 但一步预测准确率几乎不变。
  - 这是全文最有洞察力的结果之一：**长时规划真正需要的是“不会滚进非法状态”，不只是平均一步更准。**

- **附加效用信号：可做模拟器 warm-start**
  - 在 Pong 上，先在 PoE-World 里预训练 PPO，再回真实环境微调，可把“超过随机策略”所需步数从约 **1M** 降到 **200k**。
  - 这表明该世界模型不仅能规划，也能当训练环境用。

- **但别过度解读**
  - 在简单的 Pong 上，训练足够久后 **20M-step PPO** 最终明显强于 PoE-World。
  - 所以这篇论文的 capability jump 是：**样本效率 + 组合泛化 + 可解释世界建模**，而不是“全面击败成熟 model-free RL”。

### 局限性

- Fails when: 对象检测不稳定、接触关系被误判、或环境里出现演示中没覆盖的新对象/新机制时；在长时规划里若抽象图含 false edge，系统会反复试错与回滚更新。

- Assumes: 已有高质量符号对象输入，且论文实际上需要**对每个 Atari 游戏手工修补 OCAtari**；短演示必须覆盖关键因果规律；对象属性条件独立；可调用 **GPT-4o** 这类闭源 API；计算上世界模型约 **8 小时**，规划多在 **24 小时内**，构图阶段可并行到约 **100 CPUs**，每次运行还有约 **$20 OpenAI credit** 成本。

- Not designed for: 从像素端到端学习世界模型、主动探索、奖励学习、全 Atari 套件通用扩展，或在简单任务上追求最终最高回报。

### 可复用组件

- **对象变化 → 自然语言因果解释 → 代码专家** 的两阶段合成流程  
- **程序到概率分布的解释器**，可把局部规则变成可组合 world model  
- **专家权重优化 + 剪枝**，用于从大量候选规则中自动筛错  
- **hard-constraint 后处理**，专门提升长时 rollout 的可用性  
- **基于对象接触图的分层规划器**，可作为其他世界模型的上层控制模块

整体看，这篇论文最有价值的地方，是把“程序世界模型”从小玩具环境推进到更复杂的对象交互环境，并且清楚展示了一个重要认识：

> 世界模型是否能用于规划，关键不只在一步预测精度，更在于它是否把**可行状态空间**刻画对了。

![[paperPDFs/Building_World_Models_from_Language_Priors/arXiv_2025/2025_PoE_World_Compositional_World_Modeling_with_Products_of_Programmatic_Experts.pdf]]