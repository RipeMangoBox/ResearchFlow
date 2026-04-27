---
title: "LaMOuR: Leveraging Language Models for Out-of-Distribution Recovery in Reinforcement Learning"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robotic-control
  - task/out-of-distribution-recovery
  - reinforcement-learning
  - reward-code-generation
  - policy-consolidation
  - dataset/MuJoCo
  - dataset/DeepMindControlSuite
  - dataset/ManiSkill2
  - opensource/full
core_operator: 利用LVLM把OOD状态图像与任务描述编译成稠密恢复奖励和有效状态判定代码，并用门控策略巩固完成恢复重训练
primary_logic: |
  OOD状态图像 + 原任务文本 + 环境描述 → LVLM生成OOD描述、恢复行为推理、恢复奖励代码与有效状态判定器 → 无效状态下用恢复奖励重训练、有效状态下切回原任务奖励并对齐原策略 → 策略回到可继续执行原任务的有效状态
claims:
  - "LaMOuR在4个修改版MuJoCo恢复任务中，相比基于不确定性的SeRO在3个环境上具有更高恢复样本效率，并在HalfCheetah上表现相近 [evidence: comparison]"
  - "在复杂的Humanoid与PushChair恢复任务中，LaMOuR能学会从躺倒/翻倒的OOD状态恢复，而SeRO、SAC-env、SAC-zero均失败；PushChair中的Text2Reward基线也失败，且LaMOuR重训练策略从OOD起点达到79.8%成功率，接近原始策略从有效状态起点的82% [evidence: comparison]"
  - "Behavior Reasoning与few-shot代码生成对奖励正确性关键：去掉Behavior Reasoning会把原任务目标混入恢复奖励导致局部最优，去掉few-shot会生成缺少接近阶段的更难学习奖励 [evidence: ablation]"
related_work_position:
  extends: "SeRO (Kim et al. 2023)"
  competes_with: "SeRO (Kim et al. 2023); Text2Reward (Xie et al. 2024)"
  complementary_to: "N/A"
evidence_strength: strong
pdf_ref: "paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_Leveraging_Language_Models_for_Out_of_Distribution_Recovery_in_Reinforcement_Learning.pdf"
category: Embodied_AI
---

# LaMOuR: Leveraging Language Models for Out-of-Distribution Recovery in Reinforcement Learning

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2503.17125), [Project](https://lamour-rl.github.io/)
> - **Summary**: 这篇论文把“机器人掉进OOD状态后该怎么爬回来”转成一个由LVLM自动生成稠密恢复奖励的过程，从而替代复杂环境里不稳定的不确定性估计。
> - **Key Performance**: PushChair 中从 OOD 起点重训练后的成功率为 **79.8%**，接近原始策略从有效状态起点的 **82%**；在 **Humanoid / PushChair** 两个复杂环境中，LaMOuR 成功学会恢复，而 SeRO 与朴素 SAC 基线失败。

> [!info] **Agent Summary**
> - **task_path**: OOD状态图像 + 原任务文本 + 环境状态/动作描述 -> 恢复奖励代码与有效状态判定器 -> SAC重训练策略返回有效状态
> - **bottleneck**: OOD发生后缺少可扩展、动作导向的恢复信号；基于不确定性的标量奖励在复杂环境中失真
> - **mechanism_delta**: 用LVLM先把坏状态解释成恢复行为，再生成稠密奖励代码，并只在有效状态上约束原策略以避免遗忘
> - **evidence_signal**: 多环境对比显示 LaMOuR 在复杂环境中是唯一稳定学会恢复的方法，且 PushChair 成功率达79.8%
> - **reusable_ops**: [OOD-image-to-recovery-reasoning, ceval-gated-policy-consolidation]
> - **failure_modes**: [缺少Behavior Reasoning时奖励会混入原任务目标导致局部最优, 缺少few-shot时复杂操控奖励过稀疏而难以探索]
> - **open_questions**: [如何自动检测并触发OOD恢复, 如何减少对GPT-4o与环境特定状态工程的依赖]

## Part I：问题与挑战

这篇论文解决的不是“如何避免所有 OOD”，而是更现实的问题：**当 agent 已经进入 OOD 状态后，怎样学会恢复到还能继续执行原任务的有效状态**。

### 真正的问题是什么
在机器人控制里，原策略通常只在训练分布内可靠。一旦机器人跌倒、滑倒、或操作对象翻倒，原策略会输出不可靠动作，甚至进一步恶化状态。以往方法主要有两类：

1. **尽量减少 OOD 出现**：靠探索、鲁棒训练或分布约束扩大覆盖范围。  
2. **尽量别进入 OOD**：用不确定性、MPC、安全约束做预防。  

但现实部署里，这两类都无法保证“永不出错”。真正缺的是：**出错以后，怎么恢复**。

### 为什么之前的方法不够
SeRO 这类恢复方法已经提出过一个关键思路：给 OOD 状态下的 agent 一个辅助奖励，引导它回到分布内。问题在于它依赖**不确定性估计**。  
而在 Humanoid、移动操作这类高维连续控制场景里，不确定性估计很容易变差，于是辅助奖励也不再可靠。

换句话说，瓶颈不是“agent 不知道自己离训练分布有多远”，而是：  
**agent 缺少一个可操作的、面向动作的“我该怎么一步步回去”的恢复目标。**

### 输入/输出接口
这篇论文的接口非常清楚：

- **输入**：
  - 一个已经学会原任务的策略
  - 一个已知的 OOD 状态
  - 该 OOD 状态的第三人称图像
  - 原任务文本描述
  - 环境状态/动作的结构化说明
- **输出**：
  - 恢复奖励代码 `creward`
  - 有效状态判定代码 `ceval`
  - 一个重训练后的恢复策略

### 边界条件
这篇论文有明确边界，不应误读：

- 它**假设 OOD 已被检测到**，不解决 OOD detection。
- 它**假设原任务策略已经存在**，恢复是基于该策略再训练。
- 它依赖**固定视角图像**来描述 OOD 状态。
- 它做的是**重训练式恢复**，不是一次前向推理的 test-time correction。

## Part II：方法与洞察

LaMOuR 的设计哲学可以概括为一句话：

> 不再问“这个状态有多不确定”，而是直接问“从这个坏状态回到可工作状态，需要做什么”，再把答案编译成稠密奖励。

### 方法主线

#### 1. OOD Description：先把坏状态说清楚
作者用 GPT-4o 接收 OOD 图像和提示词，生成文字描述。  
作用不是简单 caption，而是把“躺地上的 humanoid”“翻倒的 chair 与机械臂相对位置”这种恢复关键信息显式化。

#### 2. Behavior Reasoning：先推“恢复行为”，再写奖励
这是全篇最关键的中间层。

模型同时看到：

- 原任务描述
- OOD 状态描述

然后先推断一个**有效状态**是什么，再推断从当前 OOD 状态回到那个有效状态需要什么行为。作者还用了 chain-of-thought 提示来让这个推理更稳定。

这一层的价值在于：  
它把“恢复子目标”从“原任务目标”里**拆出来**。  
例如 PushChair 中，原任务是把椅子推到目标点，但恢复阶段真正该做的是**先把椅子扶正**。

#### 3. Code Generation：把恢复行为编译成可执行奖励
有了恢复行为描述后，LVLM 再结合环境描述，生成两段代码：

- **恢复奖励代码**：给 agent 提供稠密恢复信号
- **有效状态判定代码 `ceval`**：判断当前是否已经回到可执行原任务的状态

这里不是训练一个 reward model，而是直接生成**环境可执行的 reward program**。  
复杂环境里作者还加了一个 few-shot 示例，帮助模型写出“分阶段”的奖励。

#### 4. 训练时的奖励切换
核心训练逻辑很直接：

- 如果还在无效状态：优化 LVLM 生成的恢复奖励
- 一旦回到有效状态：切回原任务奖励

这一步很重要，因为它把恢复阶段和原任务阶段**显式切开**，避免恢复奖励长期干扰原任务。

#### 5. LPC：只在有效状态做策略巩固
如果恢复训练全程都强行贴近原策略，会妨碍 agent 在 OOD 状态探索新动作。  
所以作者提出 **Language Model-guided Policy Consolidation**：

- **无效状态**：不约束，允许学新恢复动作
- **有效状态**：再对齐回原策略，防止遗忘原任务

这是一种很合理的门控式 continual RL 设计。

### 核心直觉

LaMOuR 真正引入的因果旋钮，不是“用了大模型”，而是：

1. **把恢复问题从标量估计改成结构化行为表示**  
   旧方法给一个“这里不确定/危险”的数。  
   新方法给一个“应该先靠近椅子、再降低倾角、再扶正”的恢复程序。

2. **把恢复目标从原任务目标中剥离出来**  
   OOD 时最需要的不是继续追原任务回报，而是先回到“任务可做”的区域。  
   `ceval` 让这个阶段切换显式化了。

3. **把灾难性遗忘控制在有效状态内处理**  
   原策略在 OOD 区域本来就不可靠，不应被拿来硬约束；  
   只有回到有效区域后，才应该把策略拉回原技能流形。

所以，LaMOuR 改变的是这个链条：

**坏状态图像 → 恢复语义 → 奖励程序 → RL 搜索方向**

而不是：

**坏状态 → 不确定性分数 → 朴素惩罚/奖励**

前者提供了更强的方向性，因此在复杂环境里更可扩展。

### 为什么这个设计有效
从因果上看，OOD 恢复本质上是一个**中间子目标发现**问题，而不是纯粹的 novelty estimation 问题。

- **Behavior Reasoning** 负责找到正确子目标；
- **Code Generation** 负责把子目标变成可优化的密集信号；
- **ceval** 负责阶段切换；
- **LPC** 负责恢复后不丢原技能。

这四者组合起来，才构成完整闭环。

### 战略取舍

| 设计选择 | 解决了什么 | 带来的能力 | 代价/风险 |
| --- | --- | --- | --- |
| 显式 Behavior Reasoning 中间层 | 防止恢复奖励直接混入原任务目标 | 恢复目标更对齐，减少局部最优 | 多一次 LVLM 调用，提示设计更复杂 |
| 生成 `ceval` 做阶段切换 | 区分“先恢复”与“再做任务” | 减少奖励冲突 | `ceval` 写错会误切换 |
| 只在有效状态做策略巩固 | 避免原策略妨碍 OOD 探索 | 同时保恢复能力与原任务能力 | 依赖原策略本身足够好 |
| few-shot 分阶段奖励 | 让复杂操控更容易探索 | 能学到“先接近、后扶正”的 staged recovery | 需要环境特定示例，迁移成本更高 |

## Part III：证据与局限

### 关键证据

#### 1. 对比实验：四个修改版 MuJoCo 恢复任务
**信号类型：comparison**  
结论：LaMOuR 在 Ant、Hopper、Walker2D 上比 SeRO 恢复更快，在 HalfCheetah 上与 SeRO 相近。

这说明把恢复问题写成**稠密方向性奖励**，通常比基于不确定性的辅助奖励更高效。

#### 2. 复杂环境压力测试：Humanoid + PushChair
**信号类型：comparison**  
结论：在复杂环境中，SeRO、SAC-env、SAC-zero 都没能学会恢复；PushChair 里连 Text2Reward 也失败，而 LaMOuR 成功。

这组实验很关键，因为它正对应作者的主张：  
**不确定性估计在复杂状态/动作空间里不可靠，但“恢复行为→奖励代码”这条路线还能工作。**

其中最硬的指标是：

- **PushChair 成功率**：
  - 原始策略从有效状态起点：**82%**
  - LaMOuR 重训练策略从 OOD 起点：**79.8%**

这说明恢复后的策略不只是“扶正椅子”，而是基本能回到原任务可执行水平。

#### 3. 消融：真正关键的是“行为推理层”
**信号类型：ablation**  
结论：

- 去掉 **Behavior Reasoning** 时，生成的奖励会混入“把椅子推到目标”的原任务项，导致 agent 可能陷入局部最优。
- 去掉 **few-shot** 时，奖励只关注椅子倾角，不鼓励先接近椅子，探索难度显著上升。
- 直接把图像送入 Behavior Reasoning 也能工作，说明**描述模态不是最关键**，关键是有无“恢复行为”这一中间语义层。

这是很强的机制证据：  
LaMOuR 的核心不是 caption，也不是单纯 codegen，而是**把恢复行为先显式推理出来**。

### 局限性

- **Fails when**: LVLM 错误理解 OOD 图像，或生成的 `ceval/creward` 与真实环境动力学不匹配时，恢复训练会被错误信号带偏；长时序、多阶段、强接触的恢复任务若没有合适 few-shot，奖励可能过稀疏。
- **Assumes**: 已有可执行原任务的原始策略；OOD 状态已被外部机制检测出来；可获取固定视角图像与环境状态/动作接口；依赖 GPT-4o 这类闭源 LVLM；复杂环境中还需要环境描述工程，甚至额外状态增强（如 PushChair 中为奖励代码补充更合适的 chair/arm 距离特征）；重训练预算不低（约 1M–4M steps）。
- **Not designed for**: 在线 OOD 检测、零样本即时恢复、纯黑盒环境下无状态接口的 reward code 执行、无需再训练的 test-time 自适应恢复。

### 可复用组件
这篇论文最值得复用的不只是“让大模型写奖励”，而是下面几个操作符：

1. **OOD image → recovery behavior** 的中间语义推理层  
2. **行为描述 → 可执行 reward / evaluator code** 的程序化奖励生成  
3. **有效状态门控的 reward switching**  
4. **仅在有效状态启用的策略巩固**

如果以后做 embodied agent 的故障恢复、跌倒自救、任务中断后重入，这四个模块都可以单独借鉴。

## Local PDF reference

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_Leveraging_Language_Models_for_Out_of_Distribution_Recovery_in_Reinforcement_Learning.pdf]]