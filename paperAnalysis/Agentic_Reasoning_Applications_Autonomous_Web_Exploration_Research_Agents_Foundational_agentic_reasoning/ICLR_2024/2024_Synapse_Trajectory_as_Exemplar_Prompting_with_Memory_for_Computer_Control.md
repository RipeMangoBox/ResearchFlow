---
title: "Synapse: Trajectory-as-Exemplar Prompting with Memory for Computer Control"
venue: ICLR
year: 2024
tags:
  - Embodied_AI
  - task/computer-control
  - task/web-navigation
  - state-abstraction
  - trajectory-prompting
  - retrieval-augmented
  - "dataset/MiniWoB++"
  - dataset/Mind2Web
  - opensource/full
core_operator: "先把原始网页状态压缩成任务相关观察，再从记忆中检索相似成功轨迹作为示例，让 LLM 以轨迹续写而非单步分类的方式生成下一段计算机操作。"
primary_logic: |
  自然语言任务 + 原始HTML/网页状态 → 状态抽象去除无关信息，并按任务元数据从记忆中检索相似成功轨迹 → 用“观察-动作”交错的轨迹级示例续写当前轨迹，输出下一段可执行网页动作
claims:
  - "在仅使用 48 个任务演示的情况下，SYNAPSE 在 MiniWoB++ 的 64 个任务上取得 99.2% 平均成功率，并成为首个报告能解决 book-flight 的 ICL 方法 [evidence: comparison]"
  - "在 Mind2Web 上，以 GPT-3.5 为底座时，状态抽象、TaE 提示和 exemplar memory 相对 MindAct 分别带来 32%、50% 和 56% 的平均 Step SR 相对提升 [evidence: ablation]"
  - "记忆检索使模型能把 48 个 MiniWoB++ 任务中的示例迁移到 16 个未见任务上，未见任务平均成功率接近 100%；但在 Mind2Web 的跨域设定中增益很小 [evidence: analysis]"
related_work_position:
  extends: "RCI (Kim et al. 2023)"
  competes_with: "RCI (Kim et al. 2023); MindAct (Deng et al. 2023)"
  complementary_to: "ReAct (Yao et al. 2022a); Reflexion (Shinn et al. 2023)"
evidence_strength: strong
pdf_ref: paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Foundational_agentic_reasoning/ICLR_2024/2024_Synapse_Trajectory_as_Exemplar_Prompting_with_Memory_for_Computer_Control.pdf
category: Embodied_AI
---

# Synapse: Trajectory-as-Exemplar Prompting with Memory for Computer Control

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [Project](https://ltzheng.github.io/Synapse)
> - **Summary**: 这篇工作把计算机控制从“看原始 HTML 后逐步做单步选择”改成“在压缩观察上参考相似成功轨迹直接续写动作”，从而提升长程网页操作与跨任务泛化。
> - **Key Performance**: MiniWoB++ 64 个任务平均成功率 99.2%；Mind2Web 上相对 MindAct 的平均 Step SR 提升 56%（GPT-3.5）

> [!info] **Agent Summary**
> - **task_path**: 自然语言任务 + 原始 HTML/网页状态 -> 可执行网页动作序列
> - **bottleneck**: 上下文被冗长网页状态占满，现有 exemplar 只给计划或 MCQ 而非完整轨迹，且 exemplar 绑定具体任务导致新任务泛化差
> - **mechanism_delta**: 用状态抽象压缩 observation，再用相似任务检索到的完整成功轨迹做 few-shot 轨迹续写，而不是每步根据 plan/MCQ 单独选一个动作
> - **evidence_signal**: 双基准对比 + 逐组件消融，显示状态抽象、TaE、记忆三者都有稳定增益
> - **reusable_ops**: [state abstraction, trajectory-level prompting]
> - **failure_modes**: [计数或字符抄写类推理错误, 跨域检索到不相关 exemplar 时收益变弱]
> - **open_questions**: [如何降低基于 LLM 的时延, 如何让记忆检索在完全未见域上更稳健]

## Part I：问题与挑战

这篇论文研究的是 **text-based computer control**：输入是自然语言任务和当前网页状态（主要是 HTML），输出是 click / type / press / select 等操作。

### 真正问题是什么？
作者认为，现有 LLM 电脑代理的瓶颈不只是“不会规划”，而是三种更底层的错配：

1. **上下文预算错配**  
   原始 HTML 太长、噪声太多，一个复杂网页就可能吃满上下文，导致 few-shot exemplar 放不下，也让 LLM 被无关元素干扰。

2. **示例结构错配**  
   以往方法常用高层 plan 或 MCQ 形式做提示，但这些都不是完整交互轨迹，无法充分表达“看到什么状态后采取什么动作、何时等待下一状态”的序列结构。结果就是每一步都重新问 LLM，长任务中误差不断累积。

3. **示例选择错配**  
   以前很多系统把“任务 -> exemplar”写死，默认每个任务都要专属示例，无法利用相似任务/相似界面的迁移关系，因此对未见任务泛化差。

### 为什么现在值得解决？
因为相比 BC/RL 或大规模微调，ICL 路线更灵活、样本效率更高，但此前 prompt 设计卡住了性能上限。  
这篇工作实际上是在回答：**如果不改模型参数，只改“给 LLM 看什么、怎么组织、从哪里取”，能不能把电脑控制显著做强？**

### 输入/输出边界
- **输入**：任务描述 + HTML/网页状态
- **输出**：浏览器/网页可执行动作
- **适用边界**：以文本 DOM/HTML 为主的网页控制
- **不在核心范围内**：纯像素 GUI、Android 端像素控制、真实在线高风险网站部署

---

## Part II：方法与洞察

SYNAPSE 由三个组件组成：**状态抽象（state abstraction）**、**轨迹即示例提示（trajectory-as-exemplar, TaE）**、**示例记忆（exemplar memory）**。

### 方法骨架

#### 1. 状态抽象：先把“长网页”变成“短观察”
作者先用 LLM 把原始网页状态转成任务相关的简洁 observation。

- **显式抽象**：用 `<state, observation>` few-shot 对，让 LLM 直接把 HTML 摘要成干净观察。
- **隐式抽象**：对于特别长的状态，让 LLM 生成一段 state-parsing code，由代码从原始状态里提取关键信息。
- 在 Mind2Web 中，他们也把已有 element ranker 当成一种简化抽象器，用更小的 top-k 元素而不是 top-50。

关键点不是“召回更多元素”，而是 **把 prompt token 留给真正有用的信息**。

#### 2. TaE prompting：把动作生成改成“轨迹续写”
以往方法常是：
- 给 plan，然后每步问下一个动作；
- 或把每步动作做成 MCQ，让 LLM 选。

SYNAPSE 改成给 LLM 看完整成功轨迹：
`<task, observation, action, observation, action, ...>`

然后把当前任务的历史轨迹接在后面，让 LLM 继续写下一段动作。

这样做的好处是：
- 条件格式与目标格式一致，便于解析动作；
- LLM 看到的是“状态-动作交替”的真实决策结构；
- 它可以一次输出多个连续原子动作，只在需要新状态时停下，形成 **temporal abstraction**，降低查询次数与误差累积。

#### 3. Exemplar memory：让示例靠相似度检索，而非手工绑定
作者把任务元数据编码成向量，并把对应的抽象 prompt / 成功轨迹存在向量库中。

- **MiniWoB++**：元数据是任务描述 + 初始状态
- **Mind2Web**：元数据是网站名 + 域名 + 任务描述

执行时先检索相似 exemplar，再把对应轨迹拿出来作为 few-shot context。这样，agent 不再依赖手工定义的 task-specific demo，而能利用相似界面和相似任务的迁移。

### 核心直觉

这篇论文最关键的变化，不是换了更强的 backbone，而是 **重写了上下文中的信息分布**：

- **原始 HTML → 压缩 observation**  
  改变了 token 分配：上下文不再浪费在 DOM 噪声上，而更多用于保留行为先例。
- **plan/MCQ → 完整轨迹 exemplar**  
  改变了决策条件：LLM 不再做局部动作分类，而是在“真实交互历史”上做轨迹补全。
- **硬编码示例 → 相似度检索**  
  改变了泛化方式：从封闭任务映射变成基于相似性的类比迁移。

因此能力变化很直接：
- 能放入更多有效 exemplar；
- 长程任务里减少 multi-round query 的误差累计；
- 能把已见任务经验迁移到未见但相似的任务。

一个很有意思的证据是：在 Mind2Web 中，作者把元素数从 top-50 大幅减到 top-5 / top-3，虽然召回从 86% 掉到 53%，但 Step SR 反而更高。  
这说明对 LLM agent 来说，**高精度、低噪声上下文** 常比“尽量全保留”更重要。

### 设计取舍

| 设计 | 改变的瓶颈 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 状态抽象 | 上下文长度与网页噪声 | 放入更多 exemplar，减少无关 DOM 干扰 | 抽象过度会丢目标元素，召回下降 |
| TaE prompting | 单步决策导致的误差累积 | 更强长程控制，可一次输出多步动作 | 更依赖高质量成功轨迹 |
| Exemplar memory | task-specific demo 无法泛化 | 可迁移到相似任务/网站 | 跨域相似度失真时会误检索 |

---

## Part III：证据与局限

### 关键证据

1. **MiniWoB++ 的主比较信号**  
   SYNAPSE 在 64 个任务上达到 **99.2% 平均成功率**，只使用了 48 个任务的演示；并且是首个报告能解决 **book-flight** 的 ICL 方法。  
   这里的能力跳跃主要体现在：
   - 复杂状态任务：如 book-flight、count-shape
   - 长程/重复动作任务：如 use-autocomplete、use-spinner、guess-number
   - 无需依赖自我纠错也能超过先前 ICL 方法

2. **Mind2Web 的递增消融信号**  
   作者不是只报最终数，而是按组件逐步加：
   - 仅加状态抽象：比 MindAct 更好
   - 再加 TaE：继续提升
   - 再加 memory：cross-task / cross-website 继续涨  
   以 GPT-3.5 为底座，最终相对 MindAct 达到 **56% 平均 Step SR 提升**。

3. **记忆模块的边界信号**  
   memory 对 **cross-task** 和 **cross-website** 有明显帮助，但对 **cross-domain** 增益很小，甚至某些设置下无益。  
   这说明它确实在做“相似任务迁移”，而不是无条件增益；一旦域差异太大，检索到的 exemplar 反而可能误导模型。

### 关键指标怎么看
- 最亮眼的指标是 **MiniWoB++ 99.2% average success rate**
- 但在更真实的 **Mind2Web** 上，虽然 Step SR 显著提升，**整任务 SR 仍然很低**（如 GPT-3.5 下 cross-task 只有 2.4）  
  这说明 SYNAPSE 明显改善了“局部动作决策”，但真实开放网页自动化离稳健完成整任务还有距离。

### 局限性
- **Fails when**: 任务需要精确计数、字符抄写或细粒度符号匹配时，LLM 仍会出现推理/转写错误；在完全未见域上，检索到的 exemplar 若不相关，memory 的收益会消失。
- **Assumes**: 需要高质量成功轨迹与状态抽象示例；主结果依赖 GPT-3.5 与 OpenAI embedding API；Mind2Web 还依赖已有 element ranker；部分隐式状态抽象代码构建涉及 GPT-4 + 人工审查。
- **Not designed for**: 低时延线上代理、纯像素 GUI/Android 控制、无 exemplar 的完全 zero-shot computer-use，以及安全敏感的真实网站部署。

### 可复用组件
- **状态抽象器**：适合任何“原始状态长、噪声大”的 agent 任务
- **轨迹级示例模板**：适合状态-动作交替的序列决策问题
- **示例记忆检索**：适合需要从已知任务迁移到相似新任务的 agent 系统

### 一句话结论
SYNAPSE 的真正贡献不是“又一个网页代理”，而是证明了：**把 computer control 重写成“压缩观察上的相似轨迹续写”后，LLM 的 few-shot 决策能力会明显提升。**

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Foundational_agentic_reasoning/ICLR_2024/2024_Synapse_Trajectory_as_Exemplar_Prompting_with_Memory_for_Computer_Control.pdf]]