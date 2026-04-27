---
title: "SAS-Prompt: Large Language Models as Numerical Optimizers for Robot Self-Improvement"
venue: arXiv
year: 2025
tags:
  - Embodied_AI
  - task/robot-self-improvement
  - task/policy-search
  - gradient-free-optimization
  - in-context-learning
  - retrieval-augmented-prompting
  - dataset/FancyGym
  - opensource/no
core_operator: 用单个SAS提示把历史执行轨迹检索、参数作用分析和新参数合成串成一次LLM调用，从而把自然语言目标转成可解释的无梯度策略搜索。
primary_logic: |
  自然语言目标 + 域描述 + 机器人历史执行轨迹 → 总结并检索最接近目标的样例、分析各衰减参数对落点/轨迹的影响 → 合成新的8维衰减参数并把新轨迹回写上下文，迭代实现自我改进
claims:
  - "在 50 次独立运行、100 步预算下，Gemini 1.5 Pro 驱动的 LLM 优化器在 2D Ackley、2D Rastrigin 和 8D Ackley 上取得所有对比方法中最低的平均最终函数值，并在 8D Rastrigin 上优于 GD、Nelder-Mead 与随机搜索但低于 Adam [evidence: comparison]"
  - "在真实机器人 10 个预定义检索查询、每个查询 100 次实验中，SAS retrieval 的平均命中率达到 Top-1 39.4%、Top-5 68.89%、Top-10 83.70% [evidence: comparison]"
  - "在 Mujoco/FancyGym 乒乓仿真的三类自改进目标中，SAS 将平均目标距离从 0.677/1.040/1.277 m 降至 0.231/0.270/0.342 m [evidence: comparison]"
related_work_position:
  extends: "Large Language Models as Optimizers (Yang et al. 2023)"
  competes_with: "Learning to Learn Faster from Human Feedback with Language Model Predictive Control (Liang et al. 2024); Eureka (Ma et al. 2024)"
  complementary_to: "Code as Policies (Liang et al. 2022); SayCan (Ahn et al. 2022)"
evidence_strength: moderate
pdf_ref: paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_SAS_Prompt_Large_Language_Models_as_Numerical_Optimizers_for_Robot_Self_Improvement.pdf
category: Embodied_AI
---

# SAS-Prompt: Large Language Models as Numerical Optimizers for Robot Self-Improvement

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2504.20459), [Project](https://sites.google.com/asu.edu/sas-llm/)
> - **Summary**: 这篇工作把“总结历史轨迹—分析参数作用—合成新参数”压进一个统一提示，让 LLM 直接充当机器人策略参数的无梯度优化器，从自然语言目标出发做可解释的自我改进。
> - **Key Performance**: 2D Ackley 上 LLM 最终值 4.38，优于 Adam 的 6.22；真实机器人检索平均 Top-1/Top-5/Top-10 为 39.4% / 68.89% / 83.70%。

> [!info] **Agent Summary**
> - **task_path**: 自然语言目标 + 域描述 + 机器人历史执行轨迹 -> 8维策略衰减参数 -> 新执行轨迹
> - **bottleneck**: 缺少一种无需手工奖励、无需梯度、还能把语言目标稳定映射为数值参数更新的机器人自改进机制
> - **mechanism_delta**: 用 SAS（Summarize-Analyze-Synthesize）提示把“样例检索 + 参数归因 + 新参数生成”合成到单次 LLM 推理里，替代外部优化器
> - **evidence_signal**: 数值优化基准上优于多种经典优化器的多组结果，加上真实机器人从仅有左侧样例出发在十余步内打到右侧
> - **reusable_ops**: [轨迹摘要表构建, 文本化梯度式参数合成]
> - **failure_modes**: [目标语义含糊时检索显著退化, 高维强耦合参数下因果归因可能失真]
> - **open_questions**: [能否扩展到高维端到端策略, 对闭源LLM与prompt模板的敏感性有多大]

## Part I：问题与挑战

这篇论文真正想解决的，不是“让 LLM 直接控制机器人”，而是一个更具体也更关键的问题：

**能不能只依赖自然语言目标和机器人自己的历史执行日志，让 LLM 自己完成参数搜索与自我改进？**

传统机器人自改进通常要显式准备三件事：

1. 选哪些特征最重要；
2. 如何设计 reward / fitness；
3. 如何写 update rule 去改参数。

这三步一旦换任务，往往都要重做。论文的切入点是：**LLM 也许本身就具备一种“文本内的随机数值优化能力”**，如果成立，那么机器人学习循环里最难手工设计的部分，可能可以被一个 prompt 吸收掉。

### 问题设定

这里的输入/输出接口非常清楚：

- **输入**
  - 域描述：如球台坐标系、范围、左右方向含义
  - 用户目标：如“尽量打到最右边”“打到上边缘中间”
  - 历史执行轨迹：每次试验的参数、球拍/球的位置序列、最终落点、是否落台
- **输出**
  - 一组新的 8 维 attenuation 参数，用来缩放底层控制器输出

### 真正的瓶颈

难点不在“是否能打到球”，而在**如何把语言目标转成可执行的数值参数更新**。具体有三层瓶颈：

- **目标表达瓶颈**：用户给的是自然语言，不是可微分损失；
- **信息瓶颈**：历史样本只是轨迹和落点，且带噪声，LLM 要自己找出哪些参数和结果有关；
- **泛化瓶颈**：如果上下文里没有现成成功样本，系统必须能**外推**，而不是只做最近邻检索。

### 边界条件

这篇方法并不是从零学一个完整控制策略，而是在一个**已经会击球的低层控制器（LLC）**之上做参数搜索。也就是说：

- LLC 已经有基本击球能力；
- LLM 只调一个 **8 维衰减向量**；
- 所以这更像是**语言条件下的残差式 policy search**，而不是端到端机器人学习。

这也是它目前能工作的关键前提：**先把搜索空间压低，再让 LLM 做“文本化优化”。**

## Part II：方法与洞察

### 方法主线

SAS-Prompt 的核心是把学习循环拆成三步，并全部放进同一个 prompt：

1. **Summarize**
   - 让 LLM 把所有 in-context 执行样例整理成表
   - 强制它逐个读样本，而不是跳过长轨迹
   - 结果是一个更结构化、更适合比较的“经验缓存”

2. **Analyze**
   - 先找出最接近目标的样例
   - 再分析参数 a-h 对落点、轨迹、是否出界的影响
   - 论文把这一步称作一种 **textual gradient**：不是显式求导，但在文本中建立“参数变化 -> 结果变化”的局部关系

3. **Synthesize**
   - 基于上面的分析，提出一组新的参数
   - 目标是“比已有样例更接近目标，但不要把球打出台面”
   - 新参数执行后产生新轨迹，再加入上下文，形成闭环

### 核心直觉

这篇文章最重要的因果旋钮是：

**把“外部 reward + 外部优化器”换成“LLM 内部的总结-归因-合成链条”。**

更具体地说：

- **原来变更前**：要手工定义 reward、写优化算法、调探索策略；
- **现在变更后**：只给语言目标和执行轨迹，LLM 自己先做样例检索，再做参数作用解释，最后生成新参数。

这改变了什么瓶颈？

- 从 **“没有可用更新信号”**  
  变成 **“可以通过文本归纳得到局部更新方向”**
- 从 **“只能匹配已有演示”**  
  变成 **“可以在已有样例附近合成未见过的新参数”**
- 从 **“学习过程不可解释”**  
  变成 **“每一步都有自然语言理由可读”**

为什么这套设计在这里有效？

1. **Summarize** 把原始长轨迹压成结构化证据，减少 LLM 漏看样本的概率；
2. **Retrieve** 让搜索从“最接近目标的局部区域”开始，而不是完全盲搜；
3. **Analyze** 在低维参数空间里建立粗糙但有用的单调假设；
4. **Synthesize** 利用 LLM 的模式补全能力，对参数做插值/外推。

本质上，它依赖一个非常现实的前提：

> 当参数空间足够低维、底层技能已经存在时，LLM 不需要真梯度，也能用“文本化局部模型”完成可用的搜索。

### 关键机制拆解

#### 1. 低层控制器 + 衰减参数

LLC 先输出 8 个执行器速度。  
LLM 不直接出动作轨迹，而是输出同维度 attenuation 向量，对 LLC 输出逐维缩放。

这一步很关键，因为它把学习问题从：

- “直接生成高频连续控制”

降成了：

- “在已有技能周围调 8 个数”

#### 2. Retrieval 不是附属模块，而是优化起点

SAS 先找“最接近用户目标”的历史样例，这相当于传统优化里先挑高 fitness 个体。  
不同的是，这里的 fitness 不是标量奖励，而是由 LLM 在表格和语言目标之间做语义匹配。

#### 3. Analyze 提供“文本化梯度”

论文反复强调，LLM 会说出类似：

- 增大 g 会让球更偏右
- h 也可能帮助右移，但稳定性不如 g
- 某些参数会增加前冲量，可能导致出界

这种描述虽然不是数学梯度，但它确实在执行一种：

- 参数归因
- 局部方向判断
- 探索/利用平衡

#### 4. Self-improvement 通过上下文增长实现

每轮新参数执行后，轨迹会被追加回 cache。  
于是 LLM 的“经验池”不断扩大，搜索不再只靠静态 few-shot，而是形成一种**in-context 的在线改进循环**。

### 战略权衡表

| 设计选择 | 带来的能力 | 代价 / 风险 | 适用前提 |
|---|---|---|---|
| 只优化 8 维 attenuation | 搜索空间小、探索快、真实机器人更安全 | 表达力有限，学不到新挥拍原语 | 底层控制器已具备基本击球能力 |
| 单 prompt 内做总结/分析/合成 | 无需外部 reward 和优化器，且过程可解释 | 对 prompt 格式和上下文质量敏感 | LLM 能稳定处理表格与数值 |
| 自然语言目标替代显式损失 | 用户接口自然、目标切换便宜 | 模糊目标会导致检索和更新方向不稳 | 目标需要足够可判别 |
| 迭代追加新轨迹 | 能在线自改进，而不是一次性检索 | 上下文越来越长，噪声和 token 成本累积 | 长上下文模型可用 |

## Part III：证据与局限

### 关键证据信号

#### 信号 1：LLM 真的像个数值优化器，而不只是“会说”
**类型：comparison**

论文先把机器人任务抽离，直接做标准数值优化基准。  
在 50 次独立运行、100 步预算下，Gemini 1.5 Pro：

- 在 **2D Ackley** 上达到 **4.38**，优于 Adam 的 **6.22**
- 在 **2D Rastrigin** 上达到 **9.02**，显著优于 GD / Adam / Nelder-Mead
- 在 **8D Ackley** 上也最好
- 但在 **8D Rastrigin** 上不如 Adam

这组结果的意义是：  
**LLM 的“优化性”不是只在机器人故事里成立，它在独立数值任务里也能表现出有用的搜索行为。**

#### 信号 2：SAS 不只是生成参数，也能做语言条件的轨迹检索
**类型：comparison**

在真实机器人检索实验中，给 10 类目标、每类 100 次评测，平均结果为：

- **Top-1: 39.4%**
- **Top-5: 68.89%**
- **Top-10: 83.70%**

这说明它确实能把“靠右、靠左、靠网、最高弧线、浅球”等语言目标映射到历史样例。  
但作者也明确展示了失败点：像 “middle of the table” 这种范围性、模糊目标，Top-1 会明显下降。

#### 信号 3：真正的能力跃迁是“没有正例也能往目标方向推”
**类型：comparison / case-study**

最有说服力的不是检索，而是真实机器人自改进实验：

- 初始 24 个上下文样例都集中在球台左侧；
- 目标却是“尽量打到最右边”；
- SAS 经过十几轮迭代后，把落点逐步推到右侧；
- 论文描述到 **约第 14 次迭代** 已能稳定打向右侧。

这说明它**不只是从 cache 里挑最像的样本**，而是在做未见参数组合的合成。

仿真里这个结论也更系统化：  
三类目标的平均距离都显著下降：

- S1 Right: **0.677 -> 0.231 m**
- S2 Top: **1.040 -> 0.270 m**
- S3 Left Corner: **1.277 -> 0.342 m**

### 1-2 个最关键指标

- **数值优化能力**：2D Ackley 最终值 **4.38**，优于 Adam 的 **6.22**
- **机器人自改进能力**：S1 目标距离从 **0.677 m** 降到 **0.231 m**

### 局限性

- **Fails when**: 目标语言本身含糊、范围型定义过宽、或需要复杂几何判别时，检索会明显退化；当参数维度更高、变量耦合更强、轨迹噪声更大时，LLM 形成的“文本化梯度”可能不稳定，甚至给出错误归因。
- **Assumes**: 必须已有一个能基本完成击球的低层控制器；必须能记录结构化执行轨迹并提供坐标系说明；方法依赖反复 rollout 与日志追加；实验核心模型是 **Gemini 1.5 Pro**，属于闭源 API 依赖；论文未给出清晰代码开源信息，复现存在门槛。
- **Not designed for**: 从原始视觉输入端到端学控制；高频闭环、毫秒级控制；需要在大规模高维连续动作空间中直接学出全新技能的场景。

### 可复用部件

- **SAS prompt 模板**：总结 -> 分析 -> 合成，适合任何“日志驱动参数搜索”任务
- **轨迹摘要表**：把原始时序日志压成更适于 LLM 比较的证据表示
- **文本化梯度归因**：用自然语言显式描述“参数变化 -> 行为变化”
- **上下文追加式优化循环**：每轮执行结果回写上下文，形成无梯度在线改进

### 一句话判断

这篇工作的价值，不在于它已经替代传统控制优化，而在于它证明了一个更有启发性的方向：

**当任务被限制在“已有技能上的低维参数搜索”时，LLM 可以不靠显式 reward 和梯度，直接充当一个可解释的数值优化器。**

![[paperPDFs/Other_Multimodal_Large_Language_Model_MLLM_based_related_Embodied_Learning_2025/arXiv_2025/2025_SAS_Prompt_Large_Language_Models_as_Numerical_Optimizers_for_Robot_Self_Improvement.pdf]]