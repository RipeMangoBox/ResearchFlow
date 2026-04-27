---
title: "CycleResearcher: Improving Automated Research via Automated Review"
venue: ICLR
year: 2025
tags:
  - Others
  - task/research-automation
  - task/automated-peer-review
  - reinforcement-learning
  - preference-optimization
  - reward-modeling
  - dataset/Review-5k
  - dataset/Research-14k
  - opensource/full
core_operator: 用自动审稿器把论文质量转成可学习的偏好奖励，让自动科研代理在“研究-评审-修订”闭环中迭代提升。
primary_logic: |
  参考文献与研究主题 → CycleResearcher按“大纲-正文-实验设计/伪结果”逐段生成论文 → CycleReviewer生成多审稿人反馈与分数 → 依据高低分论文构造偏好对并用迭代 SimPO 更新策略 → 输出更高评分的下一轮研究草稿
claims:
  - "CycleReviewer-123B 在 Review-5k 上将 Proxy MAE 从个体人类审稿人的 1.16 降到 0.92，并取得 74.24% 的录用决策准确率 [evidence: comparison]"
  - "CycleResearcher-12B 生成论文在模拟审稿中的平均分为 5.36、模拟录用率为 35.13%，高于 AI Scientist 的 4.31 和 0% [evidence: comparison]"
  - "去掉 NLL 稳定项会把 CycleResearcher-12B 的平均分从 5.36 降到 4.91，录用率从 35.14% 降到 12.03%，表明偏好优化需要语言建模正则化 [evidence: ablation]"
related_work_position:
  extends: "The AI Scientist (Lu et al. 2024)"
  competes_with: "The AI Scientist (Lu et al. 2024)"
  complementary_to: "RAG (Lewis et al. 2020); Fast-DetectGPT (Bao et al. 2024)"
evidence_strength: moderate
pdf_ref: "paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Self_evolving_agentic_reasoning/arXiv_2024/2024_CycleResearcher_Improving_Automated_Research_via_Automated_Review.pdf"
category: Others
---

# CycleResearcher: Improving Automated Research via Automated Review

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2411.00816), [Project](https://wengsyx.github.io/Researcher/)
> - **Summary**: 该工作用一个“自动写论文的策略模型 + 自动审论文的奖励模型”构成研究-评审-修订闭环，让开源 LLM 能通过自动化审稿反馈迭代改进科研写作与研究规划。
> - **Key Performance**: CycleReviewer 在 Review-5k 上将 Proxy MAE 相比个体人类审稿人降低 26.89%；CycleResearcher-12B 在模拟审稿中取得 5.36 分与 35.13% 模拟录用率，超过 AI Scientist 的 4.31 分和 0%。

> [!info] **Agent Summary**
> - **task_path**: 参考文献/研究主题 -> 论文大纲与正文 -> 自动审稿意见/分数 -> 偏好优化后的下一轮论文生成策略
> - **bottleneck**: 自动科研缺少一个可训练、低成本、接近同行评审的质量信号，导致系统只能一次性生成，无法按“被审稿指出的问题”稳定改进
> - **mechanism_delta**: 用 CycleReviewer 把完整论文转成多维审稿反馈与总分，再把高低分论文构造成偏好对，迭代更新 CycleResearcher
> - **evidence_signal**: 审稿模型在 Review-5k 上优于个体人类审稿人，且研究模型对 AI Scientist 有对比优势并有 RL/NLL/迭代训练消融支持
> - **reusable_ops**: [outline-then-section-generation, reviewer-driven-preference-loop]
> - **failure_modes**: [fabricated-experimental-results, static-reward-model-hacking]
> - **open_questions**: [how-to-ground-on-real-experiments, how-to-generalize-beyond-machine-learning]

## Part I：问题与挑战
这篇工作要自动化的不是单一步骤，而是整条科研链路：**读文献 → 提问题 → 写论文 → 被审稿 → 再改进**。

### 这篇论文在解决什么问题？
作者关注的是：**如何让开源 LLM 不只是“生成一篇像论文的文本”，而是能在审稿反馈下持续改进研究输出**。  
对应的输入/输出接口很清楚：

- **输入**：研究主题、参考文献 bib、引用论文摘要
- **输出**：LaTeX 格式论文草稿，以及对该草稿的自动审稿意见与分数
- **目标**：把“科研生成”从一次性生成，变成可迭代优化的闭环过程

### 真正瓶颈在哪里？
真正瓶颈不是“模型不会写长文”，而是 **没有可训练的科研质量反馈**：

1. **现有自动科研多是 open-loop**  
   像 idea generation、paper drafting 往往停留在 prompt 工程层面，缺少像真实同行评审那样的外部纠偏机制。

2. **论文质量难直接监督**  
   好论文不是单一维度，涉及 soundness、presentation、contribution、novelty 等多指标。没有 reviewer-like reward，就无法把科研生成变成稳定的偏好优化问题。

3. **真实实验太贵，无法在线形成 RL 环境**  
   真正的科研闭环需要跑代码、收结果、反复验证。作者因此只聚焦“研究规划 + 写作 + 审稿反馈”，而把实验执行从主体框架中剥离。

### 为什么现在做？
因为现在有三个条件刚好成熟：

- 长上下文开源 LLM 足以处理论文级长文本；
- OpenReview / arXiv 让论文和审稿数据可收集；
- SimPO 一类偏好优化方法让“自动 reviewer -> 奖励 -> 策略更新”在工程上变得可行。

### 边界条件
这篇工作的适用范围要看清：

- 主要面向 **机器学习领域**；
- 模型是 **text-only**，并不真正理解论文图像/表格；
- 训练闭环里 **实验结果是伪造的**，只是为了形成低成本虚拟环境；
- 所以它更像一个 **自动科研训练原型**，而不是可直接替代真实科学研究的系统。

## Part II：方法与洞察

### 1. 整体框架：一个写论文，一个审论文
作者构造了双模型闭环：

- **CycleResearcher**：policy model，负责读文献、做研究规划、写论文
- **CycleReviewer**：reward model，负责模拟多 reviewer 审稿并给出分数
- **Iterative SimPO**：把 reviewer 打分转成偏好对，反过来更新 policy model

相比只靠提示词的自动科研系统，这里的关键变化是：  
**把 peer review 从“后验评价”变成“训练信号”。**

### 2. 数据如何支撑这个闭环
作者构造了两个核心数据集：

- **Review-5k**：4,991 篇 ICLR 2024 论文，带 16k+ reviewer comments，用于训练 CycleReviewer
- **Research-14k**：收集 2022-2024 多个顶会的 ML 论文，并抽取出论文大纲和分段正文，用于训练 CycleResearcher

这里的关键不是数据量本身，而是数据被组织成了两类监督：

- **研究写作监督**：参考文献 -> 大纲/正文
- **审稿反馈监督**：完整论文 -> 多维评价与分数

### 3. CycleResearcher：把长文生成改成“先规划再写作”
CycleResearcher 不是一次性吐整篇论文，而是按科研流程分阶段生成：

1. 读参考文献与摘要，建立背景；
2. 先产出 motivation / main idea 的 **outline**；
3. 再生成 title / abstract / introduction / method 的 **main text**；
4. 再写 experimental setup / results / discussion；
5. 最后整合成 LaTeX 论文。

这个设计的本质，是把长文本论文生成变成一个 **带中间状态的规划问题**，从而减少直接生成 20K+ token 时的结构漂移。

### 4. CycleReviewer：把“论文质量”拆成可学习的奖励
CycleReviewer 输入完整论文，输出多 reviewer 风格的评审结果，包括：

- strengths / weaknesses
- soundness
- presentation
- contribution
- overall score
- final suggestion

它还模拟“从更严格 reviewer 到更宽松 reviewer”的多个视角，最后做汇总平均。  
这样做的意义在于：奖励不再只是一个黑盒标量，而是一个 **更接近真实同行评审的分解式质量信号**。

### 5. Iterative SimPO：怎么形成研究-评审-修订闭环
每一轮里，系统会：

1. 对同一个参考文献输入采样多篇候选论文；
2. 用 CycleReviewer 给这些论文打分；
3. 取高分样本作 win、低分样本作 lose；
4. 用偏好优化更新 CycleResearcher；
5. 下一轮继续采样与优化。

此外，作者加入了 **NLL 稳定项**，目的是防止模型只会“讨好 reward model”而产生重复、模板化、或事实更差的文本。

### 核心直觉
**真正的机制增量，是把自动科研从 open-loop generation 改成 reviewer-conditioned policy optimization。**

因果链可以概括成：

- **what changed**：从单次 prompt 生成，变为“生成论文 -> 自动审稿 -> 构造偏好对 -> 更新策略”的闭环；
- **which bottleneck changed**：把难以直接监督的学术质量，转化成 reviewer 可输出的多维反馈与偏好排序；同时用 outline 降低长文生成的信息瓶颈；
- **what capability changed**：模型更擅长生成在审稿标准下“更像一篇能过审的论文”的结构化草稿，而不是只写出表面像论文的长文本。

为什么这在机制上有效？

- **Reviewer 让目标变密**：论文好坏很稀疏，但审稿可以分解成多个局部判断；
- **Outline 让规划先于写作**：先约束研究骨架，再填正文，减少章节断裂；
- **NLL 让优化不跑偏**：偏好信号推动“更高分”，语言建模锚点维持“还能正常写”。

### 战略权衡
| 设计选择 | 改变了什么 | 获得的能力 | 代价 / 风险 |
| --- | --- | --- | --- |
| Reviewer 作为 reward model | 把论文质量变成可训练偏好信号 | 能做真正的闭环优化 | 可能 reward hacking，自评偏差难完全消除 |
| 大纲-正文交替生成 | 从一次性长文输出改成分段规划 | 结构更稳、章节更连贯 | 流程更长，依赖 outline 质量 |
| 虚拟环境里伪造实验结果 | 绕开真实实验高成本 | 能快速进行 RL 迭代 | 科学真实性与可验证性大幅下降 |
| 多 reviewer 视角汇总 | 从单点评分变成群体审稿模拟 | 奖励更接近真实 peer review | 仍受训练领域和知识时效限制 |
| 开源模型训练闭环 | 可后训练、可迭代更新 | 摆脱纯 API prompt 限制 | 需要高算力和长上下文训练基础设施 |

## Part III：证据与局限

### 关键实验信号
1. **审稿模型本身有可用性，不只是随便打中间分**  
   在 Review-5k 上，CycleReviewer-123B 的 Proxy MAE 为 **0.92**，优于个体人类审稿人的 **1.16**，也优于文中比较的闭源模型。  
   **结论**：作为 reward model，它至少具备一定的评分稳定性。

2. **研究模型相对 AI Scientist 有显著提升**  
   在模拟审稿中，CycleResearcher-12B 的平均分为 **5.36**，模拟录用率 **35.13%**；AI Scientist 分别为 **4.31** 和 **0%**。  
   **结论**：把自动审稿反馈纳入训练后，生成论文更符合 reviewer 偏好。

3. **人类评审也看到了增益，但差距仍明显**  
   三位 NLP 专家的人类评审中，CycleResearcher 平均总分 **4.8**，高于 AI Scientist 的 **3.6**，但仍低于 ICLR 2024 submitted papers 的 **5.5**。  
   **结论**：收益不只存在于“自家 reviewer”上，但离真实投稿水平还有明显差距。

4. **真正关键的不是单个技巧，而是闭环 + 稳定化**  
   消融显示：
   - 去掉 RL，性能下降；
   - 去掉 iterative training，继续下降；
   - 去掉 NLL，下降最明显，并出现重复与严重内容错误。  
   **结论**：该方法依赖“审稿反馈闭环”和“稳定正则”同时成立。

### 1-2 个最值得记住的指标
- **CycleReviewer**：Proxy MAE **0.92**（vs 个体人类审稿人 **1.16**）
- **CycleResearcher-12B**：模拟审稿平均分 **5.36**，模拟录用率 **35.13%**

### 局限性
- **Fails when**: 需要真实实验数据、图像/表格证据、跨学科知识迁移或对最新论文的新颖性进行可靠判断时，系统会明显失真；尤其本文闭环里的实验结果是伪造的。
- **Assumes**: 机器学习领域有足够开放的论文/审稿数据、8x H100 级训练资源可得、静态 CycleReviewer 可近似真实评审标准，并默认“研究写作质量”可以在不跑真实实验时被部分优化。
- **Not designed for**: 替代正式同行评审、端到端完成真实科学实验、在未披露 AI 参与的前提下直接投稿，或输出可直接信赖的科学结论。

### 对复现和扩展的现实约束
- 训练依赖较重：长上下文 Mistral/Qwen、DeepSpeed/ZeRO、8x H100。
- 数据管线依赖外部工具和许可：OpenReview、Semantic Scholar、MagicDoc、outline 抽取模型。
- Reviewer 是静态奖励模型，作者也明确承认存在 **reward exploitation** 风险。
- 域外泛化弱：目前主要针对 ML 论文，扩到其他学科需要新的论文/评审数据和新的审稿标准。

### 可复用部件
- **Review-5k / Research-14k**：研究写作与审稿建模数据
- **多 reviewer 生成式奖励模型**：适合长文质量打分与解释反馈
- **outline-then-section generation**：适合结构化长文本生成
- **reviewer-driven preference loop**：适合把主观质量目标转成可训练偏好信号
- **Fast-DetectGPT safeguard**：适合 AI 生成内容披露与治理场景

**一句话结论**：这篇工作的最大价值，不是“让 LLM 会写论文”，而是把 **自动审稿** 变成了 **自动科研的训练信号**；它证明了闭环方向可行，但由于真实实验缺失、评测高度依赖模拟 reviewer，因此更像一个有前景的原型系统，而不是可直接部署的“自动科学家”。

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Self_evolving_agentic_reasoning/arXiv_2024/2024_CycleResearcher_Improving_Automated_Research_via_Automated_Review.pdf]]