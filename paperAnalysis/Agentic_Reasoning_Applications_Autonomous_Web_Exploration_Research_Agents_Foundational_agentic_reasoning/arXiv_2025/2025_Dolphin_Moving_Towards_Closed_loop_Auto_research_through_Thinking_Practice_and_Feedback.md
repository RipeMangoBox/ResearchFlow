---
title: "DOLPHIN: Moving Towards Closed-loop Auto-research through Thinking, Practice, and Feedback"
venue: arXiv
year: 2025
tags:
  - Others
  - task/automated-scientific-research
  - retrieval-augmented-generation
  - feedback-loop
  - traceback-guided-debugging
  - dataset/ModelNet40
  - dataset/CIFAR-100
  - dataset/SST-2
  - dataset/MLE-bench
  - opensource/full
core_operator: "把文献检索、代码实验与结果记忆接成闭环，用真实实验反馈而不是仅靠语言层打分来驱动下一轮科研想法生成。"
primary_logic: |
  输入研究主题 + 基线代码 + 可检索论文 + 历史实验结果
  → 按主题相关性与任务属性筛论文，生成并过滤新想法
  → 自动写实验计划、改代码，并用 traceback 引导的局部代码结构反复调试
  → 分析实验结果并回写记忆与提示，进入下一轮想法搜索
claims:
  - "Claim 1: 在 ModelNet40 上，DOLPHIN 自动提出并实现的 PointNet-CSR 达到 93.9% OA，超过作者复现的 PointNet 基线 91.0%，并与 GPSFormer 的 93.8% 结果相当 [evidence: comparison]"
  - "Claim 2: 任务属性引导的论文排序将 20 个想法中被判定为新颖的数量从 8 提升到 19，同时平均单想法成本与 naive retrieval 基本持平（$0.184 vs. $0.187） [evidence: ablation]"
  - "Claim 3: 基于 traceback 抽取信息生成局部代码结构，可将实验成功执行率从约 33.3% 提升到 50.0% [evidence: ablation]"
related_work_position:
  extends: "The AI Scientist (Lu et al. 2024)"
  competes_with: "The AI Scientist (Lu et al. 2024); Agent Laboratory (Schmidgall et al. 2025)"
  complementary_to: "AIDE (Schmidt et al. 2024); GPT-Researcher (Assafelovic 2023)"
evidence_strength: moderate
pdf_ref: "paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Foundational_agentic_reasoning/arXiv_2025/2025_Dolphin_Moving_Towards_Closed_loop_Auto_research_through_Thinking_Practice_and_Feedback.pdf"
category: Others
---

# DOLPHIN: Moving Towards Closed-loop Auto-research through Thinking, Practice, and Feedback

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2501.03916), [Project](https://alpha-innovator.github.io/Dolphin-project-page/), [Code](https://github.com/Alpha-Innovator/Dolphin)
> - **Summary**: 这篇工作把“文献检索→想法生成→代码实现→实验分析→结果回写”串成一个自动闭环，使自动科研不再停留在“想法看起来新颖”，而是以真实 benchmark 上能否跑通、能否涨点作为下一轮搜索信号。
> - **Key Performance**: ModelNet40 上最高达到 93.9% OA（作者复现 PointNet 基线 91.0%，与 GPSFormer 93.8% 可比）；MLE-bench 的 insult detection 任务从 81.0 提升到 84.7。

> [!info] **Agent Summary**
> - **task_path**: 研究主题/基线代码/相关论文/历史结果 -> 新想法与实验计划 -> 可执行代码与 benchmark 结果 -> 下一轮反馈驱动的想法更新
> - **bottleneck**: 自动科研的关键瓶颈不是“生成更多想法”，而是如何在代码、数据与算力约束下筛出可实现、可验证、可持续改进的想法
> - **mechanism_delta**: 用“任务属性约束检索 + traceback 引导调试 + 实验结果记忆反馈”替代一次性 idea scoring 式流程
> - **evidence_signal**: 三个关键环节都有可测增益：novel idea 数 8/20→19/20，执行成功率约 33.3%→50.0%，后续 loop 改进率 2/7→4/8
> - **reusable_ops**: [task-attribute-guided paper ranking, traceback-guided local code structure debugging]
> - **failure_modes**: [project-level 复杂代码难以自动修改, 仅看标题/摘要的文献检索会遗漏关键技术细节]
> - **open_questions**: [如何抑制 LLM 训练数据导致的知识泄漏, 如何把闭环扩展到需要大规模预训练/高算力的复杂任务]

## Part I：问题与挑战

这篇论文要解决的，不是“让 LLM 写出更像研究 proposal 的文字”，而是让系统在**给定主题、参考代码和 benchmark** 的现实约束下，持续找到**能被实现、能被验证、还能反过来改进下一轮搜索**的研究想法。

### 真正的问题是什么
现有自动科研/自动 idea generation 工作有三个断点：

1. **评价错位**  
   很多方法用人类或 LLM 判断 idea 是否新颖，但“新颖”不等于“有效”。真正科研要回答的是：这个想法落到代码和实验上，是否真的带来提升？

2. **闭环缺失**  
   人类研究者会根据实验结果不断修正下一轮假设；而许多已有方法只做“生成”，不做“实验反馈到生成”的闭环。

3. **执行鸿沟**  
   即使想法方向对，LLM 改代码时也容易卡在 shape mismatch、模块嵌套、局部依赖关系等问题上，导致 idea 根本没法被验证。

### 为什么现在值得做
- LLM 已经具备一定的**文献理解、实验规划、代码修改**能力。
- 机器学习领域存在大量**公开 benchmark + 参考代码**，为自动验证提供了可操作土壤。
- 现有自动科研系统已经走到“半自动”边缘，但还没把**检索、实验、反馈**真正打通。

### 输入 / 输出接口
- **输入**：研究主题、参考代码、可检索论文（标题/摘要）、公开数据集、上一轮实验记忆
- **输出**：过滤后的新想法、可运行的修改代码、实验指标、写回下一轮的记忆与提示

### 边界条件
DOLPHIN 的适用前提很明确：
- 任务最好有**标准化指标**与**可自动执行的 benchmark**
- 最好已有**代码模板/基线实现**
- 代码复杂度不能太高，否则 LLM 很难稳定完成修改与调试

---

## Part II：方法与洞察

DOLPHIN 的核心不是“换一个更强 LLM”，而是把自动科研重写成一个**受约束的搜索过程**：  
先用文献筛选改善搜索先验，再用实验执行筛掉纸上谈兵的 idea，最后用结果反馈改变下一轮搜索方向。

### 核心直觉

**what changed → which bottleneck changed → what capability changed**

1. **从粗粒度 topic 检索，变成任务属性约束检索**  
   → 改变了参考文献分布中的噪声问题  
   → 让后续 idea 更少受相邻但不匹配任务的误导

2. **从“给 traceback 让模型猜”，变成“traceback + 局部代码结构”调试**  
   → 改变了代码调试时的信息过载瓶颈  
   → 提高代码真正跑通的概率

3. **从一次性 brainstorming，变成结果驱动的迭代搜索**  
   → 改变了系统没有 reward signal 的问题  
   → 使后续 loop 更少重复、更多朝有效方向探索

一句话讲，论文真正引入的因果旋钮是：  
**把自动科研的选择压力，从“语言上像不像好 idea”，切换成“在当前代码/数据约束下能不能执行且带来增益”。**

### 方法拆解

#### 1. 想法生成：先把“参考上下文”变干净
DOLPHIN 先从 Semantic Scholar 检索论文，但不直接拿来用，而是做两层约束：

- **主题相关性**
- **任务属性相关性**（如输入、输出、任务定义是否一致）

这一步很关键。因为“3D classification”附近很容易检到 3D detection、completion 等论文，概念相近，但方法假设不同，会把 LLM 往错误方向带。

在拿到较干净的论文集合后，系统再生成多个 ideas，并做两类过滤：

- **独立性过滤**：用 idea summary embedding 做相似度去重
- **新颖性检查**：用 LLM 判断是否与已有论文过近

这一步本质上是在控制搜索空间：  
不是让模型“更发散”，而是让它**在正确约束里发散**。

#### 2. 实验验证：把 idea 变成可执行代码
对每个通过过滤的 idea，DOLPHIN 会：

1. 生成实验计划
2. 基于参考代码做修改
3. 自动运行
4. 如果报错，则进入 traceback-guided debugging

这里的关键创新是**traceback 引导的局部代码结构分析**：

- 从异常 traceback 中提取函数名、行号、相关代码
- 只关注**自定义代码**，排除库函数噪声
- 让 LLM 先总结与错误相关的局部代码结构
- 再基于这个局部结构做修复

这相当于把“整个项目代码”压缩成“与当前错误有因果关系的局部上下文”，降低了 LLM 调试时的认知负担。

#### 3. 结果反馈：让实验结果真正影响下一轮 idea
实验跑完后，系统按相对基线的结果把 idea 分为：

- improvement
- maintenance
- decline

然后把历史信息回写到下一轮生成阶段，用于：
- 避免重复探索已验证无效/低价值的方向
- 将有效方向作为 prompt conditioning，促进后续迭代

这一步把 auto-research 从 open-loop 变成了 closed-loop。  
也就是说，系统不只是“会提案”，而是开始具备一点“试错后调整搜索策略”的能力。

### 战略性取舍

| 设计选择 | 解决的瓶颈 | 获得的能力 | 代价 / 副作用 |
|---|---|---|---|
| 任务属性引导论文排序 | 相邻任务论文污染检索结果 | idea 更贴题、更可落实 | 依赖 LLM 对任务属性和论文摘要打分；只看标题/摘要可能漏细节 |
| embedding 去重 + novelty 检查 | 重复想法浪费实验预算 | 节省验证成本，提高 idea 质量 | novelty 判断仍部分依赖 LLM |
| traceback 引导的局部代码结构 | 调试时上下文过大、因果链不清 | 代码更容易跑通 | 对多文件、跨模块项目级代码仍不够 |
| 实验结果反馈 | 没有真正 reward 的一次性生成 | 后续 loop 改进率提升 | prompt 和记忆随循环增长，成本略升 |

---

## Part III：证据与局限

### 关键证据

#### 1. 比较信号：闭环确实能找到“能涨点”的想法
最强结果出现在 3D 点云分类：

- 在 **ModelNet40** 上，DOLPHIN 找到的 PointNet-CSR 达到 **93.9% OA**
- 高于作者复现的 PointNet 基线 **91.0%**
- 与人类设计的 **GPSFormer 93.8%** 基本持平

这说明它不只是生成“像样的研究描述”，而是真的能找到有竞争力的代码改动。

其他任务上也有一致提升：
- **CIFAR-100**：81.2 → 82.0
- **SST-2**：91.0 → 92.5
- **MLE-bench** insult detection：81.0 → 84.7

结论：能力不是只在单一模态出现，而是至少在点云、图像、文本和 Kaggle 风格任务上都有可见信号。

#### 2. 机制信号：三大模块都有独立贡献
- **论文排序消融**：novel ideas 从 **8/20 → 19/20**  
  说明 task-attribute filtering 的作用不是装饰，而是实打实改变了 idea proposal 的质量。
- **调试消融**：成功执行率从约 **33.3% → 50.0%**  
  说明局部代码结构确实缓解了 LLM 调试中的上下文瓶颈。
- **反馈消融**：改进率从 **Loop 1 的 2/7 → Loop 3 的 4/8**  
  说明闭环不是形式上的“多轮”，而是让后续轮次更容易提出有效想法。

#### 3. 个案信号：自动找到的方案不只是“更准”，还可能更简洁
论文用 3D 分类 case study 比较了人类设计的 DGCNN 与 DOLPHIN 生成的 PointNet-CSR：

- PointNet-CSR 是更轻量的模块级改动
- 无需复杂可学习图构造
- 结果略优，同时每个 epoch 更快（约 6.12s vs. 20.86s）

这说明 DOLPHIN 的搜索不一定只会往“更复杂架构”走，也可能找到**更简洁但有效**的设计。

### 局限性

- **Fails when**: 任务需要 project-level 多文件代码理解、复杂环境依赖、长链实验编排、或大规模预训练资源时，当前代码 agent 很难稳定修改并验证；debug 次数上限为 5 也会截断部分可修复样本。
- **Assumes**: 有可运行的参考代码、公开 benchmark、明确指标；依赖 GPT-4o 进行 idea/ranking/novelty、DeepSeek-v2.5 进行 coding、Semantic Scholar API 做检索；默认标题/摘要足以代表论文关键信息；还存在 LLM 训练数据带来的知识泄漏风险。
- **Not designed for**: 无代码模板的全新系统搭建、需要湿实验或理论证明的科研、依赖昂贵硬件/超长训练周期的任务，以及真正需要跨学科全文深读与长期资源协调的开放式研究。

额外说一句：论文虽然做了多任务实验和组件消融，但**没有在统一 protocol 下与 AI Scientist / Agent Laboratory 做端到端 head-to-head 对比**，因此系统级优势的证据更多来自“组件有效 + 下游涨点”，这也是我把证据强度评为 **moderate** 而不是 stronger 的原因。

### 可复用组件
- **task-attribute-guided paper ranking**：适合任何“检索到的参考文献容易串题”的 agent 系统
- **idea dedup by summary embedding**：适合多轮 proposal search，减少重复实验
- **traceback-guided local code structure debugging**：适合代码 agent 在局部报错修复中的上下文压缩
- **result-conditioned prompt feedback**：适合把执行结果转成下一轮搜索偏置的闭环 agent

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Foundational_agentic_reasoning/arXiv_2025/2025_Dolphin_Moving_Towards_Closed_loop_Auto_research_through_Thinking_Practice_and_Feedback.pdf]]