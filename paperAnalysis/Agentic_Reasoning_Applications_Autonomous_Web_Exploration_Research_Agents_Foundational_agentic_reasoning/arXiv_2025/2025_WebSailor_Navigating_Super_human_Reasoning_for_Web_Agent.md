---
title: "WebSailor: Navigating Super-human Reasoning for Web Agent"
venue: arXiv
year: 2025
tags:
  - Others
  - task/web-information-seeking
  - reinforcement-learning
  - synthetic-data
  - trajectory-reconstruction
  - dataset/BrowseComp-en
  - dataset/BrowseComp-zh
  - dataset/Xbench-DeepSearch
  - dataset/GAIA
  - opensource/full
core_operator: "通过图随机游走与信息遮蔽合成高不确定性网页问答，重构短推理轨迹，并用 RFT+DUPO 后训练让 ReAct 代理学会在开放网页空间中系统性降低不确定性。"
primary_logic: |
  复杂网页问题 + search/visit 工具 →
  从真实网页图中随机游走、采样子图并做信息遮蔽，合成 Level-3 高不确定性 QA →
  用专家动作-观察轨迹重建短思维并做少量 RFT 冷启动 →
  通过 DUPO 强化学习稳定优化长程搜索与路径剪枝 →
  输出更稳健的多步检索过程与最终答案
claims:
  - "WebSailor-72B 在 BrowseComp-en/zh、Xbench-DeepSearch、GAIA 上分别达到 12.0/30.1/55.0/55.4 的 pass@1，超过文中所有已报告开源代理基线 [evidence: comparison]"
  - "SailorFog-QA 的正确轨迹呈现明显长尾工具调用分布，复杂度轮廓比 WebDancer-QA 更接近 BrowseComp-en，且大量样本需要超过 5 次工具调用 [evidence: analysis]"
  - "在 RL 前加入 RFT 冷启动，比直接从 instruction 模型做 RL 能收敛到更高最终精度，并保持更高、更稳定的工具调用长度，尤其在 BrowseComp-en 上差距更明显 [evidence: ablation]"
related_work_position:
  extends: "ReAct (Yao et al. 2023)"
  competes_with: "WebDancer (Wu et al. 2025a); WebThinker (Li et al. 2025c)"
  complementary_to: "N/A"
evidence_strength: strong
pdf_ref: "paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Foundational_agentic_reasoning/arXiv_2025/2025_WebSailor_Navigating_Super_human_Reasoning_for_Web_Agent.pdf"
category: Others
---

# WebSailor: Navigating Super-human Reasoning for Web Agent

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2507.02592), [GitHub](https://github.com/Alibaba-NLP/WebAgent)
> - **Summary**: 这篇论文把复杂网页检索的核心难点定义为“在巨大信息空间中持续降低不确定性”，并通过高不确定性数据合成、短推理轨迹重构、RFT 冷启动与 DUPO 强化学习，显著提升了开源 Web agent 的长程探索能力。
> - **Key Performance**: WebSailor-72B 在 BrowseComp-en / BrowseComp-zh 上 pass@1 为 **12.0 / 30.1**；在 Xbench-DeepSearch / GAIA 上为 **55.0 / 55.4**

> [!info] **Agent Summary**
> - **task_path**: 复杂网页问题 / ReAct 设定 -> search & visit 多轮探索 -> 事实型最终答案
> - **bottleneck**: 训练分布缺少 Level-3 高不确定性任务，RL 又因稀疏奖励难以从零学出长程搜索策略
> - **mechanism_delta**: 把训练样本从“低不确定性或固定路径多跳题”换成“图结构+信息遮蔽的高不确定性题”，再用短 CoT 轨迹重构与 DUPO RL 专门强化长程降不确定性
> - **evidence_signal**: BrowseComp-en/zh 上显著刷新开源 SOTA，且冷启动对比显示更长工具链与更高最终精度同步出现
> - **reusable_ops**: [graph-subgraph-obfuscation, concise-thought-reconstruction]
> - **failure_modes**: [context-overflow-on-long-traces, over-thinking-on-easy-queries]
> - **open_questions**: [asynchronous-agent-RL, direct-human-eval-for-superhuman]

## Part I：问题与挑战

这篇论文解决的不是普通“网页问答”，而是 **复杂信息寻址**：题目线索稀疏、表述模糊、解题路径不预先给定，模型必须靠多轮搜索、访问、交叉验证与路径剪枝，才可能收敛到答案。

### 真问题是什么？
作者把信息检索任务分成三层：

- **Level 1**：低不确定性，单次搜索或模型记忆就能答。
- **Level 2**：高不确定性，但路径清楚，本质上是线性 multi-hop。
- **Level 3**：既高不确定性，又**难以降低不确定性**；实体耦合复杂、路径非线性、起点不明显。

WebSailor 认为，现有开源 agent 失败的真正原因，不是“不会调工具”，而是**没被训练过如何在开放网页空间里系统性地降不确定性**。

### 输入/输出接口
论文采用 ReAct 范式：

- **输入**：一个复杂网页问题
- **动作空间**：`search` 和 `visit` 两个工具，以及最终 `answer`
- **输出**：若干轮 Thought-Action-Observation 后的最终短答案

其中：

- `search` 返回搜索结果标题、摘要、URL
- `visit` 返回**目标导向的网页摘要**，而不是原始 DOM/全文
- 轨迹最长限制为约 30 次工具调用

所以它解决的是 **deep information seeking**，而不是通用浏览器 UI 操作。

### 真正瓶颈
核心瓶颈有三层：

1. **数据分布错位**  
   现有训练数据大多是 Level 1/2，模型学到的是“直接搜”或“按固定链路跳转”，不是开放式探索。

2. **监督信号错位**  
   强推理模型虽能解一部分难题，但其原生 CoT 冗长、风格强，不适合直接蒸馏给长程网页代理。

3. **优化过程错位**  
   agent RL 需要多轮 rollout 且依赖外部环境，奖励又极稀疏，导致从零 RL 很难学会长链策略。

### 为什么现在值得做？
因为闭源系统如 DeepResearch 已经证明：**这种能力是可以学出来的**。  
现在的关键不再是“LLM 上限够不够”，而是开源社区缺少一套能把模型推向这种能力区间的 **后训练 recipe**。

---

## Part II：方法与洞察

### 方法链条

#### 1）SailorFog-QA：把训练分布推向 Level-3
作者先从 Wikidata 里的稀有实体出发，用网页随机游走构建实体图，再从中采样子图形成问题。

关键不是“多跳”，而是两步组合：

- **复杂结构**：随机游走让实体关系形成非线性、重叠、分叉的图结构
- **信息遮蔽**：把明确线索改写成模糊线索  
  例如日期改成“early 2010s”，名字改成首字母，数字改成区间或定性描述

这样一来，问题不再是 lookup，而变成：
**提出假设 -> 搜索验证 -> 排除错误路径 -> 汇总证据**

#### 2）轨迹监督：不直接蒸馏 teacher CoT，而是重建“短思维”
作者先让强开源 LRM 生成完整成功轨迹，但只保留：

- 动作 `a_t`
- 观察 `o_t`

然后用另一个强指令模型，为每一步重新生成**简洁、面向动作的 thought**。

这一步非常关键，因为它把监督从“模仿老师的语言风格”改成“学习这一步为什么要这样搜/访”。

结果是：

- 避免 teacher 的冗长风格污染
- 避免长轨迹 CoT 把上下文窗口挤爆
- 保留真正有用的决策逻辑

#### 3）RFT 冷启动：先让模型进入“可学区”
作者没有直接从 instruction model 做 RL，而是先做一个很小的 RFT cold start。

过滤规则很明确：

- 只保留最终答案正确的轨迹
- 去掉超过 32k token 的轨迹
- 只保留工具调用数 > 5 的轨迹

同时训练时 **mask observation token 的 loss**，只强化模型对 thought/action 的生成能力，而不去拟合环境返回文本。

这一步的作用不是“教会所有策略”，而是先让模型学会：

- 工具调用格式
- 长程 ReAct 骨架
- 不至于在 RL 初期全部拿零奖励

#### 4）DUPO：让慢速 agent RL 至少还能跑得动
agent RL 的 rollout 很慢，因为每个样本都要真实调用搜索/访问环境。  
作者提出 **DUPO (Duplicating Sampling Policy Optimization)**：

- **训练前**：先过滤掉太简单、8 个 rollout 全对的样本
- **训练中**：对 batch 内有非零方差的样本做复制补位，而不是像 DAPO 那样顺序采更多新样本

它解决的是一个很工程但很实在的问题：  
**如何在不让 wall-clock 爆炸的前提下，只保留“还有学习信号”的案例。**

奖励也很保守：

- 格式奖励
- 答案正确性奖励（LLM judge）

目的就是减少 reward hacking，同时保证轨迹仍遵循 ReAct 结构。

### 核心直觉

WebSailor 的关键，不是换了一个更会搜的 agent，而是同时改了三件事：

1. **训练分布变了**  
   从“有固定路径的问题”变成“图结构 + 模糊线索的开放问题”  
   → 改变了模型面对的搜索空间  
   → 逼它学会分支探索、交叉验证、路径剪枝

2. **监督形式变了**  
   从“模仿 teacher 的长篇 reasoning”变成“为动作提供短而干净的理由”  
   → 改变了信息瓶颈  
   → 让长程任务仍可在上下文窗口内完成

3. **优化地形变了**  
   从“直接 RL、初期几乎全零奖励”变成“先 RFT 进入可学区，再用 DUPO 强化高信息样本”  
   → 改变了稀疏奖励和训练速度约束  
   → 让模型真的能把长程搜索策略学出来

一句话概括：  
**WebSailor 不是让模型更会回答，而是让模型更会在不确定时继续找、继续排除、继续收敛。**

### 战略权衡

| 设计选择 | 解决的约束 | 带来的能力 | 代价/风险 |
|---|---|---|---|
| 图随机游走 + 子图采样 + 信息遮蔽 | 训练任务过于简单、路径过于线性 | 学到开放式探索与证据合成 | 题目可能存在歧义或多解 |
| 短 CoT 轨迹重构 | teacher CoT 冗长、风格污染、上下文过载 | 更适合长轨迹 agent 训练 | 可能丢掉 teacher 的部分隐性推理 |
| RFT 冷启动 | 纯 RL 初期奖励几乎为零 | 先学会工具格式与长程骨架 | 仍依赖高质量专家轨迹 |
| DUPO 批内复制 | agent RL rollout 太慢 | 提高训练效率，保留有信息梯度样本 | 批内多样性下降，仍受同步 RL 框架限制 |

---

## Part III：证据与局限

### 关键证据信号

- **比较信号：开源 SOTA 的主要跃迁发生在 BrowseComp**
  - WebSailor-72B 在 BrowseComp-en/zh 上达到 **12.0 / 30.1**
  - 明显高于 WebDancer-QwQ 的 **3.8 / 18.0**
  - 也高于 WebThinker-RL 的 **2.8 / 7.3**
  
  这说明它的收益主要不是“多一点工具调用”，而是**更像深搜代理而非普通 ReAct 模型**。

- **规模反证信号：增益不只是模型更大**
  - WebSailor-7B 在 BrowseComp-en 上 **6.7**
  - 已经超过一些 32B 开源代理
  
  这很重要：说明改进主要来自**训练范式**，而不是纯参数规模。

- **分布匹配信号：训练数据真的更像难题**
  - SailorFog-QA 的工具调用分布有明显长尾
  - 与 BrowseComp-en 的复杂度轮廓更接近
  - 比 WebDancer 训练集更少“2 步就解”的简单题
  
  也就是说，作者不是在堆更多数据，而是在**对齐真正困难任务的结构分布**。

- **因果信号：RFT 冷启动 + RL 提升的是稳定长链能力**
  - RL 后 Pass@1 提升明显大于 Pass@3，说明单次采样成功率更稳
  - 有冷启动的模型最终精度高于直接 RL
  - 且工具调用数保持更高、更稳定
  
  这支持论文的核心论点：  
  **能力提升来自更稳定的长程规划，而不是偶然采样出一条对的轨迹。**

- **向下兼容信号：难任务训练没有明显破坏简单事实检索**
  - 在抽样的 SimpleQA 子集上，WebSailor-72B 达到 **93.5**
  
  但这里要保守看待：SimpleQA 只抽了 200 条，不是全量结果。

### 我认为最重要的“所以呢”
WebSailor 的真正突破，不是“已经全面超越闭源 deep research 系统”，而是：

1. 证明了 **开源 Web agent 的能力瓶颈主要在后训练 recipe**
2. 证明了 **高不确定性任务分布本身就是关键训练资源**
3. 证明了 **agent RL 想有效，必须先让模型进入能拿到非零奖励的区间**

不过也要看到边界：  
最强闭源 DeepResearch 在 BrowseComp-en/zh 上仍有 **51.5 / 42.9**，尤其英文端差距仍大。  
所以更准确的表述是：**WebSailor 显著缩小了开源与闭源的差距，但尚未达到对最强闭源系统的全面 parity。**

### 局限性

- **Fails when**:  
  轨迹非常长、需要超过 32k 上下文或大量工具调用时，模型容易因上下文限制而失败；对需要数学/计算能力的 GAIA 子任务并未专门优化；对简单问题偶尔会出现 over-thinking。

- **Assumes**:  
  依赖 search/visit 工具链、Jina 抓取网页、Qwen-2.5-72B 做页面摘要；依赖强 LRM 先生成成功轨迹；训练和评测都使用 LLM judge；RL 仍运行在同步框架下，且只训练约 50 steps；合成 QA 的答案满足约束，但不总能保证唯一性。

- **Not designed for**:  
  原始浏览器 UI 操作、复杂网页交互、通用多模态 agent、以及严格意义上“超人类能力”的直接人类对照验证。标题中的 “super-human” 更接近 benchmark narrative，而不是经过正式 human study 证明的结论。

### 可复用组件

- **Level-3 任务生成器**：基于真实网页图的随机游走、子图采样、信息遮蔽
- **短推理轨迹蒸馏**：保留 action-observation，重建 concise thought
- **慢速 agent RL 的批内采样策略**：DUPO 适合 rollout 成本高、奖励稀疏的交互式训练
- **只训练决策 token**：在 agent SFT/RL 中 mask observation loss，可减少把环境文本当成监督目标的副作用

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Foundational_agentic_reasoning/arXiv_2025/2025_WebSailor_Navigating_Super_human_Reasoning_for_Web_Agent.pdf]]