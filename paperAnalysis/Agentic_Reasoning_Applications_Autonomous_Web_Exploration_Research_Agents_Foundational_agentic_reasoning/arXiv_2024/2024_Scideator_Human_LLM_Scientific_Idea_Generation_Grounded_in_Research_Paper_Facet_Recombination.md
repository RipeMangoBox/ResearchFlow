---
title: "Scideator: Human-LLM Compound System for Scientific Ideation through Facet Recombination and Novelty Evaluation"
venue: arXiv
year: 2024
tags:
  - Others
  - task/scientific-ideation
  - task/novelty-assessment
  - human-in-the-loop
  - facet-recombination
  - retrieve-then-rerank
  - dataset/SemanticScholar
  - opensource/promised
core_operator: 以论文的 purpose/mechanism/evaluation 三元 facet 为共享中间表示，让用户在 facet 级选择与替换，驱动 LLM 完成类比检索、点子重组与新颖性校验。
primary_logic: |
  输入论文与可选研究主题 → 按概念距离检索类比论文并抽取 purpose/mechanism/evaluation facets → 用户或系统重组 facets 生成候选研究点子 → 按 facet 重叠对相关工作进行检索重排并判断新颖性、给出替换建议 → 输出可迭代优化的研究想法及其文献依据
claims:
  - "在 22 名计算机科学研究者的组内用户研究中，Scideator 的创造力支持指数 CSI 中位数为 70.5，高于相同骨干 LLM 的无 facet 基线 61.0（Wilcoxon p<.01）[evidence: comparison]"
  - "在 58 个“非新颖”测试点子上，facet-based re-ranking 将“非新颖”判别准确率从一般相关性重排的 13.79% 提升到完整系统的 89.66%[evidence: ablation]"
  - "参与者最喜欢的点子更常包含其自选 facets，且在 Scideator 中生成最喜欢点子时使用的自定义文本指令更少（中位数 0 vs. 1.5），表明 facet-level steering 比自由提示更受偏好[evidence: analysis]"
related_work_position:
  extends: "SOLVENT (Chan et al. 2018)"
  competes_with: "IdeaSynth (Pu et al. 2024); PersonaFlow (Liu et al. 2024)"
  complementary_to: "AI Scientist (Lu et al. 2024)"
evidence_strength: moderate
pdf_ref: paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Foundational_agentic_reasoning/arXiv_2024/2024_Scideator_Human_LLM_Scientific_Idea_Generation_Grounded_in_Research_Paper_Facet_Recombination.pdf
category: Others
---

# Scideator: Human-LLM Compound System for Scientific Ideation through Facet Recombination and Novelty Evaluation

> [!abstract] **Quick Links & TL;DR**
> - **Links**: [arXiv](https://arxiv.org/abs/2409.14634)
> - **Summary**: 这篇工作把科研想法统一拆成“目的-机制-评估”三类 facet，并把这一表示贯穿检索、生成与新颖性验证，使研究者能用结构化方式而不是长 prompt 来和 LLM 共创研究点子。
> - **Key Performance**: novelty checker 在 58 个“非新颖”点子上的判别准确率达 89.66%；用户研究中 CSI 中位数 70.5 vs. 基线 61.0。

> [!info] **Agent Summary**
> - **task_path**: 输入论文/研究主题 -> 生成可重组的研究点子 -> 基于文献判断新颖性并给出替换建议
> - **bottleneck**: 用户意图在检索-生成-评估链路里无法稳定传递，自由文本交互也难以支撑可追溯的新颖性判断
> - **mechanism_delta**: 用统一 facet 表示替代自由提示作为跨模块控制信号，并用 facet-overlap reranking 对齐“相关文献检索”和“新颖性判断”目标
> - **evidence_signal**: facet-based reranking 将“非新颖”想法识别准确率从 13.79% 提升到 89.66%
> - **reusable_ops**: [按概念距离分层检索类比论文, 基于 facet 重叠的 retrieve-then-rerank 新颖性检查]
> - **failure_modes**: [远距离 facets 因陌生而被用户回避, “novel”判断比“not novel”更难让用户信服]
> - **open_questions**: [如何降低远距离 facets 的理解门槛, 如何扩大检索覆盖以减少 prior work 漏检]

## Part I：问题与挑战

这篇论文要解的，不是“让 LLM 多写几个研究点子”，而是**如何把科研构思中的检索、组合、判新三步，变成一个人可控且可追溯的闭环**。

### 真正的难点
现有两条路线各缺一半：

1. **facet-based ideation** 往往停在“给你类比论文做灵感”，但不会继续把 facet 真的重组为新点子，也不会继续做 novelty check。
2. **普通 LLM ideation 工具** 能直接生成文字，但主要依赖自由文本 prompting。用户每一步都要重新描述意图，系统也很难知道你究竟想保留哪个问题、替换哪个机制、还是只想改评估方式。

所以真正瓶颈是：**缺少一个能同时服务人类操控与系统推理的共享表示**。没有共享表示，检索、生成、评估是三段式拼接；有了共享表示，用户的选择才能变成可传递的结构化信号。

### 为什么现在值得做
- 文献规模持续膨胀，靠人工通读足够多论文来做类比越来越贵。
- LLM 已经足够强，能承担摘要、重组、解释与重排任务，但如果仍靠纯 prompt 驱动，用户意图会在链路中不断被“重解释”。
- 科研 ideation 里，“是否新颖”是核心问题，但过去的人机系统很少把 novelty 设计成可核查、可回路优化的一步。

### 输入 / 输出接口与边界
- **输入**：1 篇或多篇论文 + 可选研究主题。
- **输出**：多条候选研究点子；每条点子可追溯到其 purpose / mechanism / evaluation facets，并可进一步得到 novel / not novel 判断与 facet 替换建议。
- **边界条件**：
  - 主要支持的是**组合式创造**，不是从零发明全新理论；
  - 系统主要使用**标题与摘要**，不是整篇论文全文；
  - “新颖性”是**相对已检索相关文献**来定义，不是严格的全局 prior-art 证明。

## Part II：方法与洞察

Scideator 的核心做法，是把“研究想法”压缩成统一的三槽表示：**purpose（做什么问题）、mechanism（怎么做）、evaluation（怎么验证）**，然后让这一表示贯穿检索、生成、评估全流程。

### 方法骨架

#### 1. 共享 facet 表示
系统先从输入论文中抽取简短 facet 短语，并在全流程持续复用这三类 facet。

这一步的意义是双向的：
- **对用户**：交互对象不再是整段 prompt，而是可选、可替换的结构块；
- **对系统**：用户的每次操作都变成明确约束，而不是模糊自然语言。

#### 2. Module 1：按距离控制的类比论文检索
系统先抽取输入论文的 overarching purpose/mechanism，再生成不同概念距离的类比对：
- **very near**：直接相似论文
- **near**：同主题不同做法
- **far**：同子领域不同主题
- **very far**：不同子领域的高层结构类比

随后用 Semantic Scholar 检索对应论文，并从这些论文里再抽取三类 facets。  
这一步不是简单“找相似文献”，而是**把可探索的创意空间按距离展开**：既保留近邻的可行性，也引入远距的新颖性来源。

#### 3. Module 2：facet 重组式 idea generation
系统从输入论文和检索论文的 facet 池中，先在不同距离组的论文对之间生成类比，再选出更强类比，把一篇论文的 purpose 与另一篇论文的 mechanism/evaluation 组合成新点子。

它会根据用户选择自适应：
- **没选 facet**：偏向发散探索；
- **只选了 purpose 或 mechanism**：系统补全另一半；
- **两者都选了**：围绕指定组合继续细化。

因此它不是单纯“自动生成”，而是**人定方向，系统扩展组合空间**。

#### 4. Module 3：facet-based novelty verification
新颖性模块不是直接问 LLM “这个 idea 新不新”，而是先做 retrieve-then-rerank：
1. 通过已有论文、关键词查询、snippet search 召回候选文献；
2. 用 SPECTER embedding 做第一轮过滤；
3. 用 RankGPT 按 **application domain / purpose / mechanism / evaluation 重叠**做 facet-based reranking；
4. 基于 top-k 论文，让 LLM 输出 novel / not novel 及理由；
5. 若为 not novel，则分别替换 purpose / mechanism / evaluation 中的一个 facet，给出 3 个改写建议。

其 novelty 定义也被明确操作化：**只要与已检索论文在核心 facet 上重合、或其组合已被覆盖，就倾向 not novel；若至少有一个核心 facet 差异、独特组合、或迁移到新领域，则可判 novel。**

### 核心直觉

**作者真正拧动的旋钮**，是把用户与系统之间的通信，从“开放式 prompt”改成“可传播的 facet 约束”。

因果链可以概括为：

**交互从自由文本变为 facet 级选择/替换**  
→ **用户意图被固定到少数关键槽位（purpose / mechanism / evaluation）**  
→ **检索、生成、评估围绕同一表示运转，减少模块间语义漂移**  
→ **系统更容易做可控探索，用户也更容易理解和修正输出**

为什么这会有效？有三个层面的瓶颈被改掉了：

1. **信息瓶颈**  
   以前用户说“帮我更有创意一点”，模型要自己猜到底该改哪部分。现在用户可以明确保留 purpose、替换 mechanism，意图不再埋在长 prompt 里。

2. **搜索瓶颈**  
   只看近邻论文容易陷在输入论文附近；只看远距类比又容易变得不相关。距离控制检索把“相关性”和“发散性”变成可调参数，而不是随机游走。

3. **测量瓶颈**  
   novelty 的判断标准本来就依赖 facet 是否重叠，因此检索排序若只看 generic relevance，就会把“看起来相关”但不真正构成撞车证据的论文放前面。改成 facet overlap 排序后，检索目标才与判新目标一致。

所以这篇 paper 的关键贡献，不是某个更强的生成 prompt，而是**给整条 ideation pipeline 增加了稳定的中间语言**。

### 战略取舍

| 设计选择 | 带来的能力 | 代价 / 风险 |
|---|---|---|
| 统一 facet 表示 | 用户操作可跨模块传播，输出更可追溯 | facet 抽取若失真，会把误差传到后续模块 |
| 按距离分层检索 | 同时保留可行性与发散性 | 远距离 facet 常因陌生而被少用 |
| facet-based reranking | 更容易找出真正重叠的 prior work，支撑 not-novel 过滤 | 仍依赖前端召回质量；漏召回后再强的重排也无解 |
| 加入 evaluation facet | 让研究 idea 不只包含“问题+方法”，还有“如何验证” | 一些用户觉得 evaluation 没那么重要，增加交互负担 |

## Part III：证据与局限

### 关键证据信号

1. **系统级比较信号：它提升的是探索质量，而不只是输出数量**  
   在 22 名计算机科学研究者的组内实验里，Scideator 的 CSI 中位数为 **70.5**，高于同一 backbone LLM 的 baseline **61.0**。  
   最明显的提升出现在 **exploration** 和 **expressiveness**，而这两项恰好也是参与者认为对 ideation 最重要的维度。

2. **行为级信号：新概念来源从“输入论文”转向“系统 facets/ideas”**  
   baseline 中，参与者若觉得发现了新概念，往往归因于输入论文本身；而在 Scideator 中，他们更常把新概念归因于系统给出的 facets 或重组后的 ideas。  
   这说明它的能力跃迁不只是“把论文复述得更好”，而是**把外部文献中的结构化概念真正推到了用户前台**。

3. **组件级信号：novelty checker 的提升来自目标对齐**  
   在 58 个“非新颖”点子上，完整系统达到 **89.66%** 准确率；如果把 facet-based 重排换成一般相关性 RankGPT，则掉到 **13.79%**。  
   这强烈支持作者的主张：**判新时，facet overlap 比 generic relevance 更关键**。

4. **但 novelty checker 当前更像“过滤器”，不是“新颖性证明器”**  
   论文没有证明它显著提升了整体 novelty assessment 信心；更具体地说，它在判定 **not novel** 时更有用，因为用户能拿检索到的论文直接核对。  
   相反，当系统说 **novel** 时，用户更难真正相信“没有漏掉关键 prior work”。另外，这部分使用分析只基于 **17 名**按预期完成 novelty 任务的参与者，证据应保守看待。

### 局限性

- **Fails when**:  
  - 输入依赖远距离、跨子领域 facets 时，用户常因不熟悉而回避这些 facet；  
  - top-10 检索结果没覆盖关键 prior work 时，novelty 判断容易过于乐观；  
  - 需要确认“全局新颖性”而非“相对已检索文献的新颖性”时，系统证据不足。

- **Assumes**:  
  - 标题与摘要足以抽出稳定的 purpose / mechanism / evaluation；  
  - Semantic Scholar 检索覆盖较好；  
  - 闭源模型（gpt-4o、o3-mini）可用，且其推理足够稳定；  
  - 少量专家标注的 in-context examples 足以教会 novelty 分类标准。

- **Not designed for**:  
  - 全自动科研发现或实验执行；  
  - 专利级/法务级 prior-art 检索；  
  - 超出计算机科学场景的通用外推；  
  - 非组合式、强变革型创造任务。

### 复现与可扩展性的现实约束
- 依赖 **Semantic Scholar API**、**GPT-4o**、**o3-mini** 等外部服务。
- 代码尚未开源，论文只承诺“录用后发布”。
- 用户研究只覆盖 **22 位 CS 研究者**，主题集中在 HCI/NLP。
- 研究版实验还**关闭了部分完整系统功能**，所以对最终全功能交互回路的证据仍是部分的。
- 生成延迟高于 baseline（文中报告平均约 **22.04s vs. 15.21s**），这也解释了为什么 immersion 没有明显优势。
- baseline 虽使用相同 backbone LLM，但不是“最强 prompt-engineered paper-level system”，因此比较更像在验证 **facet interaction 的价值**，而不是穷尽所有替代设计。

### 可复用组件
- **共享 facet schema**：把复杂创意任务压成固定槽位表示，便于人机共控。
- **distance-controlled analogous retrieval**：用“近-远”层级组织启发来源，平衡可行性与新颖性。
- **facet-based retrieve-then-rerank novelty checking**：把判断标准前移到检索排序目标里。
- **facet swap suggestions**：把“这个想法不新”转化为“具体该改哪一槽”。

## Local PDF reference

![[paperPDFs/Agentic_Reasoning_Applications_Autonomous_Web_Exploration_Research_Agents_Foundational_agentic_reasoning/arXiv_2024/2024_Scideator_Human_LLM_Scientific_Idea_Generation_Grounded_in_Research_Paper_Facet_Recombination.pdf]]