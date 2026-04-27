---
title: 'FACT: Mitigating Inconsistent Hallucinations in LLMs via Fact-Driven Alternating Code-Text Training'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 代码-文本交替训练抑制LLM幻觉
- FACT
- FACT is the first task-agnostic par
acceptance: Poster
method: FACT
modalities:
- Text
paradigm: supervised
---

# FACT: Mitigating Inconsistent Hallucinations in LLMs via Fact-Driven Alternating Code-Text Training

**Topics**: [[T__Text_Generation]] | **Method**: [[M__FACT]] | **Datasets**: CNN, SAMSum, SQuAD V2

> [!tip] 核心洞察
> FACT is the first task-agnostic paradigm that alternates between text-to-code and code-to-text prediction to transfer the logical consistency of programming languages to LLM outputs, thereby reducing inconsistent hallucinations across diverse NLP tasks.

| 中文题名 | 代码-文本交替训练抑制LLM幻觉 |
| 英文题名 | FACT: Mitigating Inconsistent Hallucinations in LLMs via Fact-Driven Alternating Code-Text Training |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2501.0XXXX) · [Code](待公开) · [Project](待公开) |
| 主要任务 | Text Generation（摘要、问答等事实型文本生成） |
| 主要 baseline | Base LLM, Prompt engineering, SFT, SymbCoT, Lookback |

> [!abstract]
> 因为「LLM 在事实推理中频繁产生 input-conflicting 和 context-conflicting 的不一致幻觉，且现有方法局限于任务特定训练或数值推理」，作者在「标准 SFT 与端到端文本生成」基础上改了「交替代码-文本预测训练 + 事实驱动过滤 + 结构化代码推理」，在「CNN/Daily Mail、SAMSum、SQuAD V2、HaluEval」上取得「幻觉降低 2.7%-8.0%，SAMSum Coherence 98.77 对比 SymbCoT 96.37」

- **幻觉抑制**: FACT 在四个数据集上降低不一致幻觉 2.7%-8.0%
- **摘要质量**: SAMSum 上 Coherence 98.77（+2.40 对比 SymbCoT），Relevance 98.09（+2.31 对比 SymbCoT）
- **问答性能**: SQuAD V2 上 F1 88.89（+4.52 对比 Prompt），Exact Match 87.63（+1.43 对比 SymbCoT）

## 背景与动机

大型语言模型在生成事实型内容时，常常出现两种不一致幻觉：input-conflicting（生成内容与输入矛盾）和 context-conflicting（生成内容前后自相矛盾）。例如，模型在摘要任务中可能先陈述"A 公司收购了 B 公司"，后文却写成"B 公司收购了 A 公司"——这种逻辑断裂严重损害了输出的可信度。

现有方法从不同角度试图解决这一问题：
- **Prompt engineering**：通过精心设计的指令引导模型注意事实一致性，但仅依赖推理时的提示工程，无法从根本上改变模型的参数化知识表示。
- **SFT（Supervised Fine-Tuning）**：在任务特定数据上进行监督微调，然而标准 SFT 仅优化单向的 next-token prediction，缺乏对输出结构一致性的显式约束。
- **SymbCoT**：利用符号化的 chain-of-thought 进行逻辑推理，虽能提升特定推理任务的忠实度，但局限于符号推理场景，难以泛化到开放域的摘要、问答等通用任务。

这些方法的共同局限在于：**要么绑定特定任务或推理形式，要么仅在自然语言空间内操作，无法利用编程语言固有的严格逻辑结构来约束生成过程**。编程语言（如 Python）具有确定的执行语义和类型系统，其语法结构天然排斥逻辑矛盾——这正是自然语言所缺乏的。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c681d19d-ef03-4857-83f8-041f19348f2c/figures/Figure_1.png)
*Figure 1 (motivation): This figure illustrates that standard LLM responses (left) are susceptible to input-conflicting and context-conflicting hallucinations, while the right shows how FACT addresses these issues via alternating code-text training.*



本文的核心动机由此而生：能否将代码的逻辑一致性"迁移"到通用文本生成中，以一种任务无关的方式系统性抑制不一致幻觉？FACT 首次提出交替代码-文本训练范式，让模型在文本→代码→文本的双向转换中习得结构化的语义表示。

## 核心创新

核心洞察：**代码的严格语法与执行语义可作为自然语言的"逻辑约束层"**，因为编程语言不允许变量未定义、类型冲突或控制流矛盾，从而使模型在交替预测文本与代码的过程中习得内在一致性的语义空间成为可能。

| 维度 | Baseline | 本文 |
|:---|:---|:---|
| 训练目标 | 单向 next-token prediction $\mathcal{L}_{\text{SFT}} = -\sum_t \log P(x_t \| x_{<t})$ | 双向交替损失 $\mathcal{L}_{\text{FACT}} = \mathcal{L}_{\text{T}\rightarrow\text{C}} + \mathcal{L}_{\text{C}\rightarrow\text{T}}$ |
| 数据来源 | 手工合成数据或任务特定数据 | 事实驱动过滤的 Wiki-40B-en + 伪代码生成 |
| 推理方式 | 端到端文本生成 | 两阶段结构化代码推理 text→code→output |
| 质量保障 | 人工标注或 gold-standard 代码 | 两阶段伪标签质量评估（语法有效性+语义保真度） |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c681d19d-ef03-4857-83f8-041f19348f2c/figures/Figure_2.png)
*Figure 2 (pipeline): An overview of FACT begins with the cleanup of the raw base text, followed by alternating text-based and code-based self-reinforcement.*



FACT 的整体流程由五个串联模块构成，形成从原始网页文本到结构化输出的完整流水线：

1. **Fact-Driven Text Filter（事实驱动文本过滤器）**：输入原始网页语料 Wiki-40B-en，利用 LLM-based 分类器筛选出事实型文本（53.74% 保留率，93% true positive rate，92% true negative rate），过滤掉非事实型（27.85%）和无效文本（18.41%）。

2. **Pseudo-Code Generator（伪代码生成器）**：将过滤后的事实型文本转换为结构化代码伪标签，无需人工标注的 ground-truth 代码。

3. **Two-Stage Quality Assessor（两阶段质量评估器）**：对生成的伪代码进行语法有效性检查（可执行代码比例）和语义保真度验证（重建文本与原始文本的相似度），过滤低质量样本。

4. **Alternating Trainer（交替训练器）**：核心训练模块，在高质量代码-文本对上交替执行 T→C（文本到代码预测）和 C→T（代码到文本重建），建立共享语义空间。

5. **Structured Code Inference（结构化代码推理）**：测试时，输入文本先转换为结构化代码中间表示，再基于代码生成最终输出；无法转换的片段保留为纯文本作为 fallback。

```
Wiki-40B-en 原始文本
    ↓
[Fact-Driven Text Filter] ──→ 事实型文本 / 非事实型 / 无效
    ↓
[Pseudo-Code Generator] ──→ 结构化代码伪标签
    ↓
[Two-Stage Quality Assessor] ──→ 高质量 (code, text) 对
    ↓
[Alternating Trainer] ──→ 共享语义空间模型
    ↓
测试输入 → [text→code] → [code→output] + fallback → 最终输出
```

## 核心模块与公式推导

### 模块 1: 交替代码-文本训练目标（对应框架图 核心训练模块）

**直觉**: 单向的 next-token prediction 只能捕捉文本的线性共现模式，而双向交替预测迫使模型在两种表示形式之间建立可逆映射，从而将代码的结构约束编码进语义空间。

**Baseline 公式** (SFT): $$\mathcal{L}_{\text{SFT}} = -\sum_{t} \log P(x_t \text{mid} x_{<t})$$
符号: $x_t$ = 第 $t$ 个文本 token，$x_{<t}$ = 历史上下文，模型仅学习从左到右的文本生成概率。

**变化点**: 标准 SFT 完全在自然语言空间操作，缺乏显式的结构化中间表示；代码的缺失导致模型无法习得逻辑一致性约束。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathcal{L}_{\text{T}\rightarrow\text{C}} = -\sum_{t} \log P(c_t \text{mid} x, c_{<t}) \quad \text{加入文本到代码的预测损失，引入结构化约束}$$
$$\text{Step 2}: \quad \mathcal{L}_{\text{C}\rightarrow\text{T}} = -\sum_{t} \log P(x_t \text{mid} c, x_{<t}) \quad \text{加入代码到文本的重建损失，保证语义可恢复性}$$
$$\text{最终}: \mathcal{L}_{\text{FACT}} = \mathcal{L}_{\text{T}\rightarrow\text{C}} + \mathcal{L}_{\text{C}\rightarrow\text{T}}$$
符号: $c_t$ = 代码 token，$x$ = 输入文本，$c$ = 对应代码表示。两个方向共享模型参数，形成联合优化。

**对应消融**: 去掉交替训练中的任一方向（仅保留 T→C 或仅保留 C→T）会导致语义空间不对称，重建保真度下降。Figure 4/5 的消融显示完整交替机制对最终性能至关重要。

---

### 模块 2: 结构化代码推理（对应框架图 推理模块）

**直觉**: 测试时显式引入代码中间层，让生成过程受代码语法约束，从源头阻止逻辑矛盾的输出。

**Baseline 公式** (端到端生成): $$P(y \text{mid} x) = \prod_{t} P(y_t \text{mid} y_{<t}, x)$$
符号: $y$ = 输出文本，$x$ = 输入，模型直接学习输入到输出的映射，无中间约束层。

**变化点**: 端到端生成缺乏显式的结构验证步骤，矛盾只能在输出后检测；本文将代码作为"可执行的语义检查点"。

**本文公式（推导）**:
$$\text{Step 1}: \quad P(\text{code} \text{mid} \text{input}) \quad \text{将输入文本解析为结构化代码表示}$$
$$\text{Step 2}: \quad P(\text{output} \text{mid} \text{code}) \quad \text{基于代码生成最终自然语言输出}$$
$$\text{最终}: P(\text{output} \text{mid} \text{input}) = \sum_{\text{code}} P(\text{code} \text{mid} \text{input}) \cdot P(\text{output} \text{mid} \text{code})$$
符号: $\text{code}$ = 结构化代码中间表示，求和覆盖所有可能的代码解析结果。对于无法转换为代码的片段，保留原始文本作为 fallback。

**对应消融**: Table 2 / Figure 4 显示，code-based inference 在 CNN/Daily Mail 上 Consistency 为 90.24%，相比 end-to-end 的 91.08% 仅下降 0.84%，但在幻觉指标 Anah-v2 上 SQuAD V2 为 9.54% 对比 end-to-end 的 8.37%，上升 1.17%（更差），说明推理方式的 trade-off。

---

### 模块 3: 两阶段伪标签质量评估（对应框架图 数据质量控制模块）

**直觉**: 没有 ground-truth 代码时，必须自动验证伪标签质量以避免噪声训练信号。

**Baseline**: 无显式质量评估，或依赖单一启发式规则（如代码长度）。

**本文公式**:
$$\text{Quality Score} = \alpha \cdot \mathbb{1}[\text{code executable}] + \beta \cdot \text{Sim}(\text{reconstructed text}, \text{original text})$$
符号: $\mathbb{1}[\cdot]$ = 指示函数（可执行为 1，否则为 0），$\text{Sim}$ = 文本相似度度量（如 BLEU 或 embedding cosine similarity），$\alpha, \beta$ = 权重系数。

**变化点**: 单一指标无法同时保证语法正确性和语义保真度；本文将两者解耦为独立阶段，形成互补过滤。

**对应消融**: Figure 6 显示，可执行代码比例从初始 87.28% 总体上升，重建相似度分数在第三次迭代前提升最多 5.23%，之后趋于平稳——这直接支撑了三迭代训练计划的设计决策。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c681d19d-ef03-4857-83f8-041f19348f2c/figures/Table_1.png)
*Table 1 (result): Hallucination evaluation results for all methods and backbone models across datasets.*



本文在四个基准数据集上评估 FACT：CNN/Daily Mail（新闻摘要）、SAMSum（对话摘要）、SQuAD V2（阅读理解问答）、HaluEval（幻觉专项评测）。实验覆盖三个 backbone 模型：LLaMA-3.1-Instruct-8B、Ministral-Instruct-8B、Qwen-2.5-Instruct-7B。核心结果如 Table 1 和 Table 2 所示。

在摘要任务上，FACT 展现出显著优势：SAMSum 数据集上 Coherence 达到 98.77，超越最强 baseline SymbCoT（96.37）+2.40，超越 Base 模型（93.25）+5.52；Relevance 达到 98.09，超越 SymbCoT（95.78）+2.31，超越 Base（90.33）+7.76。CNN/Daily Mail 上 Coherence 97.89（+1.07 对比 Prompt 96.82），Relevance 97.26（+1.70 对比 Prompt 95.56）。在问答任务上，SQuAD V2 的 F1 为 88.89（+4.52 对比 Prompt 84.37，+5.30 对比 Base 83.59），Exact Match 87.63（+1.43 对比 SymbCoT 86.2，+8.56 对比 Base 79.07）。HaluEval 上 F1 87.26（+1.82 对比 SymbCoT 85.44），Exact Match 84.86（+2.61 对比 SymbCoT 81.73）。这些数字表明 FACT 在保持通用任务性能的同时，有效抑制了不一致幻觉。


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c681d19d-ef03-4857-83f8-041f19348f2c/figures/Table_2.png)
*Table 2 (result): Task evaluation results for all methods and backbone models across datasets.*



消融实验（Figure 4、Figure 5）进一步验证各组件贡献。最关键的消融是推理方式对比：将 code-based inference 替换为 end-to-end 后，CNN/Daily Mail 上 Consistency 从 90.24% 微降至 91.08%（实际 end-to-end 更高，差值 -0.84%），但 AlignScore 下降 0.71%，Coherence 下降 0.88%，Relevance 下降 0.54%；SQuAD V2 上 Anah-v2（幻觉指标）从 8.37% 恶化至 9.54%（+1.17%），F1 下降 2.03%，AlignScore 下降 2.03%。这说明 code-based inference 在幻觉控制上具有明确价值，尽管对个别指标有轻微代价。


![Figure 6, Figure 7](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c681d19d-ef03-4857-83f8-041f19348f2c/figures/Figure_6,_Figure_7.png)
*Figure 6, Figure 7 (result): Figure 6: The trend of data proportion of given dataset on different training steps. Figure 7: Human evaluation on CNN/Daily Mail and XSum for LLaMA backbones.*



数据迭代方面，Figure 6 揭示伪标签质量的三迭代饱和现象：可执行代码比例从 87.28% 起步总体上升，重建相似度在第三迭代前提升最多 5.23% 后 plateau。这一观察直接决定了训练计划的三迭代设计，避免了无效计算。

公平性审视：本文对比的 baseline 中，SFT 和 Prompt 为常规方法，SymbCoT 和 Lookback 是较新的专项方法，但缺少与 CodeT5、CodeBERT 等代码预训练方法的直接对比，也未在 gold-standard 代码数据上进行全量微调对比。此外，作者坦承 text filter 存在 7-8% 错误率可能引入噪声，且方法限于事实型文本，未评估创意生成场景。Code-based inference 相比 end-to-end 在部分指标上略逊，实际部署需权衡延迟与一致性收益。

## 方法谱系与知识库定位

FACT 属于 **code-augmented LLM reasoning** 方法族，直接继承自 "Language models of code are few-shot commonsense learners" [20] 的核心洞察——代码模型具备逻辑推理能力。FACT 将这一洞察从"代码模型能做常识推理"扩展为"通用 LLM 可通过交替代码-文本训练习得结构化一致性"，完成了从现象观察到系统范式的跃迁。

**改变的 slots**（相对父方法）：
- **training_recipe**: 标准 SFT → 交替 T→C / C→T 双向预测
- **data_pipeline**: 手工合成/任务数据 → 事实驱动过滤 + 伪代码 + 两阶段质量评估
- **inference_strategy**: 端到端生成 → 结构化代码中间表示推理
- **objective**: 单向 next-token → 共享语义空间的双向联合优化

**直接 baseline 对比**：
- **SFT**: FACT 替换其训练目标为交替双向损失，并引入代码中间层
- **SymbCoT**: 同为符号化推理，但 SymbCoT 局限逻辑推理任务，FACT 任务无关且无需外部求解器
- **Lookback**: 同为幻觉抑制，但 Lookback 仅检测注意力模式，FACT 从训练根源注入结构约束

**后续方向**：(1) 将交替训练扩展至多模态（代码-图像-文本三元交替）；(2) 设计可微分的代码执行反馈以替代两阶段启发式评估；(3) 探索创意生成场景下的"柔性结构化"变体。

**标签**: 模态=text | 范式=supervised fine-tuning with auxiliary code task | 场景=fact-based text generation (summarization, QA) | 机制=alternating bidirectional prediction + structured intermediate representation | 约束=no ground-truth code required, task-agnostic, limited to factual text

