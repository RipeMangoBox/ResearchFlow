---
title: 'PT-MoE: An Efficient Finetuning Framework for Integrating Mixture-of-Experts into Prompt Tuning'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 低秩分解与MoE协同的提示微调框架
- PT-MoE
- PT-MoE achieves state-of-the-art PE
acceptance: Poster
cited_by: 5
method: PT-MoE
modalities:
- Text
paradigm: supervised
---

# PT-MoE: An Efficient Finetuning Framework for Integrating Mixture-of-Experts into Prompt Tuning

**Topics**: [[T__Text_Generation]], [[T__Math_Reasoning]] | **Method**: [[M__PT-MoE]] | **Datasets**: [[D__DROP]] (其他: MRQA, Mathematical Problem Solving)

> [!tip] 核心洞察
> PT-MoE achieves state-of-the-art PEFT performance by combining matrix decomposition with MoE routing for prompt tuning, yielding complementary benefits of efficient parameter sharing and dynamic adaptation.

| 中文题名 | 低秩分解与MoE协同的提示微调框架 |
| 英文题名 | PT-MoE: An Efficient Finetuning Framework for Integrating Mixture-of-Experts into Prompt Tuning |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2505.09519) · Code · Project |
| 主要任务 | Question Answering, Mathematical Problem Solving |
| 主要 baseline | PT (Prompt Tuning), SMoP, DPT, LoRA, HydraLoRA, ATTEMPT |

> [!abstract] 因为「矩阵分解和MoE路由单独使用反而会损害提示微调性能」，作者在「SMoP 和 DPT」基础上改了「将低秩分解（rank=36）与MoE路由统一集成，使分解矩阵跨专家共享」，在「MRQA benchmark（17个QA和数学数据集）」上取得「平均F1 58.26%，比PT提升+1.49，比SMoP提升+2.01」

- **MRQA平均F1**: PT-MoE 58.26% vs. PT 56.77% vs. SMoP 56.25% vs. DPT 55.77%
- **数学推理平均准确率**: PT-MoE 56.91% vs. PT 46.16%，提升 +10.75
- **参数效率**: 仅80K可训练参数，比LoRA少25%，比Full Fine-tuning的1.2B参数少99.99%

## 背景与动机

参数高效微调（PEFT）旨在冻结预训练大模型的同时，仅优化少量额外参数以适应下游任务。Prompt Tuning（PT）是其中一类代表性方法：在输入前添加可学习的连续软提示词（soft prompt），保持模型权重不变。然而，现有研究出现了反直觉的现象：SMoP（Sparse Mixture-of-Prompts）将MoE路由引入提示微调，虽然增加了训练效率，却未在性能上全面超越标准PT；DPT（Decomposed Prompt Tuning）采用低秩矩阵分解压缩参数，在某些领域反而能提升性能，但分解本身却可能损害效果。

具体而言，SMoP [2] 使用2个专家提示词配合路由器与噪声注入，试图通过动态选择实现自适应，但MRQA上F1仅56.25%，反而低于PT的56.77%。DPT采用SVD分解将提示词压缩为低秩形式（rank=39），参数减少却导致F1降至55.77%。这两种方法各自只实现了"分解"或"路由"单一机制，未能协同发挥作用。LoRA [16] 虽然在数学推理上表现优异，但在QA任务上不如PT系列；反之PT在QA上强势，却在数学推理上明显落后（PT 46.16% vs. LoRA更高）。这种任务间的不一致性限制了实际应用。

作者的核心观察是：分解与路由并非简单叠加关系，而是存在互补潜力——分解能实现跨专家的参数共享，路由能实现输入动态适应。本文提出PT-MoE，将两者统一为协同设计，首次证明其组合效果远超各自单独使用。
![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e74236bd-be83-4733-bbc6-2a176d0b56ee/figures/fig_001.png)
*Figure: Performance comparison of PEFT methods*



## 核心创新

核心洞察：低秩分解与MoE路由的互补性源于它们解决了提示微调中两个正交的瓶颈——分解通过跨专家共享参数减少冗余，路由通过输入相关组合实现动态适应，从而使"参数更少、适应更强"的联合优化成为可能。

| 维度 | Baseline (SMoP / DPT / PT) | 本文 (PT-MoE) |
|:---|:---|:---|
| 提示词表示 | SMoP: 多个完整专家提示词直接存储; DPT: 单一低秩分解; PT: 单一完整向量 | 共享低秩基础矩阵 + 专家特定调制，rank=36 |
| 路由机制 | SMoP: 有路由但无分解; DPT/PT: 无路由 | 线性路由器 nn.Linear(embedding_dim, num_prompts)，基于嵌入维度动态选择 |
| 参数共享 | SMoP: 专家间无共享; DPT: 无多专家结构 | 分解矩阵A跨所有专家共享，B_i专家特定，实现高效共享 |
| 训练动态 | SMoP: 噪声注入; DPT/PT: 标准优化 | 联合优化共享矩阵与路由权重，kaiming_uniform_初始化A，zeros_初始化B |

关键区别在于：PT-MoE不是将DPT和SMoP作为独立模块拼接，而是将路由操作定义在分解后的低秩空间上，使路由器选择的"专家"实际上是对共享基础的差异化调制，从根本上改变了参数-功能的映射关系。

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e74236bd-be83-4733-bbc6-2a176d0b56ee/figures/fig_002.png)
*Figure: Framework of PT-MoE. Each soft prompt*



PT-MoE的数据流如下：文本输入首先经过嵌入层得到语义表示，随后分为两条路径——主路径直接进入Transformer进行编码解码，路由分支则将嵌入表示送入线性路由器生成专家组合权重。核心创新发生在提示词生成侧：共享的低秩矩阵A（经kaiming_uniform_初始化）与专家特定的矩阵B_i（经zeros_初始化）相乘重构基础提示词，再经路由器权重α_i(e)加权组合，并注入乘性噪声ε_i ~ *(1+torch.randn_like()*0.01)进行训练正则化。最终处理后的提示词与原始输入拼接，通过掩码语言建模输出预测。

各模块定义：
- **输入嵌入模块**：接收文本token，输出embedding维度的语义表示，供下游双分支使用
- **线性路由器（Router）**：nn.Linear(embedding_dim, num_prompts)，将输入嵌入映射为2个专家的归一化组合权重
- **低秩分解重构模块**：共享A ∈ ℝ^{L×r}与专家特定B_i ∈ ℝ^{r×d}，rank r=36，重构提示词维度
- **专家组合与噪声注入模块**：按路由权重加权求和，施加乘性噪声增强训练鲁棒性
- **掩码语言建模模块**：仅对非提示词位置（集合M）计算交叉熵损失，避免模型学习预测提示词本身

```
文本输入 → [嵌入层] ──┬──→ [Transformer主路径]
                    └──→ [Router] → α₁, α₂
                         ↓
                    [低秩重构: A·B₁, A·B₂] 
                         ↓
                    [加权组合 + 噪声注入]
                         ↓
                    [提示词拼接] → [Masked LM] → 输出
```

## 核心模块与公式推导

### 模块 1: 掩码交叉熵损失（对应框架图 输出端）

**直觉**: 提示词位置不应参与损失计算，否则模型会学习预测无意义的连续向量而非目标任务输出。

**Baseline 公式** (标准语言模型): $$L_{\text{LM}} = -\sum_{t=1}^{T} \log p(y_t | x_{<t})$$
符号: $y_t$ = 第t位置的目标token, $x_{<t}$ = 前文上下文, $T$ = 序列总长。

**变化点**: 标准PT同样采用此掩码策略，但PT-MoE明确继承并强调这一设计，确保路由器优化的提示词不干扰目标预测。

**本文公式**:
$$\text{Step 1}: M = \{t \text{mid} x_t \text{ 不是提示词位置}\} \quad \text{构建二元掩码区分提示词与内容}$$
$$\text{最终}: L = -\sum_{t \in M} \log p(y_t | x_{<t})$$

**对应消融**: 此为基础设计，未单独消融；但所有对比方法（PT/SMoP/DPT）均采用相同掩码，保证公平比较。

---

### 模块 2: 线性路由器（对应框架图 Router分支）

**直觉**: 输入内容决定需要激活哪些专家组合，实现"同一段文本、不同问题、不同提示策略"的动态适应。

**Baseline 公式** (SMoP, 隐式): SMoP使用路由器但架构细节不同，其路由可能基于中间层表示而非原始嵌入。

**变化点**: PT-MoE的路由器直接操作输入嵌入维度，简化设计并降低延迟；同时路由目标从"选择单一专家"变为"加权组合低秩残差"。

**本文公式**:
$$\text{Step 1}: \alpha(e) = W_r \cdot e, \quad W_r \in \mathbb{R}^{\text{num\_prompts} \times \text{embedding\_dim}} \quad \text{线性投影生成原始权重}$$
$$\text{Step 2}: \alpha_i(e) = \frac{\exp(\alpha_i)}{\sum_j \exp(\alpha_j)} \quad \text{Softmax归一化保证凸组合}$$
$$\text{最终}: \alpha(e) = \text{softmax}(W_r \cdot e) \in \mathbb{R}^k, \quad k=2 \text{ (专家数)}$$
符号: $e$ = 输入嵌入, $W_r$ = 可学习路由矩阵, $\alpha_i(e)$ = 第i个专家的动态权重。

**对应消融**: Figure 3（左/中）显示路由机制与专家数的联合影响；SMoP单独使用MoE反而-0.52 F1，证明必须与分解结合。

---

### 模块 3: 低秩分解与专家路由协同（对应框架图 核心提示词生成）

**直觉**: 共享基础捕捉跨任务的通用提示结构，专家残差捕捉任务/输入特定模式，噪声防止路由坍塌到单一专家。

**Baseline 公式**:
- PT [14]: $P \in \mathbb{R}^{L \times d}$，直接优化完整提示词矩阵
- DPT: $P = A \cdot B, A \in \mathbb{R}^{L \times r}, B \in \mathbb{R}^{r \times d}$，静态低秩分解
- SMoP: $P = \sum_{i=1}^{k} \alpha_i(x) \cdot P_i$，无分解的多专家加权

**变化点**: DPT的分解是静态的、无动态适应；SMoP的路由缺乏参数共享导致冗余。PT-MoE将路由定义在分解空间：所有专家共享同一A矩阵，仅B_i和调制系数不同，实现"共享基础+动态残差"的统一。

**本文公式（推导）**:
$$\text{Step 1}: P_{\text{base}} = A \cdot B, \quad A \in \mathbb{R}^{L \times 36}, B \in \mathbb{R}^{36 \times d} \quad \text{构建共享低秩基础}$$
$$\text{Step 2}: P_i^{\text{expert}} = A \cdot B_i + \epsilon_i, \quad \epsilon_i = \text{torch.randn\_like}(P_i) \times 0.01 \quad \text{专家特定残差+训练噪声}$$
$$\text{Step 3}: P_{\text{final}} = P_{\text{base}} + \sum_{i=1}^{k} \alpha_i(e) \cdot P_i^{\text{expert}} \quad \text{路由加权组合，保留共享基础并叠加动态残差}$$
$$\text{最终}: P = A \cdot B + \sum_{i=1}^{2} \alpha_i(e) \cdot (A \cdot B_i + \epsilon_i)$$
符号: $L=40$ = 提示词长度, $d$ = 嵌入维度, $r=36$ = 分解秩, $B_i$ = 第i个专家的特定矩阵, $\epsilon_i$ = 乘性噪声项。

**对应消融**: Table 2/相关结果显示——DPT单独55.77%（-1.0 vs PT），SMoP单独56.25%（-0.52 vs PT），而PT-MoE组合达58.26%（+1.49 vs PT），证明分解与路由的互补性必须联合才能释放。

## 实验与分析



本文在MRQA benchmark（涵盖Natural Questions、TriviaQA、DROP、SearchQA、TextbookQA、RACE等12个QA数据集）以及GSM8K、MAWPS等数学推理任务上评估PT-MoE。核心结果来自Table 2和Table 4：PT-MoE在MRQA上取得平均F1 58.26%，相比Prompt Tuning的56.77%提升+1.49，相比SMoP的56.25%提升+2.01，相比DPT的55.77%提升+2.49；Exact Match指标同样领先，47.13% vs. PT 45.52%（+1.61）。更关键的是跨任务一致性：在数学推理上PT-MoE达到56.91%平均准确率，相比PT的46.16%大幅提升+10.75，而SMoP在此任务上反而跌至41.05%（-5.11 vs PT），证明MoE单独使用可能有害，但与分解结合后产生质变。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/e74236bd-be83-4733-bbc6-2a176d0b56ee/figures/fig_003.png)
*Figure: Ablation studies on key components of PT-MoE, showing the influence of (Left) prompt length, (Center*



参数效率方面，PT-MoE仅使用80K可训练参数，对比Full Fine-tuning的1.2B参数减少99.99%，且比LoRA少25%（Figure 4）。在DROP数据集上，PT-MoE以80K参数取得F1 48.02，超越Full Fine-tuning的43.87%达+4.15，展示极端参数效率下的性能优势。



消融分析揭示关键洞察（Figure 3）：矩阵分解单独使用（DPT）导致F1 55.77%，比标准PT低1.0点；MoE路由单独使用（SMoP）导致56.25%，比PT低0.52点。两者组合后PT-MoE跃升至58.26%，提升幅度远超各自独立效果的线性叠加，验证"互补性"核心主张。Figure 3还显示提示词长度和噪声强度的敏感性分析。

公平性检查：对比基线包含PT、SMoP、DPT、LoRA（r=1, alpha=16）、HydraLoRA、ATTEMPT，覆盖提示微调和适配器两大范式。但存在潜在问题：LoRA使用r=1属于极低秩配置，标准实践常用r=8或更高，可能低估LoRA潜力；仅测试2个专家，更多专家的扩展性未验证；RACE数据集上PT-MoE相比PT有边际下降，说明并非 universally better。此外，Adapter、IA3、BitFit等强基线未纳入比较，Full Fine-tuning的完整跨数据集结果也未报告。

## 方法谱系与知识库定位

PT-MoE属于**Prompt Tuning方法族**，直接父方法为SMoP [2]（MoE路由框架）和DPT（低秩分解框架），同时根植于Lester et al.的原始Prompt Tuning [14] 与 Prefix-Tuning [16]。方法谱系中的关键演进：SMoP添加了MoE路由但缺乏参数共享 → DPT添加了矩阵分解但缺乏动态适应 → PT-MoE统一两者，使路由操作在分解空间上执行，实现跨专家参数共享与输入动态选择的协同。

**直接基线差异**：
- vs. PT [14]: 将单一静态提示词替换为"共享基础+路由残差"的动态结构
- vs. SMoP [2]: 引入低秩分解使专家间共享参数，从存储k个完整提示词变为共享A+k个小型B_i
- vs. DPT: 添加MoE路由和噪声注入，将静态分解转化为动态自适应系统
- vs. LoRA: 在提示空间而非权重空间操作，保持模型冻结，参数量更少（80K vs. LoRA更高）
- vs. ATTEMPT [1]: 使用连续路由权重而非注意力机制组合，且集成分解实现参数共享

**后续方向**：(1) 扩展至>2个专家并设计层次化路由以验证可扩展性；(2) 探索分解秩与专家数的联合优化，替代固定rank=36；(3) 将协同设计迁移至视觉-语言多模态提示微调。

**标签**: 模态=text | 范式=parameter-efficient fine-tuning, prompt tuning | 场景=question answering, mathematical reasoning | 机制=low-rank decomposition, mixture-of-experts routing, noise regularization | 约束=frozen backbone, <100K trainable parameters

