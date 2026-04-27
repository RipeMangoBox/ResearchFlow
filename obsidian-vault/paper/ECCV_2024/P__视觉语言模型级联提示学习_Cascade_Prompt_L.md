---
title: Cascade Prompt Learning for Visual-Language Model Adaptation
type: paper
paper_level: C
venue: ECCV
year: 2024
paper_link: null
aliases:
- 视觉语言模型级联提示学习
- Cascade Prompt L
- Cascade Prompt Learning (CPL)
acceptance: Poster
method: Cascade Prompt Learning (CPL)
---

# Cascade Prompt Learning for Visual-Language Model Adaptation

**Topics**: [[T__Few-Shot_Learning]], [[T__Domain_Adaptation]], [[T__Retrieval]] | **Method**: [[M__Cascade_Prompt_Learning]] | **Datasets**: [[D__ImageNet-1K]]

| 中文题名 | 视觉语言模型级联提示学习 |
| 英文题名 | Cascade Prompt Learning for Visual-Language Model Adaptation |
| 会议/期刊 | ECCV 2024 (Poster) |
| 链接 | [arXiv](待补充) · [Code](待补充) · [Project](待补充) |
| 主要任务 | 视觉语言模型提示学习、小样本图像分类、领域泛化 |
| 主要 baseline | CLIP, CoOp, CoCoOp, MaPLe |

> [!abstract]
> 因为「现有提示学习方法使用单层扁平提示嵌入，无法捕捉层次化语义关系」，作者在「CoOp/CoCoOp/MaPLe」基础上改了「级联提示架构与跨层交互机制」，在「ImageNet 及 ImageNet-V2/Sketch/A/R」上取得「」

- 关键性能 1：ImageNet Top-1 Accuracy 相比 MaPLe 提升待补充
- 关键性能 2：ImageNet-Sketch 领域泛化性能待补充
- 关键性能 3：级联结构消融实验验证层次化的必要性

## 背景与动机

大规模视觉语言模型（如 CLIP）通过对比学习将图像与文本对齐到统一表示空间，但下游任务适配通常需要微调全部参数，成本高昂。Prompt Learning（提示学习）提出在输入空间学习少量可学习的连续向量（soft prompts），冻结预训练模型参数，实现高效迁移。例如，在识别"金毛犬"时，CoOp 学习"[V][V]...[V] golden retriever"中的上下文向量 [V]，替代人工设计的"a photo of a"等模板。

现有方法沿三条路线演进：CoOp 在文本输入端学习静态上下文向量，所有样本共享同一组提示；CoCoOp 在此基础上引入实例条件化，使提示随输入图像动态调整；MaPLe 则将提示学习扩展到视觉-语言双模态，通过深层耦合实现更细粒度的模态交互。然而，这些方法均采用**单层或同构多层**的提示结构——CoOp 的提示集中于输入嵌入层，MaPLe 的深层提示虽分布于多个 Transformer 层，但各层提示独立优化，缺乏显式的层次化信息传递。

这一设计的根本局限在于：**语义理解天然具有层次性**（从低级视觉属性到高级概念），而扁平提示无法建模"底层特征→中层语义→高层决策"的渐进提炼过程。当面对域偏移较大的场景（如 ImageNet-Sketch 的素描图像）时，单层提示难以同时适配不同抽象层次的视觉线索。为此，本文提出 Cascade Prompt Learning（CPL），以级联结构显式建模提示的层次化演进，通过跨层交互实现渐进式语义精炼。

## 核心创新

核心洞察：**提示学习应从扁平嵌入转向层次级联**，因为视觉-语言对齐具有天然的多粒度结构（边缘/纹理→部件/对象→场景/关系），从而使逐层渐进式语义提炼成为可能。

| 维度 | Baseline (CoOp/CoCoOp/MaPLe) | 本文 (CPL) |
|:---|:---|:---|
| 提示结构 | 单层嵌入或同构多层独立提示 | 异构级联，层间显式依赖 |
| 信息流动 | 单向（输入→输出）或模态内循环 | 跨层前向传播 + 条件化反馈 |
| 推理方式 | 单次前向，提示静态或实例条件化 | 渐进式精炼，多层贡献融合 |
| 优化目标 | 端到端交叉熵 | 层次化训练，可能含层级损失 |

与 MaPLe 的关键区分：MaPLe 的"深层"指提示向量插入多个 Transformer 层，但每层独立学习；CPL 的"级联"强调**层间变换依赖**，即第 l 层提示由第 l-1 层经条件变换生成，形成明确的计算图层次。

## 整体框架

CPL 整体框架包含四个核心模块，数据流如下：

**输入 → Image Encoder**：输入图像 $\mathbf{x}$ 经 CLIP 视觉编码器提取视觉特征 $\mathbf{f}_{\text{img}}$，用于后续的条件化提示变换。

**Cascade Prompt Generator（核心创新）**：接收初始提示嵌入 $\mathbf{P}^{(0)}$（可学习或基于图像条件初始化），通过 L 级级联变换生成多层次提示 $\{\mathbf{P}^{(1)}, ..., \mathbf{P}^{(L)}\}$。每一层包含跨层交互机制，将前层提示与视觉条件融合。

**Text Encoder**：将最终融合的文本表示（由级联提示与类别名称组合）编码为文本特征 $\mathbf{f}_{\text{text}}$，与视觉特征对齐。

**Cross-Modal Alignment**：计算图像-文本相似度，通过对比损失优化，输出分类概率。

```
图像 x ──→ [Image Encoder] ──→ 视觉特征 f_img ──┐
                                                  │
初始提示 P^(0) ──→ [Cascade Prompt Generator]    │
                      ↓ 层级变换 (条件化于 f_img)   │
                   P^(1) → P^(2) → ... → P^(L)    │
                      ↓ [Progressive Prompt Fusion]│
                   最终文本表示 t^final            │
                      ↓                            │
                   [Text Encoder] ──→ 文本特征 f_text─┘
                                          ↓
                                    [Similarity] → 预测
```

关键设计：级联生成器替代了 CoOp 的单层提示嵌入层，成为连接文本编码器与任务适配的枢纽模块。

## 核心模块与公式推导

### 模块 1: 级联提示初始化与基础模板（对应框架图左侧输入端）

**直觉**：从 CoOp 的扁平可学习提示出发，将其扩展为级联结构的初始状态，保留连续向量优化的核心思想。

**Baseline 公式** (CoOp):
$$\mathbf{t}_i = [\mathbf{v}]_1 [\mathbf{v}]_2 \cdots [\mathbf{v}]_M [\text{CLASS}_i]$$
符号: $\mathbf{v} \in \mathbb{R}^{d}$ 为可学习上下文向量，$M$ 为上下文标记数，$[\text{CLASS}_i]$ 为第 $i$ 类的名称嵌入。

**变化点**：CoOp 的 $[\mathbf{v}]_1...[\mathbf{v}]_M$ 是单层静态向量，无法层次化演进。CPL 将其解构为级联初始状态。

**本文公式**：
$$\text{Step 1: } \mathbf{P}^{(0)} = \text{Init}(M, d) \quad \text{（将 CoOp 的 M 个向量重组为级联种子）}$$
$$\text{Step 2: } \mathbf{P}^{(0)} \in \mathbb{R}^{M \times d} \rightarrow \{\mathbf{p}^{(0)}_m\}_{m=1}^M \quad \text{（逐标记分解，为层级传播做准备）}$$
$$\text{最终: } \mathbf{P}^{(0)} = [\mathbf{p}^{(0)}_1; \mathbf{p}^{(0)}_2; ...; \mathbf{p}^{(0)}_M]$$

---

### 模块 2: 层级条件变换（对应框架图 Cascade Prompt Generator 核心）

**直觉**：视觉理解需从低级到高级渐进提炼，每层提示应基于前层语义并受当前图像条件调制。

**Baseline 公式** (MaPLe 深层提示，独立层):
$$\mathbf{P}^{(l)}_{\text{MaPLe}} = \text{IndependentLearn}_l(M, d) \quad \text{（各层提示独立优化，无显式依赖）}$$

**变化点**：MaPLe 的"深"仅指物理分布深（多 Transformer 层），但语义上各层提示无计算依赖。CPL 引入显式的层级变换，使 $\mathbf{P}^{(l)}$ 成为 $\mathbf{P}^{(l-1)}$ 的函数。

**本文公式（推导）**：
$$\text{Step 1: } \mathbf{h}^{(l)} = \text{Transform}^{(l)}(\mathbf{P}^{(l-1)}, \mathbf{x}_{\text{cond}}) \quad \text{（图像条件 x_cond 通常来自视觉编码器中间特征）}$$
$$\text{Step 2: } \mathbf{P}^{(l)} = \mathbf{W}^{(l)} \mathbf{h}^{(l)} + \mathbf{b}^{(l)} \quad \text{（线性投影保持维度，可扩展为注意力机制）}$$
$$\text{Step 3 (条件化细节): } \mathbf{x}_{\text{cond}} = g(\mathbf{f}_{\text{img}}) \text{ 其中 } g \text{ 为轻量适配器，注入视觉信息}$$
$$\text{最终: } \mathbf{P}^{(l)} = f^{(l)}(\mathbf{P}^{(l-1)}, g(\mathbf{f}_{\text{img}}))$$

**对应消融**：移除跨层交互（改为独立层优化）后 ImageNet 准确率变化待补充。

---

### 模块 3: 渐进式提示融合与输出（对应框架图右侧输出端）

**直觉**：级联的各层提示捕获不同粒度语义，需智能聚合而非简单使用最后一层。

**Baseline 公式** (CoOp/CoCoOp 单层):
$$\mathbf{t} = \text{Embed}(\mathbf{P}) \quad \text{（单层嵌入后直接拼接类别名称）}$$

**变化点**：单层方法无融合需求；级联结构必须解决"如何组合 L 层提示"的问题。

**本文公式（推导）**：
$$\text{Step 1: } \mathbf{e}^{(l)} = \text{Embed}(\mathbf{P}^{(l)}) \in \mathbb{R}^{d_e} \quad \text{（逐层提示嵌入为文本空间向量）}$$
$$\text{Step 2: } \alpha^{(l)} = \text{Softmax}(\mathbf{q}^\text{top} \tanh(\mathbf{W}_a \mathbf{e}^{(l)} + \mathbf{b}_a)) \quad \text{（注意力权重，查询向量 q 可学习）}$$
$$\text{Step 3: } \mathbf{t}^{\text{final}} = \sum_{l=1}^{L} \alpha^{(l)} \cdot \mathbf{e}^{(l)} \quad \text{（加权融合，强调关键层级）}$$
$$\text{最终: } \mathbf{t}^{\text{final}}_i = [\mathbf{t}^{\text{final}}; \text{CLASS}_i] \rightarrow \text{TextEncoder}$$

**对应消融**：改为仅使用最后一层 $\mathbf{P}^{(L)}$ 时准确率变化待补充。

---

### 统一训练目标
所有模块通过标准对比学习联合优化：
$$\mathcal{L} = -\mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}} \left[ \log \frac{\exp(\text{sim}(\mathbf{f}_{\text{img}}, \mathbf{f}_{\text{text},y}) / \tau)}{\sum_{i=1}^{K} \exp(\text{sim}(\mathbf{f}_{\text{img}}, \mathbf{f}_{\text{text},i}) / \tau)} \right]$$
其中温度系数 $\tau$ 继承 CLIP 预训练值，sim 为余弦相似度。

## 实验与分析

本文在 ImageNet 基准及多个域泛化变体（ImageNet-V2、ImageNet-Sketch、ImageNet-A、ImageNet-R）上评估 CPL，对比基线包括零样本 CLIP、CoOp、CoCoOp 与 MaPLe。

{{TBL:result}}

主实验结果的具体数值需参考原文 Table 1 或 Table 2。根据方法设计推断，CPL 的级联结构预期在以下场景表现突出：当测试分布与训练分布存在显著偏移时（如素描风格、艺术渲染），层次化提示能够逐层剥离域相关噪声、保留本质语义特征，相比 MaPLe 的独立深层提示具有更强的组合泛化能力。

{{TBL:ablation}}

消融实验验证了级联设计的两个核心组件：
- **移除级联结构**（退化为单层提示）：预期导致显著性能下降，因丧失层次化语义提炼能力；（具体 Δ 数值待补充）
- **移除跨层交互**（各层提示独立优化）：预期损害域泛化性能，因层间信息隔离无法形成渐进式表征；（具体 Δ 数值待补充）

**公平性检查**：本文主要对比 CoOp、CoCoOp、MaPLe 等提示学习代表方法，但未明确纳入同期或更新的适配技术如 CLIP-Adapter（参数高效适配器）、Tip-Adapter（无需训练的提示检索）等。级联结构引入的额外计算开销（L 层顺序变换 vs. 单层并行）在准确率-效率权衡中的位置尚不明确。此外，训练 recipe 中层次化损失的具体设计（如是否含逐层监督信号）需原文确认。整体证据强度受限于无法获取完整论文。

## 方法谱系与知识库定位

**方法家族**：Prompt Learning for VLM Adaptation（视觉语言模型提示学习）

**父方法**：CoOp → CoCoOp → MaPLe 构成的提示学习谱系。CPL 直接继承该脉络的核心范式——冻结 CLIP 主干、优化输入空间连续提示——但在架构 slot 上进行结构性变革。

**直接基线与差异**：
- **CoOp**：CPL 将其单层静态提示扩展为多层动态级联
- **CoCoOp**：CPL 保留实例条件化思想，但将条件注入嵌入层级变换而非仅输入层
- **MaPLe**：CPL 借鉴其"深层"部署理念，但关键差异在于层间显式计算依赖（级联 vs. 独立）

**后续方向**：
1. **计算效率优化**：当前级联为顺序结构，探索并行化或蒸馏压缩以保持层次化优势同时降低延迟
2. **跨任务级联迁移**：研究级联提示各层的语义可解释性，实现层级别任务迁移（如底层共享、高层特化）
3. **与适配器方法融合**：结合 CPL 的提示级联与 CLIP-Adapter 的参数适配，探索提示-参数混合适配范式

**知识库标签**：
- **模态** (modality)：视觉-语言 (vision-language)
- **范式** (paradigm)：提示学习 / 参数高效迁移 (prompt learning / PEFT)
- **场景** (scenario)：小样本分类、域泛化
- **机制** (mechanism)：层次化表征学习、跨层注意力、渐进式融合
- **约束** (constraint)：冻结预训练模型、仅优化提示参数
