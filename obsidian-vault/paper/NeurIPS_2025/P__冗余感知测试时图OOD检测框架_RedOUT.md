---
title: Redundancy-Aware Test-Time Graph Out-of-Distribution Detection
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 冗余感知测试时图OOD检测框架
- RedOUT
- RedOUT achieves superior graph OOD
acceptance: Poster
code_url: https://github.com/name-is-what/RedOUT
method: RedOUT
modalities:
- graph
paradigm:
- unsupervised test-time training
- unsupervised
---

# Redundancy-Aware Test-Time Graph Out-of-Distribution Detection

[Code](https://github.com/name-is-what/RedOUT)

**Topics**: [[T__OOD_Detection]] | **Method**: [[M__RedOUT]] | **Datasets**: TUDataset and OGB graph OOD, Anomaly, TUDataset

> [!tip] 核心洞察
> RedOUT achieves superior graph OOD detection by minimizing structural entropy at test-time through a novel Redundancy-aware Graph Information Bottleneck (ReGIB) that decouples essential information from irrelevant redundancy.

| 中文题名 | 冗余感知测试时图OOD检测框架 |
| 英文题名 | Redundancy-Aware Test-Time Graph Out-of-Distribution Detection |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2510.14562) · [Code](https://github.com/name-is-what/RedOUT) · [DOI](https://doi.org/10.48550/arxiv.2510.14562) |
| 主要任务 | Graph OOD Detection, Graph Anomaly Detection |
| 主要 baseline | GIB, GOOD-D, GOODAT, Structural Entropy Guided OOD Detection |

> [!abstract] 因为「图表示中的结构冗余导致语义偏移，损害测试时OOD检测性能」，作者在「Graph Information Bottleneck (GIB)」基础上改了「提出ReGIB目标函数与结构熵最小化编码树构造」，在「TUDataset/OGB 10个数据集对」上取得「平均6.7%提升，ClinTox/LIPO上提升17.3%」

- 在10个图OOD检测数据集对上平均AUC提升 **6.7%**，ClinTox/LIPO数据集对相比最优竞争方法提升 **17.3%**
- 测试时仅训练轻量级编码树编码器，预训练GNN完全冻结，无需端到端重训练
- 首次将GIB原理扩展至测试时设置，通过本质视图G*实现信息解耦

## 背景与动机

图神经网络(GNN)部署到实际场景时，常面临训练分布与测试分布不一致的问题——即图分布外(OOD)检测。例如，在分子性质预测中，训练集可能包含有机小分子，而测试时出现大分子药物，模型需要可靠地识别这种分布偏移。现有方法面临一个核心困境：图数据天然包含丰富的结构信息，但冗余的子结构（如无关的环状基团、重复的官能团模式）会在表示中引入噪声，导致OOD样本的语义特征被掩盖。

现有方法从不同角度尝试解决这一问题。**GIB (Graph Information Bottleneck)** 通过信息瓶颈原则优化表示压缩，目标为 $\min -I(f(G); Y) + \beta I(G; f(G))$，在训练阶段平衡预测精度与表示简洁性。**GOOD-D** 采用数据中心的框架，通过数据划分和增强赋予GNN OOD检测能力。**GOODAT** 则通过邻域塑形(neighborhood shaping)调整局部结构以区分分布内外样本。此外，**Structural Entropy Guided OOD Detection** (AAAI 2025) 尝试利用结构熵指导检测。

然而，这些方法存在关键局限：GIB依赖训练时的真实标签Y，无法直接应用于无标签的测试场景；GOOD-D和GOODAT在训练阶段固定模型参数，测试时无法适应新分布的冗余模式；结构熵方法虽利用熵概念，但未将冗余消除与信息瓶颈理论深度耦合。更根本的是，所有现有方法都直接操作原始图G，未能显式分离「本质信息」与「结构冗余」——而正是这种冗余在测试时诱导了不可控的语义偏移。

本文提出RedOUT，首次将GIB扩展至测试时设置，通过结构熵最小化构造编码树提取本质视图G*，以无监督方式消除冗余并优化OOD检测。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/1eb4e042-fd58-42fc-b984-174864db2d35/figures/fig_001.png)
*Figure: (a) A schema comparison. (b) A toy example of distinctive components within test graphs. (c) Scoring distributions before/after structural entropy (abbr. SE) minimization.*



## 核心创新

核心洞察：图结构冗余与本质信息在信息论层面是可解耦的，因为结构熵最小化能构造层次化编码树提取本质视图G*，从而使测试时无监督的冗余消除与GIB理论保证成为可能。

| 维度 | Baseline (GIB) | 本文 (RedOUT) |
|:---|:---|:---|
| 优化目标 | $-I(f(G); Y) + \beta I(G; f(G))$，需真实标签Y | ReGIB: $-I(f(G^*); f(G)) + I(f(G^*); f(G)\|\tilde{Y}) + \beta I(G^*; f(G^*))$，仅用伪标签$\tilde{Y}$ |
| 输入表示 | 原始图G直接输入 | 结构熵最小化编码树G*作为本质视图 |
| 训练范式 | 端到端训练，更新全部参数 | 测试时训练，仅优化轻量级树编码器$f_\Theta$，GNN冻结 |
| 理论保证 | 训练集上的信息瓶颈权衡 | 命题1-2给出的上下界，预测项与冗余项显式分离 |

## 整体框架



RedOUT的整体流程包含五个核心模块，形成"冻结特征提取—冗余消除—对比优化—OOD评分"的闭环：

1. **预训练GNN编码器 $f$**（冻结）：输入原始图 $G$，输出图嵌入 $Z = f(G)$。该模块在测试时完全冻结，提供稳定的初始表示。

2. **编码树构造器**（新增）：输入原始图 $G$ 和高度参数 $k$，通过MERGE（合并节点）和DROP（移除中间节点）操作最小化结构熵 $\mathcal{H}^T(G)$，输出本质视图 $G^*$（即编码树 $T$）。

3. **树编码器 $f_\Theta$**（新增，可训练）：输入编码树 $G^*$，执行树结构引导的消息传递与聚合，输出树嵌入 $Z_T = f_\Theta(G^*)$。

4. **伪标签生成器**：基于冻结GNN的输出 $Z$ 经softmax生成伪标签 $\tilde{Y}$，作为ReGIB中冗余条件的代理。

5. **ReGIB优化器**（新增）：输入 $Z$、$Z_T$、$\tilde{Y}$ 及超参数 $\lambda, \beta$，通过对比损失 $\mathcal{L}_{\text{Cl}}$ 和条件冗余消除损失 $\mathcal{L}_{\text{CRI}}$ 优化 $f_\Theta$，最终输出OOD分数 $S$。

数据流示意：
```
G ──→ [冻结GNN f] ──→ Z ──→ [伪标签生成] ──→ Ỹ
│                                          │
└──→ [编码树构造] ──→ G* ──→ [树编码器 f_Θ] ──→ Z_T
                                          ↓
                              [ReGIB优化: L_Cl + λL_CRI]
                                          ↓
                                    OOD Score S
```

## 核心模块与公式推导

### 模块1: ReGIB目标函数（对应框架图 ReGIB Objectives）

**直觉**: 标准GIB需要真实标签Y，无法用于测试时的无监督场景；同时GIB未显式分离冗余信息。ReGIB通过引入本质视图G*和伪标签Ỹ，将目标重写为可计算的预测项与冗余项之和。

**Baseline 公式 (GIB): 
$$\min \text{GIB} \triangleq -I(f(G); Y) + \beta I(G; f(G))$$
符号: $f$ = GNN编码器, $G$ = 原始图, $Y$ = 真实标签, $\beta$ = 压缩-预测权衡系数, $I(\cdot;\cdot)$ = 互信息

**变化点**: (1) 测试时无真实标签，需用伪标签Ỹ替代Y；(2) 原始图G含冗余，需用本质视图G*替代；(3) 需显式量化并消除冗余项 $I(f(G^*); f(G)|\tilde{Y})$。

**本文公式（推导）**:
$$\text{Step 1}: \min -I(f(G^*); \tilde{Y}) + \beta I(G^*; f(G^*)) \quad \text{将G替换为G*, Y替换为Ỹ}$$
$$\text{Step 2}: I(f(G^*); \tilde{Y}) = I(f(G^*); f(G)) + I(f(G^*); \tilde{Y}|f(G)) - I(f(G^*); f(G)|\tilde{Y}) \quad \text{链式法则分解}$$
$$\text{Step 3}: I(f(G^*); \tilde{Y}) \geq I(f(G^*); f(G)) - I(f(G^*); f(G)|\tilde{Y}) \quad \text{舍去非负项} I(f(G^*); \tilde{Y}|f(G)) \geq 0$$
$$\text{最终}: \min \text{ReGIB} \triangleq -I(f(G^*); f(G)) + I(f(G^*); f(G)|\tilde{Y}) + \beta I(G^*; f(G^*))$$

**对应消融**: Table 3显示移除L_Cl（预测项实例化）或L_CRI（冗余项实例化）均导致性能下降，验证两项的必要性。

### 模块2: 可计算损失实例化（对应框架图 ReGIB Optimizer）

**直觉**: 互信息项直接计算困难，需用对比学习给出紧的上下界，使ReGIB可端到端优化。

**Baseline**: 无直接可计算形式；标准对比损失如InfoNCE仅提供下界估计，未处理条件冗余项。

**变化点**: 命题2将预测项 $-I(f(G^*); f(G))$ 实例化为对比损失 $\mathcal{L}_{\text{Cl}}$；同时设计条件冗余消除损失 $\mathcal{L}_{\text{CRI}}$ 作为 $I(f(G^*); f(G)|\tilde{Y})$ 的上界实例化。

**本文公式（推导）**:
$$\text{Step 1}: I(f(G^*); f(G)) \geq -\mathcal{L}_{\text{Cl}}(G^*, G) + \log(N) \quad \text{命题2: 对比损失为互信息下界}$$
$$\text{Step 2}: \mathcal{L}_{\text{CRI}} \text{ 设计为条件冗余的上界，基于伪标签Ỹ分组计算冗余度}$$
$$\text{最终}: \mathcal{L} = \mathcal{L}_{\text{Cl}} + \lambda \mathcal{L}_{\text{CRI}}$$
符号: $\mathcal{L}_{\text{Cl}}$ = 对比损失（原始图嵌入Z与树嵌入Z_T的对比）, $\mathcal{L}_{\text{CRI}}$ = 条件冗余消除损失, $\lambda$ = 两项权衡系数, $N$ = batch size

**对应消融**: Table 3显示移除L_CRI后冗余无法有效消除，AUC显著下降；调整λ影响最终平衡。

### 模块3: 结构熵最小化编码树构造（对应框架图 Coding Tree Constructor）

**直觉**: 图的冗余源于层次化子结构的重复与噪声；结构熵能度量编码树对图结构的描述效率，最小化熵等价于提取最紧凑的本质表示。

**Baseline**: 无显式冗余消除预处理；标准GNN直接对原始邻接矩阵消息传递。

**变化点**: 引入结构熵理论，通过MERGE/DROP操作构造最优编码树，将图转化为层次化本质视图G*。

**本文公式**:
$$T = \text{arg}\min_T \mathcal{H}^T(G)$$
其中结构熵 $\mathcal{H}^T(G)$ 定义为编码树T对图G的熵编码代价，MERGE操作合并相似节点降低熵，DROP操作剪除冗余中间层。输出G*为编码树的叶节点表示，作为树编码器的输入。

**对应消融**: Table 3显示移除编码树构造（直接用G代替G*）导致本质信息提取能力丧失，性能显著退化。

## 实验与分析



本文在TUDataset和OGB的10个图OOD检测数据集对上进行评估，涵盖分子图（ClinTox, LIPO, BBBP, SIDER等）与社会网络图（IMDB-B, IMDB-M, COLLAB等）。如Table 1所示，RedOUT在所有10个数据集对上均取得最优AUC，相比 runner-up 方法GOOD-D、GOODAT及Structural Entropy Guided OOD Detection实现一致领先。核心数值：平均提升 **6.7%**；在最具挑战性的分子性质预测任务ClinTox/LIPO上，相比最优竞争方法提升 **17.3%**，显示结构冗余消除对分子图（富含重复子结构）尤为关键。



消融实验（Table 3）验证各组件贡献：移除对比损失L_Cl导致预训练模型无法获得新的最优表示；移除条件冗余消除损失L_CRI使冗余信息残留，性能下降；移除编码树构造则完全丧失本质信息提取能力。三项消融均支持ReGIB设计的理论必要性。



此外，RedOUT在图异常检测基准（vs. GKDE, OCGTL）上同样取得最优结果（Table 2/Table 10），验证方法泛化至相关任务。

公平性检验：对比基线涵盖2023-2025年最新方法（GOOD-D WSDM 2023, GOODAT 2024, Structural Entropy Guided AAAI 2025），选择合理。但需注意：(1) 性能依赖预训练GNN质量，文中未分析不同backbone强度的影响；(2) 超参数λ（损失权衡）和k（编码树高度）需逐数据集调优，敏感性分析未充分展示；(3) 局限图分类任务，节点级/边级OOD检测未探索。测试时额外开销主要来自编码树构造与E轮优化，但GNN冻结使总体计算可控（Table 9报告时间消耗）。

## 方法谱系与知识库定位

RedOUT属于**Graph Information Bottleneck (GIB) 谱系**，直接父方法为GIB（Wu et al., 2020）。核心演进路径：GIB → Test-time Graph Transformation [12] → RedOUT。

**改动槽位**: objective（GIB→ReGIB，解耦预测/冗余/压缩三项）、data_curation（新增结构熵最小化编码树）、architecture（新增树编码器f_Θ）、training_recipe（端到端训练→测试时仅优化树编码器）、inference_strategy（新增伪标签引导的OOD评分）。

**直接基线差异**:
- **GIB**: 训练时需要真实标签；RedOUT改为测试时伪标签+本质视图，首次实现GIB的测试时扩展
- **GOOD-D** [20]: 数据中心训练框架，测试时冻结；RedOUT改为测试时自适应，无需重训练
- **GOODAT** [3]: 邻域塑形调整局部结构；RedOUT从全局信息论角度解耦冗余，理论保证更明确
- **Structural Entropy Guided OOD** [10]: 同期结构熵方法但无GIB理论耦合；RedOUT将熵最小化嵌入ReGIB目标，形成端到端优化

**后续方向**: (1) 扩展至节点级/边级OOD检测；(2) 探索编码树构造的可微近似，避免离散MERGE/DROP；(3) 将测试时训练范式迁移至其他模态（图像、文本）的OOD检测。

**标签**: modality: graph / paradigm: test-time training, unsupervised learning / scenario: out-of-distribution detection / mechanism: information bottleneck, structural entropy minimization, contrastive learning / constraint: frozen pre-trained backbone, graph-level only

## 引用网络

### 直接 baseline（本文基于）

- Graph Out-of-Distribution Detection Goes Neighborhood Shaping _(ICML 2024, 直接 baseline, 未深度分析)_: Very recent (2024) graph OOD detection method; likely primary baseline that the 

