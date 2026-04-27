---
title: 'Convergent Evolution: How Different Language Models Learn Similar Number Representations'
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.20817
aliases:
- 语言模型数字表示的收敛进化分析
- CEHDLM
- 傅里叶频谱峰值（谱收敛）是一种被动的数据统计反映
modalities:
- Text
paradigm: Reinforcement Learning
---

# Convergent Evolution: How Different Language Models Learn Similar Number Representations

[Paper](https://arxiv.org/abs/2604.20817)

**Topics**: [[T__Agent]], [[T__Interpretability]], [[T__Reasoning]]

> [!tip] 核心洞察
> 傅里叶频谱峰值（谱收敛）是一种被动的数据统计反映，几乎所有在自然语言上训练的系统都会获得，因为训练数据本身的 token 频率分布就呈现周期性。真正有意义的是几何收敛——mod-T 类别的线性可分性——它要求模型不仅捕捉到频谱稀疏性，还要将其组织成几何上可分离的表示空间。Transformer 和线性 RNN 能做到这一点，LSTM 不能，差异根源在于架构的归纳偏置而非容量或数据。这一区分揭示了「模型学到了数值结构」与「模型被动反映了数据统计」之间的本质鸿沟。

| 中文题名 | 语言模型数字表示的收敛进化分析 |
| 英文题名 | Convergent Evolution: How Different Language Models Learn Similar Number Representations |
| 会议/期刊 | 2026 arXiv预印本 |
| 链接 | [arXiv](https://arxiv.org/abs/2604.20817) · [Code] · [Project] |
| 主要任务 | 跨架构语言模型数字token表示的谱收敛与几何收敛分析 |
| 主要 baseline | Transformer, Gated DeltaNet, Mamba-2, LSTM, GloVe, FastText |

> [!abstract] 因为「傅里叶频谱峰值是否等同于真正学到数值结构」这一混淆，作者在「Transformer数字表示研究」基础上建立了「谱收敛/几何收敛」双层分析框架，在「FineWeb-Edu ~9.4B tokens, ~300M参数」的跨架构控制实验上取得「Transformer/Gated DeltaNet几何收敛κ达55–96%而LSTM仅0–9%」的结果

- **谱收敛普遍性**：Transformer、Gated DeltaNet、Mamba-2、LSTM、GloVe、FastText均呈现T=2,5,10傅里叶峰值
- **几何收敛分化**：Transformer/Gated DeltaNet mod-T探针Cohen's κ = 55–96%，LSTM κ ≈ 0–9%
- **理论判定**：Theorem 1证明傅里叶稀疏性是几何可分的必要非充分条件，解释LSTM现象

## 背景与动机

当大型语言模型处理数字时，一个反直觉的现象出现了：未经任何算术监督训练，模型自发地在嵌入空间中形成周期性结构。具体而言，数字"3"、"13"、"23"的表示会聚集在一起——它们共享相同的个位数。这种周期性在傅里叶域中表现为T=2、5、10的尖锐频谱峰值，已被Transformer相关研究广泛记录。

现有方法如何处理这一问题？**Power et al. (2022)** 首次在Transformer中观察到数字嵌入的周期性，将其视为模型"理解"数值结构的证据；**Gromov et al.** 进一步将傅里叶特征与算术能力关联，暗示频谱峰值即功能学习的标志；**Nanda et al. (2023)** 通过机制可解释性方法，在小型Transformer中定位了执行模加法的电路。这些研究共同构建了一个隐式假设：傅里叶频谱峰值 = 数值结构学习 = 算术能力基础。

然而，这一等式链存在根本性断裂。**第一，谱-几何混淆**：傅里叶域的稀疏性（谱收敛）与表示空间中mod-T类别的线性可分性（几何收敛）被混为一谈，但二者数学上并不等价——一个信号可以有稀疏频谱却无法组织成几何可分的簇。**第二，架构盲区**：现有证据几乎完全来自Transformer，Mamba、xLSTM、甚至经典词嵌入是否遵循相同规律仍是黑箱。**第三，数据归因困境**：训练数据中数字token的频率分布本身即呈现T=2,5,10周期性（如"0""5"在整点、价格等语境高频出现），频谱峰值可能仅是被动反射而非主动学习。

一个尖锐的例证是LSTM：它同样展现出清晰的傅里叶峰值，却在mod-T线性探针上完全失败（κ≈0%）。这直接挑战了"频谱峰值即理解"的叙事——如果LSTM"学了"却没"学会"，那么Transformer的峰值是否也被过度解读？

本文的核心动机正是拆解这一混淆：建立谱收敛与几何收敛的严格区分框架，系统检验跨架构、跨数据条件下的收敛进化规律，并识别驱动真正几何收敛的关键因素。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/7d8bb396-993e-4b93-8847-2c3f01ab09a4/figures/Figure_1.png)
*Figure 1: Figure 1: Universality of Fourier Features and Convergent Evolution. (Left) Fourierspectrum of number embeddings across three architecture families: Transformer LLMs,non-Transformer LLMs, and classica*



## 核心创新

核心洞察：傅里叶频谱稀疏性是一种被动的数据统计反射，几乎所有在自然语言数据上训练的系统都会获得，因为原始token频率分布本身即具周期性；而几何收敛——mod-T表示的线性可分性——才是模型主动构建可用数值结构的真正标志，因为架构的归纳偏置决定了能否将频谱信息组织成几何上可操作的表示空间，从而使「区分数据反射与真正学习」成为可能。

| 维度 | Baseline（Power et al., Gromov et al.） | 本文 |
|:---|:---|:---|
| 分析层次 | 单一傅里叶频谱分析 | 谱收敛 + 几何收敛双层框架 |
| 架构范围 | 仅Transformer | Transformer、Gated DeltaNet、Mamba-2、LSTM、GloVe、FastText |
| 理论判定 | 频谱峰值暗示功能学习 | Theorem 1：频谱稀疏性是必要非充分条件 |
| 收敛解释 | 架构特定现象 | 收敛进化：共享数据约束驱动的独立趋同 |
| 失败诊断 | 未解释LSTM等反例 | 条件数cond(SW)与Φ_T量化架构差异 |

## 整体框架



本文的分析框架由三个相互衔接的层级构成，从现象描述到理论判定再到因素归因：

**输入层：统一控制条件**。所有模型在~300M参数量、FineWeb-Edu ~9.4B tokens上训练，数字token采用统一分词策略，确保跨架构比较的公平性。

**模块A：谱收敛检测**。对训练后的数字token嵌入矩阵E ∈ ℝ^(V×d)应用离散傅里叶变换（DFT），检测频率ω_k = 2πk/T处的幅度峰值，量化T∈{2,5,10}的频谱稀疏性。该模块输出"是否有傅里叶峰值"的二元判定及峰值幅度。

**模块B：几何收敛检测**。固定嵌入层，训练线性探针（T类逻辑回归）预测数字的mod-T类别。以Cohen's κ量化线性可分性：κ > 50%判定为几何收敛成功。该模块输出"频谱峰值是否转化为可用几何结构"的关键信息。

**模块C：理论-实证归因**。Theorem 1建立谱→几何的蕴含关系判定；消融实验（数据扰动、架构变体、优化器替换）识别几何收敛的充分条件；生物学类比"收敛进化"整合发现。

**输出层：跨架构对比矩阵**。每个(架构, 收敛类型)单元格填入实验结果，形成谱收敛/几何收敛的完整二分表。

数据流示意：
```
[自然语言训练数据] → [模型训练] → [嵌入矩阵E]
                                      ↓
                    ┌─────────────────┼─────────────────┐
                    ↓                 ↓                 ↓
                [DFT谱分析]      [线性探针κ]        [条件数cond(SW)]
                    ↓                 ↓                 ↓
                谱收敛判定        几何收敛判定       架构差异量化
                    └─────────────────┴─────────────────┘
                                      ↓
                          [Theorem 1必要非充分判定]
                                      ↓
                          [收敛进化：数据/架构/优化器/分词器归因]
```

## 核心模块与公式推导

### 模块1: 谱收敛检测（对应框架图 模块A）

**直觉**: 数字token嵌入的周期性应在傅里叶域呈现稀疏峰值，但需与原始数据分布的频谱对比以排除被动反射。

**Baseline公式** (Power et al., 2022): 对嵌入矩阵E的第i行（数字token i的嵌入），一维DFT为
$$\hat{E}[k] = \sum_{n=0}^{N-1} E[n] \cdot e^{-j 2\pi k n / N}$$
符号: $N$ = 数字token种类数（如0-999对应N=1000）, $k$ = 频率索引, $E[n] \in \mathbb{R}^d$ = 数字n的d维嵌入。

**变化点**: 现有研究仅报告$|\hat{E}[k]|$在$k=N/T$处的峰值存在性，未建立与数据分布频谱$|\hat{P}[k]|$（token频率的DFT）的定量比较，无法区分"模型学得的"与"数据固有的"周期性。

**本文公式（推导）**:
$$\text{Step 1}: \quad S_{model}[T] = \frac{1}{d}\sum_{j=1}^{d} |\hat{E}_j[N/T]| \quad \text{模型频谱能量聚合}$$
$$\text{Step 2}: \quad S_{data}[T] = |\hat{P}[N/T]| \quad \text{数据分布频谱能量}$$
$$\text{Step 3}: \quad \text{Spectral Gain}[T] = \frac{S_{model}[T]}{S_{data}[T]} \quad \text{增益量化主动学习成分}$$
$$\text{最终}: \quad \text{谱收敛判定}: \mathbb{1}[\text{Spectral Gain}[T] > \tau_{spec}] \land \text{峰值显著性检验}$$

**对应消融**: Figure 4显示不同数据信号条件下谱收敛始终出现，但几何收敛分化，表明Spectral Gain虽可检测峰值却无法预测功能意义。

### 模块2: 几何收敛检测（对应框架图 模块B）

**直觉**: 真正的数值学习要求表示空间中存在线性可分的mod-T结构，而非仅有频域周期性。

**Baseline公式** (Gromov et al.隐含假设): 若$|\hat{E}[N/T]|$显著，则存在线性分类器$w \in \mathbb{R}^d$使得
$$\Pr_{n \sim \mathcal{U}}[\text{arg}\max_c w_c^\text{top} E[n] = n \mod T] \approx 1$$
该假设将频谱稀疏性直接等同于几何可分性。

**变化点**: LSTM反例（κ≈0%）彻底否定这一等价关系。本文引入显式线性探针，将"假设等价"转化为"实证检验"。

**本文公式（推导）**:
$$\text{Step 1}: \quad \text{冻结嵌入} \ E, \ \text{训练} \ \{w_c\}_{c=0}^{T-1} \subset \mathbb{R}^d \ \text{最小化}$$
$$\mathcal{L}_{probe} = -\mathbb{E}_{n}\left[\log \frac{\exp(w_{n \mod T}^\text{top} E[n])}{\sum_{c'=0}^{T-1}\exp(w_{c'}^\text{top} E[n])}\right]$$
$$\text{Step 2}: \quad \hat{y}(n) = \text{arg}\max_c w_c^\text{top} E[n], \quad \kappa = \frac{p_o - p_e}{1 - p_e} \quad \text{Cohen's kappa}$$
$$\text{最终}: \quad \text{几何收敛判定}: \mathbb{1}[\kappa > \tau_{geo}], \quad \tau_{geo}=0.5$$

符号: $w_c$ = mod-T类别c的线性权重, $p_o$ = 观测一致率, $p_e$ = 期望一致率（偶然）。

**对应消融**: Figure 2直接对比Transformer/Gated DeltaNet（κ=55–96%）与LSTM（κ=0–9%），几何收敛差异达46–96个百分点。

### 模块3: 谱→几何蕴含的理论判定（对应框架图 模块C）

**直觉**: 建立数学桥梁，精确刻画频谱稀疏性何时能/不能保证几何可分性，为LSTM失败提供理论解释。

**Baseline公式**: 无显式理论基础，社区隐含假设"稀疏频谱 ⇒ 循环群表示 ⇒ 线性可分"。

**变化点**: 该假设忽略了从频谱到几何的关键中间步骤——嵌入矩阵的条件结构和周期性子空间的分离度。

**本文公式（Theorem 1推导）**:
$$\text{Step 1}: \quad \text{设} \ \Phi_T \in \mathbb{C}^{N \times r} \ \text{为频率}\ \{N/T, 2N/T, ...\}\text{的傅里叶基子矩阵}$$
$$\text{Step 2}: \quad E = SW^\text{top} + N, \quad S \in \mathbb{R}^{N \times r}, W \in \mathbb{R}^{d \times r} \ \text{低秩近似}$$
$$\text{Step 3}: \quad \text{必要条件}: \ \text{rank}(\Phi_T^* S) = r \ \text{（频谱能量集中于T-周期频率）}$$
$$\text{Step 4}: \quad \text{充分条件失效}: \ \text{cond}(SW) \gg 1 \ \text{或} \ \|\Phi_T^\perp S\|_F \ \text{过大时，几何可分性丧失}$$
$$\text{最终}: \quad \text{Theorem 1}: \ \text{谱收敛} \ \text{nRightarrow} \ \text{几何收敛} \quad \text{（必要非充分）}$$

符号: $\Phi_T^*$ = $\Phi_T$的共轭转置, $cond(SW)$ = 合成矩阵条件数, $\Phi_T^\perp$ = 正交补空间投影。

**对应消融**: LSTM深度消融（12层→4层）显示两者均有谱收敛但均无几何收敛，条件数$cond(SW)$极大，排除容量不足解释，支持Theorem 1的架构归纳偏置归因。



## 实验与分析

| Method | 谱收敛 (T=2,5,10) | 几何收敛 κ (mod-2) | 几何收敛 κ (mod-10) | 条件数 cond(SW) |
|:---|:---|:---|:---|:---|
| Transformer | ✓ | 55–96% | 55–96% | 低 |
| Gated DeltaNet | ✓ | 55–96% | 55–96% | 低 |
| Mamba-2 | ✓ |  |  |  |
| LSTM (12-layer) | ✓ | 0–9% | 0–9% | 极高 |
| LSTM (4-layer) | ✓ | 0–9% | 0–9% | 极高 |
| GloVe | ✓ |  |  | N/A |
| FastText | ✓ |  |  | N/A |


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/7d8bb396-993e-4b93-8847-2c3f01ab09a4/figures/Figure_2.png)
*Figure 2: Figure 2: A “Spiky” Fourier Spectrum Does Not Imply Good Feature Learning. (Left)Token embeddings of Transformer, Gated DeltaNet and LSTM, and even simply the NumberToken Distribution Frequency exhibi*



**核心发现分析**：

**支持核心主张的数据**：Figure 1展示了跨7种架构/模型的T=2,5,10傅里叶峰值，谱收敛具有惊人普遍性——甚至未经神经网络训练的GloVe、FastText也呈现相同模式。这强有力地支持"谱收敛是数据驱动被动反射"的论断。Figure 2的对比最为关键：Transformer与Gated DeltaNet的嵌入在PCA投影中形成清晰的mod-10环状结构，而LSTM的嵌入呈混沌分布，尽管三者频谱形态相似。这一视觉证据与κ数值（55–96% vs. 0–9%）形成互证。

**边际或待确认的数据**：Figure 1右侧将xLSTM-7B标注为"仅谱收敛"，但正文中未提供其探针κ值，该分类的实证基础不完整。Mamba-2的几何收敛状态在摘录中未明确报告，其作为线性RNN代表的地位关键但数据缺失。

**消融实验**：


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/7d8bb396-993e-4b93-8847-2c3f01ab09a4/figures/Figure_4.png)
*Figure 4: Figure 4: Spectral convergence is universal but geometric convergence depends on thedata signal. (Left) Fourier spectra of Transformer embeddings trained under data per-turbations in Table 1. All pert*



- **数据扰动**（Figure 4, Figure 6）：isolate-k配置、上下文窗口长度ℓ∈{2,4,8,64}、swap numbers等操作显示，几何收敛依赖特定数据信号——互补共现（文本-数字共现、跨数字交互）和多token加法问题。单token加法不足以驱动几何收敛。
- **优化器影响**（Figure 5）：Muon与AdamW在9-digit加法中收敛至相同谱结构但几何收敛路径不同，优化器选择影响收敛动态。
- **分词器决定**（Figure 6）：tokenization策略直接决定算术任务的收敛行为，分词器是收敛进化的关键环境约束。
- **LSTM深度消融**：12层与4层LSTM均失败，彻底排除"容量不足"假说，将失败归因于架构本身（循环连接的梯度/记忆结构无法构建分离的周期性子空间）。

**公平性检查**：
- **Baseline强度**：对比覆盖Transformer、线性RNN（Mamba, Gated DeltaNet）、经典RNN（LSTM）、非神经网络嵌入（GloVe, FastText），跨度充分。但缺少现代LSTM变体如xLSTM的完整探针数据。
- **计算/数据成本**：~300M参数、~9.4B tokens为中等规模，结论向更大规模（如7B+）的外推需验证。
- **Failure cases**：LSTM的系统性失败是本文最重要的negative result，但论文未探讨是否存在某些特殊数据分布或训练技巧可使LSTM突破此限制。



## 方法谱系与知识库定位

**方法家族**：神经网络可解释性 → 表示学习理论 → 机制可解释性（Mechanistic Interpretability）

**Parent method**：Power et al. (2022) "Grokking" 及后续Transformer数字表示研究。本文继承其对傅里叶周期性的观察，但颠覆其"频谱峰值即学习"的解释框架。

**Changed slots**：
| Slot | Parent | 本文 |
|:---|:---|:---|
| 分析目标 | 单一架构（Transformer） | 跨架构系统性对比 |
| 评估维度 | 频谱分析（一维） | 谱收敛/几何收敛（二维框架） |
| 理论基础 | 经验观察 | Theorem 1必要非充分性证明 |
| 解释范式 | 架构特定能力 | 收敛进化：数据约束驱动的独立趋同 |
| 失败诊断 | 未处理 | 条件数cond(SW)与Φ_T量化 |

**Direct baselines及差异**：
- **Power et al. (2022)**: 发现Transformer数字周期性，但未区分谱/几何层次；本文证明其观察仅为必要非充分条件。
- **Gromov et al.**: 将傅里叶特征与算术能力关联；本文显示该关联在LSTM中断裂，需几何收敛作为中介。
- **Nanda et al. (2023)**: 机制可解释性定位算术电路；本文提供跨架构的宏观框架，互补其微观机制分析。

**Follow-up directions**：
1. **规模外推**：7B+参数下LSTM变体（xLSTM, mLSTM）是否突破几何收敛障碍？规模与归纳偏置的交互待检验。
2. **功能验证**：几何收敛是否为算术能力（而非仅mod-T分类）的必要条件？需连接至下游任务性能。
3. **扩展模数**：T=2,5,10源于十进制结构，其他进制（如二进制T=2,4,8...）或连续数值的收敛规律待探索。

**知识库标签**：
- **modality**: 语言（数字token）
- **paradigm**: 理论分析 + 受控实证
- **scenario**: 预训练表示分析，无监督/自监督
- **mechanism**: 傅里叶分析、线性探针、低秩矩阵分析
- **constraint**: 跨架构公平比较、数据分布控制、分词器统一

