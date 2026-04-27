---
title: 'Bridging Time and Linguistics: LLMs as Time Series Analyzer through Symbolization and Segmentation'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- LLM时序分析的双路径符号化框架
- S2TS-LLM
- S2TS-LLM enables effective time ser
acceptance: Poster
code_url: https://github.com/JianyangQin/S2TS-LLM
method: S2TS-LLM
modalities:
- time series
- Text
paradigm: supervised
---

# Bridging Time and Linguistics: LLMs as Time Series Analyzer through Symbolization and Segmentation

[Code](https://github.com/JianyangQin/S2TS-LLM)

**Topics**: [[T__Time_Series_Forecasting]], [[T__Classification]] | **Method**: [[M__S2TS-LLM]] | **Datasets**: UEA Time Series, Long-term Forecasting

> [!tip] 核心洞察
> S2TS-LLM enables effective time series analysis by LLMs through spectral symbolization (frequency-domain representation with limited symbols) and contextual segmentation (pattern-based partitioning with reassigned positional encodings), outperforming state-of-the-art methods across time series tasks.

| 中文题名 | LLM时序分析的双路径符号化框架 |
| 英文题名 | Bridging Time and Linguistics: LLMs as Time Series Analyzer through Symbolization and Segmentation |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2505.0) · [Code](https://github.com/JianyangQin/S2TS-LLM) · [Project](待补充) |
| 主要任务 | Time Series Forecasting, Time Series Classification |
| 主要 baseline | Time-LLM, GPT4TS, PatchTST, TimesNet, TimeMixer, S2IP-LLM |

> [!abstract] 因为「时间序列的无限时域变异性难以用自然语言无损表达，且时间序列语义由连续点而非单个token传达」，作者在「Time-LLM」基础上改了「频谱符号化+上下文分段+时间-语言融合的三模块架构」，在「UEA分类基准」上取得「76.4% accuracy，超越S2IP-LLM的73.9%」

- **UEA分类**: S2TS-LLM 76.4% vs. S2IP-LLM 73.9% vs. Time-LLM 73.7%，提升+2.5%
- **ETTh1长期预测**: 完整模型MSE 0.404，去掉符号化模块(w/o Symbol)升至0.418，Δ+0.014
- **ETTh2长期预测**: 完整模型MSE 0.321，去掉分段模块(w/o Segment)升至0.347，Δ+0.026

## 背景与动机

时间序列分析面临一个根本性的表示困境：一段电力负荷曲线包含无限可能的数值组合，而LLM的词汇表是有限的；更重要的是，时间序列的语义往往由一段连续波动共同表达（如"上升趋势"需要多个点才能确定），而自然语言中每个token独立承载语义。现有方法试图将LLM repurposing用于时序任务，但都在这两个维度上存在结构性失配。

Time-LLM [9] 采用时域patching策略，将时间序列切分为 patches 后对齐到文本原型，但仍在时域操作，无法避免无限变异性的信息损失。GPT4TS [22] 统一tokenization处理多种时序任务，使用标准顺序位置编码，忽视了时间序列的连续语义结构。S2IP-LLM [7] 引入语义空间prompt学习，但未解决时域到文本的根本表示鸿沟。

这些方法的共同短板在于：第一，**表示对齐失败**——时域序列的复杂变化难以用有限词汇表达（Figure 1）；第二，**上下文对齐失败**——标准LLM为每个token顺序分配位置以捕捉上下文，但时间序列的语义由连续点群而非单点承载（Figure 2）。本文提出 S2TS-LLM，通过频谱符号化将无限时域变异压缩为有限频率符号，并通过上下文分段重构位置编码以匹配连续语义结构。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a9a5a14b-19bb-4aa2-b1a2-6ff747b7a810/figures/Figure_1.png)
*Figure 1: Figure 1: Illustration of the challenge in representation align-ment: time-domain sequences exhibit complex variationsthat are hard to express concisely, whereas frequency-domainsignals are typically*



## 核心创新

核心洞察：频率域信号天然稀疏，仅需少量显著分量即可重构关键特征，因此可以用有限ASCII符号词汇表抽象时序；同时时间序列的语义单元是模式块而非单点，因此应按模式结构而非原始顺序分配位置编码，从而使LLM的注意力机制能够正确理解时序上下文成为可能。

| 维度 | Baseline (Time-LLM/GPT4TS) | 本文 S2TS-LLM |
|:---|:---|:---|
| 数据管道 | 时域tokenization（digit序列、patch、或自定义词嵌入） | 频谱符号化：FFT→top-k幅度分量→ASCII符号映射 |
| 位置编码 | 标准顺序位置编码，逐token/patch顺序分配 | 上下文分段：按模式分块，按段结构重新分配位置编码 |
| 表示融合 | 简单求和或直接拼接时序与文本嵌入 | 交叉注意力+FFN的时间-语言融合机制 |

三个模块形成递进关系：符号化解决"用什么表示"，分段解决"如何组织顺序"，融合解决"怎样联合理解"。

## 整体框架


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a9a5a14b-19bb-4aa2-b1a2-6ff747b7a810/figures/Figure_3.png)
*Figure 3: Figure 3: The framework of S2TS-LLM. After processing the input time series through1⃝spec-tral symbolization and2⃝contextual segmentation, we obtain time-aware textual abstractions andremapped positio*



S2TS-LLM 采用三阶段流水线处理原始时间序列，最终输入预训练LLM（GPT-2 backbone）进行预测：

**阶段① 频谱符号化（Spectral Symbolization）**：输入原始时间序列 $X \in \mathbb{R}^{T \times D}$，经FFT变换到频域，提取top-k显著幅度分量，映射为有限ASCII符号词汇表中的离散token。输出为符号序列 $\mathcal{S} = \{s_i\}_{i=1}^{K}$，作为频率特征的文本化抽象。

**阶段② 上下文分段（Contextual Segmentation）**：对同一时间序列进行模式检测，划分为M个基于时序模式的块 $C = \{c_j\}_{j=1}^{M}$，并根据每个块的模式类型和块内相对位置重新分配位置编码。输出为带结构感知的分段表示。

**阶段③ 时间-语言融合（Time-Linguistic Fusion）**：将符号化表示 $H_{sym}$ 与分段表示 $H_{seg}$ 通过交叉注意力机制和前馈网络进行深度融合，生成联合表示 $H_{fuse}$，输入预训练LLM完成下游任务（预测或分类）。

```
Raw Time Series X
    ├──→ [FFT] → Top-k Amp/Phase → ASCII Symbols → H_sym (符号化路径)
    └──→ [Pattern Detection] → Block Segmentation → Reassigned PE → H_seg (分段路径)
                                    ↓
                         CrossAttn(H_sym, H_seg) + FFN([H_sym; H_seg])
                                    ↓
                              H_fuse → GPT-2 → Prediction
```

## 核心模块与公式推导

### 模块 1: 频谱符号化（对应框架图 阶段①）

**直觉**: 时域信号的无限数值组合难以用有限词汇表达，但频域表示通常稀疏——少数显著频率分量即可捕捉主要波动模式。

**Baseline 公式** (Time-LLM 时域 patching):
$$X_{patch} = \text{Reshape}(X) \in \mathbb{R}^{N \times P}, \quad H_{patch} = \text{Linear}(X_{patch})$$
符号: $N$=patch数量, $P$=patch长度。直接在时域切分，未解决数值无限性问题。

**变化点**: 时域 patching 仍面临无限数值空间；本文改为频域稀疏表示+离散符号映射。

**本文公式（推导）**:
$$\text{Step 1}: X_{freq} = \text{FFT}(X) \in \mathbb{C}^{T \times D} \quad \text{通过FFT将时序转换到频域}$$
$$\text{Step 2}: \{(f_k, A_k, \phi_k)\}_{k=1}^{K} = \text{TopK}(|X_{freq}|, K) \quad \text{按幅度选取前K个显著分量}$$
$$\text{Step 3}: \mathcal{S} = \{s_k\}_{k=1}^{K}, \quad s_k = \text{Symbolize}(f_k, A_k) \in \Sigma_{ASCII} \quad \text{映射到ASCII符号词汇表}$$
$$\text{最终}: H_{sym} = \text{LLM}_{embed}(\mathcal{S})$$
符号: $f_k$=频率, $A_k$=幅度, $\phi_k$=相位, $\Sigma_{ASCII}$=有限ASCII符号集（如26字母+数字）。

**对应消融**: Table 16 显示去掉符号化(w/o Symbol)后 ETTh1 MSE 从 0.404 升至 0.418，Δ+0.014。

---

### 模块 2: 上下文分段（对应框架图 阶段②）

**直觉**: 标准位置编码假设每个token独立等距排列，但时间序列中"上升趋势"需要连续多点共同表达，应按语义块重组位置信息。

**Baseline 公式** (标准Transformer位置编码):
$$\text{PE}_{std}(pos) = \sin(pos/10000^{2i/d})$$
符号: $pos$=序列中的绝对位置, $i$=维度索引, $d$=模型维度。严格按原始顺序编码。

**变化点**: 标准PE假设token语义独立且顺序固定；时间序列中连续点群构成语义单元，需按检测到的模式块重新分配位置编码。

**本文公式（推导）**:
$$\text{Step 1}: C = \{c_j\}_{j=1}^{M} = \text{Segment}(X) \quad \text{基于时序模式检测划分块（如变点检测）}$$
$$\text{Step 2}: \text{PatternType}(c_j) = \text{ClassifyPattern}(c_j) \in \{\text{trend}, \text{seasonal}, \text{noise}, ...\} \quad \text{识别每块模式类型}$$
$$\text{Step 3}: \text{PE}_{new}(c_j) = g(\text{PatternType}(c_j), \text{RelativePos}(c_j)) \quad \text{按模式类型和块内相对位置编码}$$
$$\text{最终}: H_{seg} = \text{LLM}_{embed}(C, \text{PE}_{new})$$
符号: $M$=分段数量, $g(\cdot)$=模式感知的位置编码函数, RelativePos=块内相对偏移。

**对应消融**: Table 16 显示去掉分段(w/o Segment)后 ETTh1 MSE 从 0.404 升至 0.416，Δ+0.012；ETTh2 MSE 从 0.321 升至 0.347，Δ+0.026。

---

### 模块 3: 时间-语言融合（对应框架图 阶段③）

**直觉**: 符号化提供"是什么频率成分"的抽象语义，分段提供"何时出现何种模式"的结构信息，二者需深度交互而非简单叠加。

**Baseline 公式** (Time-LLM/GPT4TS 简单融合):
$$H_{fuse}^{base} = H_{time} + H_{text} \quad \text{或} \quad H_{fuse}^{base} = [H_{time}; H_{text}]$$
简单求和或拼接，假设两种表示空间已对齐。

**变化点**: 时序与自然语言的表示空间存在结构性差异，简单叠加无法建立跨模态关联；需显式交叉注意力实现动态关联。

**本文公式（推导）**:
$$\text{Step 1}: H_{sym} = \text{LLM}_{embed}(\mathcal{S}), \quad H_{seg} = \text{LLM}_{embed}(C, \text{PE}_{new}) \quad \text{分别嵌入两条路径}$$
$$\text{Step 2}: H_{cross} = \text{CrossAttn}(H_{sym}, H_{seg}) = \text{softmax}\left(\frac{H_{sym} W_Q (H_{seg} W_K)^T}{\sqrt{d_k}}\right) H_{seg} W_V \quad \text{符号查询分段信息}$$
$$\text{Step 3}: H_{concat} = [H_{sym}; H_{seg}] \quad \text{保留原始表示}$$
$$\text{最终}: H_{fuse} = H_{cross} + \text{FFN}(H_{concat}) \quad \text{交叉注意力+前馈融合}$$

**对应消融**: Table 16 显示去掉融合(w/o Fusion，改为简单求和)后 ETTh1 MSE 从 0.404 升至 0.413，Δ+0.009。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a9a5a14b-19bb-4aa2-b1a2-6ff747b7a810/figures/Table_1.png)
*Table 1: Table 1: Long-term forecasting results. All the results are averaged from four different predictionhorizons {96, 192, 336, 720}. Red denotes the best performance, while Blue indicates the second-best.*



S2TS-LLM 在多个基准上进行了全面评估。长期预测结果（Table 1）显示，在ETTh1数据集上平均四个预测步长{96, 192, 336, 720}，完整模型取得MSE 0.404。短期预测（Table 2）在M4数据集上加权平均SMAPE、MASE、OWA指标。分类任务（Figure 6）在UEA十数据集上平均，S2TS-LLM达到76.4% accuracy，超越S2IP-LLM的73.9%（+2.5%）和Time-LLM的73.7%（+2.4%），在19个对比方法中取得最优。Figure 4展示了整体模型对比，涵盖预测与分类多任务。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a9a5a14b-19bb-4aa2-b1a2-6ff747b7a810/figures/Table_3.png)
*Table 3: Table 3: Few-shot learning results on 5% and 10% training data. Detailed results are provided inAppendix E.3.*



消融实验（Table 16 / Appendix E.6）验证了各组件的必要性。去掉频谱符号化(w/o Symbol)导致ETTh1 MSE上升0.014（0.404→0.418），ETTh2 MSE上升0.024（0.321→0.345），说明频率域抽象对捕捉时序本质模式至关重要。去掉上下文分段(w/o Segment)在ETTh2上损失更大（0.321→0.347, +0.026），表明模式感知的位置编码对长程结构理解尤为关键。去掉融合机制(w/o Fusion)改用简单求和，ETTh1 MSE上升0.009（0.404→0.413），验证了交叉注意力的动态关联价值。

扩展性消融显示：将ASCII符号替换为可学习词汇表(w Vocab)在ETTh1/ETTh2上取得可比甚至略优结果（ETTh1: 0.403 vs 0.404），说明符号化范式本身具有灵活性；将GPT-2 backbone换为BERT(w BERT)在ETTh2上持平(0.321)、ETTh1略降(+0.003)，表明方法对LLM架构有一定鲁棒性。但去掉预训练(w/o Pretrain)导致ETTh1 MSE大幅上升至0.434（+0.030），确认预训练知识的必要性。

公平性方面，对比基线涵盖了LLM-based方法（Time-LLM, GPT4TS, S2IP-LLM）和专用时序模型（PatchTST, TimesNet, TimeMixer），选择较为全面。但2025年同期的CALF和Freeformer未纳入对比；计算成本与专用轻量模型（如RLinear）的比较未充分展开，这是作者明确承认的局限。

## 方法谱系与知识库定位

S2TS-LLM 属于 **LLM-for-Time-Series** 方法谱系，直接继承自 Time-LLM [9] 的"repurposing预训练LLM用于时序分析"范式，但在三个关键slot上进行了结构性改造：

- **Time-LLM** [9]: 时域patching + 标准位置编码 + 简单原型对齐 → S2TS-LLM 替换为频域符号化 + 模式感知分段 + 交叉注意力融合
- **GPT4TS** [22]: 统一tokenization但仍在时域操作，标准顺序PE → S2TS-LLM 引入频域稀疏表示和结构重编码
- **S2IP-LLM** [7]: 语义空间prompt学习，未解决表示鸿沟 → S2TS-LLM 通过双路径设计显式桥接时序-语言差距

后续方向：
1. **动态符号词汇**: 当前top-k频率分量数量K为超参，可探索自适应稀疏度选择；
2. **多模态扩展**: 将频谱符号化推广至时空数据（如视频、音频），验证跨模态通用性；
3. **计算效率优化**: 与专用轻量模型（RLinear, DLinear）进行系统性的效率-精度权衡分析。

**标签**: modality=[time_series, text], paradigm=[supervised_finetuning, LLM_repurposing], scenario=[forecasting, classification, few_shot, zero_shot], mechanism=[FFT_symbolization, pattern_segmentation, cross_modal_fusion], constraint=[pretrain_dependency, hyperparameter_sensitivity]

