---
title: 'SAEMark: Steering Personalized Multilingual LLM Watermarks with Sparse Autoencoders'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 基于稀疏自编码器的黑盒LLM水印框架
- SAEMark
- SAEMARK enables scalable
acceptance: Poster
cited_by: 1
method: SAEMark
modalities:
- Text
paradigm: unsupervised
---

# SAEMark: Steering Personalized Multilingual LLM Watermarks with Sparse Autoencoders

**Topics**: [[T__Text_Generation]] | **Method**: [[M__SAEMark]] | **Datasets**: Computational Overhead, Multi-bit Scaling, Adversarial Robustness

> [!tip] 核心洞察
> SAEMARK enables scalable, quality-preserving, personalized multi-bit watermarking of LLM-generated text using only black-box API access through sparse autoencoder feature-based rejection sampling.

| 中文题名 | 基于稀疏自编码器的黑盒LLM水印框架 |
| 英文题名 | SAEMark: Steering Personalized Multilingual LLM Watermarks with Sparse Autoencoders |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2508.08211) · [DOI](https://doi.org/10.48550/arxiv.2508.08211) |
| 主要任务 | Text Generation, LLM Watermarking |
| 主要 baseline | KGW, UPV, DIP, Waterfall, PersonaMark |

> [!abstract] 因为「现有LLM水印方法需要白盒访问进行logit操纵或输出重写，无法适用于API部署的多语言场景」，作者在「PersonaMark」基础上改了「以稀疏自编码器特征为基础的拒绝采样机制，无需修改模型输出」，在「C4 English / LCSTS Chinese」上取得「99.5% F1 at 1.00× latency，10-bit (1024用户) >90%准确率」。

- **效率突破**: SAEMark F1 99.5% 仅 1.00× 基线延迟，KGW/DIP 需 3.24×/3.29× 延迟
- **多比特扩展**: 10 bits (1,024用户) 保持 >90% 准确率，13 bits (8,192用户) 保持 75%
- **对抗鲁棒性**: 5% 同义词替换攻击下 AUC-ROC 仍达 0.960，上下文感知替换攻击下 0.992

## 背景与动机

当前大型语言模型（LLM）生成的文本已广泛渗透至新闻、学术、代码等领域，追踪其来源并防止滥用成为紧迫需求。LLM水印技术旨在向生成文本中嵌入可检测的信号，以便后续识别AI生成内容并追溯至特定用户或模型。然而，现有方法面临一个根本矛盾：主流方案如 KGW（Kirchenbauer et al.）通过白盒访问操纵logit分布，将词汇表划分为"绿名单"与"红名单"并提升绿名单token概率；DIP 与 REMARK-LLM 同样依赖白盒logit偏置实现鲁棒嵌入；即便是声称黑盒的 UPV 方法，也需对输出进行重写式后处理。这些方案在API-only的部署场景（如OpenAI GPT-4、Anthropic Claude）中完全失效——服务提供商既不暴露logits，也不允许输出修改。

更深层的问题是**个性化与多语言扩展的瓶颈**。PersonaMark 虽尝试实现用户级水印归因，但仍受限于token-level的嵌入容量，难以支持大规模用户区分；Waterfall 等框架在低延迟下牺牲多比特能力。当模型通过多语言API服务全球用户时，传统方法无法跨英语、中文、代码等域保持一致的检测性能，且logit操纵会引入显著的推理延迟（3×以上），阻碍生产部署。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c5f6083a-ee80-4414-a5d8-ac86abe3d4ec/figures/Figure_1.png)
*Figure 1 (pipeline): Figure 1: An overview of SAEMark.*



本文的核心动机由此确立：**能否仅通过黑盒API访问，在不修改模型输出、不增加训练开销的前提下，实现高质量、可扩展、跨语言的个性化多比特水印？** SAEMark 通过稀疏自编码器（SAE）从隐藏状态中提取可解释特征，以特征空间匹配替代token空间操纵，彻底绕开了白盒假设。

## 核心创新

**核心洞察**：LLM隐藏状态蕴含的稀疏语义特征具有跨语言稳定性与个体区分性，因为SAE能够将高维激活分解为可解释的低维稀疏表示，从而使基于特征分布匹配的拒绝采样水印成为可能——无需触碰logits或输出文本。

| 维度 | Baseline (KGW/DIP/UPV) | 本文 SAEMark |
|:---|:---|:---|
| **模型访问** | 白盒（logits/权重）或输出重写 | 纯黑盒API，仅读取隐藏状态 |
| **嵌入机制** | Token-level logit偏置或文本改写 | 特征空间分布匹配，零输出修改 |
| **个性化容量** | 单比特或有限多比特 | 高维SAE特征空间，10-13比特可扩展 |
| **推理延迟** | 3.24×–3.29× 基线（KGW/DIP） | 1.00× 基线（N=50），N=10达98.0% F1 |
| **跨语言性** | 依赖特定语言token分布 | SAE特征跨英语/中文/代码泛化 |

## 整体框架



SAEMark 的推理流程采用**逐句生成+拒绝采样**范式，六大模块协同工作：

1. **LLM API (黑盒)**: 接收prompt，自回归生成候选句子；仅要求API返回隐藏状态激活，不修改采样策略或输出分布。
2. **SAE Encoder**: 将候选句子的隐藏状态 $h_t$ 输入预训练稀疏自编码器，编码为稀疏特征向量 $f_t$；每个非零维度对应可解释的语义概念（如"代码逻辑"、"因果关系"）。
3. **Target Generator**: 由水印密钥与用户ID经哈希函数生成目标特征序列 $\{f_1, ..., f_k\}$，实现多比特个性化嵌入（每比特对应特征空间中的一个区分维度）。
4. **FCS Calculator**: 计算**特征集中度评分（Feature Concentration Score）**，衡量候选句子的SAE特征分布与目标序列的相似度。
5. **CheckAlignment**: 应用**Range Similarity ≥ 0.95** 与 **Overlap Rate ≥ 0.95** 双重阈值过滤，验证特征分布的统计匹配性，平衡生成可行性与区分能力。
6. **Rejection Sampler**: 若候选通过过滤则接受输出；否则拒绝并重新采样（最多N次），选择FCS最优者。

```
Prompt → [LLM API] → Candidate Sentence
                          ↓
                    [SAE Encoder] → Sparse Features f_t
                          ↓
    [Target Generator] ← Key + User ID → Target Sequence
                          ↓
                    [FCS Calculator] → Similarity Score
                          ↓
                    [CheckAlignment] → Pass/Fail (≥0.95)
                          ↓
              Pass: Output  /  Fail: [Rejection Sampler] → Regenerate (≤N)
```

该架构的关键优势在于**解耦**：水印嵌入发生在特征空间，与生成模型的解码过程完全独立，因此可兼容TGI、prefix caching、CUDA kernel等优化后端。

## 核心模块与公式推导

### 模块 1: SAE 特征提取器（对应框架图 模块2）

**直觉**: LLM隐藏状态是高维且纠缠的，需分解为稀疏、可解释、跨语言稳定的语义单元，才能作为可靠的水印载体。

**Baseline**: 传统自编码器仅优化重构误差，特征密集且不可解释，无法区分"内容特征"与"背景噪声"。

**本文公式（推导）**:
$$\text{Step 1: 重构约束} \quad \mathcal{L}_{\text{recon}} = \|h_t - \text{Dec}(f_t)\|^2 \quad \text{保证特征可还原为原始隐藏状态}$$
$$\text{Step 2: 稀疏正则} \quad \mathcal{L}_{\text{sparse}} = \lambda\|f_t\|_1 \quad \text{强制大多数维度为零，仅激活语义相关的少数特征}$$
$$\text{最终}: \mathcal{L}_{\text{SAE}} = \|h_t - \text{Dec}(f_t)\|^2 + \lambda\|f_t\|_1$$

符号: $h_t \in \mathbb{R}^d$ = 时刻 $t$ 的LLM隐藏状态; $f_t \in \mathbb{R}^m$ = SAE编码的稀疏特征 ($m \ll d$ 时过完备); $\lambda$ = 稀疏惩罚系数; $\text{Dec}$ = 线性或非线性解码器。

**对应消融**: Figure 7 显示SAE特征质量直接决定后续FCS判别力；Figure 12（附录）显示移除背景特征掩码后AUC从1.0降至0.85，证明稀疏特征选择对信号纯净度的关键作用。

### 模块 2: 特征集中度评分 FCS（对应框架图 模块4）

**直觉**: 水印检测需在特征空间度量"生成文本的特征分布与目标序列的集中程度"，而非比较单个token。

**Baseline 公式** (KGW): 
$$\delta_{w,t} = \begin{cases} \Delta & \text{if } w \in G \text{ (green list)} \\ 0 & \text{otherwise} \end{cases}$$
KGW在logit空间施加确定性偏置，要求白盒访问且仅支持二值（水印/无水印）检测。

**变化点**: KGW的token-level偏置（1）无法扩展至多用户个性化，（2）破坏输出分布导致质量下降，（3）白盒假设不适用于API。SAEMark转向**分布级特征匹配**。

**本文公式（推导）**:
$$\text{Step 1: 句子级特征聚合} \quad F_{\text{text}} = \text{Pool}(\{f_t\}_{t \in \text{sentence}}) \quad \text{将时序特征聚合为分布表示}$$
$$\text{Step 2: 目标序列对齐} \quad \text{FCS}(F_{\text{text}}, T) = \text{sim}(F_{\text{text}}, \{t_1, ..., t_k\}) \quad \text{度量与目标特征序列的相似度}$$
$$\text{最终}: s_{\text{FCS}} = \text{similarity}\big(\text{SAE}(\text{text}), \text{Hash}(\text{key}, \text{user\_id})\big)$$

符号: $F_{\text{text}}$ = 文本的聚合特征分布; $T = \{t_1, ..., t_k\}$ = 由密钥哈希生成的目标特征序列; $\text{sim}$ = 分布相似度度量（如余弦相似度或KL散度的变体）。

**对应消融**: Figure 7 右图展示经验参数下的最优ROC性能，支持理论分析；Figure 4(a) 显示N=10时FCS筛选达98.0% F1，N=5时为86.8%。

### 模块 3: CheckAlignment 过滤与目标序列生成（对应框架图 模块3+5）

**直觉**: 单纯FCS可能因特征间的统计依赖性而产生假匹配，需分布级验证确保"形状相似"且"范围重叠"。

**Baseline**: 无直接对应；传统方法无显式的生成可行性检验，导致高拒绝率或低区分度。

**本文公式（推导）**:
$$\text{Step 1: 目标生成} \quad \text{target} = \text{Hash}(\text{key}, \text{user\_id}) \rightarrow \{f_1, f_2, ..., f_k\} \quad \text{高维空间编码多比特信息}$$
$$\text{Step 2: 范围相似性} \quad \text{RangeSimilarity} = \frac{|\text{range}(F_{\text{text}}) \cap \text{range}(T)|}{|\text{range}(F_{\text{text}}) \cup \text{range}(T)|} \geq 0.95$$
$$\text{Step 3: 重叠率} \quad \text{OverlapRate} = \frac{\sum_{i} \min(F_{\text{text}}^{(i)}, T^{(i)})}{\sum_{i} F_{\text{text}}^{(i)}} \geq 0.95$$
$$\text{最终}: \text{Accept if } \text{RangeSimilarity} \geq 0.95 \land \text{OverlapRate} \geq 0.95$$

**变化点**: 0.95阈值经理论最坏情况分析推导，平衡（1）生成可行性——不过度约束导致无限拒绝；（2）区分能力——不同用户的目标序列保持统计可分离。

**对应消融**: Figure 7 左图验证95%阈值的最优性；偏离该阈值时，超过1,024用户（10 bits）后准确率显著衰减。

## 实验与分析


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c5f6083a-ee80-4414-a5d8-ac86abe3d4ec/figures/Table_2.png)
*Table 2 (comparison): Table 2: Comparison of Watermarks.*




![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c5f6083a-ee80-4414-a5d8-ac86abe3d4ec/figures/Table_1.png)
*Table 1: Table 1: Dataset Statistics.*



本文在 **C4 English**、**LCSTS Chinese** 及代码数据上评估SAEMark，核心对比基线为 KGW、UPV、DIP、Waterfall。Figure 4 的计算开销分析揭示关键权衡：SAEMark以 **1.00× 基线延迟** 达到 **99.5% F1**，与DIP的99.5% F1持平，但DIP需 **3.29× 延迟**；KGW需 **3.24× 延迟** 且F1仅98.9%。这一效率优势源于SAEMark**不修改logits**，可直接部署于TGI等优化推理后端，而KGW/DIP的logit操纵阻断了这些优化。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c5f6083a-ee80-4414-a5d8-ac86abe3d4ec/figures/Figure_3.png)
*Figure 3 (result): Figure 3: Distribution analysis of ECE / KS statistics and scatter plot of quantiles.*



多比特扩展是另一核心卖点。Figure 6 显示，在固定文本长度约束下，SAEMark于 **10 bits（1,024用户）保持>90%准确率**，**13 bits（8,192用户）保持75%**，而Waterfall基线在同条件下显著衰减（目测约60-70%）。LCSTS中文数据集上跨语言泛化性能得以保持，证明SAE特征的跨语言稳定性。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c5f6083a-ee80-4414-a5d8-ac86abe3d4ec/figures/Table_3.png)
*Table 3 (quantitative): Table 3: Text quality evaluation on BIGS-Bench.*



对抗鲁棒性方面，Figure 5 的ROC曲线显示：无攻击时AUC=1.000；**5%单词删除攻击下AUC=0.949**；**5%同义词替换攻击下AUC=0.960**；**5%上下文感知替换（CTXSynonym）下AUC=0.992**。SAE特征对语义级改写具有固有鲁棒性，因为同义词替换不改变深层语义特征分布，而token-level方法（如KGW）易受此类攻击影响。


![Figure 6, Figure 7](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c5f6083a-ee80-4414-a5d8-ac86abe3d4ec/figures/Figure_6,_Figure_7.png)
*Figure 6, Figure 7 (ablation): Figure 6: SAE watermarks for human interpretable representations; Figure 7: Framework component ablation studies.*



消融研究（Figure 7）进一步验证组件必要性：CheckAlignment的95%阈值对超过1,024用户的扩展至关重要；背景特征掩码的移除导致AUC从1.0骤降至0.85（Figure 12，附录），证明过滤 ubiquitous 特征（标点、基础语法）对信号质量的决定性作用。

**公平性审视**：基线选择涵盖主流白盒（KGW、DIP）与黑盒（UPV、Waterfall）方案，但缺少与 **PersonaMark**、**SemStamp**、**PostMark** 的直接数值对比——后三者分别代表个性化、语义鲁棒、黑盒鲁棒的最先进方法，其缺席限制了"SOTA"声明的完备性。延迟比较部分反映基础设施优势（TGI、CUDA kernels），基线方法未必获得同等优化实现。Table 3（BIGGen-Bench）的文本质量评估数据未在提取材料中呈现，质量-效率权衡的完整图景有待补充。此外，固定长度假设下的多比特实验可能低估真实场景中变长文本的挑战。

## 方法谱系与知识库定位

**方法家族**: LLM水印 → 个性化水印 → 黑盒/特征水印

**父方法**: **PersonaMark** [12] — 共享"用户级水印归因"目标，但PersonaMark依赖token-level方案且需白盒访问；SAEMark彻底替换为SAE特征空间机制，实现黑盒化与容量扩展。

**直接基线差异**:
- **KGW** [3]: 白盒logit绿名单偏置 → SAEMark零logit修改，特征空间匹配
- **DIP/REMARK-LLM** [14]: 白盒鲁棒logit操纵 → SAEMark同F1（99.5%）但3.29×更快
- **UPV** [7]: 黑盒但输出重写 → SAEMark黑盒且不修改输出文本
- **Waterfall** [13]: 低延迟但有限多比特 → SAEMark同低延迟（1.00× vs 1.06×）但10-13比特扩展

**改动槽位**: architecture（新增SAE编码器）、objective（FCS替代logit偏置）、training_recipe（SAE预训练，水印零训练）、inference_recipe（拒绝采样替代贪婪生成）、data_curation（跨语言/跨域评估）

**后续方向**:
1. **动态SAE适配**: 当前依赖预训练SAE质量，探索针对水印任务端到端优化的SAE结构
2. **自适应比特分配**: 根据文本长度与领域动态调整嵌入容量，突破固定长度假设
3. **联邦水印协作**: 多模型API服务商共享SAE特征空间标准，实现跨平台水印互认

**标签**: 模态=text | 范式=推理时特征操控 | 场景=黑盒API水印/多用户归因 | 机制=稀疏自编码器+拒绝采样 | 约束=零模型修改/零训练/跨语言

## 引用网络

### 直接 baseline（本文基于）

- Unbiased Watermark for Large Language Models _(ICLR 2024, 直接 baseline, 未深度分析)_: Foundational LLM watermarking paper (Kirchenbauer et al.); standard baseline tha

