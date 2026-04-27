---
title: 'MedConclusion: A Benchmark for Biomedical Conclusion Generation from Structured Abstracts'
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.06505
aliases:
- MedConclusion：生物医学结论生成基准与评估框架
- MedConclusion
code_url: https://github.com/Harvard-AI-and-Robotics-Lab/MedConclusion
method: MedConclusion
modalities:
- Text
---

# MedConclusion: A Benchmark for Biomedical Conclusion Generation from Structured Abstracts

[Paper](https://arxiv.org/abs/2604.06505) | [Code](https://github.com/Harvard-AI-and-Robotics-Lab/MedConclusion)

**Topics**: [[T__Text_Generation]], [[T__Benchmark_-_Evaluation]], [[T__Reasoning]] | **Method**: [[M__MedConclusion]] | **Datasets**: MedConclusion

| 中文题名 | MedConclusion：生物医学结论生成基准与评估框架 |
| 英文题名 | MedConclusion: A Benchmark for Biomedical Conclusion Generation from Structured Abstracts |
| 会议/期刊 | arXiv (Cornell University) (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.06505) · [Code](https://github.com/Harvard-AI-and-Robotics-Lab/MedConclusion) · [Project](https://github.com/Harvard-AI-and-Robotics-Lab/MedConclusion) |
| 主要任务 | biomedical conclusion generation（生物医学结构化摘要结论生成） |
| 主要 baseline | PubMed 200k RCT (Shieh et al., 2019); Evaluating unsupervised argument aligners via generation of conclusions of structured scientific abstracts; GPT-5.4, Gemini 3.1 Pro/3 Flash, DeepSeek-V3.2, Gemma-3-27B/2-9B, GLM-4.6V |

> [!abstract]
> 因为「生物医学结论生成缺乏大规模专门基准，现有数据集规模小、领域窄且将结论生成混同于摘要任务」，作者在「PubMed 200k RCT」基础上改了「构建569万条记录的PubMed结构化摘要语料库，引入五维LLM-as-a-judge评估与四模式话语功能对照实验」，在「MedConclusion 30K子集」上取得「结论生成与摘要生成的语义相似度差异达-8.23点，格式约束提升数值一致性+3.12点」

- **数据规模**: 5,692,839条结构化摘要记录，3,772种期刊，141个学科类别
- **核心发现**: Mode A（结论生成）vs Mode D（摘要生成）语义相似度 73.22 vs 64.99（GPT-5.4评判）
- **评估创新**: 五维LLM评判（语义/风格/非矛盾/数值/正式度）+ 双评判器稳健性验证

## 背景与动机

生物医学论文的结构化摘要通常包含 Background、Methods、Results、Conclusion 等明确标注的段落。其中 Conclusion 段承担独特的**推断性功能**——它并非简单压缩前文，而是基于证据提出解释、临床意义与未来方向。然而，当前研究面临三重困境：

**第一，数据瓶颈。** 现有生物医学摘要数据集规模受限：PubMed 200k RCT（Shieh et al., 2019）仅覆盖20万篇随机对照试验，用于句子分类而非生成；Tang et al. (2022) 的超声心动图笔记、Gao et al. (2024) 与 Bastan et al. (2022) 的结论重建数据集均不足20万样本，且缺乏期刊元数据与学科分层信息。

**第二，任务混淆。** 此前工作将结论生成视为标准摘要（summarization）的附属任务或辅助训练目标。例如，"Evaluating unsupervised argument aligners via generation of conclusions of structured scientific abstracts" 虽直接研究结论生成，但未系统区分结论的话语功能与摘要的信息压缩功能，导致模型行为难以解释。

**第三，评估粗糙。** 传统 ROUGE、BLEU 等词汇重叠指标无法捕捉科学写作的关键质量维度——事实非矛盾性、数值一致性、正式程度等。单一维度评估掩盖了生成结论在临床安全与科学准确性上的风险。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/74258b67-e76a-4281-b8b8-10de617237c5/figures/Figure_1.png)
*Figure 1: Figure 1: Overview of MedConclusion and the evaluation pipeline. Left: an exampleMedConclusion instance, including article metadata, subject categories, and a structuredabstract, where the non-conclus*



本文的核心动机由此确立：构建一个**百万级、跨学科、带期刊质量元数据**的专门基准，通过**多维度评估协议**和**话语功能对照实验**，将生物医学结论生成为独立研究任务，系统检验大语言模型的证据→结论推理能力。

## 核心创新

核心洞察：**结论生成在行为层面与摘要写作存在系统性差异**，因为结构化摘要中的 Conclusion 段具有独特的推断性话语功能（discourse function）——它要求模型从 Methods/Results 的证据中推导临床意义与机制解释，而非仅压缩信息——从而使「通过受控提示消融量化话语功能效应、并设计匹配其多维质量需求的混合评估协议」成为可能。

| 维度 | Baseline（传统方法） | 本文（MedConclusion） |
|:---|:---|:---|
| **数据规模与覆盖** | <20万条，窄领域（RCT/专科），无期刊元数据 | **569万条**，全生物医学领域，**SJR期刊质量评分+141学科分层** |
| **任务定义** | 结论生成 = 标准摘要或辅助训练目标 | **四模式对照**：结论/摘要 × 有/无格式约束，显式分离话语功能 |
| **评估指标** | ROUGE/BLEU 单一词汇重叠 | **五维LLM评判**（语义/风格/非矛盾/数值/正式度）+ 传统指标 + 轻量诊断 |
| **评估可靠性** | 单指标或单维度 | **双评判器稳健性协议**（GPT-5.4-mini + Gemini 3 Flash交叉验证） |

## 整体框架



MedConclusion 的整体框架分为**数据构建管线**与**评估引擎**两大阶段，共六个核心模块：

**阶段一：数据构建**
1. **PubMed EDirect 查询**：输入为 PubMed 数据库，施加 `hasstructuredabstract` 约束，限定时间范围 2000–2025；输出为候选文献 UID 列表。
2. **XML 解析与去重**：将 PubMed XML 解析为 JSONL，提取元数据、关键词及（段落标签, NLM类别, 文本）三元组；基于 PMID、DOI、归一化标题进行多键去重。
3. **规则清洗与结论识别**（新增）：输入去重记录，通过** curated conclusion label variant matcher** 匹配归一化的结论段标签变体（附录G）；输出满足 `≥3个摘要段落 ∧ ≥1个结论段落 ∧ 英文 ∧ 有效书目字段` 的清洗记录。
4. **SJR 元数据增强**（新增）：将清洗后记录的期刊名称与 SCImago Journal & Country Rank 数据库匹配，附加年度 SJR 评分与学科分类；输出带期刊质量评分的 5,692,839 条记录语料库。

**阶段二：评估引擎**
5. **四模式生成评估**（新增）：输入为移除结论后的结构化摘要 `x` 与提示模板（A/B/C/D）；输出为生成文本 `ŷ` 及长度诊断指标。模式 A 要求"正式学术结论"，模式 B 要求"正式学术摘要"，模式 C/D 在 A/B 基础上增加显式的句数/词数约束。
6. **混合评分引擎**（新增）：输入为 `(y⋆, ŷ)` 对；输出五维LLM评判分数、传统参考指标（ROUGE-1/2/L, BLEU）、轻量诊断（词数比、句数比、嵌入余弦相似度、困惑度）。

```
PubMed (2000-2025) → EDirect查询 → XML解析 → 多键去重 
    → [规则清洗+结论识别] → [SJR增强] → 5.7M语料库
    → 四模式提示(A/B/C/D) → LLM生成 → 混合评分引擎
        ├── GPT-5.4-mini 五维评判
        ├── Gemini 3 Flash 二次验证
        └── ROUGE/BLEU/诊断指标
```

## 核心模块与公式推导

### 模块 1: 记录过滤与数据清洗（对应框架图阶段一模块3）

**直觉**: 结构化摘要的结论生成需要确保输入包含完整的证据段落和明确的结论标签，同时排除非英语和低质量书目记录。

**Baseline 公式**（传统PubMed数据集过滤）: 无显式结论段要求，通常仅按语言或文献类型过滤。

**本文公式（推导）**:
$$\text{Step 1: } \text{LabelMatch}(r) = \mathbb{1}\left[\text{Normalize}(\text{section\_label}(r)) \in \mathcal{V}_{\text{conclusion}}\right] \quad \text{（curated变体标签集合匹配）}$$
$$\text{Step 2: } \text{Retain } r \iff |\text{segments}(r)| \geq 3 \land |\text{conclusion}(r)| \geq 1 \land \text{lang}(r) = \text{en} \land \text{valid\_bib}(r)$$

符号: $r$ = 单条记录, $\mathcal{V}_{\text{conclusion}}$ = 人工整理的结论标签变体集合（如 "Conclusion", "Conclusions", "Summary and Conclusion" 等，见Appendix G）, $\text{valid\_bib}(r)$ = 核心书目字段有效性布尔函数。

**对应消融**: 无直接消融，但过滤后语料库从原始查询结果缩减至 5,692,839 条（保留率。

---

### 模块 2: 五维LLM-as-a-Judge 评分函数（对应框架图阶段二模块6）

**直觉**: 科学结论的质量是多维的——词汇重叠无法检测事实矛盾或数值错误，而人类审稿人关注语义忠实度、写作规范、逻辑一致性等。LLM 可模拟这一多维评判过程。

**Baseline 公式**（传统参考指标）:
$$\text{ROUGE-}n = \frac{\sum_{S \in \{\text{Reference}\}} \sum_{\text{gram}_n \in S} \text{Count}_{\text{match}}(\text{gram}_n)}{\sum_{S \in \{\text{Reference}\}} \sum_{\text{gram}_n \in S} \text{Count}(\text{gram}_n)}$$

符号: $\text{gram}_n$ = n-gram, $\text{Count}_{\text{match}}$ = 匹配次数。

**变化点**: ROUGE 仅衡量词汇召回，无法评估（1）生成结论是否引入原文不存在的新声明（非矛盾性）、（2）数值是否与原文一致（数值一致性）、（3）学术正式程度是否匹配（正式度相似度）。

**本文公式（推导）**:
$$\text{Step 1: } \mathbf{s} = J(y^\star, \hat{y}) \rightarrow [s_{\text{semantic}}, s_{\text{style}}, s_{\text{non-contradiction}}, s_{\text{numeric}}, s_{\text{formality}}] \in [0, 100]^5$$
$$\text{Step 2: } \text{DualJudge}(y^\star, \hat{y}) = \left(J_{\text{GPT-5.4-mini}}(y^\star, \hat{y}),\; J_{\text{Gemini-3-Flash}}(y^\star, \hat{y})\right) \quad \text{（双评判器稳健性验证）}$$
$$\text{最终: } \text{Score}(y^\star, \hat{y}) = \left(\mathbf{s}_{\text{judge}},\; \text{ROUGE-1/2/L},\; \text{BLEU},\; \text{CosSim}_{\text{emb}},\; \text{PPL},\; \frac{|\hat{y}|_{\text{words}}}{|y^\star|_{\text{words}}},\; \frac{|\hat{y}|_{\text{sents}}}{|y^\star|_{\text{sents}}}\right)$$

符号: $y^\star$ = 参考结论, $\hat{y}$ = 生成结论, $J$ = 评判模型, $s_{\text{semantic}}$ = 语义内容匹配度, $s_{\text{style}}$ = 写作风格相似度, $s_{\text{non-contradiction}}$ = 事实非矛盾性, $s_{\text{numeric}}$ = 数值一致性, $s_{\text{formality}}$ = 正式程度相似度。

**对应消融**: Table 5 显示 GPT-5.4-mini 与 Gemini 3 Flash 作为评判器的评分存在系统性偏移（具体数值。

---

### 模块 3: 话语功能效应度量与四模式对照（对应框架图阶段二模块5）

**直觉**: 若结论生成确为独立任务，则改变提示中的目标类型（结论 vs. 摘要）应导致可量化的行为差异，而非仅格式变化。

**Baseline 公式**（单提示评估）:
$$\hat{y} = f_\theta(P_{\text{conclusion}}(x))$$

**变化点**: 单一提示无法分离「任务类型」与「格式约束」的效应。本文引入 2×2 因子设计：目标类型（结论/摘要）× 约束条件（有/无显式长度/风格控制）。

**本文公式（推导）**:
$$\text{Step 1: } \mathcal{P} = \{P_A, P_B, P_C, P_D\} \text{ where}$$
$$P_A: \text{"write a formal academic conclusion"},\quad P_B: \text{"write a formal academic summary"}$$
$$P_C: P_A + \text{ explicit sentence/word-count targets},\quad P_D: P_B + \text{ same targets}$$
$$\text{Step 2: } \Delta_{\text{semantic}}^{(A-B)} = s_{\text{semantic}}^{(A)} - s_{\text{semantic}}^{(B)} \quad \text{（核心话语功能效应度量）}$$
$$\text{最终: } \Delta_{\text{format}}^{(A-C)} = s_{\text{numeric}}^{(C)} - s_{\text{numeric}}^{(A)} \quad \text{（格式约束效应度量）}$$

**对应消融**: Table 4 显示关键结果：
- $\Delta_{\text{semantic}}^{(A-D)} = 73.22 - 64.99 = -8.23$（GPT-5.4，语义相似度下降）
- $\Delta_{\text{style}}^{(A-D)} = 71.20 - 60.13 = -11.07$（写作风格相似度下降）
- $\Delta_{\text{numeric}}^{(A-C)} = 91.36 - 88.24 = +3.12$（GPT-5.4，数值一致性提升）
- $\Delta_{\text{numeric}}^{(A-C)} = 91.82 - 86.45 = +5.37$（Gemini 3 Flash，数值一致性提升）

## 实验与分析

**主实验结果（30K随机子集，Mode A）**

| Method | Semantic Similarity (GPT-5.4-mini) | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU |
|:---|:---|:---|:---|:---|:---|
| GPT-5.4 | **73.22** | 0.34 |  |  |  |
| Gemini 3 Flash | 71.33 | — | — | — | — |
| Gemini 3.1 Pro | ~72（"within a few points"） | — | — | — | — |
| DeepSeek-V3.2 | ~72（"within a few points"） | **0.35** | **最优** | **最优** | **并列最优** |
| Gemma-3-27B |  | — | — | — | — |
| Gemma-2-9B |  | — | — | — | — |
| GLM-4.6V |  | — | — | — | — |


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/74258b67-e76a-4281-b8b8-10de617237c5/figures/Figure_2.png)
*Figure 2: Figure 2 plots journal-level SJR scores against each evaluation metric for GPT-5.4 undersetting A . Most reference-based and judge-based metrics show small but statistically signifi-cant positive asso*



**核心发现分析**：

1. **分数压缩现象（Score Compression）**：GPT-5.4、Gemini 3.1 Pro、DeepSeek-V3.2 在五维评判指标上差距仅"几分"（few points），Table 2 显示强模型聚类。这一发现支持论文核心主张——当前自动指标对强模型的区分度不足，任务可能过易或存在评估天花板效应。

2. **指标不一致性**：DeepSeek-V3.2 在 ROUGE-1/2/L 上最优（0.35），但 LLM-as-a-judge 语义分数并非最高；GPT-5.4 评判分数领先但 ROUGE 略低。这暴露传统词汇指标与语义质量评估的错位，验证混合评估的必要性。

3. **话语功能效应**（Table 4，
![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/74258b67-e76a-4281-b8b8-10de617237c5/figures/Figure_3.png)
*Figure 3: Figure 3: Radar chart comparison of the top 5 and bottom 5 biomedical categories rankedby both mean Semantic Similarity and mean ROUGE-L for GPT-5.4 under setting A . Eachaxis is normalized to [0, 1].*

）：
   - Mode A（结论）→ Mode D（摘要+约束）：语义相似度 -8.23，写作风格 -11.07，数值一致性 -14 至 -28 点
   - 该消融直接证明结论生成与摘要写作是行为不同的任务，非简单标签替换

4. **格式约束效果**（Table 4）：
   - Mode A → Mode C（增加句数/词数约束）：GPT-5.4 数值一致性 +3.12（88.24→91.36），Gemini 3 Flash +5.37（86.45→91.82）
   - 但语义/风格相似度轻微下降，说明约束带来可预测的 trade-off

**公平性检查**：
- **基线强度**：未覆盖 GPT-4o/o3、Claude 3.5/4、Llama 3.3/4、Med-PaLM/Meditron 等专用生物医学LLM，基线非最强可用
- **计算/数据成本**：30K子集评估因成本约束，完整569万数据集未使用；无训练开销（零样本提示）
- **失败案例/局限**：无人工或专家临床验证；困惑度使用 GPT-2 计算可能过时；LLM评判器固有偏见未完全消除

## 方法谱系与知识库定位

**方法谱系**：MedConclusion ← **PubMed 200k RCT (Shieh et al., 2019)**

谱系父节点 PubMed 200k RCT 确立了「PubMed 结构化摘要 + 段落标签」的数据范式，但仅限20万篇RCT的句子分类任务。MedConclusion 完成三重跃迁：从**句子分类数据集**到**生成任务基准**，从**单一领域**到**全生物医学覆盖**，从**数据资源**到**完整研究基础设施**（数据+协议+评估工具）。

**直接基线对比**：
- **PubMed 200k RCT (Shieh et al., 2019)**：提供结构化摘要范式，但规模小29倍、无结论生成任务、无期刊元数据
- **Evaluating unsupervised argument aligners...**：最接近的前期结论生成工作，但规模小、无生物医学特化、无话语功能对照
- **Gao et al., 2024; Bastan et al., 2022**：将结论重建作为辅助训练目标，非独立评估任务

**变更 slots**：
| Slot | 变更类型 | 具体变化 |
|:---|:---|:---|
| data_pipeline | 替换 | 20万RCT → 569万全领域，新增SJR期刊质量评分与141学科分类 |
| task_formulation | 修改 | 标准摘要/辅助目标 → 四模式因子设计，显式分离结论与摘要的话语功能 |
| evaluation_strategy | 替换 | ROUGE/BLEU单一指标 → 五维LLM评判 + 传统指标 + 诊断指标，双评判器验证 |
| inference_strategy | 新增 | 单提示 → 系统提示变体 + 无新声明指令 + 长度比诊断 |

**后续方向**：
1. 扩展至完整569万数据集评估，降低采样方差
2. 引入临床专家人工评判与专用生物医学LLM（Med-PaLM/Meditron）对比
3. 利用141学科元数据开发学科特异性评估，探索任务难度的领域变异性
4. 解决分数压缩：设计更具区分度的生成任务或引入对抗性/事实性检验

**知识库标签**：
- **modality**: text
- **paradigm**: zero-shot evaluation, benchmark construction
- **scenario**: biomedical NLP, scientific document generation
- **mechanism**: LLM-as-a-judge, controlled prompt ablation, dual-judge validation
- **constraint**: large-scale data curation, discourse-function-aware evaluation, journal-quality metadata enrichment

