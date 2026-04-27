---
title: 'MMAR: A Challenging Benchmark for Deep Reasoning in Speech, Audio, Music, and Their Mix'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 音频深度推理层次化基准MMAR
- MMAR
- MMAR is a challenging benchmark tha
acceptance: Poster
cited_by: 82
method: MMAR
modalities:
- Audio
- Text
paradigm: supervised
---

# MMAR: A Challenging Benchmark for Deep Reasoning in Speech, Audio, Music, and Their Mix

**Topics**: [[T__Benchmark_-_Evaluation]], [[T__Visual_Question_Answering]] | **Method**: [[M__MMAR]] | **Datasets**: MMAR

> [!tip] 核心洞察
> MMAR is a challenging benchmark that evaluates deep reasoning in audio-language models across hierarchical reasoning layers (Signal, Perception, Semantic, Cultural) and mixed audio modalities.

| 中文题名 | 音频深度推理层次化基准MMAR |
| 英文题名 | MMAR: A Challenging Benchmark for Deep Reasoning in Speech, Audio, Music, and Their Mix |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2505.13032) · [Code](待发布) · [Project](待发布) |
| 主要任务 | 音频问答 / 基准测试与评估 |
| 主要 baseline | MMAU, Mellow, GAMA, CompA, Audio Flamingo 2, Qwen-Audio/Qwen2-Audio, Listen-Think-Understand, Gemini 2.0 Flash, GPT-4o, OpenAI o1 |

> [!abstract] 因为「音频领域缺乏类似文本域MMLU-Pro的严格深度推理基准，现有benchmark局限于单域浅层理解」，作者在「MMAU」基础上改了「四层层次化分类体系（Signal/Perception/Semantic/Cultural）+ 混合模态覆盖 + Chain-of-Thought标注」，在「MMAR自构建基准」上揭示「当前音频语言模型存在关键推理缺陷，OpenAI o1表现最优但仍有限」。

- **规模**: 1,000条音频-问题-答案三元组，覆盖声音、音乐、语音及混合模态
- **推理深度**: 部分问题需研究生级别知识，四层递进式难度设计
- **评估发现**: 现有LALMs/LARMs表现低下，LRMs（如OpenAI o1）相对最优但仍显不足

## 背景与动机

音频语言模型（LALMs）近年来在语音识别、音乐理解、环境音分类等任务上取得显著进展，但当面对需要多步深度推理的复杂问题时，这些模型往往暴露出根本性缺陷。例如，给定一段包含背景音乐的人声对话视频，模型不仅需要识别"有人在说话"和"有钢琴声"，还需推断说话者的情绪状态、音乐所暗示的场景氛围，乃至文化背景下的社交含义——这种跨层次推理远超现有benchmark的评估范畴。

现有方法如何处理音频理解？**MMAU**作为最直接的对比基准，提供了大规模多任务音频理解评测，但局限于单域覆盖（声音/音乐/语音各自独立），且以表层识别为主，缺乏推理深度的系统性分层。**Mellow**专为音频推理设计，聚焦小规模模型的推理能力，但未涉及混合模态场景与层次化难度递进。**GAMA**强调复杂推理能力，**Audio Flamingo 2**具备专家级推理，**CompA**针对组合推理，然而这些工作均以模型改进为目标，而非构建严格的分层推理评测体系。

它们的核心短板在于：**缺乏显式的推理难度分层机制**。现有benchmark采用扁平化评估结构，无法区分模型是在进行低级的信号特征匹配，还是高级的语义/文化推理；同时，单域设计忽略了真实世界中声音、音乐、语音交织的混合模态场景；更关键的是，几乎没有benchmark提供Chain-of-Thought标注来支撑可解释性评估与模型训练。文本域已有MMLU-Pro等严格推理基准，音频域却长期空白。

本文提出MMAR，首次构建覆盖四层推理层次、包含混合模态、附带CoT标注的音频深度推理基准，系统揭示当前模型的关键局限。
![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/6f5eb931-7791-49b2-9092-fc5bb0cfaff3/figures/fig_001.png)
*Figure: Examples from the MMAR benchmark, illustrating challenges at the signal, perceptual,*



## 核心创新

核心洞察：音频推理的难度本身具有层次结构——从物理信号属性到感知组织、再到语义意义、最终达至文化专家知识，因为真实世界的音频理解天然遵循这种由浅入深的认知递进，从而使系统性的分层评估与针对性诊断成为可能。

| 维度 | Baseline (MMAU等) | 本文 (MMAR) |
|:---|:---|:---|
| 领域覆盖 | 单域独立（声音/音乐/语音分开评测） | 混合模态：声音+音乐+语音+任意组合 |
| 推理结构 | 扁平化，无难度分层 | 四层层次化：Signal → Perception → Semantic → Cultural |
| 标注形式 | 仅有答案，无推理过程 | 每题附Chain-of-Thought逐步推理标注 |
| 难度设计 | 通用知识为主 | 部分题目需研究生级别专业知识 |
| 评估目标 | 识别准确率 | 分层错误分析，定位模型失效层级 |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/6f5eb931-7791-49b2-9092-fc5bb0cfaff3/figures/fig_002.png)
*Figure: (a) The data distribution of single and mixed modalities in MMAR. (b) The hierarchical*



MMAR的构建遵循"源头采集 → 层次分类 → 推理标注 → 多维评估"的完整pipeline，共四个核心模块：

**模块1：音频源头采集（Audio source collection）**
输入为真实世界互联网视频，输出为经筛选的多样化音频片段。不同于现有benchmark复用静态数据集（AudioSet、Clotho等），MMAR直接从原始视频提取，确保混合模态场景的自然分布——如带解说的纪录片、有背景音乐的访谈、环境音与语音交织的户外场景等。

**模块2：层次化分类体系应用（Hierarchical taxonomy application）**
输入为原始音频内容，输出为按四层体系分类的推理问题。该模块是MMAR的组织核心：Signal层考察物理属性（频率、时长、声源数量）；Perception层考察听觉场景分析（空间定位、流分离）；Semantic层考察意义与上下文（情绪、意图、事件因果）；Cultural层考察领域专家知识（音乐风格史、声学工程原理、方言文化背景）。

**模块3：Chain-of-Thought推理标注（CoT rationale generation）**
输入为问题-答案对，输出为详细的逐步推理标注。每道题配备完整CoT，既支撑可解释性评估，也为未来训练推理模型提供潜在监督信号。

**模块4：多模型分层评估（Multi-model evaluation）**
输入为benchmark问题与模型响应，输出为分层准确率及错误类型分析。覆盖五类模型：LALMs（大型音频语言模型）、LARMs（音频-文本检索模型）、OLMs（专用音频模型）、LLMs（纯文本大模型）、LRMs（大型推理模型如OpenAI o1）。

```
互联网视频 → [音频提取] → 多样化音频片段
                                    ↓
                    [层次化分类: Signal/Perception/Semantic/Cultural]
                                    ↓
                    [问题生成 + 答案标注 + CoT推理链生成]
                                    ↓
                    [1,000条三元组: 音频 + 问题 + (答案, CoT)]
                                    ↓
                    [五类模型评估 → 分层准确率 + 错误分析]
```

## 核心模块与公式推导

### 模块1：泊松二项分布性能建模（对应框架图 评估模块）

**直觉**：传统benchmark假设所有题目难度相同（二项分布），但MMAR中每道题的推理复杂度差异显著，需用变概率模型刻画真实性能分布。

**Baseline 公式** (标准二项分布): $$X \sim \text{Binomial}(n, p)$$
符号: $n$ = 总题数, $p$ = 统一正确率。该假设要求各题独立同分布，显然不适用于分层难度设计。

**变化点**：MMAR的四层taxonomy导致不同题目具有异质的正确概率 $p_i$——Signal层题目对强模型可能 $p_i \approx 0.9$，而Cultural层可能 $p_i \approx 0.1$。统一参数 $p$ 会严重扭曲方差估计与置信区间。

**本文公式（推导）**:
$$\text{Step 1}: X_i \sim \text{Bernoulli}(p_i) \quad \text{第}i\text{题独立但异质，加入题目难度层级参数}$$
$$\text{Step 2}: X = \sum_{i=1}^{n} X_i \quad \text{总正确数为独立异质伯努利试验之和}$$
$$\text{最终}: X \sim \text{PoissonBinomial}(p_1, p_2, \ldots, p_n)$$
其中 $p_i$ 由题目所属层级（Signal/Perception/Semantic/Cultural）及模型类型联合决定。该分布的PMF需通过FFT或递归算法计算，但其均值与方差有闭式解：$\mathbb{E}[X] = \sum p_i$，$\text{Var}(X) = \sum p_i(1-p_i)$，允许更精确的模型能力估计与跨模型显著性检验。

**对应消融**：Figure 4展示该统计建模的应用，区分了不同模型在各层的 $p_i$ 分布差异。

### 模块2：四层层次化错误分解（对应框架图 分类模块）

**直觉**：传统accuracy无法诊断"模型错在哪里"，需将错误归因到特定推理层级以指导后续改进。

**Baseline 公式** (标准准确率): $$\text{Acc} = \frac{1}{n}\sum_{i=1}^{n}\mathbb{1}[\hat{y}_i = y_i]$$
符号: $\hat{y}_i$ = 模型预测, $y_i$ = 真实答案, $\mathbb{1}[\cdot]$ = 指示函数。该指标完全混淆了信号层失误与语义层失误。

**变化点**：MMAR要求显式追踪错误发生的认知层级。以Table 4中的"井深估计"问题为例，模型可能在Signal层（水声频率分析）、Perception层（混响时间推断）、Semantic层（物理公式应用）或Cultural层（潜水文化常识）任一环节失败。

**本文公式（推导）**:
$$\text{Step 1}: \text{Acc}_l = \frac{1}{n_l}\sum_{i \in \mathcal{Q}_l}\mathbb{1}[\hat{y}_i = y_i] \quad \text{分层准确率，其中}\mathcal{Q}_l\text{为层级}l\text{的题目集合}$$
$$\text{Step 2}: \text{ErrType}(i) = \text{arg}\min_{l} \{l \text{mid} \text{CoT}_i^{(l)} \text{ contains error}\} \quad \text{基于CoT标注定位最早错误层级}$$
$$\text{最终}: \text{LayeredScore} = \sum_{l=1}^{4} w_l \cdot \text{Acc}_l, \quad w_l = \frac{l}{\sum_{k=1}^{4}k} = \frac{l}{10}$$
权重设计体现递进难度：Signal(0.1), Perception(0.2), Semantic(0.3), Cultural(0.4)。

**对应消融**：Table 4显示井深估计问题的错误类型分解，揭示多数模型在Perception→Semantic的过渡层失败；Table 6显示音频替换为噪声后性能退化，验证错误非文本偏见所致。

### 模块3：人类-LLM协作质量管控（对应框架图 全pipeline）

**直觉**：纯人工构建难以覆盖研究生级专业知识，纯LLM生成难以保证事实准确性，需迭代协作。

**Baseline 流程** (传统人工或纯自动): $$\text{Question} \leftarrow \text{Human}_\text{only} \text{ or } \text{LLM}_\text{only}$$

**变化点**：单一来源要么受限于专家人力瓶颈，要么产生幻觉错误。MMAR采用三轮迭代：人类专家定义taxonomy框架 → LLM批量生成候选问题 → 人类专家验证并修正 → LLM辅助生成CoT → 人类终审。

**本文流程**:
$$\text{Step 1}: \mathcal{T} = \text{HumanDesign}(\{\text{Signal, Perception, Semantic, Cultural}\}) \quad \text{人类设计四层框架}$$
$$\text{Step 2}: \mathcal{Q}^{(0)} = \text{LLMGenerate}(\mathcal{T}, \mathcal{A}) \quad \text{LLM基于框架与音频生成初稿}$$
$$\text{Step 3}: \mathcal{Q}^{(1)} = \text{HumanFilter}(\mathcal{Q}^{(0)}, \text{difficulty} \geq \tau) \quad \text{人类筛选高难度题目}$$
$$\text{Step 4}: (\mathcal{Q}, \text{CoT}) = \text{IterativeRefine}(\mathcal{Q}^{(1)}, \text{Human} \leftrightarrow \text{LLM}, k=3) \quad \text{三轮人机迭代精修}$$
该流程确保1,000条数据的专家级质量与跨域覆盖度。

## 实验与分析



MMAR对五类模型进行系统评估，覆盖LALMs（如Qwen-Audio、Qwen2-Audio）、LARMs、OLMs、LLMs（纯文本基线）、LRMs（OpenAI o1等推理专用模型）。Figure 5展示核心结果：当前音频语言模型在MMAR上普遍表现不佳，即使是最强的通用多模态模型也远未达到饱和。OpenAI o1作为显式推理模型，在总体表现上领先于其他类别，但其绝对准确率仍显示显著的提升空间——这表明MMAR成功构建了一个具有挑战性的评测上界。

分层评估揭示关键模式：模型在Signal层（基础物理属性识别）表现相对最好，但随着层级提升至Cultural（专家域知识），性能急剧下降。这种分层衰减曲线证明了四层taxonomy的有效性——它确实捕捉到了不同深度的推理能力差异，而非简单的题目难度噪声。LLMs在音频输入被替换时的表现进一步说明，文本偏见无法解释模型行为，音频信号本身对推理至关重要。



Table 6的噪声消融实验提供关键验证：将原始音频替换为等长白噪声后，所有模型的准确率显著退化。这一结果确认模型并非依赖问题文本中的统计偏见进行"猜测"，而是确实在处理音频信号内容。Table 4针对特定问题（井深估计）的错误类型分解显示，多数失败发生在Perception到Semantic的过渡——模型能感知水声特征，但无法正确推断物理深度，这精确指向了当前音频模型在"感知→语义"映射上的结构性缺陷。

公平性审视：本文baseline选择较为全面，覆盖了专用音频模型（Mellow、GAMA、Audio Flamingo 2）、通用音频语言模型（Qwen系列）、显式推理模型（Listen-Think-Understand、OpenAI o1）及商用多模态大模型（Gemini 2.0 Flash、GPT-4o）。但值得注意的是，部分最新模型未纳入（如Gemini 2.5 Pro、Claude 3.5/3.7 Sonnet的音频能力、DeepSeek-R1的音频适配版本），且1,000样本量虽保证质量却可能带来排名高方差。作者坦承未发布代码、未报告人工性能天花板、未展示基于MMAR训练后的模型改进——这些均为后续工作留出空间。

## 方法谱系与知识库定位

MMAR属于**音频语言理解基准**家族，直接继承自**MMAU**（Massive Multi-Task Audio Understanding Benchmark）作为谱系父节点，但在四个关键slot上完成结构性变革：**data_pipeline**从单域静态数据集扩展为混合模态真实视频；**architecture**从扁平结构升级为四层层次化taxonomy + CoT标注；**inference_strategy**从单层评估改为递进难度测评；**objective**从单一准确率增加分层错误分解指标。

直接baseline差异速览：
- **MMAU**: 最可比前身，MMAR将其从"多任务理解"推进到"深度推理"维度
- **Mellow/GAMA/Audio Flamingo 2**: 专用音频推理模型，MMAR作为benchmark对其统一评测而非改进模型架构
- **CompA**: 聚焦组合推理，MMAR覆盖更广的推理类型与层次
- **Qwen-Audio/Qwen2-Audio**: 通用音频语言模型代表，MMAR揭示其在深层推理上的瓶颈
- **OpenAI o1/DeepSeek-R1**: 文本域推理范式向音频域的迁移需求被MMAR激发

后续方向：（1）基于MMAR CoT数据训练专用音频推理模型，验证benchmark的训练价值；（2）扩展至10K+规模并覆盖更多文化域知识；（3）开发自动化的分层错误诊断工具，将MMAR从评估基准升级为模型改进的闭环基础设施。

**标签**: 模态[audio+text] / 范式[benchmark/evaluation] / 场景[deep reasoning] / 机制[hierarchical taxonomy, chain-of-thought, human-LLM collaboration] / 约束[small-scale high-quality curation, graduate-level difficulty]

