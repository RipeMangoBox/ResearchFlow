---
title: 'Video-Bench: Human-Aligned Video Generation Benchmark'
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- Video-Bench：人类对齐的视频生成评测基准
- Video-Bench
acceptance: poster
cited_by: 25
code_url: https://github.com/Video-Bench/Video-Bench
method: Video-Bench
baselines:
- 多模态LLM评判能力基准测试_MLLM-as-a-Judge_
---

# Video-Bench: Human-Aligned Video Generation Benchmark

[Code](https://github.com/Video-Bench/Video-Bench)

**Topics**: [[T__Benchmark_-_Evaluation]], [[T__Video_Generation]] | **Method**: [[M__Video-Bench]] | **Datasets**: Human alignment, Video quality dimensions, Video-Condition Alignment, Inter-rater agreement, Video

| 中文题名 | Video-Bench：人类对齐的视频生成评测基准 |
| 英文题名 | Video-Bench: Human-Aligned Video Generation Benchmark |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2504.04907) · [Code](https://github.com/Video-Bench/Video-Bench) · [Project](https://video-bench.github.io) |
| 主要任务 | 视频生成质量评估（Video Generation Evaluation），涵盖视频-条件对齐（Video-Condition Alignment）与视频质量（Video Quality）两大类别共9个维度 |
| 主要 baseline | EvalCrafter、VBench、ComBench、MLLM-as-a-Judge、LLMScore |

> [!abstract] 因为「现有视频生成评测指标（如CLIP、ViCLIP、GRiT）与人类主观评分相关性低，且MLLM-based方法多为单轮交互导致信息捕获不足」，作者在「GPT-4V as a Generalist Evaluator」基础上改了「引入Chain of Query多轮交互机制与Few-shot Scoring参考视频校准」，在「FETV/GenAI-Bench prompt suite」上取得「Spearman相关性0.733，超越最佳baseline ComBench*达+0.100」

- **人类对齐度**：Spearman相关系数 0.733，较最佳baseline CompBench*（0.633）提升 +0.100，较最佳传统指标GRiT（0.469）提升 +0.264
- **评分者一致性**：Krippendorff's α 达到 0.50，接近人类-人类一致性 0.52，显著优于单轮GPT方案 0.41
- **稳定性**：三轮重复实验Total Agreement Rate @3 达到 0.67

## 背景与动机

当前文本到视频（Text-to-Video）生成模型（如CogVideoX、VideoCrafter2、Show-1等）快速发展，但如何可靠地评估生成视频质量仍是核心瓶颈。现有评测体系存在根本性矛盾：传统指标（如基于CLIP的语义相似度、基于MUSIQ的图像质量评估）计算高效但与人类感知脱节；而新兴的MLLM-as-a-Judge方法虽能利用多模态大模型的理解能力，却受限于单轮交互的信息捕获不足。

具体而言，现有方法在以下场景表现不佳：当用户输入复杂提示词如"一只金毛犬在沙滩上追逐飞盘，背景有日落和海浪"时，需要同时评估视频-文本语义一致性、时序动态连贯性、成像质量、物理合理性等多个维度。EvalCrafter [35] 通过组合多个传统指标进行评分，但各指标间缺乏协同，且无法捕捉人类对"自然度"的综合判断；VBench [23] 构建了细粒度维度体系，但仍依赖预训练模型的特征距离，与人类偏好的相关性有限；ComBench [52] 尝试引入MLLM进行单轮视频描述与评分，然而单次前向传播难以充分挖掘视频中的时序细节与跨模态语义关联，导致对复杂组合属性的评估失真。

核心痛点在于：**单轮MLLM评估存在"信息瓶颈"**——模型一次性接收全部视频帧，既无法针对关键帧深入追问，也缺乏与人类评分标准对齐的校准机制。这直接导致了MLLM评分与人类评分的一致性显著低于人类之间的一致性（单轮GPT方案Krippendorff's α仅0.41 vs 人类-人类0.52）。本文提出Video-Bench，通过Chain of Query多轮交互与Few-shot Scoring参考校准，系统性解决上述对齐缺陷。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ede66910-c81a-439c-8f56-5bc66d2b7b67/figures/Figure_1.png)
*Figure 1: Figure 1. Overview of Video-Bench. Left: We introduce comprehensive evaluation dimensions in two main categories:video-condition alignment (Sec. 3.1.1) and video quality (Sec. 3.1.2). Right: For these*



## 核心创新

核心洞察：**视频评估需要"渐进式信息挖掘"而非"一次性信息倾倒"**，因为MLLM在单轮交互中受限于上下文长度分配与注意力分散，而多轮迭代查询能够逐层细化视觉-语义对齐判断，从而使Few-shot参考校准真正发挥作用成为可能。

| 维度 | Baseline（ComBench/VBench等） | 本文（Video-Bench） |
|:---|:---|:---|
| 交互模式 | 单轮视频描述+评分（Single-round） | Chain of Query多轮迭代查询 |
| 评分校准 | 无参考示例，绝对分数直接输出 | Few-shot Scoring提供参考视频锚定评分尺度 |
| 视觉输入 | 逐帧或均匀采样独立输入 | Grid-view网格视图统一呈现，保留时空关系 |
| 信息捕获 | 单次前向，被动接收 | 主动追问机制，针对模糊区域深入验证 |
| 人类对齐 | Spearman 0.633（CompBench*） | Spearman 0.733，提升+0.100 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ede66910-c81a-439c-8f56-5bc66d2b7b67/figures/Figure_2.png)
*Figure 2: Figure 2. Chain-of-query for video-condition alignmentevaluation. An iterative process where MLLM transformsvideo content into text descriptions, enabling detailed as-sessment of video-condition align*



Video-Bench的评估流程包含五个核心模块，形成从原始输入到多维评分的完整pipeline：

1. **Video + Prompt Input（输入层）**：接收待评估生成视频及其对应文本提示词，作为评估的原始素材。

2. **Grid-view Frame Extractor（网格视图帧提取器）**：将视频帧按时间轴排列为网格化视觉呈现（Grid-view），替代传统的逐帧独立输入或简单均匀采样，使MLLM能够直观把握时序演进与空间布局的关联。

3. **Few-shot Reference Provider（少样本参考提供者）**：从参考视频库中检索与当前评估维度相关的示例视频（含已知人类评分），构建评分上下文。该模块为MLLM建立与人类评分标准一致的"锚点尺度"。

4. **Chain of Query Engine（查询链引擎）**：核心创新模块。针对Video-Condition Alignment维度（如视频-文本一致性、主体-属性绑定、空间关系等），执行多轮迭代查询：首轮提取视频内容描述，后续轮次针对前一轮的模糊或矛盾点进行追问，逐步细化跨模态语义比对。

5. **MLLM Scorer (GPT-4o) + Dimension Aggregator（评分与聚合层）**：GPT-4o基于多轮查询结果与少样本参考进行综合判断，输出9个维度的分数；最终聚合为完整的评估报告。

```
[Video, Prompt] → [Grid-view Extractor] → [Few-shot Reference] ─┐
                                                                ↓
[Chain of Query Engine] ←────多轮交互反馈←──── [GPT-4o Scorer] ←┘
       ↓
[Dimension Aggregator] → [9-Dimension Scores]
```

## 核心模块与公式推导

### 模块 1: Grid-view 视觉编码（对应框架图 输入层→Grid-view Extractor）

**直觉**：将视频时序信息压缩为二维空间布局，使MLLM能够像"看图说话"一样快速建立全局时空认知，避免逐帧输入导致的注意力碎片化。

**Baseline 形式**（传统均匀采样）：
$$\{f_1, f_2, ..., f_k\} = \text{UniformSample}(V, k)$$
符号：$V$为原始视频，$k$为采样帧数，$f_i$为第$i$帧图像。

**变化点**：传统方法将采样帧作为独立图像序列输入，MLLM需自行推断帧间时序关系；Grid-view将帧按时间轴排列为$\sqrt{k} \times \sqrt{k}$的二维网格$G$，显式编码邻近帧的空间-时间邻接性。

**本文公式**：
$$G = \text{GridArrange}(\{f_1, f_2, ..., f_k\}, r, c) \quad \text{其中 } r \times c = k, \text{ 按时间顺序填充}$$
$$\text{Input}_{\text{MLLM}} = [\text{Prompt}; G; \text{Task Instruction}]$$

**对应消融**：Table 3显示，移除Grid-view组件后（即"GPT"配置），Human-GPT一致性从HU-HA的0.50降至0.41，证明视觉呈现格式对对齐度有实质性贡献。

---

### 模块 2: Few-shot Scoring 评分校准（对应框架图 Few-shot Reference Provider）

**直觉**：MLLM的绝对分数输出缺乏跨样本稳定性，通过提供带有人类评分的参考视频作为"标尺"，可将相对判断转化为与人类标准对齐的绝对分数。

**Baseline 公式**（零样本直接评分）：
$$s_{\text{zero}} = \text{GPT-4o}(V, P, C) \in [1, 10]$$
符号：$V$为待评视频，$P$为文本提示，$C$为评分准则描述。

**变化点**：零样本评分中，MLLM对"8分"与"9分"的界限缺乏稳定认知，导致跨样本、跨维度分数不可比；引入参考视频后，模型通过类比推理建立评分尺度。

**本文公式（推导）**：
$$\text{Step 1}: R = \{(V_1^{\text{ref}}, s_1^{\text{human}}), ..., (V_m^{\text{ref}}, s_m^{\text{human}})\} \quad \text{检索} m \text{个参考样本}$$
$$\text{Step 2}: \text{Context} = \text{Format}(R) = "\text{Example 1: [video] → Score: } s_1^{\text{human}}; ..."$$
$$\text{最终}: s_{\text{few-shot}} = \text{GPT-4o}(V, P, C, \text{Context})$$

**对应消融**：Table 4(a)显示，移除Few-shot Scoring后，Imaging Quality维度的Spearman相关性从0.733降至0.639，降幅$\Delta = -0.094$；所有视频质量维度均出现显著下降，证明参考校准是达成人类对齐的关键组件。

---

### 模块 3: Chain of Query 多轮交互（对应框架图 Chain of Query Engine）

**直觉**：视频-条件对齐（尤其是组合属性、空间关系、时序动作等）无法通过单次观察充分验证，需要"描述→质疑→验证"的迭代过程来逼近人类评审的细致程度。

**Baseline 公式**（ComBench式单轮评估）：
$$D = \text{MLLM}_{\text{describe}}(V), \quad s = \text{MLLM}_{\text{score}}(D, P, C)$$
符号：$D$为单次视频描述文本，随后基于描述与提示词的文本匹配进行评分。

**变化点**：单轮描述存在"幻觉"风险——MLLM可能遗漏关键视觉细节或错误推断时序关系；且描述-评分分离导致信息损失。Chain of Query将描述与评分融合为多轮主动探询。

**本文公式（推导）**：
$$\text{Step 1}: q_0 = \text{InitQuery}(P, C), \quad a_0 = \text{GPT-4o}(V, q_0) \quad \text{初始内容提取}$$
$$\text{Step 2}: q_t = \text{FollowUp}(a_{t-1}, P, C, \text{Uncertainty}(a_{t-1})), \quad a_t = \text{GPT-4o}(V, q_t, \{q_i, a_i\}_{i<t})$$
$$\text{Step 3}: \text{当 } \text{Confidence}(a_t) > \tau \text{ 或 } t = T_{\max}: \quad s = \text{ExtractScore}(a_t)$$
$$\text{最终}: s_{\text{CoQ}} = \text{AggregateRound}(\{a_t\}_{t=0}^{T})$$

其中$\text{Uncertainty}(\cdot)$检测前一轮回答中的模糊实体或未验证属性，驱动下一轮针对性追问。

**对应消融**：Table 4显示，移除Chain of Query后，Video-Condition Alignment平均Spearman从0.7336降至0.679，$\Delta = -0.0546$；其中Video-text Consistency维度降幅最大，从0.732降至0.671，$\Delta = -0.061$，验证多轮交互对语义对齐评估的决定性作用。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ede66910-c81a-439c-8f56-5bc66d2b7b67/figures/Table_1.png)
*Table 1: Table 1. Video-Bench Leaderboard. Higher scores indicate better performance. The best score in each dimension is high-lighted in bold. “Avg Rank” is the average rank of multiple dimensions, the lower*



本文在FETV与GenAI-Bench的prompt suite上开展系统评测，核心结论围绕"人类对齐度"展开。如Table 2所示，Video-Bench的Spearman相关系数达到0.733，显著超越所有对比方法：相比复现于相同评测体系的CompBench*（0.633）提升+0.100，相比传统指标中表现最佳的GRiT（0.469）提升+0.264，相比CLIP（0.260）提升近3倍。这一差距在Video-text Consistency维度尤为突出——Video-Bench达0.732，而CompBench*为0.633，+0.099的增量直接印证了Chain of Query对语义对齐评估的增益。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ede66910-c81a-439c-8f56-5bc66d2b7b67/figures/Table_3.png)
*Table 3: Table 3. Inter rater agreement degree (Krippendorff’s α). Higher score indicates better performance. “HU” stands forhuman, “HA” stands for Video-Bench and “GPT” stands for evaluations from single-GPT*



评分者间一致性分析（Table 3）揭示了更深层的意义：Video-Bench与人类评分的一致性（Krippendorff's α = 0.50）已逼近人类之间的一致性（0.52），而单轮GPT方案仅为0.41。这表明本文提出的多轮交互+少样本校准组合，基本消除了"MLLM评判者"与"人类评判者"之间的系统性认知偏差。


![Table 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ede66910-c81a-439c-8f56-5bc66d2b7b67/figures/Table_4.png)
*Table 4: Table 4 shows the ablation study on alignment withhumans. We observe that our proposed componentsare all necessarily effective in reaching higher align-ment with humans. Adding each component leads to*



消融实验（Table 4）进一步量化各组件贡献：Few-shot Scoring的缺失导致Imaging Quality维度相关性骤降0.094（0.733→0.639），是所有单一组件中最大降幅；Chain of Query的缺失导致Video-Condition Alignment平均下降0.0546，其中Video-text Consistency单项下降0.061。两者结合才能实现全维度的最优对齐。


![Table 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ede66910-c81a-439c-8f56-5bc66d2b7b67/figures/Table_5.png)
*Table 5: Table 5. Evaluation results on different base models. For each dimension, we randomly select 30 prompts for comparison.*



扩展实验方面，Table 5显示GPT-4o作为基础模型显著优于Gemini与Qwen2vl；Table 6验证了对简单/复杂提示词的鲁棒性。公平性审视：本文未与MLLM-as-a-Judge [10]进行同benchmark直接对比，且CompBench*为作者复现版本；此外，近期更强的MLLM评判者（如Qwen2vl、Gemini）仅在Table 5中作为替代base model出现，未进入主对比。人类标注指南细节存放于附录，可复现性受限。

## 方法谱系与知识库定位

Video-Bench隶属于**MLLM-as-Judge**方法谱系，直接继承自**GPT-4V(ision) as a Generalist Evaluator** [方法父节点]。该谱系的演进路径为：GPT-4V建立通用视觉-语言评估范式 → MLLM-as-a-Judge [10] 系统化评判者能力分析 → Video-Bench针对视频生成场景引入时序感知的多轮交互机制。

**关键slot变更**：
- **Inference Strategy**：单轮评估 → Chain of Query多轮迭代（替换）
- **Data Pipeline**：零样本评分 → Few-shot参考视频校准（新增）
- **Input Format**：独立帧/序列 → Grid-view网格视图（修改）

**直接baseline差异**：
- **EvalCrafter [35]**：传统指标组合，无MLLM参与；Video-Bench全面转向MLLM-based端到端评估
- **VBench [23]**：细粒度维度体系+传统指标；Video-Bench保留维度框架但替换评分内核为GPT-4o+CoQ
- **ComBench [52]**：单轮MLLM描述-评分；Video-Bench以多轮主动查询替代，Video-Condition Alignment提升+0.093
- **MLLM-as-a-Judge [10]**：通用VLM评判者基准；Video-Bench专注视频生成，引入时序特有的参考校准

**后续方向**：(1) 将Chain of Query机制扩展至更长视频（>10s）的时序一致性评估；(2) 结合视频理解专用MLLM（如Video-LLaMA）替代通用GPT-4o以降低API成本；(3) 构建可微分的评估proxy，使Video-Bench分数可直接用于生成模型的训练优化。

**标签**：modality=video | paradigm=MLLM-as-Judge | scenario=text-to-video generation evaluation | mechanism=multi-round chain-of-query + few-shot calibration | constraint=human-alignment prioritized over computational efficiency

## 引用网络

### 直接 baseline（本文基于）

- [[P__多模态LLM评判能力基准测试_MLLM-as-a-Judge_]] _(直接 baseline)_: Directly related work on using MLLMs as judges for vision-language tasks; likely

