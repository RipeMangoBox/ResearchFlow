---
title: 'MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark'
type: paper
paper_level: C
venue: ICML
year: 2024
paper_link: null
aliases:
- 多模态LLM评判能力基准测试
- MLLM-as-a-Judge
- MLLM-as-a-Judge Benchmark
acceptance: Oral
cited_by: 337
code_url: https://mllm-judge.github.io
method: MLLM-as-a-Judge Benchmark
followups:
- 多模态模型评判器LLaVA-Cr_LLaVA-Critic
- 多模态模型自训练评估器LLaVA_LLaVA-Critic
- Video-Bench：人类对齐_Video-Bench
---

# MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark

[Code](https://mllm-judge.github.io)

**Topics**: [[T__Benchmark_-_Evaluation]], [[T__Visual_Reasoning]] | **Method**: [[M__MLLM-as-a-Judge_Benchmark]] | **Datasets**: MLLM-as-a-Judge, Consistency

| 中文题名 | 多模态LLM评判能力基准测试 |
| 英文题名 | MLLM-as-a-Judge: Assessing Multimodal LLM-as-a-Judge with Vision-Language Benchmark |
| 会议/期刊 | ICML 2024 (Oral) |
| 链接 | [arXiv](https://arxiv.org/abs/2402.04788) · [Code](https://mllm-judge.github.io) · [Project](https://mllm-judge.github.io) |
| 主要任务 | 评估多模态大语言模型（MLLM）作为评判者（judge）的能力，涵盖打分、成对比较、批排序三种评判设置 |
| 主要 baseline | GPT-4V、Gemini/Gemini-Pro-Vision、LLaVA、CogVLM、Qwen-VL-Max |

> [!abstract] 因为「现有MLLM评判能力缺乏系统评估，且单轮直接判断存在偏置和不一致性问题」，作者在「传统直接判断（direct judgment）」基础上改了「引入Analyze-then-Judge链式推理协议、三种评判格式（Score/Pair/Batch）以及多数一致性标准MCC」，在「覆盖8种能力的10数据集基准」上取得「GPT-4V在人类一致率上显著优于Gemini（Pair Comparison +9.5%，Batch Ranking相对差距+32.4%），但所有MLLM仍存在长度偏置和一致性问题」

- **关键性能 1**：GPT-4V在Pair Comparison任务上人类一致率达到79.3%，相比Gemini的72.4%提升+6.9pp（+9.5%相对提升）
- **关键性能 2**：GPT-4V的Score MCC（多数一致性标准）为61.1%，而Gemini仅为5.4%，差距超过10倍
- **关键性能 3**：Gemini在Diffusion数据集上完全失效，CogVLM无法完成Batch Ranking任务，暴露出现有MLLM的脆弱性

## 背景与动机

随着GPT-4V、Gemini等多模态大语言模型（MLLM）能力的快速提升，研究者开始将这些模型用作"评判者"（judge）来自动评估其他模型的输出质量——例如给图像描述打分、比较两个回答的优劣、或对多个候选答案进行排序。然而，一个根本问题尚未解决：**MLLM作为评判者时，其判断是否真的可靠？** 例如，当GPT-4V给某个图像描述打4分时，这个分数与人类专家的判断有多接近？当要求Gemini比较两个回答时，它是否会因为回答长度而非内容质量做出选择？

现有工作主要采用三种方式处理MLLM评判：
- **单轮直接判断（Direct Judgment）**：如早期VQA评估，模型直接输出分数或标签，但缺乏推理过程，导致判断不透明且易受训练数据偏置影响；
- **简单成对比较（Pairwise Comparison）**：如RLHF中的奖励模型，但通常局限于文本模态，未考虑视觉理解错误对评判的干扰；
- **人工评估基准**：如传统图像字幕数据集的人类标注，但成本高昂且无法规模化。

这些方法的核心缺陷在于：**没有系统评估MLLM在视觉-语言联合推理中的评判可靠性**。具体而言，现有基准要么仅覆盖单一能力（如OCR或VQA），要么缺乏对评判一致性的量化（同一问题多次询问结果是否稳定），更重要的是，未发现并纠正MLLM固有的**长度偏置（length bias）**和**高分偏置（high-score bias）**——即模型倾向于给更长或更"安全"的回答更高分数。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ec077948-fd43-41f4-87d8-335d3dc4fffc/figures/fig_001.png)
*Figure: Comparative performance of different MLLMs across three judging settings in 10 datasets, each is the average of three iterations.*



本文首次构建了一个覆盖8种视觉-语言能力的综合评判基准，通过三种评判格式和六重复一致性检验，系统量化MLLM-as-a-Judge的真实能力与局限。

## 核心创新

**核心洞察**：MLLM的评判可靠性不能通过单一任务或单次判断来评估，因为视觉-语言推理涉及多能力耦合且模型输出存在显著方差；因此需要**Analyze-then-Judge链式推理协议**配合**多格式输出**和**重复一致性检验**，才能暴露并量化隐藏的判断偏置。

与baseline的差异：

| 维度 | Baseline（直接判断） | 本文 |
|:---|:---|:---|
| 推理协议 | 单轮直接输出判断，无显式分析过程 | Analyze-then-Judge CoT：先分析图像-文本关系，再给出结构化判断 |
| 评判格式 | 单一格式（仅打分或仅比较） | 三种格式：Scoring Evaluation（1-5分）、Pair Comparison（A赢/B赢/平）、Batch Ranking（全排序） |
| 一致性评估 | 单次判断，无重复检验 | 6次重复实验 + Majority Consistency Criterion（MCC）量化稳定性 |
| 能力覆盖 | 单一任务（如VQA或字幕） | 10数据集 × 8能力：Recognition、OCR、Knowledge、Reasoning、Generation、Hallucination、Bias、Safety |
| 偏置诊断 | 无系统性偏置分析 | 长度偏置密度图、高分偏置统计、训练数据泄露检测 |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ec077948-fd43-41f4-87d8-335d3dc4fffc/figures/fig_002.png)
*Figure: An overview of MLLM-as-a-Judge.*



整体流程遵循"输入→理解→推理→结构化输出→评估验证"的五阶段范式：

1. **Image-Question Input**：输入为图像 + 评判指令（如"请评估这个图像描述的质量"或"比较回答A和回答B哪个更好"）。指令根据三种评判格式模板化，确保不同MLLM接收一致的评估请求。

2. **MLLM Encoder**：商用或开源MLLM（GPT-4V/Gemini/LLaVA/CogVLM/Qwen-VL-Max）对视觉-语言输入进行联合编码，生成隐层表示。此模块为现有模型，本文不修改其参数。

3. **Analyze-then-Judge CoT**（核心创新模块）：替代传统的直接判断，要求MLLM先进行显式分析（如"图像中有一只狗在奔跑，描述A提到'狗'但遗漏'奔跑'，描述B完全准确..."），再基于分析给出最终判断。该设计使判断过程可解释，同时通过延长推理链暴露模型的真实推理能力。

4. **Three-format Output**：根据任务类型输出结构化结果——
   - **Score**：1-5离散分数（Scoring Evaluation）
   - **Pair**：A获胜 / B获胜 / 平局（Pair Comparison）
   - **Batch**：完整排序列表，如[B, A, C, D]（Batch Ranking）

5. **Human Comparison & Consistency Verification**：将MLLM判断与3名独立人类标注者的共识结果对比计算Human Agreement；同时对同一样本重复查询6次，用MCC检验输出稳定性。

```
图像 + 评判指令 → [MLLM Encoder] → 视觉-语言理解
                              ↓
              [Analyze-then-Judge CoT] → 推理分析文本
                              ↓
              [Three-format Output] → Score/Pair/Batch
                              ↓
              ├─→ Human Comparison → Human Agreement %
              └─→ 6× Repetition → MCC Consistency Score
```

## 核心模块与公式推导

### 模块 1: Human Agreement Percentage（对应框架图"Human Comparison"阶段）

**直觉**：量化MLLM判断与人类专家共识的吻合程度，是评判可靠性的首要指标。

**Baseline 公式**（传统单标注者评估）：
$$\text{Accuracy} = \frac{|\{\text{pred} = \text{human}_1\}|}{|\text{Total samples}|}$$
符号：$\text{pred}$ = MLLM预测，$\text{human}_1$ = 单名标注者标签

**变化点**：单标注者存在主观方差，且无法检测MLLM与"平均人类"vs"专家人类"的差异。本文改为**3人多数共识**作为gold standard。

**本文公式**：
$$\text{Human Agreement} = \frac{|\{\text{MLLM judgment} = \text{Human consensus}\}|}{|\text{Total samples}|}$$
其中 $\text{Human consensus}$ 定义为3名独立标注者中出现次数≥2的选项（即多数投票）。对于Batch Ranking任务，由于全排序空间巨大，采用归一化Levenshtein Distance衡量排序差异：
$$\text{Batch Distance} = \frac{\text{Levenshtein Distance}(\text{pred}, \text{human})}{\max(\text{len}(\text{pred}), \text{len}(\text{human}))}$$
Human Agreement for Batch = 1 − Batch Distance（数值越低表示越接近人类排序）。

**对应消融**：Table 3显示GPT-4V在Batch Ranking上距离为0.621，Gemini为0.469（数值越低越好），相对差距+32.4%。

---

### 模块 2: Majority Consistency Criterion (MCC)（对应框架图"Consistency Verification"阶段）

**直觉**：单次判断无法区分"模型确实确信"和"模型随机猜测"；通过6次重复查询，检测MLLM输出的随机性。

**Baseline**：无。现有MLLM评估均为单次pass，缺乏一致性量化。

**本文公式（推导）**：

$$\text{Step 1: 定义重复响应集合} \quad R = \{r_1, r_2, ..., r_6\}, \quad r_i \in \mathcal{C}$$
其中$\mathcal{C}$为所有可能的响应类别（如Score的$\{1,2,3,4,5\}$，Pair的$\{A, B, Tie\}$）。

$$\text{Step 2: 计算众数出现次数} \quad n_{\max} = \max_{c \in \mathcal{C}} \sum_{i=1}^{6} \mathbb{1}[r_i = c]$$
加入了"过半阈值"以排除偶然一致：要求某响应在6次中出现超过3次（即严格多数），而非简单plurality。

$$\text{最终: MCC} = \mathbb{1}\left[n_{\max} > 3\right] = \mathbb{1}\left[\max_c \sum_{i=1}^{6} \mathbb{1}[r_i = c] > 3\right]$$
符号：$\mathbb{1}[\cdot]$ = 指示函数，$c$ = 响应类别，$r_i$ = 第$i$次重复响应。

**加权扩展**：对于Score任务，除二元MCC外，还计算6次分数的加权平均以捕捉细微波动：
$$\text{Score Average} = \frac{1}{6}\sum_{i=1}^{6} r_i \quad (r_i \in \{1,2,3,4,5\})$$

**对应消融**：Table 4显示GPT-4V的Score MCC为61.1%，Gemini仅为5.4%；Score Average上GPT-4V为79.6%，Gemini为53.1%，差距+49.9%相对提升。这表明**CoT推理虽延长输出，但并未牺牲一致性**——反而GPT-4V的高一致性源于其稳定的推理链，而Gemini的低一致性暴露了其判断的随机性。

---

### 模块 3: Analyze-then-Judge CoT Protocol（对应框架图核心推理阶段）

**直觉**：强制MLLM先分析后判断，模拟人类专家的审慎评估过程，同时使隐藏偏置（如长度偏置）在分析文本中显性化。

**Baseline 公式**（Direct Judgment prompt）：
$$\text{Prompt}_{\text{direct}}: \text{"Evaluate the quality of [response] for [image]. Output: [Score/Pair/Batch]"}$$

**变化点**：直接判断导致模型依赖训练数据中的表面相关性（如长回答=高质量），且无法诊断错误来源。本文改为两步结构化提示：

$$\text{Step 1: 分析指令} \quad \text{"First, analyze the [image] and [response]: identify key objects, check factual accuracy, note omissions..."}$$
$$\text{Step 2: 判断指令} \quad \text{"Based on your analysis, provide your final judgment: [Score/Pair/Batch format]"}$$

**关键观察**：Figure 3和Figure 7显示，GPT-4V在CoT设置下输出长度显著增加，但长度与分数的相关性模式被显性记录——通过比较分析文本长度与最终分数的联合分布，可量化诊断**长度偏置**（length bias）。

**对应消融**：Table 3及Figure 4显示，CoT设置下GPT-4V的人类一致率与直接判断基本持平，但输出方差降低；而Gemini和开源模型（LLaVA/CogVLM）的CoT输出暴露更严重的高分偏置——LLaVA在Scoring Evaluation中倾向于给几乎所有回答打4-5分，导致与人类标注的吻合度极低。

## 实验与分析



本文在自建的10数据集基准上评估了5个MLLM：GPT-4V、Gemini-Pro-Vision、Qwen-VL-Max、LLaVA-1.5、CogVLM。核心结果汇总于Table 3（三种评判格式的人类一致率）和Table 4（6次重复一致性）。**关键发现**：GPT-4V在所有设置中均为最强评判者，但绝对性能远未饱和；开源模型与商用模型存在显著鸿沟；所有模型均暴露特定偏置。

具体而言，在**Scoring Evaluation**（1-5分打分）中，GPT-4V的人类一致率为69.9%，Gemini为67.7%，差距仅+3.2%——这是三种格式中差距最小的，说明离散打分任务相对简单，但两者均未超过70%，表明**即使是最佳MLLM，在细粒度质量评估上仍有约30%的判断与人类专家分歧**。在**Pair Comparison**（成对比较）中，GPT-4V的优势扩大至79.3% vs 72.4%（+6.9pp, +9.5%相对提升），这与Pairwise任务需要相对而非绝对判断、更能暴露模型偏置有关。最显著的差距出现在**Batch Ranking**（批排序）：GPT-4V的归一化Levenshtein距离为0.621，Gemini为0.469（数值越低越接近人类），相对差距达+32.4%；CogVLM在此任务上完全失败，无法输出完整排序而被排除统计。这说明**复杂排序任务对MLLM的规划能力和全局一致性要求极高**，当前模型尚未具备可靠的批排序评判能力。



一致性分析（Table 4）揭示了更深层的脆弱性。GPT-4V的Score MCC为61.1%，意味着仅约六成样本在6次重复中保持严格多数一致；而Gemini的MCC暴跌至5.4%，**几乎等同于随机抛硬币**。Score Average指标同样分化：GPT-4V 79.6% vs Gemini 53.1%。这一对比表明，**Gemini的高人类一致率部分源于其与人类共享的系统性偏置（如长度偏置），而非稳定的推理能力**——当多次查询时，其判断剧烈波动，暴露了其内部决策机制的不稳定性。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ec077948-fd43-41f4-87d8-335d3dc4fffc/figures/fig_003.png)
*Figure: Length distribution in responses for different MLLMs.*



Figure 4的密度图直观展示了这些偏置：左图（Pair Comparison）显示GPT-4V的判断分布与人类标注高度重叠，而Gemini和LLaVA明显右偏（倾向于选A或判平局而非严格区分优劣）；右图（Scoring Evaluation）显示LLaVA的分数密度集中在4-5分区间，呈现严重的高分偏置。Figure 6和Figure 7进一步量化了长度偏置——在多个数据集中，MLLM给出的分数与回答长度呈正相关，而人类标注则无此相关性。

**公平性检验**：本文作为benchmark论文，其"公平性"主要体现在基准覆盖度而非方法比较。当前评估的模型均为2024年初可用版本，缺少后续更强的模型如GPT-4o、Claude-3-Opus-vision、Gemini Ultra、InternVL、Yi-VL等。此外，Gemini在DiffusionDB数据集上完全失效（输出格式错误），CogVLM无法完成Batch Ranking，这些排除可能低估了开源模型的真实能力差距。作者未控制不同模型的prompt engineering差异，且6次重复对于捕捉完整方差可能不足。Figure 5显示6次重复的consistency checking结果，但未报告随重复次数增加的收敛曲线。

## 方法谱系与知识库定位

**方法族**：MLLM Evaluation / Benchmark Construction

**直接继承**：无单一父方法——本文属于benchmark构建工作，而非模型改进。其方法论上最接近**LLM-as-a-Judge**（如Zheng et al., 2023的Chatbot Arena）的文本评判范式，但将其扩展至视觉-语言多模态场景，并系统引入了CoT推理协议和一致性检验机制。

**与直接baseline的差异**：
- **GPT-4V**：本文将其从"被评估的通用MLLM"重新定位为"judge baseline"，发现其虽为最强但仍远非完美评判者
- **Gemini-Pro-Vision**：暴露其高人类一致率背后的低一致性陷阱——与人类一致≠可靠
- **LLaVA/CogVLM**：开源模型因训练数据偏置（high-score bias）和指令遵循能力不足，在评判任务上差距显著

**后续方向**：
1. **Judge-Tuning**：基于本文诊断的偏置类型（长度、分数、一致性），专门微调MLLM以提升评判可靠性
2. **Multi-Judge Ensemble**：利用本文MCC指标筛选高一致性模型，构建加权ensemble judge
3. **Dynamic Benchmark**：当前为静态数据集，未来可扩展为对抗性动态评测，针对MLLM新暴露的偏置持续迭代

**知识库标签**：
- **模态（modality）**：vision-language, image-text
- **范式（paradigm）**：benchmark, evaluation protocol, chain-of-thought reasoning
- **场景（scenario）**：model-as-a-judge, automatic evaluation, human alignment
- **机制（mechanism）**：consistency checking, bias diagnosis, multi-format judgment
- **约束（constraint）**：no training, black-box API evaluation, repeated inference budget

## 引用网络

### 后续工作（建立在本文之上）

- [[P__多模态模型评判器LLaVA-Cr_LLaVA-Critic]]: Core methodology for using MLLMs as judges; directly enables the paper's evaluat
- [[P__多模态模型自训练评估器LLaVA_LLaVA-Critic]]: Core methodology for using MLLMs as judges, directly relevant to paper's approac
- [[P__Video-Bench：人类对齐_Video-Bench]]: Directly related work on using MLLMs as judges for vision-language tasks; likely

