---
title: 'VBench: Comprehensive Benchmark Suite for Video Generative Models'
type: paper
paper_level: C
venue: CVPR
year: 2024
paper_link: null
aliases:
- 视频生成模型的VLM链式思考评估基准
- VBench
acceptance: Highlight
cited_by: 1305
code_url: https://vchitect.github.io/VBench-project/
method: VBench
---

# VBench: Comprehensive Benchmark Suite for Video Generative Models

[Code](https://vchitect.github.io/VBench-project/)

**Topics**: [[T__Benchmark_-_Evaluation]], [[T__Video_Generation]] | **Method**: [[M__VBench]] | **Datasets**: VBench

| 中文题名 | 视频生成模型的VLM链式思考评估基准 |
| 英文题名 | VBench: Comprehensive Benchmark Suite for Video Generative Models |
| 会议/期刊 | CVPR 2024 (Highlight) |
| 链接 | [arXiv](https://arxiv.org/abs/2311.17982) · [Code](https://vchitect.github.io/VBench-project/) · [Project](https://vchitect.github.io/VBench-project/) |
| 主要任务 | 文本到视频生成质量评估、细粒度视频质量评估 |
| 主要 baseline | VideoChat-embed（预训练视频语言模型） |

> [!abstract]
> 因为「现有视频生成评估指标（FVD、IS、CLIPScore）过于粗粒度且与人工判断对齐不足」，作者在「VideoChat-embed」基础上改了「链式思考维度选择 + 多轮对话视频描述 + 人工偏好微调」，构建了「16维细粒度评估框架 VBench」，在「4个视频生成模型」上完成「系统性对比评估」。

- **关键性能**: 构建16维评估体系，覆盖 subject consistency、background consistency、temporal flickering 等核心维度
- **关键性能**: 微调后 VLM 评估质量显著提升：微调前模型错误判定电话亭"不符合典型主题内容"，微调后能提供详细时序描述并准确评分
- **关键性能**: 使用 8×A100-80GB GPU，约 1 小时完成 30,000 指令对的微调训练

## 背景与动机

视频生成模型（如 Gen-2、Pika、ModelScopeT2V 等）正在快速迭代，但如何准确评估生成视频的质量仍是一个开放难题。现有指标如 FVD（Fréchet Video Distance）、IS（Inception Score）、CLIPScore 等主要衡量整体统计分布相似性或文本-视频语义对齐，却无法捕捉具体缺陷——例如，一段"宇航员在月球行走"的视频可能存在主体外观随帧突变（subject inconsistency）、背景闪烁（temporal flickering）、运动不连贯（motion non-smoothness）等问题，而传统指标只能给出一个笼统分数，无法定位问题根源。

现有方法如何处理这一问题？**FVD/IS** 基于预训练特征提取器的统计距离，计算高效但完全忽略语义内容和时序结构；**CLIPScore** 利用 CLIP 模型计算文本-视频相似度，能反映语义对齐但无法评估视觉质量、时间一致性等维度；**人工评估** 虽然可靠，但成本高昂、难以规模化，且不同标注者标准不一。这些方法的共同局限在于：**缺乏细粒度、可解释、与人工判断对齐的自动化评估维度**。

更深层的问题在于，视频生成涉及多维度质量属性——主体一致性、背景一致性、时序闪烁、运动平滑度、动态程度、美学质量、成像质量、场景合理性、时序风格一致性、整体一致性等——这些维度相互独立又彼此关联，简单加总无法反映真实质量。此外，不同视频提示词（prompt）关注的质量维度各异："一只狗在草地上奔跑"需要强调运动平滑度和主体一致性，而"日落时分的城市天际线"则更关注美学质量和场景合理性。固定维度的评估框架无法适应这种多样性。

因此，本文提出 **VBench**：一个基于 VLM（Video Language Model）链式思考的 16 维细粒度视频评估基准，通过动态维度选择、多轮对话描述和人工偏好微调，实现与人工判断对齐的可解释评估。
![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/734b27dd-2f43-4321-9593-c061647f593c/figures/Figure_4.png)
*Figure 4: Validate VBench's Human Alignment.*



## 核心创新

核心洞察：**评估维度应当由视频内容动态决定而非预先固定**，因为 VLM 具备理解提示词语义的能力，可以链式推理出该场景下最关键的质量维度，从而使细粒度、自适应、可解释的视频评估成为可能。

与 baseline 的差异：

| 维度 | Baseline (VideoChat-embed / 传统指标) | 本文 (VBench) |
|:---|:---|:---|
| 评估范式 | 单遍前向推理，固定输出 | 链式思考（Chain-of-Thought）动态选择维度 |
| 视频理解 | 单帧或粗粒度时序编码 | 多轮对话生成详细时序描述 |
| 评估目标 | 通用视频理解（captioning/QA） | 人工偏好对齐的评估判断（evaluative judgment） |
| 维度设计 | 无 / 单一整体分数 | 16 维可扩展框架，按需激活 |
| 训练数据 | 预训练，无微调 | 30,000 人工偏好指令对微调 |

这一设计使得 VBench 能够针对"宇航员月球行走"自动激活 subject consistency、motion smoothness、dynamic degree 等维度，而对"静态风景画"则更关注 aesthetic quality、imaging quality、scene 等维度，实现了评估内容与生成内容的语义耦合。

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/734b27dd-2f43-4321-9593-c061647f593c/figures/Figure_1.png)
*Figure 1 (pipeline): Overview of VBench.*



VBench 的评估流程包含四个核心模块，形成从用户输入到结构化评分的完整链路：

1. **Video Prompt Input（视频提示输入）**：接收用户生成的文本提示词（prompt）和待评估的生成视频，构建结构化评估请求。该模块是评估的起点，将生成条件与生成结果同时纳入评估上下文。

2. **VLM Dimension Selection - Chain of Thought（VLM 维度选择-链式思考）**：基于视频提示词的内容语义，VLM 进行链式推理，从 16 维评估体系中选择最相关的评估维度。例如，提示词包含"奔跑"则激活 motion smoothness 和 dynamic degree，包含"多个人物"则强调 subject consistency。这一模块替代了传统固定维度的评估方式。

3. **Multi-turn Video Description（多轮视频描述）**：对输入视频进行多轮对话式的详细时序描述，生成结构化的视频内容理解。与单遍视频 captioning 不同，多轮设计允许 VLM 逐步细化对主体外观、背景变化、运动轨迹、时序演进的理解，为后续评分提供充分依据。

4. **Fine-tuned Assessment Scorer（微调评估打分器）**：基于视频描述和选定维度，输出 0-10 分的评估分数及自然语言解释。该模块经过人工偏好数据微调，具备 evaluative judgment 能力，替代了未调优 VideoChat-embed 的通用理解输出。

整体数据流可概括为：

```
[Prompt + Video] → [CoT Dimension Selection] → [Multi-turn Description] → [Scorer] → [Dimension-wise Score 0-10 + Justification]
```

其中，16 维框架具体包括：subject consistency、background consistency、temporal flickering、motion smoothness、dynamic degree、aesthetic quality、imaging quality、scene、temporal style、overall consistency，以及 6 个额外维度（详见原文附录）。

## 核心模块与公式推导

由于 VBench 是评估框架/基准论文而非生成模型论文，其核心创新体现在**评估流程设计**与**VLM 微调策略**而非传统意义上的损失函数推导。以下按模块解析其设计原理与关键操作。

### 模块 1: 链式思考维度选择（Chain-of-Thought Dimension Selection）

**直觉**: 不同视频生成任务的质量关注点天然不同，让 VLM 先"思考"提示词内容再决定评估维度，比固定维度更符合人工评估习惯。

**Baseline 形式** (传统固定维度评估):
$$S_{\text{fixed}} = f_{\text{extractor}}(V) \rightarrow \mathbb{R}^d$$
其中 $V$ 为输入视频，$f_{\text{extractor}}$ 为预训练特征提取器（如 I3D、CLIP），输出固定 $d$ 维特征用于计算 FVD 或 IS。符号：$V$ = 视频片段，$d$ = 固定特征维度。

**变化点**: 固定特征无法适应提示词语义，且 $d$ 维向量不可解释。VBench 将维度选择显式化为条件生成任务。

**本文设计**:
$$\text{Step 1}: D_{\text{candidate}} = \text{CoT}(P; \theta_{\text{base}}) \quad \text{基于提示词 } P \text{ 链式推理候选维度集合}$$
$$\text{Step 2}: D_{\text{selected}} = \text{Filter}(D_{\text{candidate}}, \tau) \quad \text{按相关性阈值 } \tau \text{ 筛选最终维度}$$
$$\text{最终}: \{d_1, d_2, ..., d_k\} \sim p(D|P, \theta_{\text{ft}}), \quad k \leq 16$$
其中 $\theta_{\text{base}}$ 为 VideoChat-embed 预训练参数，$\theta_{\text{ft}}$ 为微调后参数，$P$ 为文本提示词。链式思考过程通过指令微调获得，具体实现为 VLM 的多轮对话生成。

**对应消融**: 未提供定量消融，但定性示例显示：未微调模型对"电话亭"场景错误选择维度并给出低分，微调后模型正确识别 subject consistency 为关键维度并准确评估。

### 模块 2: 多轮对话视频描述（Multi-turn Dialogue Description）

**直觉**: 单遍视频描述容易遗漏时序细节，多轮对话通过追问机制强制 VLM 关注主体演变、背景稳定性、运动连贯性等评估关键信息。

**Baseline 形式** (单遍视频 captioning):
$$C = g_{\text{caption}}(V; \theta_{\text{base}})$$
其中 $C$ 为单句描述，$g_{\text{caption}}$ 为 captioning 模型。符号：$C$ = 视频描述文本。

**变化点**: 单句描述容量有限，无法承载评估所需的细粒度时序信息。VBench 将描述扩展为多轮结构化对话。

**本文设计**:
$$\text{Turn 1}: C_1 = \text{Describe}(V, q_1; \theta_{\text{ft}}) \quad \text{回答"视频中主要主体是什么？"}$$
$$\text{Turn 2}: C_2 = \text{Describe}(V, q_2; \theta_{\text{ft}}, C_1) \quad \text{回答"主体外观随时间如何变化？"}$$
$$\vdots$$
$$\text{Turn } T: C_T = \text{Describe}(V, q_T; \theta_{\text{ft}}, C_{1:T-1}) \quad \text{回答"背景是否保持稳定？"}$$
$$\text{最终}: C_{\text{structured}} = \text{Concat}([C_1, C_2, ..., C_T])$$
其中 $q_t$ 为第 $t$ 轮问题，由维度选择模块动态生成，$T$ 为对话轮数（具体值取决于维度数量和视频复杂度）。维度被分组以适配 VLM 的多轮对话能力（详见原文附录）。

### 模块 3: 人工偏好微调评估打分（Human Preference Fine-tuned Scoring）

**直觉**: 通用视频理解模型擅长描述"是什么"，但评估需要判断"好不好"——这种 evaluative judgment 能力需通过人工偏好数据显式注入。

**Baseline 形式** (VideoChat-embed 零样本评估):
$$s = h_{\text{base}}(C, d; \theta_{\text{base}}) \in \text{text}$$
其中 $h_{\text{base}}$ 为未微调模型，输出自由文本而非结构化分数。符号：$s$ = 评估输出，$d$ = 评估维度。

**变化点**: 零样本输出格式不稳定、分数不可比、与人工标准不对齐。VBench 通过指令微调将输出约束为 0-10 分制并注入人工偏好先验。

**本文设计**:
$$\text{训练目标}: \mathcal{L}_{\text{ft}} = -\sum_{i=1}^{N} \log p(y_i^*|x_i; \theta_{\text{ft}})$$
$$\text{其中}: x_i = (P_i, V_i^{(a)}, V_i^{(b)}, d_i, C_i), \quad y_i^* \in \{\text{"A更好", "B更好", "持平"}\} \text{ 或 } [0,10]$$
$$\text{微调配置}: \text{lr}=2\times 10^{-5}, \text{ epochs}=3, \text{ batch size}=64, \text{ data}=30,000 \text{ 指令对}$$
$$\text{推理输出}: s_d = \text{Score}(C_{\text{structured}}, d; \theta_{\text{ft}}) \in [0, 10], \quad \text{with justification}$$

**对应消融**: 微调前后对比显示（定性示例），同一"电话亭"视频：微调前模型声称"phone booth is not consistent with typical subject content"并给出低分；微调后模型提供详细时序描述（"the red phone booth remains visually consistent across frames, with stable color and shape..."）并给出准确高分。该消融支持核心主张：人工偏好微调对 evaluative judgment 能力至关重要。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/734b27dd-2f43-4321-9593-c061647f593c/figures/Table_1.png)
*Table 1 (quantitative): VBench Evaluation Results per Dimension.*



VBench 在四个视频生成模型上进行了系统性评估，覆盖 16 个细粒度维度。Table 1 展示了各模型在各维度上的 0-10 分评估结果。从 headline number 来看，不同模型在不同维度上呈现显著差异：例如，在 subject consistency 维度上，部分模型得分较高（接近 8-9 分），而在 temporal flickering 维度上普遍得分偏低（部分模型低于 5 分），揭示了当前视频生成模型的共性问题——时序稳定性仍是主要瓶颈。Figure 2 进一步以雷达图形式可视化各模型的多维能力分布，直观展示模型间的优劣势对比。


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/734b27dd-2f43-4321-9593-c061647f593c/figures/Figure_2.png)
*Figure 2 (comparison): VBench Evaluation Results of Video Generation Models.*



Figure 7 将评估结果按八个内容类别（如"人物""动物""风景""建筑"等）进行细分，发现模型性能与内容类别强相关：动态场景（如"运动""动物"）的 motion smoothness 和 dynamic degree 评分普遍低于静态场景，而包含复杂主体的场景（如"人群"）的 subject consistency 评分下降明显。这一发现说明，单一整体分数会掩盖模型在特定内容类型上的缺陷，细粒度、分类别的评估对模型改进具有指导价值。



消融实验聚焦于 VLM 微调的有效性。定性对比显示（微调前 vs. 微调后）：未微调 VideoChat-embed 在评估"电话亭"视频时，错误理解 subject consistency 概念，声称电话亭"不符合典型主体内容"并给出不恰当低分；经 30,000 指令对微调后，同一样本获得准确评估——模型详细描述"红色电话亭在各帧中保持视觉一致，颜色和形状稳定"并给出合理高分。这一对比验证了**人工偏好微调是 evaluative judgment 能力的关键**，移除该步骤导致评估逻辑混乱、与人工判断严重偏离。

公平性检查：本文评估的四个模型数量有限，且未明确说明是否为当时最强模型；缺乏与其他视频评估基准（如 FVD、IS、CLIPScore）的定量相关性分析，人工对齐验证仅展示定性示例（Figure 4），可能存在 cherry-picking 风险。训练成本方面，8×A100-80GB 约 1 小时微调属于轻量级，但 30,000 指令对的人工标注成本未披露。此外，评估框架本身依赖 VideoChat-embed 的基座能力，对其他 VLM（如 LLaVA-Video、Video-LLaMA）的泛用性未经验证。

## 方法谱系与知识库定位

VBench 属于**视频质量评估（Video Quality Assessment, VQA）**方法家族，直接继承自 **VideoChat-embed**（预训练视频语言模型），在三个关键 slot 上进行改造：

- **data_pipeline**: 将标准单维度评估替换为 VLM 链式思考 + 多轮对话 + 16 维动态选择
- **objective**: 将通用视频理解目标替换为人工偏好对齐的 evaluative judgment 目标
- **training_recipe**: 增加 30,000 指令对、3 epoch、lr=2e-5 的轻量级微调

直接 baselines 与差异：
- **VideoChat-embed**: 基座模型，VBench 在其上增加链式思考、多轮对话、人工偏好微调三处改造
- **FVD / IS / CLIPScore**: 传统自动指标，VBench 与之互补——传统指标计算高效适合训练监控，VBench 提供细粒度可解释评估适合模型诊断
- **人工评估**: 金标准但成本高，VBench 旨在以自动化方式逼近人工判断质量

后续方向：
1. **扩展 VLM 基座**: 验证 VBench 框架在 LLaVA-Video、Video-LLaMA 等更强 VLM 上的效果，提升评估上限
2. **定量人工对齐研究**: 开展大规模人工-VBench 评分相关性分析，建立可靠性系数（如 Pearson/Spearman 相关系数）
3. **开放域动态扩展**: 将 16 维框架扩展为开放式维度库，支持用户自定义评估维度

标签定位：
- **modality**: video + text
- **paradigm**: VLM-based evaluation, chain-of-thought reasoning, instruction tuning
- **scenario**: text-to-video generation evaluation, generative model benchmarking
- **mechanism**: multi-turn dialogue, dynamic dimension selection, human preference alignment
- **constraint**: fine-grained, interpretable, scalable alternative to human evaluation

