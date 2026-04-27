---
title: 'T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation'
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- T2V组合生成综合评测基准
- T2V-CompBench
acceptance: poster
cited_by: 120
method: T2V-CompBench
---

# T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation

**Topics**: [[T__Benchmark_-_Evaluation]], [[T__Video_Generation]] | **Method**: [[M__T2V-CompBench]] | **Datasets**: T2V-CompBench Human Evaluation Correlation

| 中文题名 | T2V组合生成综合评测基准 |
| 英文题名 | T2V-CompBench: A Comprehensive Benchmark for Compositional Text-to-video Generation |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2407.14505) · [Code] · [Project] |
| 主要任务 | 文本到视频生成的组合性评估（7大类别：一致属性绑定、动态属性绑定、空间关系、运动绑定、动作绑定、物体交互、生成数量关系） |
| 主要 baseline | CLIP-Score, BLIP-CLIP, BLIP-BLEU, BLIP-VQA, ViCLIP, VPEval-S, M-GDino |

> [!abstract]
> 因为「现有T2V评估指标（CLIP-Score/ViCLIP等）无法准确衡量视频的组合性能力」，作者在「传统文本-图像/视频相似度指标」基础上改了「设计7类组合提示+类别专用指标（Grid-LLaVA/D-LLaVA/G-Dino/DOT）」，在「T2V-CompBench人工相关性验证」上取得「所提指标在各自类别上与人类判断相关性最高」。

- **关键性能1**：Grid-LLaVA 在 consistent attribute binding、action binding、object interactions 三类上人类相关性最优
- **关键性能2**：D-LLaVA 在 dynamic attribute binding 上人类相关性最优；G-Dino 在 spatial relationships 和 generative numeracy 上最优；DOT 在 motion binding 上最优
- **关键性能3**：传统统一指标（CLIP-Score/ViCLIP）在所有组合类别上人类相关性均显著低于所提专用指标

## 背景与动机

当前文本到视频（T2V）生成模型发展迅速，但如何准确评估生成视频是否真正遵循了文本提示中的组合性要求，仍是一个未解决的核心问题。组合性指的是模型能否正确地将多个属性、对象、关系和动作按照文本描述组合在一起——例如，"一个红色的球在蓝色的盒子左边滚动"要求同时满足颜色属性（红/蓝）、空间关系（左边）、运动（滚动）和对象交互。

现有评估方法主要分为两类：一是基于文本-视觉相似度的统一指标，如 **CLIP-Score** 计算CLIP文本与图像嵌入的余弦相似度，**ViCLIP** 进一步提取视频级特征进行相似度计算；二是基于生成-再描述的方法，如 **BLIP-CLIP** 用BLIP生成图像描述后计算文本-文本相似度，**BLIP-BLEU** 用BLEU度量描述质量，**BLIP-VQA** 借助视觉问答能力验证内容。**VPEval-S** 和 **M-GDino** 则尝试引入检测机制评估空间关系和物体运动方向。

然而，这些方法存在根本性缺陷：CLIP-Score/ViCLIP 等统一指标将视频压缩为单一向量，丢失了细粒度的组合信息；BLIP系列依赖再描述质量，对动态变化和复杂交互无能为力；VPEval-S 和 M-GDino 仅覆盖单一维度且从T2I任务简单迁移，无法适应视频的时序复杂性。具体而言，当评估"动态属性绑定"（如物体颜色随时间变化）时，帧平均的CLIP-Score完全无法捕捉时序变化；评估"物体交互"（如A把B递给C）时，ViCLIP的单一视频向量无法解析多主体关系。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5895ed36-56f5-44ea-a4c0-c0e0e19dadba/figures/fig_001.png)
*Figure: Overview of T2V-CompBench. We propose T2V-CompBench, a comprehensive compositional text-to-video generation bench-*



因此，本文提出 T2V-CompBench，首次系统性地将T2V组合性评估分解为7个独立类别，并为每类设计专用评估指标，以实现对模型组合能力的精确诊断。

## 核心创新

核心洞察：组合性评估必须「分而治之」，因为不同组合维度（属性、空间、运动、交互等）需要截然不同的感知能力，从而使「为每类组合设计专用指标并验证其与人类判断的一致性」成为可能。

| 维度 | Baseline（CLIP/ViCLIP/BLIP系列） | 本文（T2V-CompBench） |
|:---|:---|:---|
| 评估粒度 | 单一统一分数，全局相似度 | 7类独立分数，细粒度诊断 |
| 特征提取 | 文本-视频整体向量对齐 | 检测/跟踪/LLM推理，按需选择 |
| 时序处理 | 帧平均或简单视频编码 | D-LLaVA显式对比帧间变化，DOT跟踪运动轨迹 |
| 空间推理 | 无显式空间建模 | G-Dino基于GroundingDINO的检测框关系推理 |
| 人类对齐 | 未系统验证 | 651视频人工标注，Kendall's τ/Spearman's ρ验证 |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5895ed36-56f5-44ea-a4c0-c0e0e19dadba/figures/fig_002.png)
*Figure: 1) Consistent attribute*



T2V-CompBench 的评估流程包含四个核心阶段：

**阶段一：Prompt Generation（GPT-4模板生成）** — 输入为7类组合类别的定义与结构化模板，输出为覆盖一致属性绑定、动态属性绑定、空间关系、运动绑定、动作绑定、物体交互、生成数量关系的大规模评估提示集。该模块通过层次化元类结构确保名词/动词分布的多样性与平衡性。

**阶段二：Video Generation（T2V模型生成）** — 输入为上述文本提示，输出为各待评估T2V模型（如VideoCrafter、ModelScope、AnimateDiff等）生成的视频片段。此阶段不限制具体模型，支持开放式评估。

**阶段三：Category-Specific Metric Router（类别专用指标路由）** — 输入为生成视频与原始提示，根据类别自动选择最优评估指标：Grid-LLaVA（video LLM）处理 consistent attribute binding / action binding / object interactions；D-LLaVA（image LLM）处理 dynamic attribute binding；G-Dino（detection-based）处理 spatial relationships / generative numeracy；DOT（tracking-based）处理 motion binding。PLLaVA 作为备选video LLM跨类别测试。

**阶段四：Score Aggregation & Human Validation（分数聚合与人类验证）** — 输出为每类归一化至[0,1]的分数，并通过Amazon Mechanical Turk收集人类判断，计算 Kendall's τ 和 Spearman's ρ 验证自动指标与人类偏好的一致性。

```
GPT-4模板 → 结构化组合提示 → T2V模型生成视频 → [Metric Router]
                                                        ↓
                    ┌─────────────┬─────────────┬─────────────┬─────────────┐
                    │ Grid-LLaVA  │  D-LLaVA    │   G-Dino    │    DOT      │
                    │ (video LLM) │ (image LLM) │(detection)  │ (tracking)  │
                    │ 属性/动作/  │  动态属性   │  空间/数量  │   运动绑定  │
                    │   交互      │   绑定      │   关系      │             │
                    └──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┘
                           └───────────────┴───────────────┴───────────────┘
                                           ↓
                              归一化分数 + 人类相关性验证 (τ/ρ)
```

## 核心模块与公式推导

### 模块 1: CLIP-Score / ViCLIP-Score 基线公式（框架图：传统评估路径）

**直觉**: 现有主流方法通过计算文本与视觉特征的余弦相似度来度量对齐程度，是本文所有改进的出发点。

**Baseline 公式** (CLIP-Score [15]):
$$\text{CLIP-Score}(t, v) = \frac{E_{\text{text}}(t) \cdot E_{\text{image}}(v)}{\|E_{\text{text}}(t)\| \|E_{\text{image}}(v)\|}$$

符号: $t$ = 文本提示, $v$ = 视频帧, $E_{\text{text}}$ = CLIP文本编码器, $E_{\text{image}}$ = CLIP图像编码器。视频分数通过对所有帧取平均得到。

**Baseline 公式** (ViCLIP-Score [21]):
$$\text{ViCLIP-Score}(t, v) = \frac{E_{\text{text}}(t) \cdot E_{\text{video}}(v)}{\|E_{\text{text}}(t)\| \|E_{\text{video}}(v)\|}$$

符号: $E_{\text{video}}$ = ViCLIP视频编码器，提取时序聚合的视频级特征。

**变化点**: 这两个基线将视频压缩为单一向量（帧平均或视频编码），丢失了组合性所需的细粒度信息——无法区分"红球在蓝盒左边"和"蓝球在红盒左边"的分数差异，也无法捕捉动态变化。

---

### 模块 2: D-LLaVA 动态属性绑定评估（框架图：动态属性分支）

**直觉**: 动态属性绑定要求评估物体属性随时间的变化（如"苹果从绿变红"），需要显式对比帧间差异而非全局平均。

**Baseline 不足**: CLIP-Score 帧平均会抹平时序变化信息，绿→红与红→绿可能得到相同分数。

**本文公式（推导）**:
$$\text{Step 1}: \{f_1, f_2, ..., f_n\} = \text{SampleFrames}(v, n) \quad \text{均匀采样n帧以覆盖时序}$$
$$\text{Step 2}: \{c_1, c_2, ..., c_n\} = \text{D-LLaVA}(\{f_i\}, Q_{\text{dynamic}}) \quad \text{每帧回答动态属性问题，得到属性描述}$$
$$\text{Step 3}: \text{Score} = \text{Match}(\{c_i\}, t_{\text{dynamic}}) \quad \text{对比帧间属性变化序列与文本描述的一致性}$$

符号: $f_i$ = 第i采样帧, $Q_{\text{dynamic}}$ = 针对动态属性的结构化查询（如"这帧中苹果是什么颜色？"）, $t_{\text{dynamic}}$ = 提示中的动态属性描述。

**核心设计**: D-LLaVA 作为 image LLM，对每帧独立进行视觉问答，通过对比多帧答案的演变来验证动态绑定，而非依赖视频级全局特征。

**对应消融**: 使用 CLIP-Score 替代 D-LLaVA 评估 dynamic attribute binding 时，人类相关性显著下降（具体Δ值。

---

### 模块 3: Grid-LLaVA 视频级组合评估（框架图：属性/动作/交互分支）

**直觉**: 一致属性绑定、动作绑定和物体交互需要理解视频全局内容，但传统 video LLM 直接处理长视频存在稳定性问题，需通过网格化帧采样提升可靠性。

**Baseline 不足**: 直接将整个视频输入 PLLaVA 等 video LLM 会导致帧间信息淹没，且计算不稳定；CLIP-Score 更无法处理"A把B递给C"这类需要理解完整事件序列的交互。

**本文公式（推导）**:
$$\text{Step 1}: G = \text{GridSample}(v, k \times k) \quad \text{将视频帧排列为} k \times k \text{网格图像}$$
$$\text{Step 2}: \text{response} = \text{Grid-LLaVA}(G, Q_{\text{category}}) \quad \text{对网格图像进行单一VQA推理}$$
$$\text{Step 3}: \text{Score} = \text{Parse}(\text{response}) \in \{0, 1\} \text{或} [0,1] \quad \text{解析答案为二元或连续分数}$$

符号: $G$ = 网格化视觉输入, $Q_{\text{category}}$ = 类别特定问题模板（如"视频中是否始终有[属性]的[物体]？""[物体A]是否将[物体B]传递给[物体C]？"）。

**核心设计**: Grid-LLaVA 将时序信息空间化为网格布局，使 image LLM（LLaVA）能够"一眼看尽"视频关键帧，既保留时序概览又避免长视频处理的稳定性问题。Figure 11 展示了 Grid-LLaVA 随采样帧数增加的稳定性优于直接帧堆叠的 D-LLaVA（Figure 12）。

**对应消融**: Table 1 显示 Grid-LLaVA 在 consistent attribute binding、action binding、object interactions 上取得最高 Kendall's τ，替换为 CLIP-Score 或 ViCLIP 后相关性显著降低（具体Δ值。

## 实验与分析



本文在自建的 T2V-CompBench 上进行实验验证，核心结论来自两方面：自动指标与人类判断的相关性验证，以及6个主流T2V模型的组合能力诊断。

在**人工相关性验证**方面，作者通过 Amazon Mechanical Turk 收集了651个视频的人类标注，计算各自动指标与人工评分的 Kendall's τ 和 Spearman's ρ。结果显示：Grid-LLaVA 在 consistent attribute binding、action binding、object interactions 三类上相关性最高；D-LLaVA 专为 dynamic attribute binding 设计，相关性优于所有基线；G-Dino 在 spatial relationships 和 generative numeracy 上借助 GroundingDINO 的检测框推理能力取得最优；DOT 通过跟踪运动轨迹在 motion binding 上最可靠。相比之下，CLIP-Score、ViCLIP 等统一指标在所有7类上均未能取得最优，验证了「分而治之」策略的必要性。



在**模型诊断**方面，Table 2 展示了6个T2V模型（VideoCrafter、ModelScope、AnimateDiff、LaVie、Show-1、T2VZero）在7类上的归一化分数。各模型表现差异显著：部分模型在 spatial relationships 上得分较高但在 motion binding 上表现薄弱，另一些则相反。这种细粒度诊断是传统单一分数无法提供的。Table 4 进一步展示子维度评估结果，揭示模型在具体能力上的短板。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5895ed36-56f5-44ea-a4c0-c0e0e19dadba/figures/fig_003.png)
*Figure: Illustration of prompt categories. We show the number*



**消融分析**：去除类别专用指标选择、统一使用 CLIP-Score 或 ViCLIP 时，所有组合类别的人工相关性均下降。其中 dynamic attribute binding 和 motion binding 受影响最大，因为帧平均机制完全丢失了时序变化信息（具体Δ值。

**公平性检查**：本文比较的基线覆盖了T2V评估的主流方法（CLIP系列、BLIP系列、ViCLIP、VPEval-S、M-GDino），但缺少 FVD、IS 等视频质量指标以及2024年后新提出的视频专用指标。人工评估仅覆盖6个模型和651视频，样本量对稳健相关性估计可能偏紧。Amazon Mechanical Turk 标注质量存在个体差异，作者未报告标注者间一致性（IRR）数据。Figure 7 披露了 object interactions 类别中部分模型的典型失败案例。

## 方法谱系与知识库定位

T2V-CompBench 属于**文本到视频生成评估**方法族，其直接技术源头是T2I组合性评估（如T2I-CompBench、VPEval）向视频领域的扩展，但进行了根本性重构——从简单迁移到视频专用的多维度指标设计。

**直接基线与差异**：
- **CLIP-Score / ViCLIP**：本文保留其作为通用对齐基线，但证明其无法处理组合性，以之为反面教材推动类别专用化
- **VPEval-S / M-GDino**：继承其检测/跟踪思想，但 G-Dino 扩展至 numeracy，DOT 重新设计为通用 motion binding 指标而非仅方向判断
- **BLIP系列**：彻底放弃再描述范式，转向 LLM-based 推理（Grid-LLaVA/D-LLaVA）以处理复杂交互

**改动槽位**：数据流程（7类结构化提示替代通用提示）、目标函数（5类专用指标替代统一相似度）、推理策略（检测/跟踪/LLM按需路由替代帧平均）。

**后续方向**：(1) 将 Grid-LLaVA/D-LLaVA 升级为原生 video LLM（如 GPT-4V、Gemini）以提升复杂推理能力；(2) 扩展至长视频（>10秒）和更细粒度的时间定位评估；(3) 结合本文指标作为奖励信号，直接优化T2V模型的组合生成能力。

**知识库标签**：模态(video) / 范式(基准评测, evaluation benchmark) / 场景(文本到视频生成) / 机制(组合性分解, 类别专用指标路由, 检测+跟踪+LLM混合评估) / 约束(人工相关性验证, 细粒度诊断)

