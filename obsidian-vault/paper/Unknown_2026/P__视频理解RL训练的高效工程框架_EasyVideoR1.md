---
title: 'EasyVideoR1: Easier RL for Video Understanding'
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.16893
aliases:
- 视频理解RL训练的高效工程框架
- EasyVideoR1
- 视频RL训练的主要瓶颈不在算法层面
method: EasyVideoR1
modalities:
- Image
paradigm: Reinforcement Learning
---

# EasyVideoR1: Easier RL for Video Understanding

[Paper](https://arxiv.org/abs/2604.16893)

**Topics**: [[T__Video_Understanding]], [[T__Reinforcement_Learning]], [[T__Visual_Reasoning]] | **Method**: [[M__EasyVideoR1]]

> [!tip] 核心洞察
> 视频RL训练的主要瓶颈不在算法层面，而在工程层面——冗余的视频解码和缺乏视频感知的评估基础设施。通过将视频预处理从训练热路径中剥离（离线缓存），并为视频模态的异构任务类型构建统一的奖励路由系统，可以在不改变核心RL算法的前提下显著提升训练效率和任务覆盖范围。本质上，这是一个「把视频当一等公民」的工程框架重构，而非算法创新。

| 中文题名 | 视频理解RL训练的高效工程框架 |
| 英文题名 | EasyVideoR1: Easier RL for Video Understanding |
| 会议/期刊 | arXiv preprint (2026) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.16893) · [Code](https://github.com/） · [Project](https://arxiv.org/abs/2604.16893) |
| 主要任务 | 视频理解（多选题、OCR、时序定位、空间定位、目标跟踪、密集分割等11种任务类型） |
| 主要 baseline | Qwen3-VL-8B-Instruct, Qwen3-VL-8B-Think, EasyR1 |

> [!abstract] 因为「视频RL训练中每个视频被冗余解码3N次、缺乏视频感知的评估基础设施、异构任务奖励难以统一」，作者在「EasyR1」基础上改了「离线张量缓存机制 + 任务感知奖励路由 + 混合离线-在线训练 + 联合图像-视频训练 + 异步多基准评估」，在「10个视频理解基准」上取得「平均准确率从62.1提升至64.4（+2.3），推理密集型任务VideoMathQA +6.7 / Video-Holmes +6.6，训练吞吐量提升1.47×」

- **精度提升**: 10基准平均 +2.3，VideoMathQA +6.7，Video-Holmes +6.6，Video-MME +2.1，MVBench +3.5
- **效率提升**: 32×H200上每步时间从194s降至132s，token吞吐797→1175 tok/s，Rollout阶段1.5×加速，参考模型前向2.9×加速
- **覆盖范围**: 支持11种任务类型、22个评估基准、混合图像-视频批次训练

## 背景与动机

将大语言模型中验证有效的可验证奖励强化学习（RLVR）扩展至视频理解任务，面临一个根本性的工程困境：视频不是图像的简单序列，其高维数据特性使得训练流程中的每一个环节都成为瓶颈。以时序事件定位任务为例，模型需要从数分钟的视频中精确识别事件起止时间；以密集像素级分割为例，输出需要与帧级mask进行IoU计算——这些异构任务对奖励函数、输入格式、评估协议的要求截然不同，而现有框架并未为视频模态做专门设计。

现有开源RL训练框架主要沿两条路线发展。**veRL** 提供通用的分布式RL基础设施，但完全未针对视频模态优化，视频解码由用户自行处理。**EasyR1** 在veRL基础上简化了GRPO等算法的实现，支持图像-文本多模态训练，但其数据加载、Rollout生成、Actor训练三阶段各自独立处理原始视频文件，未考虑跨阶段冗余。**OneThinker** 同样面向通用多模态推理，采用在线解码策略，在视频场景下同样面临重复解码问题。这些框架的共同假设是：视频预处理代价可以忽略，或可通过简单缓存解决——这与视频数据的高计算开销现实严重不符。

具体而言，现有框架存在三重缺陷：**第一**，视频任务类型极度多样（多选题、OCR、时序定位、空间定位、跟踪、分割等），奖励设计缺乏统一路由，研究者需为每种任务重复实现评分逻辑；**第二**，帧采样、缩放、归一化等预处理在数据集处理、Rollout生成、Actor训练三阶段各自独立执行，每个视频被冗余解码最多3N次（N为训练步数），严重拖慢训练吞吐量；**第三**，视频基准评估对帧采样策略、最大视觉token预算、FPS、分辨率、提示模板等超参数极为敏感，现有框架缺乏能忠实复现官方精度的评估代码，导致基线精度被系统性低估，RL增益难以准确衡量。

EasyVideoR1的核心判断是：视频RL训练的主要瓶颈不在算法层面，而在工程层面——需要将视频作为「一等公民」进行框架级重构。

## 核心创新

**核心洞察：将视频预处理从训练热路径中剥离并通过统一路由接口处理异构任务奖励，因为视频解码的冗余开销和任务碎片化是现有框架无法扩展的根本约束，从而使在不改变核心RL算法的前提下显著提升训练效率和任务覆盖范围成为可能。**

本质上，这不是算法创新，而是一个「把视频当一等公民」的工程框架重构：通过离线缓存消除跨阶段冗余解码，通过任务感知奖励系统统一异构评分逻辑，通过混合离线-在线训练平衡探索与质量，通过联合图像-视频训练扩展数据多样性，通过异步评估确保结果可信。

| 维度 | Baseline (EasyR1) | 本文 (EasyVideoR1) |
|:---|:---|:---|
| 视频解码 | 三阶段各自在线解码，每视频3N次 | 离线预处理为.pt缓存，每视频1次 |
| 任务奖励 | 需用户自行实现每种任务评分 | 11种任务类型统一路由，模块化扩展 |
| 数据来源 | 纯在线rollout采样 | 混合预收集离线轨迹 + 在线rollout |
| 模态支持 | 图像-文本为主，视频为兼容模式 | 图像与视频同批次混合，独立像素预算 |
| 评估框架 | 无标准化视频基准评估 | 22基准异步评估，复现官方精度 |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/aeff065d-4103-44e3-ba5a-92bd13a64088/figures/Figure_1.png)
*Figure 1 (pipeline): Overview of the EasyVideoR1 training pipeline. Videos are preprocessed offline into .pt cache files. During training, each worker loads cached frames locally.*



EasyVideoR1的训练流程可分为五个核心阶段，数据流向如下：

**输入端**：原始视频数据集（约100K视频样本，来源于OneThinker、Video-R1、VideoChat-R1）→ **离线预处理模块** → 输出为.pt格式的张量缓存文件（含已采样帧、已缩放归一化的像素张量）。此步骤在训练前一次性完成，将视频从可变长原始格式转为固定结构张量。

**训练循环**：缓存文件 → **数据加载器**（直接从.pt读取，跳过解码）→ **混合批次构造器**（同一batch内混合静态图像与视频片段，视频每帧262,144像素，图像1,048,576像素，独立预算配置）→ **GRPO训练引擎**（采用DAPO非对称裁剪变体，ε_low=0.2，ε_high=0.28，禁用KL惩罚，rollout组大小n=8，全局批大小256）→ **Actor模型更新**。

**奖励计算分支**：模型输出 → **任务感知奖励路由器**（根据问题类型自动分发至对应评分模块，覆盖多选题、OCR、时序事件定位、空间定位、目标跟踪、密集像素级分割等11种类型）→ **准确率/IoU/编辑距离等量化奖励** → 回传至GRPO优化器。

**数据过滤分支**：在线rollout输出 → **pass-rate过滤器**（k=8次rollout，仅保留0 < pass rate < 1的样本，即有一定难度但非完全不可解的样本）→ 筛选后样本进入训练循环。

**评估分支**：训练后模型 → **异步多基准评估框架**（基于vLLM AsyncLLMEngine，贪心解码，覆盖22个视频理解基准）→ 精度报告。

```
原始视频 → [离线预处理] → .pt缓存 → [数据加载] → 混合批次(图像+视频)
                                              ↓
                    [任务感知奖励] ← 模型输出 ← [GRPO引擎] → Actor更新
                          ↑                      ↑
                    [pass-rate过滤] ← [在线rollout + 离线轨迹]
                                              ↓
                                    [异步评估: 22基准]
```

## 核心模块与公式推导

### 模块 1: 离线张量缓存与冗余消除（对应框架图 左侧输入端）

**直觉**: 视频解码是训练热路径上的纯计算开销，不应在每次训练迭代中重复执行。

**Baseline 流程 (EasyR1/OneThinker)**: 三阶段独立在线解码
$$T_{\text{base}} = N \times (t_{\text{data}} + t_{\text{rollout}} + t_{\text{actor}}) = N \times 3 \times t_{\text{decode}} + t_{\text{other}}$$
其中 $N$ = 训练步数，$t_{\text{decode}}$ = 单视频完整解码+预处理时间，三阶段各自独立解码同一视频。

**变化点**: 帧采样（2FPS，最多128帧）、缩放、归一化等操作具有确定性，结果与训练动态无关，可完全前置。原始流程中每个视频被解码 $3N$ 次，其中仅第1次必要，其余 $3N-1$ 次为冗余。

**本文公式（推导）**:
$$\text{Step 1}: \quad V_{\text{raw}} \text{xrightarrow}{\text{sample}(\text{FPS}=2, \max=128)} F \in \mathbb{R}^{T \times H \times W \times 3} \quad \text{确定性格式化采样，固定时序分辨率}$$
$$\text{Step 2}: \quad F \text{xrightarrow}{\text{resize, normalize}} \tilde{F} \in \mathbb{R}^{T \times h \times w \times 3} \quad \text{像素级预处理，输出标准化张量}$$
$$\text{Step 3}: \quad \tilde{F} \text{xrightarrow}{\text{torch.save}} \text{cache}_i.pt \quad \text{持久化缓存，训练时直接内存映射加载}$$
$$\text{最终}: \quad T_{\text{ours}} = 1 \times t_{\text{decode}} + N \times t_{\text{load}}^{\text{(memmap)}} + t_{\text{other}}, \quad t_{\text{load}} \ll t_{\text{decode}}$$

**对应消融**: 32×H200受控实验中，每步时间从194s降至132s（1.47×），token吞吐797→1175 tok/s；Rollout生成阶段加速1.5×，参考模型前向加速2.9×，Actor更新阶段时间不变（验证了解码瓶颈定位准确）。

---

### 模块 2: 任务感知奖励路由与GRPO训练目标（对应框架图 中部训练引擎）

**直觉**: 视频理解任务的答案形式高度异构（选项字母/时间戳/边界框/掩码/轨迹），需统一接口将不同格式转化为标量奖励，才能接入标准RL优化。

**Baseline 公式 (标准GRPO/EasyR1)**: 
$$L_{\text{GRPO}} = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(O|q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left( r_{i,t} \hat{A}_{i,t}, \text{clip}(r_{i,t}, 1-\varepsilon, 1+\varepsilon) \hat{A}_{i,t} \right) \right]$$
其中 $r_{i,t} = \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})}$ 为重要性采样比，$\hat{A}_{i,t}$ 为组内相对优势（基于组内奖励归一化），$G$ 为组大小。

**变化点**: 标准GRPO假设奖励 $R(o_i)$ 为标量且计算方式统一。视频任务中，$R$ 的实现因任务类型 $c \in \mathcal{C}$（$|\mathcal{C}|=11$）而异：多选题需精确匹配选项字母，OCR需编辑距离，时序定位需IoU或编辑距离容忍，空间定位需IoU，跟踪需MOTA类指标，分割需mIoU。此外，DAPO发现对称裁剪限制过强，采用非对称裁剪扩展探索空间。

**本文公式（推导）**:
$$\text{Step 1}: \quad c = \text{Router}(q), \quad c \in \{\text{MCQ, OCR, TempLoc, SpatialLoc, Track, Seg, ...}\} \quad \text{任务类型自动识别}$$
$$\text{Step 2}: \quad R(o_i) = R_c(o_i, o_i^*) \in [0, 1] \quad \text{任务专属评分函数，输出归一化奖励}$$
$$\text{其中}: R_{\text{MCQ}} = \mathbb{1}[o_i = o_i^*], \quad R_{\text{TempLoc}} = \text{EditDistanceTol}(o_i, o_i^*), \quad R_{\text{Seg}} = \text{mIoU}(o_i, o_i^*)$$
$$\text{Step 3}: \quad \hat{A}_{i,t} = \frac{R(o_i) - \text{mean}(\{R(o_j)\}_{j=1}^G)}{\text{std}(\{R(o_j)\}_{j=1}^G)} \quad \text{组内归一化优势（标准GRPO）}$$
$$\text{Step 4}: \quad L_{\text{DAPO}} = \mathbb{E}\left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left( r_{i,t} \hat{A}_{i,t}, \text{clip}(r_{i,t}, 1-\varepsilon_{\text{low}}, 1+\varepsilon_{\text{high}}) \hat{A}_{i,t} \right) \right]$$
$$\text{最终}: \quad L_{\text{final}} = L_{\text{DAPO}} - \lambda_{\text{KL}} \cdot \text{KL}[\pi_\theta \| \pi_{\text{ref}}], \quad \lambda_{\text{KL}} = 0 \text{（显式禁用）}$$

**符号说明**: $\varepsilon_{\text{low}}=0.2$, $\varepsilon_{\text{high}}=0.28$ 为非对称裁剪边界（DAPO变体），$G=8$ 为rollout组大小，全局批大小256，学习率 $1\times 10^{-6}$（AdamW）。

**对应消融**: —— 论文未报告任务路由系统、DAPO非对称裁剪、KL惩罚禁用各组件的独立消融贡献。

---

### 模块 3: 混合离线-在线数据训练（对应框架图 数据过滤分支）

**直觉**: 纯在线rollout对高难度任务采样效率低，预收集的离线轨迹可提供稳定的奖励信号锚点。

**Baseline 流程 (标准在线GRPO)**: 
$$\mathcal{D}_{\text{online}} = \{(q, \{o_i\}_{i=1}^G, \{R(o_i)\}) \text{mid} q \sim P(Q), o_i \sim \pi_{\theta_{\text{old}}}(\cdot|q)\}$$
每步仅从当前策略采样，对pass rate接近0或1的任务无有效梯度信号。

**变化点**: 引入预收集的高质量离线轨迹 $\mathcal{D}_{\text{offline}}$（来自OneThinker、Video-R1、VideoChat-R1的精选数据），与在线数据混合构造每批次训练样本。同时通过pass-rate过滤筛选"可学习"样本：完全已掌握（pass rate=1）和完全未掌握（pass rate=0）的样本被丢弃。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathcal{D}_{\text{offline}} = \{(q, o^*, R(o^*))\}_{\text{curated}} \quad \text{预收集专家/高质量轨迹}$$
$$\text{Step 2}: \quad \text{For each } q: \quad \{o_i\}_{i=1}^k \sim \pi_{\theta_{\text{old}}}(\cdot|q), \quad \text{pass-rate}(q) = \frac{1}{k}\sum_{i=1}^k \mathbb{1}[R(o_i)=1]$$
$$\text{Step 3}: \quad \mathcal{D}_{\text{online}}^{\text{filtered}} = \{(q, \{o_i\}) \text{mid} 0 < \text{pass-rate}(q) < 1\} \quad \text{保留有挑战性的样本，} k=8$$
$$\text{最终}: \quad \mathcal{D}_{\text{batch}} = \text{Mix}(\mathcal{D}_{\text{offline}}, \mathcal{D}_{\text{online}}^{\text{filtered}}) \text{xrightarrow}{\text{GRPO}} \Delta\theta$$

**对应消融**: —— 论文未报告混合比例、离线数据独立贡献、pass-rate阈值 $k=8$ 的消融验证。

## 实验与分析


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/aeff065d-4103-44e3-ba5a-92bd13a64088/figures/Figure_2.png)
*Figure 2 (result): Benchmark performance comparison. The number above each blue bar indicates the accuracy change relative to the Instruct baseline.*



**主实验结果**（Qwen3-VL-8B-Instruct为基础模型，200步GRPO训练）：

| Method | Video-MME | MVBench | VideoMathQA | Video-Holmes | TempCompass | MLVU | Video-MMMU | 10基准平均 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| Qwen3-VL-8B-Instruct | 62.1基线 | 62.1基线 | 62.1基线 | 62.1基线 | 62.1基线 | 62.1基线 | 62.1基线 | 62.1 |
| + EasyVideoR1 (200步) | +2.1 | +3.5 | +6.7 | +6.6 | -0.3 | -0.6 | -1.7 | +2.3 |
| Qwen3-VL-8B-Think | | | | | | | | |

（注：原始报告以Instruct为统一基线给出变化量，具体各基准绝对值未完整披露）

**核心结论分析**：
- **支持核心 claim 的数据**：推理密集型任务提升最显著——VideoMathQA +6.7、Video-Holmes +6.6，表明RLVR对需要多步推理的视频任务确实有效；通用理解基准Video-MME +2.1、MVBench +3.5验证了一致正向收益；10基准平均+2.3在200步短训练周期内具有实际意义。
- **边际/负面结果**：TempCompass -0.3、MLVU -0.6、Video-MMMU -1.7出现负向变化，说明RL训练收益分布不均匀，可能与任务类型与奖励设计的匹配度有关（Video-MMMU为知识密集型，非纯推理）。
- **效率验证**：
![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/aeff065d-4103-44e3-ba5a-92bd13a64088/figures/Figure_3.png)
*Figure 3 (comparison): Training efficiency comparison between cache-based loading and on-the-fly video decoding.*

 显示缓存机制带来1.47×吞吐量提升，其中Rollout生成1.5×、参考模型前向2.9×加速，Actor更新无变化——精准定位了解码瓶颈，而非虚假的全流程加速。

**消融与公平性检查**：
- **缺失消融**：混合离线-在线训练的独立收益、联合图像-视频训练的互增强效果、pass-rate阈值 $k=8$ 的选择、任务路由系统的模块化贡献均无实验验证，停留在设计描述层面。
- **基线强度**：仅与Instruct和Think对比，未与同期视频RL工作（Video-R1、VideoChat-R1训练的模型）直接对比，无法判断+2.3在领域内的相对位置。
- **计算成本**：32×H200集群，200步训练，约100K视频样本——资源门槛较高，但效率优化降低了同等精度下的时间成本。
- **评估可信度**：声称"复现精度与官方报告高度一致"，但缺乏完整的官方vs.本文评估结果对比表格支撑该 claim 的量化程度。

## 方法谱系与知识库定位

**方法家族**: 多模态大模型后训练（视觉-语言模型 + RLVR）

**父方法**: **EasyR1** —— 本文明确声明在其基础上构建，继承了其GRPO实现和分布式训练基础设施。核心改动是将EasyR1的通用多模态设计专项化为视频一等公民架构。

**改变的插槽**:
- **training_recipe**: 引入混合离线-在线数据流、pass-rate过滤、联合图像-视频批次
- **data_curation**: 离线张量缓存（工程前置）、约100K视频样本汇集（OneThinker/Video-R1/VideoChat-R1）
- **inference/evaluation**: 异步多基准评估框架（22基准，vLLM AsyncLLMEngine）
- **architecture/objective**: 任务感知奖励路由系统（11种类型统一接口）

**直接基线与差异**:
- **EasyR1**: 通用多模态RL框架，视频为兼容模式；本文专项化视频流程，消除冗余解码
- **OneThinker**: 强调思考链生成，在线解码；本文聚焦训练效率，离线缓存+混合数据
- **Video-R1 / VideoChat-R1**: 同期视频RL工作，提供数据源但未直接对比性能；本文定位为基础设施框架而非单点算法

**后续方向**:
1. **算法-工程协同优化**: 当前效率提升与精度提升相对独立，需探索缓存机制是否支持更大规模在线rollout（如 $G>8$）以进一步提升RL信号质量
2. **负向基准诊断**: Video-MMMU等负向变化任务需针对性分析，可能需任务自适应奖励权重或课程学习
3. **闭源模型适配**: 当前基于Qwen3-VL开源模型，框架是否可迁移至GPT-4V等API-based模型的RL微调（需解决缓存与API格式的兼容性）

**知识库标签**: 
- modality: video + image
- paradigm: RLVR / GRPO / post-training
- scenario: video understanding (multi-task)
- mechanism: offline tensor caching / task-aware reward routing / hybrid offline-online training
- constraint: engineering efficiency / reproducible evaluation / heterogeneous task unification

