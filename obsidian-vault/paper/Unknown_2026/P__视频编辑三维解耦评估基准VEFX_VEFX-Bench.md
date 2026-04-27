---
title: 'VEFX-Bench: A Holistic Benchmark for Generic Video Editing and Visual Effects'
type: paper
paper_level: C
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.16272
aliases:
- 视频编辑三维解耦评估基准VEFX-Bench
- VEFX-Bench
- 视频编辑评估的失败根源在于「维度混淆」——将指令执行、渲染质量和内容保
method: VEFX-Bench
modalities:
- Image
- Text
---

# VEFX-Bench: A Holistic Benchmark for Generic Video Editing and Visual Effects

[Paper](https://arxiv.org/abs/2604.16272)

**Topics**: [[T__Benchmark_-_Evaluation]], [[T__Video_Generation]], [[T__Image_Editing]] | **Method**: [[M__VEFX-Bench]]

> [!tip] 核心洞察
> 视频编辑评估的失败根源在于「维度混淆」——将指令执行、渲染质量和内容保留压缩为单一分数，导致评估信号模糊且无法指导优化。VEFX-Reward的有效性来自两个相互强化的设计：一是用三维解耦标签替代单一标量，使模型学习到更精确的质量信号；二是以视频编辑专用数据训练，避免了图像编辑或视频生成模型迁移时的领域不匹配。本质上，这是一个「专用数据+专用标签+专用模型」三位一体的解决方案，而非单一技术创新。

| 中文题名 | 视频编辑三维解耦评估基准VEFX-Bench |
| 英文题名 | VEFX-Bench: A Holistic Benchmark for Generic Video Editing and Visual Effects |
| 会议/期刊 | arXiv 2026 (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.16272) · Code · Project |
| 主要任务 | 视频编辑质量评估、奖励模型训练、编辑系统基准测试 |
| 主要 baseline | EditReward、VE-Bench |

> [!abstract] 因为「视频编辑评估缺乏三维解耦的人工标注数据集和专用奖励模型，现有方法将IF/RQ/EE压缩为单一标量且存在严重领域不匹配（EditReward在RQ维度SRCC=-0.211）」，作者在「VE-Bench、EditReward」基础上改了「构建5,049样本的VEFX-Dataset（三维解耦标注1-4分）+ 基于Qwen3-VL的VEFX-Reward（CORN Loss序数回归）」，在「849样本测试集」上取得「VEFX-Reward-32B整体SRCC 0.780 vs EditReward 0.558 vs VE-Bench 0.214，组内成对准确率0.872」

- **VEFX-Reward-32B整体SRCC 0.780**，超越EditReward 39.8%（相对提升），超越VE-Bench 264.5%
- **组内成对准确率0.872/0.863**（32B/4B），EditReward仅0.792，VE-Bench仅0.665
- **VEFX-Dataset 5,049样本**，覆盖9大类32子类，首个大规模三维解耦（IF/RQ/EE）人工标注视频编辑数据集

## 背景与动机

指令引导的视频编辑已成为AI辅助影视制作的核心环节，但评估体系存在根本性缺口。想象一个场景：用户要求"将视频中的汽车变成红色"，编辑系统A正确变色但留下了明显的伪影，系统B视觉干净却把整辆车替换成了不相关的物体，系统C完美执行但意外改变了背景天空。现有评估方法只能输出"A=3分、B=3分、C=3分"的单一标量，无法区分这三种本质不同的失败模式，更无法指导针对性优化。

现有资源如何应对这一挑战？**EditBoard、FiVE-Bench、IVE-Bench**等基准仅提供编辑指令而无实际编辑输出，无法用于端到端评估；**OpenVE**依赖自动化生成而非人工标注，质量参差不齐；**VE-Bench**虽包含编辑视频和人工分数，但将质量压缩为单一标量，且基于老旧编辑系统构建，难以反映当前SOTA水平；**EditReward**作为专用奖励模型，专注于图像编辑场景，迁移至视频编辑时出现严重领域不匹配——在渲染质量（RQ）维度上甚至呈现负相关（SRCC = -0.211）。

这些方法的共同缺陷在于**维度混淆**：将指令遵循（Instruction Following, IF）、渲染质量（Rendering Quality, RQ）、编辑局部性（Editing Extent, EE）三个相互独立的质量维度压缩为一维信号。单一标量无法捕捉"语义正确但视觉粗糙"或"视觉干净但过度编辑"的复杂情况，导致评估信号模糊、优化方向不明。此外，现有自动评估方法要么依赖昂贵的人工审查，要么使用通用VLM作为评判者，这类模型并非专为视频编辑质量设计，缺乏对"编辑操作"这一核心概念的理解能力。

本文的核心动机正是填补这一空白：构建首个大规模三维解耦标注的视频编辑数据集，并训练专用的视频编辑奖励模型，使评估信号精确、可解释、可优化。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a2bfc7f4-2f2d-492b-b936-96c62bff94bb/figures/Figure_1.png)
*Figure 1: Figure 1 Overview of our framework. We construct VEFX-Dataset, a human-annotated dataset with 5,049 video editingexamples across 9 categories and 32 subcategories, scored along three decoupled dimensi*



## 核心创新

核心洞察：视频编辑质量评估需要同时理解「做了什么」（指令）、「做得怎样」（渲染）和「改了哪里」（局部性），这三个维度需要联合推理而非独立评分，且必须以视频编辑专用数据训练才能避免领域迁移失败——因为通用VLM或图像编辑奖励模型缺乏对视频时序一致性和编辑操作边界的先验知识，导致跨领域时产生系统性偏差。

| 维度 | Baseline (VE-Bench / EditReward) | 本文 (VEFX-Bench) |
|:---|:---|:---|
| 标签设计 | 单一整体标量分数 | IF / RQ / EE 三维解耦，各1-4分 |
| 数据来源 | 老旧编辑系统 / 图像编辑数据集 | 商业+开源+智能体混合生成，覆盖9类32子类 |
| 模型架构 | 通用VLM直接推断 / 图像编辑专用编码器 | Qwen3-VL视频专用，4 FPS采样，奖励token提取 |
| 损失函数 | 直接回归 / 对比学习 | CORN Loss序数回归，建模有序评分条件概率 |
| 评估维度 | 整体相关性 | 维度级SRCC + 组内成对准确率 + 雷达图任务分析 |

与baseline的本质差异：VE-Bench是「无专用模型的通用评估」，EditReward是「有专用模型但跨领域失败」，本文是「专用数据+专用标签+专用模型」三位一体的闭环方案。

## 整体框架



VEFX-Bench的整体框架由三个相互配套的组件构成，形成「数据构建→模型训练→基准测试」的完整闭环：

**输入层**：源视频（经质量过滤：分辨率>720p、>40帧、无场景切换）+ 编辑指令 + 编辑后视频（由商业系统、开源模型、智能体编辑流水线混合生成）。

**VEFX-Dataset（数据构建模块）**：通过多阶段人工标注流程，由训练有素的标注员对每个样本沿IF、RQ、EE三个维度打分。输出为5,049个带三维解耦标签的视频编辑样本，覆盖9大类（如风格迁移、物体替换、背景编辑等）32个子类。

**VEFX-Reward（模型训练模块）**：以Qwen3-VL为骨干网络，实例化为4B和32B两个规模。输入源视频、编辑指令和编辑后视频（以4 FPS采样，最大帧分辨率399,360像素），通过特殊奖励token提取隐藏状态，经共享线性奖励头输出三个维度的质量分数。训练目标采用CORN Loss进行序数回归。

**VEFX-Bench（评估模块）**：包含300个精心筛选的视频-提示对，用于标准化比较不同编辑系统，配合VEFX-Reward作为自动评估器。

数据流示意：
```
原始视频 ──→[质量过滤]──→ 候选视频池 ──→[编辑系统生成]──→ 编辑三元组(源/指令/结果)
                                                              ↓
标注员三维打分(IF/RQ/EE) ←──[多阶段人工标注]←─────────────────┘
         ↓
    VEFX-Dataset (5,049样本)
         ↓
    [Qwen3-VL + 奖励token + CORN Loss]
         ↓
    VEFX-Reward-4B / VEFX-Reward-32B
         ↓
    [300对标准测试集]──→ VEFX-Bench 系统排名与任务分析
```

## 核心模块与公式推导

### 模块 1: 三维解耦标注体系（对应框架图 VEFX-Dataset 部分）

**直觉**: 单一标量无法区分"指令错误但视觉完美"和"指令正确但视觉崩溃"，必须解耦为独立维度才能提供可优化的精确信号。

**Baseline 形式** (VE-Bench): 
$$s_{\text{overall}} \in \mathbb{R}$$
符号: 单一实数，无维度区分。

**变化点**: VE-Bench的整体分数将IF、RQ、EE的信息压缩到一维，导致不同失败模式相互抵消；本文假设三个维度条件独立且各有有序离散分布，评分空间从$\mathbb{R}$扩展为$\{1,2,3,4\}^3$。

**本文设计**:
$$\text{Step 1}: \quad \mathbf{s} = (s_{\text{IF}}, s_{\text{RQ}}, s_{\text{EE}}) \in \{1,2,3,4\}^3$$
$$\text{Step 2}: \quad p(\mathbf{s}) = p(s_{\text{IF}}) \cdot p(s_{\text{RQ}}) \cdot p(s_{\text{EE}}) \quad \text{(维度间假设条件独立)}$$
$$\text{最终}: \quad \text{每个维度} \; s_d \in \{1,2,3,4\}, \; d \in \{\text{IF}, \text{RQ}, \text{EE}\}$$

**对应消融**: Figure 3 Panel (a) 显示IF-RQ-EE的常见分数模式，验证了三维解耦的必要性——三个维度呈现弱相关性，单一标量会丢失关键信息。

---

### 模块 2: CORN Loss 序数回归（对应框架图 VEFX-Reward 训练部分）

**直觉**: 直接回归将有序标签视为等距实数（1和2的差距等于2和3），忽略了评分的有序离散本质；分类则丢弃了顺序信息。CORN通过条件概率链同时保留有序性和离散性。

**Baseline 公式** (直接回归):
$$L_{\text{MSE}} = \frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2$$
符号: $\hat{y}_i \in \mathbb{R}$为预测分数，$y_i \in \{1,2,3,4\}$为真实标签。

**变化点**: MSE假设标签间隔相等且支持实数域，但人工评分是主观有序的离散值（4分"优秀"与3分"良好"的差距未必等于2分"一般"与1分"差"）；本文改用条件有序回归，将$P(Y=k)$分解为条件概率的乘积，每个条件概率用sigmoid二分类建模。

**本文公式（推导）**:
$$\text{Step 1}: \quad P(Y > k \text{mid} \mathbf{x}) = \sigma(f_k(\mathbf{x})), \quad k \in \{1,2,3\}$$
$$\text{加入了K-1个二分类器以建模累积概率，保留有序结构}$$

$$\text{Step 2}: \quad P(Y = k \text{mid} \mathbf{x}) = \begin{cases} 1 - P(Y > 1 \text{mid} \mathbf{x}) & k=1 \\ P(Y > k-1 \text{mid} \mathbf{x}) - P(Y > k \text{mid} \mathbf{x}) & 1 < k < K \\ P(Y > K-1 \text{mid} \mathbf{x}) & k=K \end{cases}$$
$$\text{重归一化以保证概率和为1且有序约束自动满足}$$

$$\text{Step 3}: \quad L_{\text{CORN}} = -\sum_{i=1}^{N}\sum_{k=1}^{K-1}\left[\mathbb{1}_{[y_i > k]}\log P(Y_i > k) + \mathbb{1}_{[y_i \leq k]}\log(1 - P(Y_i > k))\right]$$
$$\text{最终}: \quad L_{\text{final}} = L_{\text{CORN}}^{\text{IF}} + L_{\text{CORN}}^{\text{RQ}} + L_{\text{CORN}}^{\text{EE}}$$

符号: $\mathbf{x}$为(源视频, 指令, 编辑视频)的联合表示；$f_k(\cdot)$为共享骨干上的第k个线性头；$\sigma(\cdot)$为sigmoid；$\mathbb{1}_{[\cdot]}$为指示函数；三个维度损失相加。

**对应消融**: 

---

### 模块 3: 奖励Token提取机制（对应框架图 VEFX-Reward 推理部分）

**直觉**: 通用VLM的[CLS]或平均池化特征包含大量与编辑质量无关的语义信息，需在输出层引入显式的"质量查询"机制以聚焦评估目标。

**Baseline 形式** (标准VLM):
$$\mathbf{h}_{\text{pool}} = \text{Pool}(\text{VLM}(\mathbf{v}_{\text{src}}, \mathbf{t}, \mathbf{v}_{\text{edit}}))$$
$$\hat{s} = \text{MLP}(\mathbf{h}_{\text{pool}})$$
符号: $\mathbf{v}_{\text{src}}, \mathbf{v}_{\text{edit}}$为源/编辑视频帧序列，$\mathbf{t}$为文本指令。

**变化点**: 标准池化丢失细粒度编辑定位信息；本文在Qwen3-VL的输出序列中插入特殊奖励token（类似BERT的[MASK]或T5的sentinel token），以其隐藏状态作为质量感知的显式表示。

**本文公式**:
$$\text{Step 1}: \quad \mathbf{H} = \text{Qwen3-VL}([\mathbf{v}_{\text{src}}^{4\text{FPS}}; \mathbf{t}; \mathbf{v}_{\text{edit}}^{4\text{FPS}}]) \in \mathbb{R}^{L \times d}$$
$$\text{4 FPS降采样以平衡时序覆盖与计算效率}$$

$$\text{Step 2}: \quad \mathbf{h}_{\text{reward}} = \mathbf{H}[\text{pos}(\texttt{<reward>})] \in \mathbb{R}^{d}$$
$$\text{提取特殊token位置对应的隐藏状态作为质量嵌入}$$

$$\text{Step 3}: \quad \hat{\mathbf{s}} = \mathbf{W} \cdot \mathbf{h}_{\text{reward}} + \mathbf{b}, \quad \mathbf{W} \in \mathbb{R}^{3 \times d}, \mathbf{b} \in \mathbb{R}^{3}$$
$$\text{最终}: \quad \hat{\mathbf{s}} = (\hat{s}_{\text{IF}}, \hat{s}_{\text{RQ}}, \hat{s}_{\text{EE}}) \in \mathbb{R}^3 \text{xrightarrow}{\text{CORN解码}} \{1,2,3,4\}^3$$

**对应消融**: VEFX-Reward-4B在RQ维度略优于32B，表明渲染质量预测在较小规模已接近饱和，暗示奖励token机制的有效性对模型规模不敏感。

## 实验与分析

主实验结果（849样本测试集，SRCC/KRCC/PLCC/RMSE）：

| Method | Overall SRCC | Overall KRCC | Overall PLCC | Overall RMSE | 组内成对准确率 |
|:---|:---|:---|:---|:---|:---|
| VE-Bench | 0.214 |  |  |  | 0.665 |
| EditReward | 0.558 |  |  |  | 0.792 |
| **VEFX-Reward-4B** |  |  |  |  | **0.863** |
| **VEFX-Reward-32B** | **0.780** |  |  |  | **0.872** |


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a2bfc7f4-2f2d-492b-b936-96c62bff94bb/figures/Figure_4.png)
*Figure 4: Figure 4 Predicted overall scores versus human overall scores for VEFX-Reward-32B, EditReward, and VE-Bench. Herethe human overall score is defined as the mean of IF, RQ, and EE. VEFX-Reward-32B exhib*



核心结论支撑：VEFX-Reward-32B的Overall SRCC 0.780显著超越EditReward（0.558）39.8%，超越VE-Bench（0.214）264.5%，验证了三维解耦+专用数据+专用模型的核心假设。组内成对准确率0.872/0.863表明模型能可靠区分同一组内的质量高低，这对偏好优化训练至关重要。

维度级分析（Figure 4散点图）：VEFX-Reward-32B预测分数与人类分数呈现紧密单调关系，而EditReward在RQ维度出现负相关（SRCC=-0.211），直观展示了领域不匹配的严重后果。Figure 5的软分数分布显示不同编辑系统在三维空间中的差异化表现——商业系统可能在RQ上集中但IF分散，开源模型可能呈现相反模式。

任务级分析（Figure 6雷达图）：
![Figure 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a2bfc7f4-2f2d-492b-b936-96c62bff94bb/figures/Figure_6.png)
*Figure 6: Figure 6 Task-wise Overall (GeoAgg) profiles of the benchmarked video editing systems. Each radar plot uses the sameradial scale, allowing the profile shape and absolute score level of each model to b*

 9大类任务的Overall (GeoAgg)轮廓揭示各系统的相对优劣势，允许研究者识别特定任务类型的瓶颈。

消融与边际分析：VEFX-Reward-4B与32B性能接近（成对准确率0.863 vs 0.872），提升边际仅0.9%，暗示4B规模已覆盖大部分可学习信号；32B在整体SRCC上的优势可能来自更好的IF推理能力。CORN Loss的具体消融数据未披露。

公平性检查：
- **基线强度**：奖励模型基线仅含EditReward和VE-Bench，未纳入VideoScore、VideoReward等视频生成奖励模型，竞争强度可能低估；
- **计算成本**：32B模型推理需处理4 FPS视频帧，最大帧分辨率399,360像素，计算开销显著高于轻量级基线；
- **数据成本**：5,049样本的人工三维标注成本高昂，但论文未披露具体预算和标注周期；
- **失败案例**：跨域泛化未验证，所有评估均在同一数据集分布上进行；CORN Loss对标注员一致性敏感，极端评分分布（如大量4分）可能削弱条件概率建模优势。

## 方法谱系与知识库定位

**方法家族**: 视频质量评估（VQA）→ 视频编辑质量评估 → 奖励模型用于生成内容评估

**父方法**: Qwen3-VL（多模态大语言模型）+ CORN Loss（序数回归，源自一般有序分类文献）

**改动插槽**: 
- **数据策展（data_curation）**: 全新构建，从0到5,049样本的三维解耦人工标注数据集
- **架构（architecture）**: Qwen3-VL + 特殊奖励token + 共享线性奖励头，非原生设计
- **目标函数（objective）**: CORN Loss适配三维输出，损失空间从单维扩展到$\mathbb{R}^3$
- **训练配方（training_recipe）**: 视频编辑专用微调，区别于通用VLM或图像编辑预训练
- **推理（inference）**: 4 FPS采样+帧分辨率限制，针对长视频效率优化

**直接基线与差异**: 
- **VE-Bench**: 通用VLM无微调评估 → 本文专用模型+专用数据，SRCC从0.214→0.780
- **EditReward**: 图像编辑奖励模型跨域应用 → 本文原生视频编辑设计，解决RQ维度负相关问题
- **OpenVE/IVE-Bench等**: 无编辑输出或无人工标注 → 本文提供完整三元组+三维标签

**后续方向**:
1. **跨域验证**: 在VEFX-Dataset分布外测试泛化，特别是向电影级长视频（>1分钟）扩展
2. **偏好优化闭环**: 将VEFX-Reward作为奖励函数，直接训练视频编辑模型（类似RLHF/DP0流程）
3. **实时评估**: 当前4 FPS采样和32B规模仍较重，探索蒸馏至更小模型或帧级早期退出机制

**标签**: 
- **模态（modality）**: 视频-文本联合
- **范式（paradigm）**: 监督学习（序数回归）
- **场景（scenario）**: 视频编辑质量评估、AIGC内容审核
- **机制（mechanism）**: 三维解耦标注、特殊token特征提取、条件有序回归
- **约束（constraint）**: 人工标注成本高、推理计算量大、跨域泛化未验证

