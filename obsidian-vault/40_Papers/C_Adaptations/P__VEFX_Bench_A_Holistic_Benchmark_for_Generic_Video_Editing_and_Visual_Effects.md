---
title: 'VEFX-Bench: A Holistic Benchmark for Generic Video Editing and Visual Effects'
type: paper
paper_level: C
venue: ''
year: 2026
acceptance: null
cited_by: null
facets:
  modality:
  - Image
  - Text
  domain:
  - Agent
core_operator: 视频编辑评估的失败根源在于「维度混淆」——将指令执行、渲染质量和内容保留压缩为单一分数，导致评估信号模糊且无法指导优化。VEFX-Reward的有效性来自两个相互强化的设计：一是用三维解耦标签替代单一标量，使模型学习到更精确的质量信号；二是以视频编辑专用数据训练，避免了图像编辑或视频生成模型迁移时的领域不匹配。本质上，这是一个「专用数据+专用标签+专用模型」三位一体的解决方案，而非单一技术创新。
paper_link: https://arxiv.org/abs/2604.16272
structurality_score: 0.25
---

# VEFX-Bench: A Holistic Benchmark for Generic Video Editing and Visual Effects

## Links

- Mechanism: [[C__task_specific_reward_modeling]]

> 视频编辑评估的失败根源在于「维度混淆」——将指令执行、渲染质量和内容保留压缩为单一分数，导致评估信号模糊且无法指导优化。VEFX-Reward的有效性来自两个相互强化的设计：一是用三维解耦标签替代单一标量，使模型学习到更精确的质量信号；二是以视频编辑专用数据训练，避免了图像编辑或视频生成模型迁移时的领域不匹配。本质上，这是一个「专用数据+专用标签+专用模型」三位一体的解决方案，而非单一技术创新。

> **适配/插件型**。可快速浏览，看改了哪个 slot 和效果。

## 核心公式

$$
\text{PairAcc} = \frac{\sum_{g=1}^{G} \sum_{(i,j)\in P_g} \text{Acc}_{ij}}{\sum_{g=1}^{G} |P_g|}
$$

> 定义了组内成对准确率，用于衡量奖励模型在同一源视频和指令下对候选编辑结果的相对排序是否与人类判断一致，是评估局部偏好一致性的核心指标。
> *Slot*: Group-wise preference evaluation metric

## 关键图表

**Table 4**
: Standard IQA/VQA metrics (SRCC, KRCC, PLCC, RMSE) for all methods across IF, RQ, EE, and Overall dimensions
> 证据支持: VEFX-Reward 在标准相关性指标上显著优于先前奖励模型（EditReward、VE-Bench）及多数 VLM-as-Judge 基线，支持其与人类判断更强对齐的核心主张。

**Table 5**
: Group-wise preference evaluation using Pairwise Accuracy across IF, RQ, EE, and Overall
> 证据支持: VEFX-Reward-4B 和 32B 在组内成对准确率上大幅超越 EditReward 和 VE-Bench，验证了模型在局部偏好排序任务中的有效性。

**Figure 4**
: Scatter plots of predicted overall scores vs. human overall scores for VEFX-Reward-32B, EditReward, and VE-Bench
> 证据支持: 可视化地证明 VEFX-Reward-32B 预测分数与人类分数呈紧密单调关系，而 EditReward 非线性压缩、VE-Bench 离散度极大，支持定量结果的可靠性。

**Figure 1 / Table 1**
: Framework overview and dataset comparison table showing VEFX-Dataset properties vs. existing datasets
> 证据支持: VEFX-Dataset 是唯一同时满足包含编辑输出、人工标注质量分、多维度标签三个条件的数据集，支持其作为奖励模型训练基础的必要性主张。

## 详细分析

# VEFX-Bench: A Holistic Benchmark for Generic Video Editing and Visual Effects

## Part I：问题与挑战

指令引导的视频编辑已成为AI辅助影视制作的核心环节，但该领域的评估体系存在根本性缺口。现有评估资源面临三重困境：其一，缺乏包含完整编辑三元组（源视频、编辑指令、编辑结果）的大规模人工标注数据集——EditBoard、FiVE-Bench、IVE-Bench等基准仅提供指令而无编辑输出，OpenVE依赖自动化生成而非人工标注，VE-Bench虽包含编辑视频和人工分数但将质量压缩为单一标量且基于老旧编辑系统构建；其二，现有自动评估方法要么依赖昂贵的人工审查，要么使用通用视觉语言模型（VLM）作为评判者，这类模型并非专为视频编辑质量设计，无法区分指令执行、渲染质量和内容保留三个相互独立的质量维度；其三，现有奖励模型（如EditReward）专注于图像编辑或视频生成质量，迁移至视频编辑场景时出现严重的领域不匹配——EditReward在渲染质量维度上甚至呈现负相关（SRCC = -0.211）。这三重困境共同导致视频编辑系统的系统性基准测试和基于偏好的优化训练均难以推进，是制约该领域发展的核心瓶颈。

## Part II：方法与洞察

本文提出三个相互配套的贡献，共同构建视频编辑评估的完整闭环。

**VEFX-Dataset**：通过多阶段流程构建包含5,049个视频编辑样本的人工标注数据集。原始视频经严格质量过滤（分辨率>720p、>40帧、无场景切换等），覆盖9大类32个子类的编辑任务。编辑结果由商业系统、开源模型和智能体编辑流水线混合生成，确保多样性。每个样本由训练有素的标注员沿三个解耦维度打分：指令遵循（IF）、渲染质量（RQ）和编辑局部性（EE），评分范围1-4分。这种三维解耦设计是核心创新之一——一个编辑结果可能语义错误但视觉干净，或正确执行指令但破坏了未编辑区域，单一标量无法捕捉这种差异。

**VEFX-Reward**：基于VEFX-Dataset训练的专用奖励模型，以Qwen3-VL为骨干网络，实例化为4B和32B两个规模。模型联合输入源视频、编辑指令和编辑后视频（以4 FPS采样，最大帧分辨率399,360像素），通过特殊奖励token提取隐藏状态，经共享线性奖励头输出三个维度的质量分数。训练目标采用序数回归（CORN Loss），利用条件概率建模有序评分分布，比直接回归更适合离散有序标签。

**VEFX-Bench**：包含300个精心筛选的视频-提示对，用于标准化比较不同编辑系统，配合VEFX-Reward作为自动评估器使用。

方法的核心洞察在于：视频编辑质量评估需要同时理解「做了什么」（指令）、「做得怎样」（渲染）和「改了哪里」（局部性），这三个维度需要联合推理而非独立评分，且必须以视频编辑专用数据训练才能避免领域迁移失败。

### 核心直觉

视频编辑评估的失败根源在于「维度混淆」——将指令执行、渲染质量和内容保留压缩为单一分数，导致评估信号模糊且无法指导优化。VEFX-Reward的有效性来自两个相互强化的设计：一是用三维解耦标签替代单一标量，使模型学习到更精确的质量信号；二是以视频编辑专用数据训练，避免了图像编辑或视频生成模型迁移时的领域不匹配。本质上，这是一个「专用数据+专用标签+专用模型」三位一体的解决方案，而非单一技术创新。

## Part III：证据与局限

实验在849样本测试集上进行，覆盖标准IQA/VQA指标（SRCC/KRCC/PLCC/RMSE）和组内成对准确率两类评估协议。

**定量结果**：VEFX-Reward-32B在整体SRCC上达到0.780，显著优于EditReward（0.558）和VE-Bench（0.214）；VEFX-Reward-4B在RQ维度略优于32B，表明渲染质量预测在4B规模已接近饱和。在组内成对准确率上，VEFX-Reward-32B达到0.872，VEFX-Reward-4B达到0.863，均大幅超越EditReward（0.792）和VE-Bench（0.665）。散点图分析（Figure 4）直观展示了VEFX-Reward-32B预测分数与人类分数的紧密单调关系。

**局限性**：（1）奖励模型基线仅包含EditReward和VE-Bench，VideoScore、VideoReward等视频生成奖励模型未纳入对比，可能低估竞争强度；（2）所有评估均在同一数据集分布上进行，跨域泛化能力未经验证；（3）关于「揭示商业与开源系统差距」的结论在论文提供的文本片段中缺乏具体数值支撑；（4）VEFX-Dataset的数据集比较基于论文写作时的已知资源，可能遗漏同期工作；（5）CORN Loss的具体实现细节和特殊奖励token的架构细节未完整披露。
