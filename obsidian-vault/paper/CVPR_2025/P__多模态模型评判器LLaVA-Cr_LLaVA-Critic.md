---
title: 'LLaVA-Critic: Learning to Evaluate Multimodal Models'
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- 多模态模型评判器LLaVA-Critic
- LLaVA-Critic
acceptance: poster
cited_by: 121
code_url: https://llava-vl.github.io/blog/2024-10-03-llava-critic/
method: LLaVA-Critic
baselines:
- 测试时计算最优扩展超越参数扩展_Optimal_Test-Tim
- 视觉语言模型的校准自奖励优化_CSR_(Calibrated_
- 多模态大模型的自举偏好优化对齐_BPO_(Bootstrappe
- 多模态LLM评判能力基准测试_MLLM-as-a-Judge_
---

# LLaVA-Critic: Learning to Evaluate Multimodal Models

[Code](https://llava-vl.github.io/blog/2024-10-03-llava-critic/)

**Topics**: [[T__Benchmark_-_Evaluation]], [[T__Visual_Reasoning]], [[T__Cross-Modal_Matching]] | **Method**: [[M__LLaVA-Critic]] | **Datasets**: [[D__LLaVA-Bench]], [[D__LLaVA-Wilder]], [[D__WildVision-Bench]], [[D__LiveBench]], [[D__MMHal-Bench]]

| 中文题名 | 多模态模型评判器LLaVA-Critic |
| 英文题名 | LLaVA-Critic: Learning to Evaluate Multimodal Models |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2410.02712) · [Code](https://llava-vl.github.io/blog/2024-10-03-llava-critic/) · [Project](https://llava-vl.github.io/blog/2024-10-03-llava-critic/) |
| 主要任务 | 多模态模型评估（逐点评分、成对排序）、偏好对齐、推理时Best-of-N采样 |
| 主要 baseline | LLaVA-RLHF、SIMA、CSR、RLAIF-V、LLaVA-OneVision、GPT-4o |

> [!abstract] 因为「现有偏好学习方法依赖昂贵的人类反馈或GPT-4o作为奖励源」，作者在「LLaVA-OneVision」基础上改了「训练专用多模态评判模型替代外部奖励模型」，在「LLaVA-Bench等7个基准」上取得「LLaVA-W 73.5（+10.1 over base）、LLaVA-Wilder 57.2（+3.0）、WildVision-Bench 29.2（+8.8）」

- **LLaVA-Bench**: 73.5 vs LLaVA-v1.5-7B base 63.4，提升 **+10.1**，优于所有对比方法
- **逐点评分一致性**: LLaVA-Critic-7B 与 GPT-4o 的 Pearson-r 达到 **0.732**，远超 LLaVA-OV-7B 的 0.364
- **偏好对齐**: OV-7B + LLaVA-Critic 在 LLaVA-W 达到 **100.3**，比 LLaVA-RLHF 的 97.5 高 **+2.8**

## 背景与动机

大型多模态模型（LMM）的能力快速提升，但如何可靠地评估其输出质量并据此优化模型，仍是一个核心瓶颈。具体而言，当模型生成对图像问题的回答后，我们需要一个高质量的"裁判"来判断回答好坏——这个裁判的评分将直接用于强化学习或偏好优化。

现有方法主要沿三条路径解决这一问题：

- **LLaVA-RLHF** 依赖预训练的人类反馈奖励模型，需要昂贵的人工标注偏好数据来训练奖励模型；
- **SIMA** 采用 in-context self-critic 提示，让模型通过精心设计的 prompt 自我评判并给出成对排序，但受限于模型固有的评估偏见；
- **RLAIF-V** 使用分而治之策略，将句子级判断组合为整体奖励分数，但依赖 CLIP-score 等外部信号校准，流程复杂且难以扩展。

这些方法的共同短板在于：**要么依赖昂贵的人类反馈或闭源 GPT-4o（成本高、不可复现），要么利用模型自身提示能力（一致性差、分数分布偏斜）**。尤其关键的是，现有开源模型作为评判者时，与 GPT-4o 的一致性极低——例如 LLaVA-OneVision-7B 与 GPT-4o 的 Pearson 相关系数仅 0.364，几乎无法提供可靠的奖励信号。

本文的核心动机是：**训练一个专门的多模态评判模型，使其在评估能力上逼近甚至替代 GPT-4o，从而为多模态偏好学习提供一个可复现、零成本的奖励源**。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/fbfe19bf-44e3-49a0-a231-eeed0b196727/figures/Figure_1.png)
*Figure 1: Figure 1. Data statistic of LLaVA-Critic-113k training dataset. In the pointwise setting, we categorize datasets by instruction sourcesand select data based on the task type corresponding to each eval*



## 核心创新

核心洞察：**将"评估能力"本身作为可学习的任务进行专门训练**，因为通用多模态模型的优化目标与评估目标存在本质错位——生成模型被训练为"回答问题"，而非"判断答案质量"，从而使专用评判模型能够提供与 GPT-4o 高度一致的奖励信号成为可能。

| 维度 | Baseline (LLaVA-RLHF/SIMA/RLAIF-V) | 本文 (LLaVA-Critic) |
|:---|:---|:---|
| **奖励源** | 人类反馈预训练模型 / in-context 自批判 / 分治式CLIP校准 | **专用训练的多模态评判模型**，提供逐点评分与成对排序 |
| **数据** | 人类偏好数据集或自生成批判，无结构化评判训练 | **LLaVA-Critic-113k**，11.3万条多样化评估指令 |
| **推理策略** | 单响应生成 | 新增 **Best-of-N 采样**，用 Critic 选择最优响应 |
| **成本** | 需 GPT-4o API 或人工标注 | **零成本替代**，完全开源可复现 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/fbfe19bf-44e3-49a0-a231-eeed0b196727/figures/Figure_2.png)
*Figure 2: Table 2. Results on in-domain pointwise scoring. LLaVA-Critic consistently outperforms baselines across 7 multimodal benchmarks.*



LLaVA-Critic 的整体框架包含四个核心阶段，形成从输入到优化输出的完整闭环：

1. **输入阶段**：接收图像-问题对（Image-Question pairs），同时送入策略模型和评判模型
2. **策略模型生成（Policy Model）**：基于 LLaVA-v1.5-7B 或 LLaVA-OneVision-7B/72B 生成候选响应（candidate responses）
3. **LLaVA-Critic 评估**：接收图像-问题-响应三元组，输出两种形式的评估信号：
   - **逐点评分（pointwise scoring）**：标量质量分数
   - **成对排序（pairwise ranking）**：比较两个响应的优劣判断
4. **下游优化**：利用 Critic 的评估信号进行两种增强：
   - **DPO 训练**：用成对排序构造偏好对，执行 3 epoch 的 Direct Preference Optimization
   - **Best-of-N 采样**：推理时生成多个候选，由 Critic 选择最高分响应

```
Image + Question ──→ [Policy Model] ──→ Candidate Response(s)
       │                                    │
       └──────────────→ [LLaVA-Critic] ←───┘
                             │
                    ┌────────┴────────┐
                    ↓                 ↓
              Pointwise Score    Pairwise Rank
                    │                 │
                    ↓                 ↓
            Best-of-N Selection    DPO Training
                    │                 │
                    └────────┬────────┘
                             ↓
                    Improved Response / Aligned Model
```

## 核心模块与公式推导

### 模块 1: 评判模型训练目标（对应框架图 Critic 模块）

**直觉**: 通用多模态模型被优化为生成答案，而非判断答案质量；需通过专门数据将评估能力"蒸馏"到模型中。

**Baseline 形式** (标准指令微调): 
$$\mathcal{L}_{\text{SFT}} = -\sum_{t} \log P_\theta(y_t | x, y_{<t})$$

符号: $x$ = 图像+问题+候选响应的联合输入, $y_t$ = 第 $t$ 个评估 token, $\theta$ = 模型参数

**变化点**: 标准 SFT 仅训练模型跟随评估指令格式，但缺乏对"评估准确性"的显式优化；本文将评估任务拆解为两个可监督的子任务，并构造大规模专用数据集。

**本文公式推导**:

$$\text{Step 1 (逐点评分)}: \quad s = f_\theta^{\text{score}}(x, r) \in \mathbb{R}$$
其中 $r$ 为候选响应，输出标量分数，训练目标为最小化与参考评分（如 GPT-4o）的均方误差：
$$\mathcal{L}_{\text{score}} = \mathbb{E}_{(x,r,s^*)} \left[ (f_\theta^{\text{score}}(x,r) - s^*)^2 \right]$$

$$\text{Step 2 (成对排序)}: \quad p_\theta(r_1 \text{succ} r_2 | x) = \sigma\left( f_\theta^{\text{rank}}(x, r_1) - f_\theta^{\text{rank}}(x, r_2) \right)$$
采用 Bradley-Terry 模型形式，训练目标为交叉熵损失：
$$\mathcal{L}_{\text{rank}} = -\mathbb{E}_{(x, r_w, r_l)} \left[ \log p_\theta(r_w \text{succ} r_l | x) \right]$$
其中 $r_w$ = 优胜响应, $r_l$ = 落败响应

$$\text{最终联合目标}: \mathcal{L}_{\text{critic}} = \mathcal{L}_{\text{score}} + \lambda \cdot \mathcal{L}_{\text{rank}}$$

**对应消融**: 数据规模消融显示，使用完整 LLaVA-Critic-113k 相比 v0.5 子集，在各基准上的 Pearson-r 均有显著提升，验证了数据多样性和规模的必要性。

### 模块 2: DPO 偏好对齐（对应框架图 DPO Training 模块）

**直觉**: 利用 Critic 的成对排序替代人工标注或 GPT-4o，实现低成本偏好优化。

**Baseline 公式** (标准 DPO, Rafailov et al.):
$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

符号: $\pi_\theta$ = 待优化策略, $\pi_{\text{ref}}$ = 参考策略, $\beta$ = 温度系数, $y_w/y_l$ = 胜/负响应

**变化点**: 标准 DPO 依赖人类或 GPT-4o 标注的 $(y_w, y_l)$ 对；本文用 LLaVA-Critic 的成对排序生成偏好对，且固定训练 3 epochs。

**本文公式**:
$$\text{Step 1 (Critic 生成偏好对)}: \quad (y_w, y_l) = \text{Sort}_2\left( \{r_1, ..., r_N\}, f_\theta^{\text{rank}} \right)$$
从策略模型采样 $N$ 个响应，由 Critic 排序选取最优/最差对

$$\text{Step 2 (DPO with Critic preferences)}: $$
$$\mathcal{L}_{\text{DPO-Critic}} = -\mathbb{E}_{(x, y_w, y_l) \sim \pi_{\text{Critic}}} \left[ \log \sigma\left( \beta \Delta_{\text{KL}} \right) \right]$$
其中 $\Delta_{\text{KL}} = \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$

**对应消融**: Table 5 显示，LLaVA-v1.5-7B + LLaVA-Critic 在 LLaVA-W 达 73.5，相比 base 63.4 提升 +10.1；LLaVA-RLHF 仅 63.7，SIMA 66.1，CSR 71.1，RLAIF-V 72.7。

### 模块 3: Best-of-N 采样（对应框架图 Inference 模块）

**直觉**: 推理时利用已训练的 Critic 进行响应选择，无需额外训练即可提升输出质量。

**Baseline 形式** (贪婪解码或随机采样):
$$y^* = \text{arg}\max_{y} \log P_{\pi_\theta}(y|x) \quad \text{或} \quad y \sim P_{\pi_\theta}(\cdot|x)$$

**变化点**: 单点估计忽略了响应空间中的高质量候选；本文通过 Critic 评分进行事后选择。

**本文公式**:
$$\text{Step 1 (生成候选)}: \quad \{y_1, ..., y_N\} \sim P_{\pi_\theta}(\cdot|x)$$
从策略模型采样 $N$ 个响应

$$\text{Step 2 (Critic 评分)}: \quad s_i = f_{\text{Critic}}^{\text{score}}(x, y_i), \quad i = 1, ..., N$$

$$\text{最终选择}: y^* = \text{arg}\max_{y_i} s_i$$

**对应消融**: Table 7 显示 Best-of-N 采样结果，使用 Critic-7B 选择最佳响应可进一步提升性能（具体数值。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/fbfe19bf-44e3-49a0-a231-eeed0b196727/figures/Table_1.png)
*Table 1: Table 1. An example of LLaVA-Critic training data. The top block shows pointwise scoring, where LLaVA-Critic predicts a scoreto evaluate a single response’s quality; the bottom block illustrates pairw*



本文在三个层面验证 LLaVA-Critic 的有效性：评判能力、偏好对齐效果、以及推理时增强。

**评判能力验证**（Table 2, Table 3）。在逐点评分任务上，LLaVA-Critic-7B 与 GPT-4o 的 Pearson 相关系数达到 **0.732**，而同为 7B 级别的 LLaVA-OV-7B 仅 0.364、Qwen2-VL-7B-Instruct 仅 0.352、LLaMA3.2-11B-Vision-Instruct 仅 0.359——LLaVA-Critic 将一致性提升了一倍以上。放大到 72B 规模，LLaVA-Critic-72B 达到 **0.754**，相比 LLaVA-OV-72B 的 0.634 提升 **+0.120**。在成对排序任务上（Table 3），LLaVA-Critic 与人工评估者的一致性可与 GPT-4V 媲美。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/fbfe19bf-44e3-49a0-a231-eeed0b196727/figures/Table_3.png)
*Table 3: Table 3. Results on in-domain pairwise ranking. LLaVA-Critic iscomparable with GPT-4V in alignment with human evaluators.*



**偏好对齐效果**（Table 5, Table 6）。以 LLaVA-v1.5-7B 为初始策略，使用 LLaVA-Critic 进行 DPO 训练后，在 **LLaVA-Bench (LLaVA-W)** 达到 **73.5**，相比 base 的 63.4 提升 **+10.1**，超越所有对比方法：LLaVA-RLHF 63.7、SIMA 66.1、CSR 71.1、RLAIF-V 72.7。在 **WildVision-Bench** 上提升尤为显著，达到 **29.2** vs base 20.4（**+8.8**），而 LLaVA-RLHF 甚至下降至 19.8。在 **LLaVA-Wilder** 上为 **57.2**（+3.0），同样最优。以更强的 LLaVA-OneVision-7B 为 base 时（Table 6），LLaVA-Critic 在 LLaVA-W 达到 **100.3**，比 LLaVA-RLHF 的 97.5 高 **+2.8**；OV-72B base 时达到 **104.4**，比 LLaVA-RLHF 的 103.2 高 **+1.2**。


![Table 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/fbfe19bf-44e3-49a0-a231-eeed0b196727/figures/Table_6.png)
*Table 6: Table 6. Comparison between LLaVA-Critic and baselines in pref-erence alignment. “Base”: the initial LMM checkpoint for DPO.*



**消融与公平性检查**。数据规模消融表明，LLaVA-Critic-113k 的完整数据相比 v0.5 子集显著提升了评判一致性，验证了评估指令多样性和数据量的必要性。但需注意几点限制：其一，**MMHal-Bench** 上 LLaVA-Critic 仅 2.07，低于 RLAIF-V 的 3.04，说明在幻觉检测维度存在 trade-off；其二，GPT-4o 虽作为逐点评分的参考标准，但未在 Table 5 中直接作为 DPO 奖励源进行对比，无法完全确认成本替代的经济性优势；其三，MMHal-Bench 因 API 弃用改用 gpt-4-0613 评估，跨评估器比较需谨慎。此外，缺少与 Prometheus-Vision 等其他开源评判模型的直接对比。

## 方法谱系与知识库定位

**方法家族**: LLaVA 系列多模态模型 → **父方法**: LLaVA-OneVision（7B/72B 作为 Critic 训练的 base model）

**修改槽位**:
- **reward_design**: 用专用训练的 Critic 模型替代人类反馈/GPT-4o/CLIP 等外部奖励源
- **data_pipeline**: 新增 LLaVA-Critic-113k 评判专用训练数据
- **training_recipe**: DPO 训练使用 Critic 生成的偏好信号，固定 3 epochs
- **inference_strategy**: 新增 Best-of-N 采样利用 Critic 评分

**直接基线对比**:
- **LLaVA-RLHF**: 人类反馈预训练奖励模型 → 本文用训练好的 Critic 替代，无需人工标注
- **SIMA**: in-context 自批判提示 → 本文通过专门训练获得更稳定、更多样化的评分分布
- **RLAIF-V**: 分治式句子级判断+CLIP校准 → 本文端到端训练，流程更简洁

**后续方向**:
1. **Critic 迭代自举**: 用对齐后的更强模型重新生成更高质量的评判数据，形成"策略-Critic"协同提升循环
2. **细粒度评判维度**: 从单一总分扩展到多维度评估（准确性、安全性、完整性等），服务更精细的偏好优化
3. **跨模态扩展**: 将 Critic 机制扩展至视频、3D 等多模态场景，构建统一评估框架

**标签**: 视觉-语言 / 偏好优化 / 模型评估-as-a-service / 奖励模型替代 / 开源可复现

## 引用网络

### 直接 baseline（本文基于）

- [[P__测试时计算最优扩展超越参数扩展_Optimal_Test-Tim]] _(方法来源)_: Core methodology paper on test-time compute scaling; likely inspires the paper's
- [[P__视觉语言模型的校准自奖励优化_CSR_(Calibrated_]] _(直接 baseline)_: Self-rewarding VLM with calibration; closely related method likely compared agai
- [[P__多模态大模型的自举偏好优化对齐_BPO_(Bootstrappe]] _(直接 baseline)_: Bootstrapped preference optimization for MLLMs; direct methodological competitor
- [[P__多模态LLM评判能力基准测试_MLLM-as-a-Judge_]] _(方法来源)_: Core methodology for using MLLMs as judges; directly enables the paper's evaluat

