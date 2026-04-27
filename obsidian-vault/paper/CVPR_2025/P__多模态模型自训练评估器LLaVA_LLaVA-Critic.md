---
title: 'LLaVA-Critic: Learning to Evaluate Multimodal Models'
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- 多模态模型自训练评估器LLaVA-Critic
- LLaVA-Critic
acceptance: poster
cited_by: 4
code_url: https://llava-vl.github.io/blog/2024-10-03-llava-critic/
method: LLaVA-Critic
baselines:
- 测试时计算最优扩展超越参数扩展_Optimal_Test-Tim
- 多模态LLM评判能力基准测试_MLLM-as-a-Judge_
---

# LLaVA-Critic: Learning to Evaluate Multimodal Models

[Code](https://llava-vl.github.io/blog/2024-10-03-llava-critic/)

**Topics**: [[T__Benchmark_-_Evaluation]], [[T__Visual_Reasoning]], [[T__Cross-Modal_Matching]] | **Method**: [[M__LLaVA-Critic]] | **Datasets**: [[D__LLaVA-Wilder]], [[D__WildVision-Bench]], [[D__LiveBench]], [[D__MMHal-Bench]] (其他: LLaVA-W)

| 中文题名 | 多模态模型自训练评估器LLaVA-Critic |
| 英文题名 | LLaVA-Critic: Learning to Evaluate Multimodal Models |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2410.02712) · [Code](https://llava-vl.github.io/blog/2024-10-03-llava-critic/) · [Project](https://llava-vl.github.io/blog/2024-10-03-llava-critic/) |
| 主要任务 | 多模态模型对齐（Multimodal Model Alignment）、视觉语言评估（Vision-Language Evaluation） |
| 主要 baseline | LLaVA-RLHF、SIMA、CSR、RLAIF-V、LLaVA-OneVision、Qwen2-VL、LLaMA3.2-Vision |

> [!abstract] 因为「依赖GPT-4o等昂贵API提供reward signal导致多模态模型对齐成本高且不可扩展」，作者在「LLaVA-OneVision」基础上改了「reward design（用自训练的critic model替代外部reward model）、training recipe（迭代DPO仅9.4k prompts）、data pipeline（LLaVA-Critic-113k合成数据集）和inference strategy（Best-of-N采样）」，在「LLaVA-W / LLaVA-Wilder / WildVision-Bench」上取得「73.5 / 57.2 / 29.2，分别超越RLAIF-V +0.8 / +0.8 / +10.0」

- **LLaVA-W**: 73.5，超越base model +10.1，超越RLAIF-V +0.8
- **WildVision-Bench**: 29.2，超越base model +8.8，超越RLAIF-V +10.0
- **LLaVA-Critic-72B与GPT-4o相关性**: 平均Pearson-r 0.754，超越LLaVA-OV-72B +0.120

## 背景与动机

大型多模态模型（LMM）的对齐质量高度依赖于可靠的reward signal，但现有方案面临严重的成本与可扩展性瓶颈。以GPT-4o作为评估器为例，仅3轮评估就需要约$690的API费用，这使得大规模迭代优化变得不现实。

现有方法主要从三个方向尝试解决这一问题：**LLaVA-RLHF** 利用基于人类反馈预训练的reward model，但需要昂贵的人工标注；**SIMA** 开发了in-context self-critic prompt进行成对判断，但依赖模型自身的反思能力，缺乏系统性的评估训练；**CSR** 引入sentence-level beam search结合CLIP-score校准，然而CLIP-score与真实人类偏好的对齐度有限；**RLAIF-V** 采用分而治之策略组合sentence-level judgment计算总体reward，但流程复杂且对细粒度错误敏感。

这些方法的共同短板在于：要么依赖昂贵的外部评估器（GPT-4o、人工反馈），要么依赖未经专门训练的模型自身生成reward signal，导致评估质量不稳定、与human preference的一致性不足。具体而言，现有开源LMM（如LLaVA-OV-7B、Qwen2-VL-7B、LLaMA3.2-11B-Vision）在pointwise scoring任务上与GPT-4o的Pearson相关性仅约0.35-0.36，远不能满足可靠reward signal的需求。

本文的核心动机是：能否训练一个专门的多模态critic model，使其评估能力与GPT-4o高度一致，同时完全开源、可本地部署，从而替代昂贵的外部reward model？基于此，作者提出了LLaVA-Critic——一个自训练的多模态评估器，通过大规模合成数据学习pointwise scoring和pairwise ranking能力。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ae88a0a6-2e66-4537-b311-7f576ba75f94/figures/fig_001.jpeg)
*Figure: Data statistic of LLaVA-Critic-113k training dataset. In the pointwise setting, we categorize datasets by instruction sources*



## 核心创新

核心洞察：**专门训练的critic model可以可靠地替代GPT-4o作为reward signal来源**，因为通过GPT-4o作为teacher合成的大规模多样化评估数据（LLaVA-Critic-113k），能够将closed-loop评估能力蒸馏到开源模型中，从而使低成本、可扩展的迭代偏好优化和推理时搜索成为可能。

| 维度 | Baseline (LLaVA-RLHF/SIMA/CSR/RLAIF-V) | 本文 (LLaVA-Critic) |
|:---|:---|:---|
| **Reward来源** | 人工反馈预训练RM / in-context prompt / CLIP-score / 分治sentence judgment | 自训练多模态critic model（pointwise + pairwise） |
| **数据成本** | 人工标注或模型自生成，无系统critic训练 | GPT-4o合成113k数据，零人工标注 |
| **训练效率** | RLAIF-V用33.8k prompts | 仅9.4k prompts即可超越RLAIF-V |
| **推理扩展** | 单次生成 | Best-of-N采样，critic作为scorer |
| **部署成本** | 依赖GPT-4o API (~$690/3轮) | 完全本地推理，零API成本 |

## 整体框架



LLaVA-Critic的整体框架包含四个核心模块，形成从数据合成到模型对齐的完整闭环：

1. **Base LMM（LLaVA-OV-7B/72B）**：输入image-question pair，生成候选response。这是被评估和优化的目标模型。

2. **LLaVA-Critic Scorer**：输入image-question-response triplet，输出pointwise score（1-10分）或pairwise preference（A>B/B>A/Tie）。这是本文新增的核心模块，替代外部reward model。

3. **DPO Trainer**：输入来自LLaVA-RLHF数据集的prompts，配合LLaVA-Critic标注的preference pairs，执行3个epoch的Direct Preference Optimization，输出对齐后的policy model。

4. **Best-of-N Sampler**：推理时从policy model生成n=5个候选response（temperature=0.7, top-p=0.9），由LLaVA-Critic选择最高分的response作为最终输出。

数据流示意：
```
Image + Question → [Base LMM] → Candidate Responses
                                      ↓
Image + Question + Response → [LLaVA-Critic] → Score / Preference
                                      ↓
                    ┌─────────────────┴─────────────────┐
                    ↓                                   ↓
            [DPO Trainer]                      [Best-of-N Sampler]
            (Training)                         (Inference)
                    ↓                                   ↓
            Aligned Policy                    Best Response Output
```

关键数据资产LLaVA-Critic-113k包含pointwise scoring和pairwise ranking两种标注，由GPT-4o作为teacher合成，覆盖多种instruction source和任务类型。

## 核心模块与公式推导

### 模块 1: Pointwise Scoring（对应框架图 LLaVA-Critic Scorer）

**直觉**：让critic model直接预测一个连续质量分数，与GPT-4o的评分高度相关，从而提供细粒度的reward signal。

**Baseline 公式** (标准回归/分类)：
$$L_{\text{base}} = \frac{1}{N}\sum_{i=1}^{N}(s_\theta(x_i, q_i, r_i) - s^*_i)^2$$

符号：$\theta$ = critic model参数，$(x_i, q_i, r_i)$ = (image, question, response)，$s_\theta(\cdot)$ = 模型预测分数，$s^*_i$ = GPT-4o teacher分数。

**变化点**：标准MSE回归无法充分利用多模态模型的生成能力；本文将scoring任务重新建模为**生成式数值预测**，让模型以自然语言形式输出分数并解析，同时引入多样化instruction template增强泛化。

**本文公式**：
$$\text{Step 1}: \quad \tilde{s}_i = \text{Decode}\big(\text{LM}_\theta(x_i, q_i, r_i, \text{prompt}_{\text{score}})\big) \quad \text{将数值预测转化为条件生成任务}$$
$$\text{Step 2}: \quad s_i = \text{ExtractNumber}(\tilde{s}_i) \in [1, 10] \quad \text{从生成文本中提取标准化分数}$$
$$\text{最终}: \quad L_{\text{score}} = \frac{1}{N}\sum_{i=1}^{N}(s_i - s^*_i)^2 + \lambda \cdot \text{FormatReg} \quad \text{加入格式正则化保证可解析性}$$

**对应消融**：数据规模从v0.5到full data的扩展显示，更大规模和多样性的数据对Pearson-r提升至关重要（具体Δ值。

---

### 模块 2: Pairwise Ranking（对应框架图 LLaVA-Critic Scorer）

**直觉**：人类偏好本质上是相对比较而非绝对分数，pairwise ranking更直接地对齐RLHF/DPO的需求。

**Baseline 公式** (Bradley-Terry / DPO implicit reward)：
$$L_{\text{BT}} = -\mathbb{E}_{(x,q,r_w,r_l)\sim\mathcal{D}}\left[\log\sigma\big(r_\theta(x,q,r_w) - r_\theta(x,q,r_l)\big)\right]$$

符号：$r_w$ = win response，$r_l$ = lose response，$\sigma$ = sigmoid，$r_\theta(\cdot)$ = implicit reward function。

**变化点**：标准BT模型需要单独训练reward model；本文让**critic model直接输出比较结果**（A>B / B>A / Tie），将ranking任务转化为三分类生成问题，与pointwise scoring共享同一模型 backbone。

**本文公式**：
$$\text{Step 1}: \quad p_i = \text{Softmax}\big(\text{LM}_\theta(x_i, q_i, r_i^A, r_i^B, \text{prompt}_{\text{compare}})\big) \quad \text{三分类：A赢/B赢/平局}$$
$$\text{Step 2}: \quad \hat{y}_i = \text{arg}\max p_i \in \{\text{``A''}, \text{``B''}, \text{``Tie''}\} \quad \text{离散比较决策}$$
$$\text{最终}: \quad L_{\text{rank}} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c \in \mathcal{C}}\mathbb{1}[y_i=c]\log p_i(c) + \gamma \cdot L_{\text{score}} \quad \text{联合pointwise损失多任务学习}$$

**对应消融**：Table 5显示，使用LLaVA-Critic-7B作为reward model进行迭代DPO，在LLaVA-W上达到100.3，超越OV-7B base +9.6，超越LLaVA-RLHF +2.8。

---

### 模块 3: Iterative DPO with Critic Reward（对应框架图 DPO Trainer）

**直觉**：利用训练好的critic model生成preference pairs，以极小数据量实现高效的策略优化。

**Baseline 公式** (标准DPO)：
$$L_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x,q,r_w,r_l)\sim\mathcal{D}}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(r_w|x,q)}{\pi_{\text{ref}}(r_w|x,q)} - \beta\log\frac{\pi_\theta(r_l|x,q)}{\pi_{\text{ref}}(r_l|x,q)}\right)\right]$$

符号：$\pi_\theta$ = 策略模型，$\pi_{\text{ref}}$ = reference模型，$\beta$ = temperature参数，$(r_w, r_l)$ = 由reward model判定的win/lose pair。

**变化点**：标准DPO依赖外部reward model或人工标注的preference；本文用**LLaVA-Critic替代reward model**，且仅使用9.4k prompts（vs. RLAIF-V的33.8k），通过迭代训练逐步提升。

**本文公式**：
$$\text{Step 1}: \quad (r_w, r_l) = \text{RankByCritic}\big(\{r^{(j)}\}_{j=1}^{K}, \text{LLaVA-Critic}(\cdot)\big) \quad \text{critic标注偏好对}$$
$$\text{Step 2}: \quad \mathcal{D}_{\text{critic}}^{(t)} = \{(x_i, q_i, r_{w,i}^{(t)}, r_{l,i}^{(t)})\}_{i=1}^{9400} \quad \text{每轮迭代重新采样并标注}$$
$$\text{最终}: \quad L_{\text{iter-DPO}}^{(t)} = L_{\text{DPO}}(\pi_\theta^{(t)}; \pi_{\text{ref}}) + \alpha \cdot \text{KL}[\pi_\theta^{(t)} \| \pi_{\text{ref}}] \quad \text{迭代t轮，critic动态更新偏好}$$

**对应消融**：Table 5中，+ LLaVA-Critic在LLaVA-Wilder上57.2 vs. + RLAIF-V 56.4（+0.8），在WildVision-Bench上29.2 vs. 19.2（+10.0），显示critic-based reward在开放域场景优势显著。

## 实验与分析



本文在多个benchmark上评估了LLaVA-Critic的两类能力：一是**作为评估器**的pointwise scoring准确性（与GPT-4o的相关性），二是**作为reward model**对base model进行偏好对齐后的性能提升。

在**评估能力**方面，Table 2显示LLaVA-Critic-7B在in-domain pointwise scoring上达到平均Pearson-r 0.732，显著超越同尺寸的LLaVA-OV-7B（0.364）、Qwen2-VL-7B（0.352）和LLaMA3.2-11B-Vision（0.359），提升幅度超过+0.36。更大规模的LLaVA-Critic-72B进一步达到0.754，超越其base model LLaVA-OV-72B（0.634）+0.120，验证了critic training的有效性可扩展至大模型。



在**偏好对齐**方面，Table 5的核心结果显示：基于LLaVA-Critic-7B reward signal的迭代DPO，在LLaVA-W上达到73.5，超越base model LLaVA-v1.5-7B（63.4）+10.1，超越此前最优的RLAIF-V（72.7）+0.8；在更具挑战性的WildVision-Bench上，优势更为显著——29.2 vs. RLAIF-V 19.2，差距达+10.0。值得注意的是，MMHal-Bench上LLaVA-Critic（2.07）仍落后于RLAIF-V（3.04），作者归因于该benchmark使用GPT-4V而非GPT-4o进行评估（API deprecation导致），可能引入不一致性。



消融实验关注**数据规模**的影响：从v0.5到full LLaVA-Critic-113k数据的扩展显示，数据规模和多样性对critic model的评估质量至关重要。在偏好对齐设置中（Table 6），LLaVA-Critic-7B使OV-7B base从90.7提升至100.3，超越LLaVA-RLHF（97.5）+2.8，且仅使用9.4k prompts（vs. LLaVA-RLHF的完整数据）。推理时Best-of-N采样（Table 7）进一步带来增益：LLaVA-W上+1.7（100.3→102.0），LLaVA-Wilder上+3.2（71.6→74.8）。

**公平性检查**：对比存在一定限制——Table 5中RLAIF-V使用33.8k prompts而LLaVA-Critic仅9.4k，数据量差异可能影响公平性；MMHal-Bench使用GPT-4V评估而非GPT-4o；缺少与Prometheus-VISION等专门评估模型的直接对比，以及DPO变体（IPO、KTO）的消融。此外，GPT-4o作为直接reward model for DPO的实验未被实际运行，仅作为成本参考。

## 方法谱系与知识库定位

**方法家族**：LLaVA系列 → LLaVA-OneVision → **LLaVA-Critic**

**父方法**：LLaVA-OneVision（架构基础，提供7B/72B视觉语言模型backbone）

**改变的slots**：
- **reward_design**：外部reward model → 自训练多模态critic（pointwise + pairwise）
- **training_recipe**：标准偏好优化 → 迭代DPO，仅9.4k prompts
- **data_pipeline**：人工/自生成数据 → LLaVA-Critic-113k合成数据集（GPT-4o as teacher）
- **inference_strategy**：单次生成 → Best-of-N采样（n=5, temperature=0.7, top-p=0.9）

**直接baseline对比**：
- **LLaVA-RLHF**：用人工反馈预训练RM；本文用自训练critic替代，零人工成本
- **SIMA**：in-context self-critic prompt；本文通过专门训练获得更稳定评估能力
- **CSR**：CLIP-score校准；本文用学习到的critic score替代浅层语义相似度
- **RLAIF-V**：分治sentence-level judgment；本文用统一critic model简化流程

**后续方向**：(1) 将critic能力扩展至更多模态（视频、3D）；(2) 探索critic model与policy model的联合迭代训练而非固定critic；(3) 开发更高效的蒸馏策略进一步降低对GPT-4o teacher的依赖。

**标签**：modality=vision-language | paradigm=critic-based RLHF / self-improvement | scenario=multimodal model alignment & evaluation | mechanism=distillation from strong teacher + iterative DPO | constraint=cost-efficient / open-source / no human annotation

## 引用网络

### 直接 baseline（本文基于）

- [[P__测试时计算最优扩展超越参数扩展_Optimal_Test-Tim]] _(方法来源)_: Core algorithmic idea about test-time compute scaling likely inspires the paper'
- [[P__多模态LLM评判能力基准测试_MLLM-as-a-Judge_]] _(方法来源)_: Core methodology for using MLLMs as judges, directly relevant to paper's approac

