---
title: Unified Multimodal Chain-of-Thought Reward Model through Reinforcement Fine-Tuning
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 多模态思维链奖励模型的统一强化微调
- UNIFIEDREWARD-TH
- UNIFIEDREWARD-THINK
- Incorporating explicit long chains
acceptance: Poster
cited_by: 69
code_url: https://codegoat24.github.io/UnifiedReward/think
method: UNIFIEDREWARD-THINK
modalities:
- Image
- Video
- Text
paradigm: reinforcement fine-tuning with GRPO
---

# Unified Multimodal Chain-of-Thought Reward Model through Reinforcement Fine-Tuning

[Code](https://codegoat24.github.io/UnifiedReward/think)

**Topics**: [[T__Image_Generation]], [[T__Visual_Reasoning]] | **Method**: [[M__UNIFIEDREWARD-THINK]] | **Datasets**: VLRewardBench

> [!tip] 核心洞察
> Incorporating explicit long chains of thought into reward reasoning significantly enhances reliability and robustness, and once internalized, improves even direct response accuracy through implicit reasoning capabilities.

| 中文题名 | 多模态思维链奖励模型的统一强化微调 |
| 英文题名 | Unified Multimodal Chain-of-Thought Reward Model through Reinforcement Fine-Tuning |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2505.03318) · [Code](https://codegoat24.github.io/UnifiedReward/think) · [Project](https://codegoat24.github.io/UnifiedReward/think) |
| 主要任务 | Multimodal Reward Modeling (Image/Video Understanding & Generation Evaluation) |
| 主要 baseline | UnifiedReward, GPT-4o, LLaVA-Critic, PickScore, HPSv2, ImageReward, DeepSeek-R1, Visual-RFT |

> [!abstract] 因为「当前多模态奖励模型仅输出直接响应或浅层推理，导致奖励信号不可靠且缺乏可解释性」，作者在「UnifiedReward」基础上改了「三阶段训练范式（冷启动CoT蒸馏→拒绝采样精炼→GRPO强化微调）+ 多维度长思维链推理机制」，在「VLRewardBench」上取得「Overall Accuracy 73.8 vs UnifiedReward 67.5（+6.3）」

- **VLRewardBench Image Understanding Overall Accuracy**: 73.8，超越 UnifiedReward 67.5（+6.3）、GPT-4o 65.8（+9.6）
- **VLRewardBench Hallucination 子任务**: 72.7，大幅领先 UnifiedReward 58.1（+14.6）
- **VideoGen-RewardBench**: 82.3，超越 baseline 77.2（+5.1）

## 背景与动机

当前视觉语言模型的对齐质量高度依赖奖励模型的信号准确性，但现有方案存在根本性缺陷：当面对一张AI生成的图片与其文本描述是否匹配的判断任务时，传统奖励模型往往直接输出一个标量分数（如 PickScore、HPSv2）或简单的二元偏好判断（如 UnifiedReward 的直接响应），既无法解释"为什么这张图更好"，也难以捕捉多维度质量差异——例如图像的文本忠实度、美学质量、细节丰富度可能相互矛盾，需要分步权衡。

现有方法的处理方式各有局限：**UnifiedReward** 虽统一了图像/视频理解与生成任务的奖励建模，但推理深度仅限于直接响应或浅层思考，缺乏结构化分析；**PickScore / HPSv2 / ImageReward** 等专用奖励模型仅输出单标量分数，完全丧失可解释性；**LLaVA-Critic** 虽引入评估能力，但未针对奖励建模任务优化长推理链。这些方法的共同瓶颈在于：推理过程与最终判断耦合于黑盒参数中，导致错误难以追溯、复杂场景下信号不稳定。

更深层的问题在于，视觉内容的评估天然需要多维度分析——一个视频奖励模型需同时考量时序一致性、运动自然度、文本-视频对齐度等，而现有方法无法显式拆解这些维度并展示权衡过程。这直接激励了本文的核心探索：能否让多模态奖励模型像人类评审一样，显式地展开长思维链（CoT）进行多维度逐步分析，再通过强化学习内化这种能力？


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f442e1f3-10ee-4e74-9111-5f2b074de427/figures/Figure_1.png)
*Figure 1 (comparison): Figure 1. Overview of Comparison Results. (a) Our method enables multi-dimensional and long CoT reasoning in reward modeling for image and video understanding tasks. (b) Comparison results demonstrate our superiority in both video understanding and generation overall tasks.*



本文提出 UNIFIEDREWARD-THINK，首次将显式长思维链推理引入统一的多模态奖励模型，并设计三阶段训练范式使其在理解与生成任务上均实现可解释的高精度评估。

## 核心创新

核心洞察：奖励模型的"判断过程"本身应当被显式生成并训练，因为视觉内容的多维度质量冲突无法被单标量压缩，从而使长思维链推理成为奖励信号可靠性的来源而非仅可解释性的装饰。

与 baseline 的差异：

| 维度 | Baseline (UnifiedReward) | 本文 (UNIFIEDREWARD-THINK) |
|:---|:---|:---|
| 推理形态 | 直接响应或浅层思考，无显式中间步骤 | 多维度长思维链，逐步独立评分后聚合判断 |
| 训练范式 | 标准监督微调（SFT） on 偏好数据 | 三阶段：冷启动CoT蒸馏 → 拒绝采样精炼 → GRPO强化微调 |
| 数据构建 | 原始多模态偏好数据集，无推理痕迹 | 蒸馏GPT-4o生成CoT推理链，一致性过滤保留5K高质量样本 |
| 探索机制 | 无显式探索 | GRPO采样8个响应，专门利用错误样本驱动多样化推理路径探索 |
| 奖励设计 | 单标量或简单排序 | 结构化多维度独立评分，显式聚合函数生成最终判断 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f442e1f3-10ee-4e74-9111-5f2b074de427/figures/Figure_2.png)
*Figure 2 (pipeline): Figure 2. Method Overview. (1) Cold Start: We first collect GPT-4V reasoning process via a small set of seed data, then synthesize long CoT data for reward reasoning. (2) Rejection Sampling: We train a reward model to perform unified CoT reward generalization by distinguishing good and bad reasoning. (3) GRPO: We sample and pair the rejected responses and use them to train the policy model with our unified CoT reward.*



UNIFIEDREWARD-THINK 的整体数据流遵循"数据构建 → 格式习得 → 质量精炼 → 探索增强"的四层递进逻辑：

**输入**：多模态内容（图像/视频）及其候选对（如两张生成图或理解任务的答案对），附评估维度提示。

**模块 1: 冷启动CoT蒸馏（Cold-start CoT Distillation）** —— 输入 10K 随机采样的 HPD/EvalMuse/OIP 偏好数据，经 GPT-4o 生成带推理过程的 CoT 响应；通过一致性过滤（最终判断匹配真实标签）筛选出 5K 样本构成 ImageGen-CoT-Reward-5K。输出结构化 CoT 格式模板，教会模型"如何分步思考"。

**模块 2: 冷启动SFT（Cold-start SFT）** —— 输入 ImageGen-CoT-Reward-5K，以标准监督微调让模型掌握多维度评分的语言格式与结构。输出具备基础 CoT 生成能力的检查点。

**模块 3: 拒绝采样精炼（Rejection Sampling Refinement）** —— 输入大规模统一多模态偏好数据（冷启动后剩余数据），模型自举生成多条推理路径，仅保留最终判断正确的样本进行迭代训练。输出推理质量初步稳定的中期模型。

**模块 4: GRPO强化微调（GRPO RFT）** —— 输入拒绝采样阶段的错误推理样本，每组查询采样 N=8 个响应，利用组相对优势估计进行策略优化，KL 惩罚系数 β=0.04 约束偏离。输出最终 UNIFIEDREWARD-THINK 模型，具备探索驱动的多样化推理能力。

**模块 5: 多维度评分聚合器（Multi-dimensional Scoring Aggregator）** —— 推理时，模型对输入内容在 n 个评估维度独立评分，经聚合函数（如加权求和）输出最终偏好判断与完整 CoT 推理链。

```
多模态偏好数据 → [GPT-4o蒸馏+过滤] → ImageGen-CoT-Reward-5K
                                      ↓
[冷启动SFT] ← 5K CoT样本 → 基础CoT模型
                                      ↓
[拒绝采样] ← 大规模偏好数据 → 精炼模型（保留正确轨迹）
                                      ↓
[GRPO RFT] ← 错误样本(N=8采样) → UNIFIEDREWARD-THINK
                                      ↓
推理: 输入内容 → 多维度独立评分 → 聚合 → CoT推理链 + 最终判断
```

## 核心模块与公式推导

### 模块 1: 冷启动蒸馏损失与一致性过滤（对应框架图 阶段一）

**直觉**: 模型首先需要"学会说话的方式"——即长推理链的结构化格式，而非直接优化判断准确性；通过强教师蒸馏获取高质量示范，再以一致性过滤剔除"推理过程与结论矛盾"的噪声样本。

**Baseline 公式** (标准 SFT on preference data): $$\mathcal{L}_{SFT} = -\mathbb{E}_{(x,y_{win},y_{lose}) \sim \mathcal{D}} \left[ \log \pi_\theta(y_{label}|x, y_{win}, y_{lose}) \right]$$
符号: $x$ = 多模态输入（图像/视频+文本），$y_{win}, y_{lose}$ = 偏好对，$y_{label}$ = 二元偏好标签，$\theta$ = 模型参数。

**变化点**: 标准 SFT 直接拟合标签，不生成可解释的推理过程；本文要求模型输出完整 CoT 推理链 $c$ 且最终结论 $c_{final}$ 需与标签一致。

**本文公式（推导）**:
$$\text{Step 1 (蒸馏)}: \mathcal{L}_{distill} = -\mathbb{E}_{(x,y_{win},y_{lose},c) \sim \mathcal{D}_{raw}} \left[ \log \pi_\theta(c|x, y_{win}, y_{lose}) \right] \quad \text{最大化GPT-4o生成推理链的似然}$$
$$\text{Step 2 (过滤)}: \mathcal{D}_{filtered} = \{(x,y_{win},y_{lose},c) \in \mathcal{D}_{raw} : \text{Judge}(c) = y_{gt}\} \quad \text{仅保留推理结论与真实标签匹配的样本}$$
$$\text{Step 3 (冷启动损失)}: \mathcal{L}_{cold} = -\mathbb{E}_{(x,y_{win},y_{lose},c) \sim \mathcal{D}_{cot}} \left[ \log \pi_\theta(c|x, y_{win}, y_{lose}) \cdot \mathbb{1}[c_{final} = y_{label}] \right]$$
其中 $\mathcal{D}_{cot}$ = ImageGen-CoT-Reward-5K，$\mathbb{1}[\cdot]$ 为指示函数，最终过滤后保留约 5K/10K = 50% 样本。

**对应消融**: Table 7（未在 figures_available 中列出，但文本提及）显示以 Qwen2.5-VL-72b 替代 GPT-4o 蒸馏后，经后续训练可达到可比性能，说明冷启动主要教授格式而非决策质量。

---

### 模块 2: 多维度评分聚合机制（对应框架图 推理阶段）

**直觉**: 单标量奖励无法表达"美学优秀但文本错配"的复杂情况；显式拆解维度、独立评分再聚合，使推理过程可追溯、可干预。

**Baseline 公式** (传统标量奖励模型如 PickScore): $$s = R_{\phi}(x, y) \in \mathbb{R} \quad \text{(单标量输出)}$$
符号: $R_{\phi}$ = 奖励网络，$x$ = 条件，$y$ = 生成内容。

**变化点**: 单标量压缩了所有质量维度，无法解释；本文将评分显式分解为 n 个维度的独立评估。

**本文公式（推导）**:
$$\text{Step 1 (维度分解)}: \{s_{dim_1}, s_{dim_2}, ..., s_{dim_n}\} = \text{Score}_{\theta}^{multi}(x, y) \quad \text{模型对每个维度输出独立分数}$$
$$\text{Step 2 (聚合)}: s_{final} = f(s_{dim_1}, s_{dim_2}, ..., s_{dim_n}) = \text{Aggregate}(\{s_{dim_j}\}_{j=1}^n) \quad \text{聚合函数如加权求和或多数投票}$$
$$\text{Step 3 (偏好判断)}: \hat{y} = \text{arg}\max_{y \in \{y_{win}, y_{lose}\}} s_{final}(x, y) \quad \text{基于聚合分输出最终偏好}$$
此结构化设计减少了中间推理与最终结论之间的不一致性，因为每个维度的评分可独立验证。

---

### 模块 3: GRPO 强化微调目标（对应框架图 阶段三）

**直觉**: 拒绝采样后的模型已能生成"正确"推理，但缺乏对"错误边界"的探索；利用 GRPO 在错误样本上采样多样化响应，通过组内相对优势驱动策略改进，无需额外训练 critic 网络。

**Baseline 公式** (标准 PPO with critic):
$$\mathcal{L}^{CLIP}(\theta) = \hat{\mathbb{E}}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}\left(r_t(\theta), 1-\epsilon, 1+\epsilon\right)\hat{A}_t\right)\right]$$
$$A_i^{PPO} = r_i + \gamma V(s_{i+1}) - V(s_i) \quad \text{(需训练critic网络 } V\text{)}$$
符号: $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ = 重要性采样比率，$\hat{A}_t$ = 优势估计，$V$ = 价值函数。

**变化点**: PPO 需要独立的 critic 网络，显存开销大；GRPO 通过组内归一化消除 critic，且本文专门针对错误样本进行探索性训练。

**本文公式（推导）**:
$$\text{Step 1 (组采样)}: \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(\cdot|q), \quad G=8 \quad \text{每个查询采样8个响应}$$
$$\text{Step 2 (组相对优势)}: A_i = \frac{r_i - \text{mean}(\{r_1, ..., r_G\})}{\text{std}(\{r_1, ..., r_G\})} \quad \text{用组内均值和标准差归一化，无需critic}$$
$$\text{Step 3 (裁剪目标)}: \mathcal{L}_{GRPO}(\theta) = \mathbb{E}_{q, \{o_i\}} \left[ \frac{1}{G} \sum_{i=1}^G \min\left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right) A_i \right) \right]$$
$$\text{Step 4 (KL约束)}: - \beta \mathbb{D}_{KL}(\pi_\theta \| \pi_{ref}), \quad \beta = 0.04 \quad \text{防止策略偏离参考策略过远}$$
$$\text{最终}: \mathcal{L}_{GRPO}^{final}(\theta) = \mathcal{L}_{GRPO}(\theta) - \beta \mathbb{D}_{KL}(\pi_\theta \| \pi_{ref})$$

**对应消融**: Table 3 显示移除 CoT 推理（Ours w/o CoT）后，VLRewardBench Overall Accuracy 从 73.8 降至 73.1（-0.7），Macro Accuracy 从 72.3 降至 71.3（-1.0），验证了显式 CoT 的增益；Table 5 评估拒绝采样阶段的贡献。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f442e1f3-10ee-4e74-9111-5f2b074de427/figures/Table_1.png)
*Table 1 (comparison): Table 1. Image Understanding Assessment Comparison. We evaluate baselines across different categories on image understanding and generation tasks.*



本文在 VLRewardBench、GenAI-Bench（含图像/视频生成）、VideoGen-RewardBench 三个核心基准上评估。VLRewardBench 覆盖图像理解的 General、Hallucination、Reasoning 三个子维度，是视觉语言奖励模型最具挑战性的测试平台之一。


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f442e1f3-10ee-4e74-9111-5f2b074de427/figures/Table_2.png)
*Table 2 (comparison): Table 2. Image and Video Generation Assessment Comparison. 'uvs' indicates that accuracy is evaluated by user study.*



核心结果显示，UNIFIEDREWARD-THINK 在 VLRewardBench Image Understanding 上达到 Overall Accuracy 73.8，较直接 baseline UnifiedReward 的 67.5 提升 +6.3，较商业强模型 GPT-4o 的 65.8 提升 +9.6。更值得关注的是 Hallucination 子任务：72.7 vs UnifiedReward 58.1，提升高达 +14.6，说明长 CoT 推理对识别视觉-文本不一致性具有显著优势。即使在去除显式 CoT 的推理模式（Ours w/o CoT）下，模型仍保持 73.1 的 Overall Accuracy，暗示经过 CoT 训练后内化的隐式推理能力。GenAI-Bench 图像生成评估中，GPT-4o 蒸馏版本达 73.8，视频生成 72.5；VideoGen-RewardBench 上达 82.3，超越 baseline 77.2（+5.1）。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f442e1f3-10ee-4e74-9111-5f2b074de427/figures/Table_3.png)
*Table 3 (ablation): Table 3. Ablation Results of Image Understanding Assessment. We conduct ablation experiments on the training paradigm and reward modeling. 'uCoT' denotes our unified CoT reward model.*



消融实验进一步验证各组件必要性。Table 3 的图像理解消融表明，CoT 推理机制本身贡献 0.7 的 Overall Accuracy 增益（73.8 vs 73.1），而 Macro Accuracy 增益达 1.0（72.3 vs 71.3）。Table 4 的生成任务消融显示类似趋势。Table 5 专门评估拒绝采样阶段的贡献，该阶段作为冷启动与 GRPO 之间的桥梁，通过迭代保留正确推理轨迹实现质量精炼。Table 6 对比不同骨干网络，验证方法对 Qwen2.5-VL 系列的适应性。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/f442e1f3-10ee-4e74-9111-5f2b074de427/figures/Figure_3.png)
*Figure 3 (qualitative): Figure 3. Qualitative Results of Video Generation CoT Reward Reasoning. Given a pair of video and its detailed description, we analyze reward reasoning to assess semantic consistency, temporal coherence, and truthfulness through CoT reasoning.*



定性结果（Figure 3、Figure 4）展示模型在视频生成 CoT 推理、图像与视频理解任务上的逐步分析能力，包括维度拆解、矛盾识别与聚合判断的完整链条。

公平性检查：对比的 baseline 覆盖专用奖励模型（PickScore、HPSv2、ImageReward）、统一模型（UnifiedReward）、商业模型（GPT-4o、Gemini-1.5-Pro）及评估专用模型（LLaVA-Critic），选择较为全面。但需注意：GRPO 阶段使用 64 块 NVIDIA H20，计算成本显著高于 baseline 的标准训练；冷启动依赖 GPT-4o 蒸馏虽可用 Qwen2.5-VL-72b 替代，但仍构成资源门槛。作者披露的限制包括：未测试真实部署规模、未与 OpenAI o1 多模态版本直接对比、以及 "Ours w/o CoT" 的消融设计细节（是否同架构仅关闭 CoT 输出）未完全明确。

## 方法谱系与知识库定位

UNIFIEDREWARD-THINK 属于**多模态奖励模型**方法族，直接父方法为 **UnifiedReward**（统一架构支持图像/视频理解与生成奖励任务）。谱系关系上，本文沿 UnifiedReward 的跨任务统一思路，在五个关键 slot 进行结构性改造：

| Slot | 父方法 | 本文改造 |
|:---|:---|:---|
| architecture | Qwen2.5-VL 基础 VLM | 继承，增加 CoT 输出头 |
| objective | 单标量/直接响应 | 多维度结构化评分 + 显式推理链生成 |
| training_recipe | 标准 SFT on 偏好数据 | 三阶段：冷启动蒸馏 → 拒绝采样 → GRPO |
| data_curation | 原始偏好数据集 | 蒸馏 CoT 推理痕迹 + 一致性过滤 |
| inference | 直接输出判断 | 长 CoT 推理后聚合判断，支持 w/ CoT 与 w/o CoT 双模式 |

直接对比的近期工作：**DeepSeek-R1** 提供 RL 驱动 CoT 的范式灵感，但针对纯文本数学推理；**Visual-RFT** 为视觉强化微调的紧密相关工作，但未统一理解与生成任务，亦未显式构建长 CoT 推理；**GRPO**（源自 DeepSeekMath）作为算法组件被适配到多模态奖励场景，关键差异在于利用错误样本进行探索而非仅正确样本优化。

后续可拓展方向：(1) 降低冷启动对 GPT-4o 的依赖，探索弱监督或自举蒸馏；(2) 将 CoT 推理机制迁移至视频理解、3D 生成等更复杂模态；(3) 结合在线学习实现奖励模型的持续更新，避免静态训练后的分布偏移。

**标签**: 模态(image+video+text) / 范式(reinforcement fine-tuning + chain-of-thought) / 场景(multimodal reward modeling, preference evaluation) / 机制(group relative policy optimization, multi-dimensional scoring, rejection sampling) / 约束(unified across understanding & generation tasks, interpretable reasoning)

