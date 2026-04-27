---
title: 'UFO-RL: Uncertainty-Focused Optimization for Efficient Reinforcement Learning Data Selection'
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 基于单次不确定性估计的高效RL数据选择
- UFO-RL
- UFO-RL enables efficient RL trainin
acceptance: Poster
cited_by: 2
method: UFO-RL
modalities:
- Text
paradigm: Reinforcement Learning
baselines:
- 面向目标指令微调的影响力数据选择_LESS
- DAPO：大规模LLM强化学习开_DAPO
---

# UFO-RL: Uncertainty-Focused Optimization for Efficient Reinforcement Learning Data Selection

**Topics**: [[T__Math_Reasoning]] | **Method**: [[M__UFO-RL]] | **Datasets**: [[D__GSM8K]] (其他: GSM8K training time, DAPO-MATH-17K, Training Efficiency, Training Time, Data Evaluation)

> [!tip] 核心洞察
> UFO-RL enables efficient RL training by using single-pass uncertainty estimation to identify data within the model's Zone of Proximal Development, achieving comparable or superior performance with only 10% of data while reducing training time by up to 16x.

| 中文题名 | 基于单次不确定性估计的高效RL数据选择 |
| 英文题名 | UFO-RL: Uncertainty-Focused Optimization for Efficient Reinforcement Learning Data Selection |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2505.12457) · [Code](待公开) |
| 主要任务 | Math Reasoning (数学推理) |
| 主要 baseline | DAPO, LiMR, LESS, LIMO, PPO |

> [!abstract] 因为「RL微调需要多采样评估数据难度导致计算成本极高」，作者在「PPO + DAPO式多采样筛选」基础上改了「单次前向传播不确定性估计 + ZPD数据选择」，在「GSM8K / DAPO-MATH-17K」上取得「仅用10%数据达到全数据可比性能，训练时间最高加速16倍」

- 数据评估速度：单次不确定性估计 vs. 多采样准确率，加速 **185×**
- 训练时间：Qwen2.5-0.5B 从 1815s 降至 **140s**（**11×**）；Mistral 7B 从 22955s 降至 **1454s**（**16×**）
- 数据效率：仅使用 **10%** 选定数据，在 GSM8K 和 DAPO-MATH-17K 上达到与全数据训练可比或更优的准确率

## 背景与动机

当前大语言模型的强化学习微调面临严峻的计算瓶颈。以数学推理任务为例，现有方法（如 DAPO）需要对每个训练样本进行多次采样生成，通过统计正确率来评估数据难度和信息量，进而筛选出对模型学习最有价值的样本。这种多采样评估虽然有效，但意味着每轮数据筛选都需要执行数十次甚至上百次前向传播与自回归生成，计算开销随模型规模和数据量线性膨胀，成为 RL 训练流程中不可忽视的瓶颈。

具体而言，DAPO [23] 采用多采样准确率（Multiple-Sample Accuracy）作为数据信息量的代理指标：对同一问题采样 K 次，计算正确比例，过滤掉始终正确（太简单）或始终错误（太难）的样本。LiMR [10] 和 LESS [19] 分别从 RL 缩放和 SFT 角度探索数据选择，但仍依赖迭代式评估或梯度匹配等昂贵操作。LIMO [22] 展示了少量高质量数据即可激发推理能力，但未解决如何高效筛选的问题。这些方法的共同痛点在于：**数据信息量评估本身就需要大量计算**，形成"为了省训练而先花更多计算评估"的悖论。


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/67ba7fa6-d590-422d-98a5-42e2454eebdd/figures/Table_1.png)
*Table 1 (quantitative): Distributions of Multiple-Sample Accuracy on the GSM8K Training Set for the Qwen2.5...*



核心观察在于：模型输出概率分布中已蕴含不确定性信息，无需通过多次采样来间接估计。如果能从单次前向传播直接提取可靠的不确定性分数，就能彻底绕过多采样的计算陷阱。受此启发，本文提出 UFO-RL，将教育心理学中的"最近发展区"（Zone of Proximal Development, ZPD）理论引入数据选择：模型处于中等不确定性的样本——既非已掌握的简单题，也非完全不可解的难题——恰恰是最具教学价值的训练数据。本文通过单次不确定性估计快速定位这些"黄金样本"，实现数据评估和 RL 训练的双重效率飞跃。

## 核心创新

核心洞察：**单次前向传播的不确定性足以替代多采样蒙特卡洛估计**，因为模型输出分布的熵/置信度结构与多次采样的经验正确率高度相关，从而使 185× 数据评估加速和 10% 数据高效训练成为可能。

| 维度 | Baseline (DAPO/多采样) | 本文 (UFO-RL) |
|:---|:---|:---|
| 数据评估方式 | K 次独立生成，统计正确率 | 单次前向传播，提取置信度/不确定性 |
| 计算复杂度 | O(K × seq_len) 自回归生成 | O(1) 前向传播，无 next-token 迭代 |
| 数据选择标准 | 过滤极端（全对/全错）样本 | ZPD 区间：中等不确定性样本 |
| 数据使用比例 | 通常保留 30-50% | 仅 10% 精选数据 |
| RL 训练目标 | 标准 PPO 在全数据或粗筛子集 | 标准 PPO 在 ZPD 子集，目标函数不变 |

关键差异在于：UFO-RL 不修改 PPO 的核心优化目标，仅改变"喂给 PPO 什么数据"——这是一个轻量级的即插即用替换，却带来数据评估和训练时间的数量级提升。

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/67ba7fa6-d590-422d-98a5-42e2454eebdd/figures/Figure_1.png)
*Figure 1 (result): Impact of Sampling Accuracy-Based Data Selection Strategy on Model Learning Efficiency*



UFO-RL 采用三阶段流水线架构，将数据选择从 RL 训练的内循环中解耦出来：

**阶段一：单次不确定性估计（Single-Pass Uncertainty Estimation）**
输入为完整训练数据集 D 和当前策略模型 π_θ。对每个样本 x 执行一次前向传播（无自回归生成），从输出分布中提取不确定性分数 u(x) ∈ [0,1]。该模块替代了 DAPO 中需要 K 次采样的多准确率估计，是整个框架的计算瓶颈突破点。

**阶段二：ZPD 数据选择（ZPD-Based Selection）**
输入为所有样本的不确定性分数集合 {u(x_i)}。根据预设的上下阈值 τ_low 和 τ_high，筛选出中等不确定性样本构成 ZPD 子集 S_ZPD，通常 |S_ZPD| ≈ 0.1|D|。该模块实现"最近发展区"理论：太低的不确定性意味着模型已掌握，太高的不确定性意味着超出当前能力，中间区域才是最优学习区。

**阶段三：标准 RL 训练（PPO Training）**
输入为选定的 ZPD 子集 S_ZPD 和策略模型 π_θ。执行标准的 PPO 训练，使用 outcome-based reward（如答案正确性）。该模块完全复用现有 RL 基础设施，无需任何修改。

```
完整数据 D ──→ [单次前向传播] ──→ 不确定性分数 {u(x)}
                                      ↓
                              [ZPD 阈值过滤: τ_low < u(x) < τ_high]
                                      ↓
                              选定子集 S_ZPD (≈10%数据)
                                      ↓
                              [PPO 训练] ──→ 更新策略 π_θ'
```

整个框架的关键设计是"计算换数据"：用极轻量的单次估计（约 1/185 计算）换取 90% 数据的裁剪，最终训练时间随数据量线性下降，实现最高 16× 端到端加速。

## 核心模块与公式推导

### 模块 1: 单次不确定性估计（对应框架图 阶段一）

**直觉**: 模型对正确答案的置信度分布本身即蕴含不确定性信息，无需通过多次采样来经验估计。

**Baseline 公式** (DAPO 多采样准确率):
$$u_{\text{multi}}(x) = \frac{1}{K}\sum_{k=1}^{K} \mathbb{1}[\text{generate}(x; \pi_\theta, k) \text{ is correct}]$$
符号: K = 采样次数, generate(·) = 自回归生成完整回答, 𝟙[·] = 指示函数（正确为1，错误为0）。该公式需要 K 次完整的自回归生成，每次生成长度与答案序列长度成正比。

**变化点**: 多采样估计虽然无偏，但计算成本 O(K × L) 极高；且对于大模型，K 次独立采样可能消耗大量 GPU 内存和 wall-clock 时间。本文假设模型输出的 token-level 概率分布（特别是答案部分的平均置信度或熵）与经验正确率高度相关，可直接作为不确定性的代理。

**本文公式（推导）**:
$$\text{Step 1}: \quad p(y|x; \pi_\theta) = \prod_{t=1}^{L} \pi_\theta(y_t | x, y_{<t}) \quad \text{（标准前向传播，仅计算概率无生成）}$$
$$\text{Step 2}: \quad \text{conf}(x) = \frac{1}{L}\sum_{t=1}^{L} \log \pi_\theta(\hat{y}_t | x, \hat{y}_{<t}) \quad \text{（提取答案序列的平均对数似然/置信度）}$$
$$\text{Step 3}: \quad u(x) = g(\text{conf}(x)) \in [0,1] \quad \text{（归一化为不确定性分数，高置信度→低不确定性）}$$
$$\text{最终}: \quad u(x) = f(\text{forward\_pass}(x; \pi_\theta))$$
其中 g(·) 为单调递减映射函数（如基于训练集分布的百分位归一化），将置信度转换为与多采样准确率同量纲的不确定性分数。

**对应消融**: Table 2 显示多采样准确率与单次置信度分数的分布统计高度一致，验证了替代的有效性；Table 4 显示计算效率从多采样的高耗时降至单次估计的 1/185。

---

### 模块 2: ZPD 数据选择准则（对应框架图 阶段二）

**直觉**: 教育心理学中的"最近发展区"指出，最有效的学习发生在"跳一跳够得着"的区域——对应到模型即中等不确定性区间。

**Baseline 公式** (DAPO/标准过滤):
$$S_{\text{DAPO}} = \{x \in D : u_{\text{multi}}(x) \notin \{0, 1\}\} \quad \text{或} \quad S = D \text{ (全数据)}$$
符号: S = 选定子集, D = 完整数据集。DAPO 仅过滤掉极端样本（始终正确或始终错误），仍保留约 30-50% 数据；全数据训练则不进行任何选择。

**变化点**: 简单过滤极端值忽略了数据难度的连续分布特性。本文发现，模型在**中等不确定性**（非高非低）样本上学习收益最大：低不确定性样本无新信息，高不确定性样本超出当前能力导致梯度噪声大。

**本文公式（推导）**:
$$\text{Step 1}: \quad \text{计算全数据集不确定性分布: } \{u(x_i)\}_{i=1}^{|D|}$$
$$\text{Step 2}: \quad \text{确定 ZPD 区间: } [\tau_{\text{low}}, \tau_{\text{high}}] = \text{Percentile}(\{u(x_i)\}, p_{\text{low}}), \text{Percentile}(\{u(x_i)\}, p_{\text{high}})$$
$$\text{Step 3}: \quad \text{通常取 } p_{\text{low}} \approx 40\text{th}, p_{\text{high}} \approx 60\text{th} \text{ 以保留约 10\% 数据}$$
$$\text{最终}: \quad S_{\text{ZPD}} = \{x \in D : \tau_{\text{low}} < u(x) < \tau_{\text{high}}\}, \quad |S_{\text{ZPD}}| \approx 0.1|D|$$

**对应消融**: Table 3 显示使用最容易数据分位数（低不确定性）训练的 curriculum 显著劣于中等置信度选择，验证了 ZPD 假设；Table 5 显示不同选择策略中，UFO-RL 的 Top-Mid（中等不确定性）策略最优。

---

### 模块 3: ZPD 约束的 RL 目标函数（对应框架图 阶段三）

**直觉**: 在精选的 ZPD 子集上执行标准 PPO，不改变优化目标本身，仅优化数据分布。

**Baseline 公式** (标准 PPO):
$$\mathcal{L}_{\text{PPO}}^{\text{full}}(\theta) = \mathbb{E}_{(x,y) \sim D} [\mathcal{L}_{\text{PPO}}(\theta; x, y)]$$
符号: θ = 策略参数, D = 完整数据集, L_PPO = 带裁剪的替代目标 + 价值函数损失。

**变化点**: 无本质变化——这是本文的刻意设计。UFO-RL 作为即插即用模块，完全兼容现有 RL 训练栈，仅将采样分布从 D 替换为 S_ZPD。

**本文公式**:
$$\mathcal{L}_{\text{RL}}(\theta) = \mathbb{E}_{(x,y) \sim S_{\text{ZPD}}} [\mathcal{L}_{\text{PPO}}(\theta; x, y)]$$

**对应消融**: Table 6 显示在保持相同 PPO 超参数和奖励函数的前提下，仅改变数据子集即可实现 11-16× 训练加速且性能不下降。

## 实验与分析


![Table 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/67ba7fa6-d590-422d-98a5-42e2454eebdd/figures/Table_5.png)
*Table 5 (result): Performance of Different Data Selection Strategies*



本文在数学推理基准上进行了系统性评估。核心结果见 Table 5：UFO-RL 在 GSM8K 和 DAPO-MATH-17K 上仅使用 10% 的选定数据，即可达到与全数据训练（Full Data）以及 DAPO、LiMR、LESS、LIMO 等基线相当或更优的准确率。具体而言，在 Qwen2.5-7B 等主流模型上，Top-Mid 置信度选择策略稳定优于随机选择和极端过滤策略，验证了 ZPD 假设的有效性。这一结果的关键意义在于：它证明了数据"质"的选择可以弥补"量"的缩减，打破了"RL 训练必须遍历全量数据"的隐含假设。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/67ba7fa6-d590-422d-98a5-42e2454eebdd/figures/Table_3.png)
*Table 3 (comparison): Comparison of Qwen2.5-7B training performance using the curriculum derived from the multiple-sample accuracy*



效率方面，Table 6 展示了端到端训练时间的对比：对于 Qwen2.5-0.5B，UFO-RL 仅需 140 秒，相比全数据训练的 1815 秒实现 **11× 加速**；对于 Mistral 7B，从 22955 秒降至 1454 秒，实现 **16× 加速**。这一加速比接近数据压缩比（10×）的线性放大，说明数据选择的开销被充分摊薄。Table 4 进一步揭示，单次不确定性估计本身相比多采样准确率计算实现了 **185× 的数据评估加速**，这是整体训练效率提升的根源。


![Table 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/67ba7fa6-d590-422d-98a5-42e2454eebdd/figures/Table_6.png)
*Table 6 (comparison): RL Training Time Comparison: Training on UFO (Top-Mid) Confidence Data vs. Full Data*



消融实验（Table 3）显示，若将 ZPD 选择替换为最容易数据分位数（Lowest uncertainty），模型学习曲线显著劣化，说明"中等难度"而非"简单数据"才是高效学习的关键。Table 2 通过统计相关性分析，验证了单次置信度分数与多采样准确率的分布一致性，为替代方案的可靠性提供了实证支撑。

公平性检查：本文对比的基线涵盖了当前数据高效 RL 的代表性工作（DAPO、LiMR、LESS、LIMO），且全数据训练作为天然上界，比较设置合理。但需注意：评估局限于数学推理领域（GSM8K、DAPO-MATH-17K），MMLU 仅作辅助提及；10% 的选择比例未见敏感性分析；未开源代码影响独立复现；对于超大规模模型（如 70B+），不确定性估计的质量是否保持稳定尚待验证。此外，185× 和 16× 的加速数字分别对应"数据评估"和"端到端训练"两个不同环节，需避免混淆。

## 方法谱系与知识库定位

UFO-RL 属于 **数据高效的强化学习训练** 方法家族，直接构建于 **PPO** [14] 之上作为 lineage parent。核心改动 slot 为：
- **data_pipeline**: 用单次不确定性估计替换多采样评估（DAPO 式）
- **training_recipe**: 引入 ZPD 理论指导的数据选择，通常选 10% 中等不确定性数据
- reward_design / architecture / inference: 无修改（即插即用兼容）

直接 baselines 及差异：
- **DAPO** [23]: 使用多采样准确率过滤数据 → UFO-RL 用单次估计替代，185× 评估加速
- **LiMR** [10]: RL 缩放中的数据选择 → UFO-RL 从"少即是多"扩展到"单次即知少"
- **LESS** [19]: SFT 场景的影响力数据选择 → UFO-RL 将数据选择思想迁移至 RL 设置
- **LIMO** [22]: 极少数据激发推理能力 → UFO-RL 解决"如何高效筛选这极少数据"

后续可拓展方向：
1. **跨领域验证**: 从数学推理扩展至代码生成、科学问答等需要长程推理的任务
2. **动态 ZPD 调整**: 训练过程中模型能力演化，ZPD 区间应自适应移动而非固定阈值
3. **不确定性估计增强**: 结合模型集成或贝叶斯神经网络提升不确定性校准，服务更大规模模型

标签: **modality=text** | **paradigm=reinforcement_learning** | **scenario=math_reasoning** | **mechanism=uncertainty_estimation + curriculum_learning** | **constraint=compute_efficiency, data_efficiency**

## 引用网络

### 直接 baseline（本文基于）

- [[P__面向目标指令微调的影响力数据选择_LESS]] _(直接 baseline)_: Direct precursor on data selection for instruction tuning; 'LESS' is closely rel
- [[P__DAPO：大规模LLM强化学习开_DAPO]] _(实验对比)_: Open-source RL system; likely used as comparison baseline for implementation and

