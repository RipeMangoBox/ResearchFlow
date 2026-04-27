---
title: 'Model Capability Dominates: Inference-Time Optimization Lessons from AIMO 3'
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2603.27844
aliases:
- AIMO 3竞赛揭示推理时优化瓶颈
- MCDITO
code_url: https://github.com/nat-nischw/model-capability-dominates-lessons-aimo3
modalities:
- Text
---

# Model Capability Dominates: Inference-Time Optimization Lessons from AIMO 3

[Paper](https://arxiv.org/abs/2603.27844) | [Code](https://github.com/nat-nischw/model-capability-dominates-lessons-aimo3)

**Topics**: [[T__Math_Reasoning]], [[T__Benchmark_-_Evaluation]]

| 中文题名 | AIMO 3竞赛揭示推理时优化瓶颈 |
| 英文题名 | Model Capability Dominates: Inference-Time Optimization Lessons from AIMO 3 |
| 会议/期刊 | arXiv (Cornell University) (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2603.27844) · [Code](https://github.com/nat-nischw/model-capability-dominates-lessons-aimo3) · [Project](https://arxiv.org/abs/2603.27844) |
| 主要任务 | 数学竞赛推理（AIMO 3）、推理时优化（inference-time optimization）、majority voting策略分析 |
| 主要 baseline | Qwen3.5-35B-A3B, gpt-oss-120b, gpt-oss-20b, 标准 majority voting (N=3,8,16,32,48,64) |

> [!abstract] 因为「推理时扩展（inference-time scaling）被广泛认为能提升LLM推理性能，但其在高难度数学竞赛中的实际收益边界不明」，作者在「标准majority voting和多次采样策略」基础上做了「系统性的竞赛级实证分析，涵盖模型能力、投票数N、per-problem成功率ρ与per-attempt准确率p的关系」，在「AIMO 3竞赛 leaderboard」上取得「42分（对比winner 46分，揭示模型能力主导而非推理时技巧）」

- **关键性能 1**: 标准 majority voting 在 N=16 时 µ=39.3, σ=1.7，Mixer 策略仅 µ≈39.0, σ=2.0，无实质提升（Figure 7）
- **关键性能 2**: Qwen3.5-35B-A3B 在10道本地题目上的消融实验：baseline 8/10，所有改进尝试均未超过 8/10，部分导致 crash（Figure 4）
- **关键性能 3**: AIMO 3 竞赛 winner 46 分，本工作 42 分；历史趋势显示 winner 从 29→34→46，而投票数 N 从 48→64→8，呈反比关系（Figure 6）

## 背景与动机

当前LLM推理的一个核心信念是：通过增加推理时计算（inference-time compute）——如更多采样路径、更复杂的投票聚合策略、自我验证机制——可以弥补模型能力的不足。具体而言，在数学竞赛等需要严格多步推理的场景中，研究者普遍假设"只要采样足够多次，正确答案终会浮现"。例如，面对一道IMO级别的几何证明题，标准做法是使用 Qwen 或 GPT 系列模型生成 N=16 或 N=64 个候选解答，然后通过 majority voting 选出最常见答案。

现有方法主要沿三条路径展开：（1）**Majority voting / self-consistency**（Wang et al., 2023）：简单多数决，假设独立样本的错误互不相关；（2）**Weighted voting / reward model reranking**：引入过程奖励模型（PRM）或结果奖励模型（ORM）对候选答案排序，如 Best-of-N 采样；（3）**Tree search with verifier**（如 Monte Carlo Tree Search + LLM）：在推理空间中进行系统性搜索，依赖验证器剪枝。

然而，这些方法在 AIMO（AI Mathematical Olympiad）这类顶级竞赛中的实际表现揭示了一个被忽视的瓶颈：**per-attempt准确率 p 过低时，任何投票聚合策略的边际收益急剧衰减**。Figure 1 显示，当单模型单次尝试的准确率 p̂ 低于约 0.5 时，即使 Binomial voting 将 N 扩展到 32，期望的 majority-vote score 也远未饱和。更关键的是，Figure 3 表明不同模型（Qwen N=16, gpt-oss-120b N=8, gpt-oss-20b N=8）的 per-problem 成功率 ρ̂ 与 p̂ 高度线性相关，意味着**问题层面的难度差异无法通过简单增加 N 来平滑**。Figure 4 的消融实验更是直接：Qwen3.5-35B-A3B 在10道本地题目上，所有推理时优化尝试（underperform 7/10, crashed）均未超过 baseline 的 8/10。

本文的核心动机由此明确：在 AIMO 3 的极端难度下，**模型固有能力（capability）而非推理时技巧（inference-time trick）才是性能的决定性因素**，这一结论与当前社区过度投资 inference-time optimization 的趋势形成张力。作者通过竞赛级实证，系统解构了"更多采样=更好结果"的隐含假设。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/7f03dfae-d021-45b9-9807-0ec785bf941c/figures/Figure_1.png)
*Figure 1: Figure 1: Model capability dominates. Per-attempt accuracy ˆp vs. expected majority-votescore under Binomial voting at N=3, 8, 16, 32. Seven data points across four model families.At equal N=8, the 8-*



## 核心创新

核心洞察：per-attempt 准确率 p 是推理时扩展收益的上界决定因素，因为 Binomial voting 的期望得分本质是 p 和 N 的函数 E[score] = f(p, N)，而当 p 低于临界阈值时，增大 N 的边际收益被问题间方差吞噬，从而使"模型能力投资优先于推理时技巧投资"成为最优策略。

| 维度 | Baseline（标准 majority voting + 大 N 采样） | 本文 |
|------|------------------------------------------|------|
| 核心假设 | 增加 N 可以补偿低 p，最终收敛到正确答案 | p 是硬瓶颈，低 p 时投票聚合收益有上界 |
| 优化目标 | 最大化 E[score] 通过增大 N | 识别 p-ρ 关系的线性约束，重新分配优化资源 |
| 验证场景 | 标准 benchmark（GSM8K, MATH） | AIMO 3 竞赛级难度（历史最高分 46/100） |
| 关键指标 | 绝对得分 | per-problem ρ̂ vs. per-attempt p̂ 的散点关系（Figure 3） |
| 策略结论 | 继续扩展 N 或改进投票机制 | 优先提升 base model 的 p，而非 inference-time 技巧 |

## 整体框架


![Figure 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/7f03dfae-d021-45b9-9807-0ec785bf941c/figures/Figure_6.png)
*Figure 6: Figure 6: AIMO competition progression. Orange bars: winner/top LB scores (29→34→46).Blue bar: this work (42). Red line: voters N (48→64→8). High-N voting gives way to high-ˆpcapability.*



本文的实证框架是一个**竞赛驱动的诊断系统**，旨在解构 inference-time optimization 的真实收益边界。数据流如下：

**输入层**：AIMO 3 竞赛题目（高难度数学竞赛，历史得分范围 29-46）+ 多个开源/闭源 base model（Qwen3.5-35B-A3B, gpt-oss-120b, gpt-oss-20b）。

**模块 A - 采样生成（Generation）**：对每个问题，各模型生成 N 个独立候选解答（N ∈ {3, 8, 16, 32, 48, 64}，不同模型配置不同）。输入：题目文本 + 模型权重；输出：N 个候选解答字符串。

**模块 B - 答案提取与匹配（Answer Parsing & Matching）**：从候选解答中提取最终数值/表达式答案，与 ground truth 比对。输入：N 个候选解答；输出：per-attempt 二元正确标签（correct/incorrect）。

**模块 C - 聚合策略（Aggregation）**：应用标准 majority voting（Binomial 假设）及改进策略（Mixer 等）。输入：N 个二元标签；输出：单题预测答案 + 聚合置信度。

**模块 D - 诊断分析（Diagnostic Analysis）**：计算 per-attempt 准确率 p̂、per-problem 成功率 ρ̂、期望 majority-vote score 的理论曲线，并与实际观测对比。输入：聚合结果 + 真实标签；输出：p̂-ρ̂ 散点、score 分布、策略消融结论。

**输出层**：竞赛得分、策略对比结论、资源重分配建议。

```
[AIMO 3 Problem] → [Generation: N samples] → [Answer Parsing] → [Aggregation: voting] → [Diagnostic: p̂, ρ̂, E[score]] → [Conclusion]
                                    ↑___________________________________________________________↓
                                                        (feedback: model capability assessment)
```

## 核心模块与公式推导

### 模块 1: Binomial Majority-Vote 期望得分（对应框架图 模块 C）

**直觉**: 在独立同分布假设下，N 次尝试中正确答案出现次数的分布服从二项分布，由此可计算 majority voting 的期望性能上界。

**Baseline 公式** (标准 self-consistency / majority voting):
$$E[\text{score}_i] = \sum_{k=\lceil N/2 \rceil}^{N} \text{binom}{N}{k} p_i^k (1-p_i)^{N-k}$$
符号: $p_i$ = 模型对问题 $i$ 的 per-attempt 准确率, $N$ = 投票数/采样次数, $E[\text{score}_i]$ = 问题 $i$ 的期望得分（二元正确为1，否则0）。

**变化点**: 标准分析假设 $p_i$ 可通过 scaling 任意提升，或 $N$ 可无限增大。但在 AIMO 3 中，实际观测到 $p_i \ll 0.5$ 对大量问题成立，此时 $E[\text{score}_i]$ 随 $N$ 增长极慢（Figure 1 的平坦区域）。

**本文公式（推导）**:
$$\text{Step 1}: \quad \hat{p}_i = \frac{1}{N}\sum_{j=1}^{N} \mathbb{1}[\text{answer}_{ij} = \text{truth}_i] \quad \text{（从采样估计单题准确率）}$$
$$\text{Step 2}: \quad \hat{\rho}_i = \mathbb{1}[\text{majority\_vote\_correct}_i] \quad \text{（实际观测的问题级成功率）}$$
$$\text{Step 3}: \quad \text{拟合关系}: \hat{\rho}_i = g(\hat{p}_i) + \epsilon_i \quad \text{（检验理论二项界与实际的相关性）}$$
$$\text{最终}: \quad \text{Figure 3 显示 } \hat{\rho} \text{ vs. } \hat{p} \text{ 呈近似线性，斜率 } < 1 \text{，说明问题间异质性导致聚合效率损失}$$

**对应消融**: Figure 1 显示，当 $p \approx 0.3$ 时，$N$ 从 16 增至 32 仅提升 E[score] 约 0.05；Figure 6 显示竞赛历史 winner 的 $N$ 从 48→64→8 递减，而得分从 29→34→46 递增，反向验证模型能力主导。

---

### 模块 2: Mixer 策略与方差分析（对应框架图 模块 C 改进策略）

**直觉**: 若标准 voting 收益饱和，尝试通过"混合"不同策略（如结合过程奖励模型打分、多样性采样、自我修正）打破独立同分布假设，检验是否能超越二项上界。

**Baseline 公式** (标准 Best-of-N / weighted voting):
$$\text{score}_{\text{BoN}} = \max_{j \in [N]} r(\text{answer}_j) \cdot \mathbb{1}[\text{answer}_j \text{ parsed}]$$
其中 $r(\cdot)$ 为奖励模型分数。

**变化点**: 在 AIMO 3 中，奖励模型本身受限于 base model 能力，$r(\cdot)$ 的排序可靠性随问题难度下降；且自我修正（self-correction）常引入错误累积（Figure 4 的 "crashed" 案例）。

**本文公式（推导）**:
$$\text{Step 1}: \quad \text{Mixer}(\{\text{answer}_j\}_{j=1}^N) = \text{Vote}(\{\text{answer}_j\}_{j \in S_{\text{high}}} \cup \{\text{answer}_j^{\text{(revised)}}\}_{j \in S_{\text{low}}})$$
$$\quad \text{其中 } S_{\text{high}} = \{j: r_j > \tau\}, \quad S_{\text{low}} = \{j: r_j \leq \tau\} \text{ 触发 revision}$$
$$\text{Step 2}: \quad \text{方差分解}: \sigma^2_{\text{Mixer}} = \sigma^2_{\text{model}} + \sigma^2_{\text{revision}} + 2\text{Cov}(\text{model}, \text{revision})$$
$$\text{最终}: \quad \text{Figure 7 显示 Mixer } (\mu \approx 39.0, \sigma = 2.0) \text{ vs. Baseline } (\mu = 39.3, \sigma = 1.7)$$
$$\text{Mixer 均值更低、方差更高，且均未达到目标 42 分（红线），策略失效}$$

**对应消融**: Figure 4 的 Qwen3.5-35B-A3B 消融：baseline 8/10，underperform 7/10，crashed 若干，**没有任何改进超过 baseline**，直接证明 Mixer 类策略在竞赛级难度下的负收益或零收益。

---

### 模块 3: 竞赛级诊断指标 ρ̂-p̂ 线性模型（对应框架图 模块 D）

**直觉**: 若模型能力主导，则跨模型、跨问题的 ρ̂-p̂ 关系应呈现稳定的统计规律，可用于预测新模型的 inference-time 收益天花板。

**Baseline 公式** (理想二项模型):
$$\rho_i^{\text{ideal}} = \mathbb{1}\left[\sum_{j=1}^{N} X_{ij} > \frac{N}{2}\right], \quad X_{ij} \sim \text{Bernoulli}(p_i) \text{ i.i.d.}$$

**变化点**: 实际中 $X_{ij}$ 非独立（模型对同类错误有系统性偏见），且 $p_i$ 跨问题高度异质（部分问题 $p_i \approx 0$，部分 $p_i > 0.5$）。

**本文公式（推导）**:
$$\text{Step 1}: \quad \text{去理想化}: \quad X_{ij} = p_i + \delta_i + \epsilon_{ij}$$
$$\quad \text{其中 } \delta_i \sim \mathcal{N}(0, \sigma^2_{\text{problem}}) \text{ 为问题级随机效应，} \epsilon_{ij} \text{ 为尝试级噪声}$$
$$\text{Step 2}: \quad \text{聚合正确概率}: \quad P(\rho_i=1) = P\left(\sum_j X_{ij} > \frac{N}{2}\right) = \Phi\left(\frac{Np_i - N/2}{\sqrt{N\sigma^2_{\text{eff}}}}\right)$$
$$\text{Step 3}: \quad \text{Figure 3 实证拟合}: \quad \hat{\rho}_i = \alpha + \beta \hat{p}_i + \epsilon_i, \quad \beta < 1 \text{（跨模型一致）}$$
$$\text{最终}: \quad \text{线性关系意味着 } \frac{\partial \rho}{\partial N}\bigg|_{p \text{ fixed}} \approx 0 \text{ 当 } p < p_{\text{crit}} \approx 0.4$$

**对应消融**: Figure 3 中四种符号（Qwen ○, gpt-oss-120b □, gpt-oss-20b ◆, hollow ◇）落在同一趋势线附近，验证该关系的**跨模型稳健性**，即无论模型架构如何，低 p 区域的 inference-time 扩展收益均被锁定。

## 实验与分析

| Method | AIMO 3 Score | N (voters) | σ (stability) | 关键结论 |
|--------|-----------|-----------|--------------|---------|
| Winner/Top LB | 46 | 8 | — | 历史最高分，低 N 高能力 |
| This work | **42** | — | — | 本文最终提交 |
| Baseline majority voting | µ=39.3 | 16 | 1.7 | Figure 7 蓝色分布 |
| Mixer (改进策略) | µ≈39.0 | 16 | 2.0 | Figure 7 橙色分布，**无提升** |
| Historical winner (prev.) | 34 | 64 | — | 高 N 低得分 |
| Historical winner (earlier) | 29 | 48 | — | 更高 N 更低得分 |


![Figure 7](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/7f03dfae-d021-45b9-9807-0ec785bf941c/figures/Figure_7.png)
*Figure 7: Figure 7: Left: score distributions. Baseline (µ=39.3, σ=1.7, blue) vs. Mixer (µ≈39.0, σ≈2.0,orange). Red dashed line: target score 42. Right: cumulative probability of max ≥42 over Ksubmissions. Base*



**核心数字解读**：
- **42 vs. 46 的 4 分差距**：本文最终得分 42 接近但未达 winner 的 46，关键差异在于模型 base capability 而非 inference-time 技巧——Figure 6 显示 winner 仅用 N=8 即获 46 分，而历史高分伴随 N 递减（48→64→8），形成强烈的**反比证据**。
- **39.3 vs. 39.0 的 Mixer 失效**：Figure 7 左图显示 Mixer 策略（橙色）相比 baseline（蓝色）均值更低（-0.3）、方差更高（+0.3），且两者均未触及 42 分目标线（红色虚线）。这说明**复杂的推理时策略在统计上无显著收益，甚至可能引入不稳定性**。
- **Figure 4 的消融崩溃**：Qwen3.5-35B-A3B 在10道本地题上的所有改进尝试——包括可能的 prompt engineering、self-correction、工具调用——均未超过 baseline 的 8/10，部分直接 crash。这是**最直接的负向消融证据**。


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/7f03dfae-d021-45b9-9807-0ec785bf941c/figures/Figure_4.png)
*Figure 4: Figure 4: Qwen3.5-35B-A3B ablation on 10 local problems.Blue: baseline (8/10).Orange:underperform (7/10). Red: crashed. Nothing improves beyond baseline.*



**公平性检验**：
- **Baseline 强度**：对比的 gpt-oss-120b/20b 和 Qwen3.5-35B-A3B 均为当时主流开源/开放权重模型，非刻意弱 baseline。
- **计算成本**：Figure 6 的 N 递减趋势暗示 winner 可能使用了更高效的推理策略或更强的 base model，但核心结论是"模型能力允许低 N 高得分"，与本文论点一致。
- **失败案例**：Figure 4 的 "crashed" 标记和 Figure 7 的右图（未展示细节但暗示分布尾部风险）表明，inference-time 技巧在极端情况下可能**主动损害**性能。
- **数据局限**：本地消融仅 10 题，但竞赛级全量结果（Figure 6-7）与之一致，增强可信度。

## 方法谱系与知识库定位

**方法家族**：Inference-time scaling / Test-time compute optimization → **诊断性实证研究**（非新算法，而是对现有范式的系统性解构）。

**Parent method**: Majority voting / self-consistency (Wang et al., 2023) 与 Best-of-N sampling。本文不提出新聚合算法，而是**用竞赛数据重新标定这些方法的适用边界**。

**变动槽位**：
- **objective**: 从"最大化 E[score]"转向"识别 p 的硬瓶颈"
- **training_recipe**: 无——纯推理时分析，不涉及训练
- **data_curation**: AIMO 3 竞赛级难度（区别于 GSM8K/MATH 等标准 benchmark）
- **inference**: 系统性消融 N 和策略，而非单一最优配置

**直接 Baselines 及差异**：
| Baseline | 本文差异 |
|---------|---------|
| Standard majority voting (N=16,32,64) | 揭示低 p 时 E[score] 饱和，N 的边际收益递减 |
| Best-of-N with reward model | 证明奖励模型本身受 base model 能力限制，排序不可靠 |
| Self-correction / revision loops | 实证显示可能 crash 或 underperform，负收益风险 |
| Historical AIMO winners (N=48,64) | 反证：winner 的 N 递减而得分递增，模型能力主导 |

**后续方向**：
1. **Pre-training / post-training 投资优先**：将资源从 inference-time 技巧转向提升 base model 的 p，特别是数学推理的预训练数据质量和后训练 RLHF/RLAIF。
2. **Problem-adaptive N**：基于 p̂ 的实时估计动态调整 N，对高 p 问题少采样、对低 p 问题放弃或转交更强模型。
3. **跨竞赛验证**：在 Putnam、IMO Shortlist 等更高难度基准上检验 ρ̂-p̂ 线性关系的稳健性，建立 universal scaling law。

**知识库标签**：
- **modality**: text / mathematical reasoning
- **paradigm**: empirical analysis / diagnostic study / competition benchmark
- **scenario**: high-stakes mathematical olympiad / extreme difficulty regime
- **mechanism**: majority voting / Binomial statistics / inference-time compute scaling
- **constraint**: low per-attempt accuracy / model capability bottleneck / diminishing returns of sampling

