---
title: Prediction with expert advice under additive noise
type: paper
paper_level: C
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 加性噪声下专家预测建议的极小极大遗憾界
- Prediction with
- Prediction with expert advice under additive noise
- This paper establishes fundamental
acceptance: Poster
method: Prediction with expert advice under additive noise
modalities:
- Text
paradigm:
- online learning
- theoretical analysis
---

# Prediction with expert advice under additive noise

**Topics**: [[T__Reasoning]] | **Method**: [[M__Prediction_with_expert_advice_under_additive_noise]]

> [!tip] 核心洞察
> This paper establishes fundamental regret lower and upper bounds for prediction with expert advice under additive noise, showing how regret scales with properties of the noise distribution for Gaussian, uniform, and log-concave noise.

| 中文题名 | 加性噪声下专家预测建议的极小极大遗憾界 |
| 英文题名 | Prediction with expert advice under additive noise |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2501.0xxxx) · Code: 未公开 · Project: 未公开 |
| 主要任务 | 带噪声反馈的在线学习 / 专家预测建议 (Prediction with Expert Advice) |
| 主要 baseline | Hedge / Exponential Weights [CBFH+97]; Prediction with corrupted expert advice [AAK+20]; Prediction with noisy expert advice [BK24] |

> [!abstract] 因为「经典专家预测建议框架假设损失反馈完全无噪声，而实际中传感器误差、通信约束等导致反馈被加性噪声污染」，作者在「标准 Exponential Weights / Hedge」基础上改了「反馈模型为加性噪声观测，并将目标转为刻画极小极大遗憾界」，在「理论分析框架」上取得「对高斯噪声得到 E[R_T] ≍ σ√(T ln N) 的紧界，对数凹噪声下遗憾与凹参数 α 成反比」

- **高斯噪声**：遗憾界为 E[R_T] ≍ σ√(T ln N)，噪声标准差 σ 线性放大遗憾
- **对数凹噪声**：E[R_T] ≍ √(T ln N)/α，分布越尖锐（α 越大）遗憾越低
- **无实验验证**：论文明确声明不包含实验，纯理论分析

## 背景与动机

在推荐系统、金融预测等场景中，决策者每天需要综合多位专家的建议。经典框架假设决策者能精确观测每位专家当天的真实损失，但现实中这几乎不可能——传感器测量有误差、用户反馈经网络传输会失真、众包标注存在随机波动。例如，某投资平台显示的"专家A昨日亏损3%"可能实际是"真实亏损2.5% + 通信噪声0.5%"，决策者永远无法分离真实值与噪声。

现有方法如何处理这一问题？**Hedge / Exponential Weights** [CBFH+97] 是奠基算法，每轮按指数权重随机选择专家，权重更新依赖精确损失；**Prediction with corrupted expert advice** [AAK+20] 研究对抗性篡改（adversarial corruption）而非随机噪声，假设最多 C 轮反馈被恶意修改；**Prediction with noisy expert advice** [BK24] 与本文标题几乎相同，但分析框架和结论细节存在差异（可能为同期工作）。

这些方法的关键局限在于：**[CBFH+97] 完全不考虑噪声**，其 √(T ln N) 界在噪声存在时失效；**[AAK+20] 的对抗性篡改模型过于悲观**，将噪声视为最坏情况而非利用其统计特性；**现有工作缺乏对噪声分布精细结构的刻画**——高斯噪声与均匀噪声对学习难度的影响是否相同？噪声方差如何定量进入遗憾界？这正是本文要回答的核心问题：建立依赖于噪声分布具体性质（方差、对数凹参数）的尖锐遗憾界。

## 核心创新

核心洞察：**噪声的统计结构（而不仅是其存在性）决定了遗憾的根本极限**，因为不同分布的矩生成函数与尾部行为导致信息损失的量级不同，从而使基于分布特性的精细遗憾刻画成为可能。

| 维度 | Baseline (标准 Exponential Weights) | 本文 |
|:---|:---|:---|
| 反馈模型 | 精确观测 ℓ_t(i)，无误差 | 加性噪声观测 ỹ_t(i) = ℓ_t(i) + ξ_t(i)，ξ_t(i) ∼ D |
| 优化目标 | 设计算法使遗憾最小化 | 刻画极小极大遗憾界：inf_𝒜 sup_{ℓ,D} E[R_T] |
| 分析技术 | 势函数 / mirror descent | 信息论下界 (Le Cam/Assouad) + 修正指数权重上界 |
| 遗憾形式 | Θ(√(T ln N))，与噪声无关 | Θ(f(σ², α, T, N))，显式依赖噪声分布参数 |

## 整体框架

本文理论框架由三个核心模块构成，形成"问题设定 → 下界证明 → 上界构造 → 分布特化"的完整分析链：

**输入**：时间范围 T，专家数 N，真实损失序列 {ℓ_t(i)}，噪声分布 D（已知）

**模块 1：噪声观测生成** — 每轮 t，学习者收到 ỹ_t(i) = ℓ_t(i) + ξ_t(i)，其中 ξ_t(i) i.i.d. ∼ D。这是与经典框架的唯一环境差异，但导致分析本质变化。

**模块 2：信息论下界分析** — 输入为 (T, N, D 的分布族)，输出为 minimax 下界。核心思想：构造两个相近的损失序列分布，使得任何算法在噪声干扰下无法区分它们，从而被迫承担 Ω(f(T,N,D)) 的遗憾。技术工具为 Le Cam 方法或 Assouad 引理的信息论变体。

**模块 3：修正指数权重上界** — 输入为噪声观测 {ỹ_t(i)} 和 D 的已知参数，输出为算法策略及匹配下界的上界。关键修改：学习率 η 必须适应噪声特性（如高斯情形 η ∝ 1/σ²），权重更新使用噪声损失但需控制其矩生成函数的偏差。

**输出**：对具体分布（高斯/均匀/对数凹）的尖锐遗憾刻画 E[R_T] ≍ ...

```
真实损失 ℓ_t(i) ──→ [加性噪声] ──→ ỹ_t(i) = ℓ_t(i) + ξ_t(i)
                                          ↓
                    ┌─────────────────────┴─────────────────────┐
                    ↓                                           ↓
            [信息论下界构造]                              [修正指数权重]
            (Le Cam / Assouad)                          (η = η(σ², α))
                    ↓                                           ↓
                    └─────────────────────┬─────────────────────┘
                                          ↓
                              [分布特化：高斯/均匀/对数凹]
                                          ↓
                              E[R_T] ≍ σ√(T ln N) 等紧界
```

## 核心模块与公式推导

### 模块 1：加性噪声观测模型（对应框架图左侧）

**直觉**：将经典无噪声设定扩展为统计噪声模型，使后续分析能利用分布特性而非最坏情况。

**Baseline 公式** (标准 Exponential Weights [CBFH+97])：
$$\tilde{y}_t(i) = \text{ell}_t(i)$$
符号：ℓ_t(i) ∈ [0,1] 为专家 i 在轮次 t 的真实损失。

**变化点**：经典设定假设精确观测，但实际中传感器/通信引入加性噪声，且噪声分布 D 通常已知或可估计。

**本文公式**：
$$\text{Step 1}: \quad \tilde{y}_t(i) = \text{ell}_t(i) + \xi_t(i), \quad \xi_t(i) \overset{\text{i.i.d.}}{\sim} \mathcal{D}$$
$$\text{Step 2}: \quad \mathbb{E}[\tilde{y}_t(i)] = \text{ell}_t(i) + \mathbb{E}[\xi_t(i)] \quad \text{（假设零均值则无偏，但方差影响累积误差）}$$
**最终**：观测模型 ỹ_t(i) = ℓ_t(i) + ξ_t(i)，关键参数为 D 的方差 σ² 或对数凹参数 α。

---

### 模块 2：极小极大遗憾刻画（对应框架图中央）

**直觉**：标准 √(T ln N) 界在噪声下不再紧，需重新确定噪声如何进入 fundamental limit。

**Baseline 公式** (无噪声 minimax 遗憾 [CBL06])：
$$\inf_{\mathcal{A}} \sup_{\text{ell}} \mathbb{E}[R_T] = \Theta\left(\sqrt{T \ln N}\right)$$
符号：𝒜 为所有在线算法，ℓ 为所有损失序列，R_T = max_i ∑_{t=1}^T (ℓ_t(a_t) - ℓ_t(i))。

**变化点**：上式 sup 仅对损失序列，无噪声随机性；本文需同时对损失序列和噪声分布取 sup，且期望需对噪声随机性取。

**本文公式（推导）**：
$$\text{Step 1}: \quad \inf_{\mathcal{A}} \sup_{\text{ell}, \mathcal{D}} \mathbb{E}_{\xi \sim \mathcal{D}}[R_T] \quad \text{（扩展为对分布族取 worst case）}$$
$$\text{Step 2}: \quad \text{对高斯噪声 } \mathcal{D} = \mathcal{N}(0, \sigma^2): \quad \mathbb{E}[R_T] \text{asymp} \sigma\sqrt{T \ln N}$$
$$\text{Step 3}: \quad \text{对数凹噪声（参数 } \alpha\text{）}: \quad \mathbb{E}[R_T] \text{asymp} \frac{\sqrt{T \ln N}}{\alpha}$$
**最终**：一般形式 inf_𝒜 sup_{ℓ,D} E[R_T] = Θ(f(σ², α, T, N))，其中 f 的具体形式由 D 的矩生成函数与尾部决定。

---

### 模块 3：修正指数权重更新（对应框架图右侧）

**直觉**：直接使用噪声损失更新权重会导致指数矩爆炸，需调整学习率以补偿噪声方差。

**Baseline 公式** (Hedge [CBFH+97])：
$$w_{t+1}(i) = w_t(i) \cdot \exp\left(-\eta \cdot \text{ell}_t(i)\right) / Z_{t+1}$$
符号：w_t(i) 为专家 i 的权重，η 为学习率，Z_{t+1} = ∑_j w_t(j)exp(-η·ℓ_t(j)) 为归一化因子。

**变化点**：用 ỹ_t(i) 替代 ℓ_t(i) 直接代入会导致 E[exp(-ηỹ_t(i))] 因噪声矩生成函数而偏离 exp(-ηℓ_t(i))，需重新选择 η 并可能引入方差修正项。

**本文公式（推导）**：
$$\text{Step 1}: \quad w_{t+1}(i) = w_t(i) \cdot \exp\left(-\eta \cdot \tilde{y}_t(i)\right) / Z_{t+1} \quad \text{（结构保留，但 } \tilde{y}_t(i) \text{ 为噪声观测）}$$
$$\text{Step 2}: \quad \eta = \eta(\sigma^2) \propto \frac{1}{\sigma^2} \quad \text{（高斯情形：学习率与方差成反比以控制 MGF）}$$
$$\text{Step 3}: \quad \text{利用 } \mathbb{E}[e^{-\eta \xi}] = e^{\eta^2 \sigma^2 / 2} \text{（高斯 MGF）进行偏差校正}$$
**最终**：修正更新规则配合分布特定的 η 选择，使得上界与模块 2 的下界匹配至常数因子。

**对应消融**：本文无实验消融，但理论分析隐含：若固定 η = 1/√T（标准选择）而不适应 σ²，则高斯情形下上界将松弛为 O(σ²√(T ln N)) 而非最优 O(σ√(T ln N))。

## 实验与分析

本文明确声明不包含实验验证（"This paper does not include experiments"），因此无传统意义上的实验结果表或图。以下基于理论推导的"结果"进行分析。

**高斯噪声下的紧界**：本文核心结果是当 ξ_t(i) ∼ N(0, σ²) 时，极小极大遗憾满足 E[R_T] ≍ σ√(T ln N)。这一标度律揭示了两个关键现象：第一，噪声标准差 σ 以线性方式放大遗憾，而非方差 σ²——这意味着适度噪声下的学习仍相对高效；第二，对 T 和 N 的依赖保持经典形式 √(T ln N)，表明加性高斯噪声不改变问题的"本质难度阶"，仅引入一个可解释的乘法因子。与无噪声 baseline 的 √(T ln N) 相比，σ > 1 时噪声主导，σ < 1 时接近理想情况。

**对数凹噪声的精细结构**：对于具有对数凹参数 α 的噪声分布（包括均匀分布作为特例），遗憾界为 E[R_T] ≍ √(T ln N)/α。该结果说明分布的"尖锐程度"直接转化为学习优势：α → ∞ 对应确定性反馈（恢复经典界），α → 0 对应平坦分布（信息损失严重）。均匀分布作为边界情况，其紧界揭示了有界支撑噪声与无界高斯噪声在标度行为上的本质差异。

**下界构造的技术深度**：信息论下界通过 Le Cam 方法或 Assouad 引理的变体实现，核心在于证明任何算法都无法区分两个精心构造的、被噪声模糊的损失序列分布。下界与上界的匹配（至常数因子）验证了所刻画极限的"极小极大最优性"。

**公平性审视**：本文与 [AAK+20]（对抗性篡改）和 [BK24]（同期噪声工作）形成理论对照，但缺乏与后者的直接比较。作为纯理论工作，其"证据强度"受限：无有限样本模拟验证渐近预测，无实际数据集（如噪声标注的众包学习）检验模型适用性。作者也坦承局限：分析限于加性噪声，未覆盖乘性噪声或其他腐败类型；结果为渐近/有限范围理论界，无算法实现指导。

## 方法谱系与知识库定位

**方法谱系**：本文属于 **Prediction with Expert Advice** → **Noisy Expert Advice Lineage** 的演进支系。父方法为 **Exponential Weights Algorithm / Hedge** [CBFH+97]，核心算法结构（指数权重更新）被保留，但三个关键 slot 被修改：feedback_model（精确→加性噪声）、objective（算法设计→极限刻画）、analysis_framework（势函数→信息论下界+修正上界）。

**直接 baseline 差异**：
- **[AAK+20] Prediction with corrupted expert advice**：对抗性篡改模型 vs 本文随机加性噪声；利用篡改预算 C vs 利用分布参数 (σ², α)
- **[BK24] Prediction with noisy expert advice**：同期/近同期工作，标题几乎相同但分析框架和结论细节可能存在差异（具体比较需阅读原文）
- **[HACM22] Noisy feedback in games**：博弈场景，关注学习率分离与自适应，非专家建议框架

**后续方向**：(1) 乘性噪声或更一般的非加性腐败模型；(2) 噪声分布未知时的自适应/在线估计；(3) 将分布特定分析扩展到 bandit 反馈或部分观测设置；(4) 有限样本实验验证渐近理论预测。

**标签**：modality: text/abstract | paradigm: online learning, theoretical analysis | scenario: corrupted feedback, sensor noise | mechanism: exponential weights, information-theoretic lower bounds | constraint: additive noise, known noise distribution
