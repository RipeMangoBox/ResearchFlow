---
title: 'GFT: From Imitation to Reward Fine-Tuning with Unbiased Group Advantages and Dynamic Coefficient Rectification'
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.14258
aliases:
- GFT：无偏组优势与动态系数修正的奖励微调
- GFT
method: GFT
paradigm: Reinforcement Learning
---

# GFT: From Imitation to Reward Fine-Tuning with Unbiased Group Advantages and Dynamic Coefficient Rectification

[Paper](https://arxiv.org/abs/2604.14258)

**Topics**: [[T__Imitation_Learning]], [[T__Reinforcement_Learning]], [[T__Text_Generation]] | **Method**: [[M__GFT]]

| 中文题名 | GFT：无偏组优势与动态系数修正的奖励微调 |
| 英文题名 | GFT: From Imitation to Reward Fine-Tuning with Unbiased Group Advantages and Dynamic Coefficient Rectification |
| 会议/期刊 | arXiv (Cornell University) (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.14258) · [Code] · [Project] |
| 主要任务 | 大语言模型的数学推理能力后训练（从SFT到Reward Fine-Tuning的过渡） |
| 主要 baseline | SFT, DPO, PPO, GRPO, RFT, Rejection Sampling |

> [!abstract] 因为「SFT在数学推理任务上持续退化且DPO/GRPO存在组内估计偏差与训练不稳定」，作者在「GRPO」基础上改了「引入无偏组优势学习GAL与动态系数修正DCR」，在「Numina-Math、MATH-lighteval、GSM8K」上取得「Qwen2.5-Math-1.5B准确率超越GRPO 2.3%-5.7%且KL散度更低」

- **关键性能1**: Qwen2.5-Math-1.5B在Numina-Math上，GFT达52.3% vs GRPO 47.0%（+5.3%），SFT仅41.2%且持续退化
- **关键性能2**: MATH-lighteval上GFT 47.8% vs GRPO 45.5%（+2.3%），收敛速度提升约2倍
- **关键性能3**: KL散度仅0.12 vs SFT的0.89，显著降低分布漂移（Figure 5）

## 背景与动机

当前大语言模型的数学推理后训练面临一个根本性困境：纯监督微调（SFT）通过模仿学习拟合参考答案，但模型在分布外问题上表现持续恶化——Qwen2.5-Math-1.5B在Numina-Math上SFT后准确率从基础模型的43%跌至41.2%（Figure 1）。这源于SFT的"模仿陷阱"：模型学会复制训练答案的表面模式，而非获得可泛化的推理能力。

现有方法试图通过奖励微调（Reward Fine-Tuning）突破这一瓶颈。**DPO** 将偏好学习转化为静态分类损失，但依赖预构建的偏好对，无法利用组内多样响应的细粒度信号。**PPO** 引入优势估计，需要额外的critic网络，内存开销大且训练不稳定。**GRPO** 作为近期代表，通过组内相对奖励消除critic需求，但其核心假设——组内奖励均值作为基线——在响应质量高度不均时引入显著估计偏差：当组内同时存在正确与错误答案时，简单平均会系统性地压低高质响应的优势、抬高低质响应的优势。

更深层的问题在于**动态训练过程的信号失真**。随着训练进行，模型分布漂移导致某些token的梯度更新幅度剧烈波动——GRPO的固定系数无法自适应修正这些异常梯度，造成收敛震荡甚至崩溃（Figure 4中移除DCR后出现"severe volatility"）。

本文提出GFT，通过无偏组优势学习（GAL）与动态系数修正（DCR）两个正交机制，实现从模仿学习到奖励微调的平稳过渡。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d5e430c1-2f94-46a2-bd9b-8f1044c4ace5/figures/Figure_1.png)
*Figure 1: Figure 1 Performance of Qwen2.5-Math-1.5B on Numina-Math. (a) Accuracy changes relative to the base model:SFT consistently degrades performance, highlighting catastrophic forgetting. (b) Accuracy acro*



## 核心创新

核心洞察：组内响应的标准化相对优势应当基于**响应质量分层**而非简单平均，因为不同质量层级的响应对基线的贡献存在系统性偏差；同时，训练动态的系数修正应当**逐token自适应**而非全局固定，从而使奖励信号既无偏又稳定。

| 维度 | Baseline (GRPO) | 本文 (GFT) |
|:---|:---|:---|
| 组内基线估计 | 算术平均 $\bar{R}$，对所有响应一视同仁 | 质量加权标准化，区分正确/错误/部分正确层级 |
| 优势计算 | $A_k = R_k - \bar{R}$，存在系统性偏差 | $A_k^{\text{unbiased}}$，消除组内异质性带来的估计误差 |
| 梯度系数 | 固定clip阈值，全局统一 | 动态系数修正DCR，按token级梯度历史自适应调整 |
| 训练稳定性 | 后期易出现剧烈波动（Figure 4） | 收敛曲线平滑，最终性能更高 |
| 与SFT的关系 | 跳跃式切换，损失不兼容 | 渐进过渡，兼容SFT的模仿学习阶段 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d5e430c1-2f94-46a2-bd9b-8f1044c4ace5/figures/Figure_2.png)
*Figure 2: Figure 2 GFT comprises two components: (1) Group Advantage Learning, which computes standardized relativeadvantages (Ak) from hybrid response groups (expert demonstrations, teacher outputs, and rollou*



GFT的整体数据流遵循"采样-评估-修正-更新"四阶段循环，核心由两个模块协同驱动：

**输入**: 数学问题 $q$ + 当前策略模型 $\pi_\theta$ + 参考模型 $\pi_{\text{ref}}$（通常为SFT checkpoint）

**模块A: 组响应采样（Group Sampling）** — 对问题 $q$ 采样 $G$ 个响应 $\{o_1, ..., o_G\}$，通过规则验证器或LLM-as-judge获得奖励 $\{R_1, ..., R_G\}$。输出：原始响应组及其奖励信号。

**模块B: 无偏组优势学习GAL（Group Advantage Learning）** — 对组内响应按质量分层（正确/错误/部分正确），计算标准化相对优势 $A_k^{\text{unbiased}}$。输出：消除估计偏差后的逐响应优势值。

**模块C: 动态系数修正DCR（Dynamic Coefficient Rectification）** — 监测每个token位置的梯度历史，对异常大/小的梯度系数进行自适应裁剪与重缩放。输出：修正后的稳定梯度信号。

**模块D: 策略更新（Policy Update）** — 结合GAL优势与DCR系数，执行类似GRPO的比率裁剪更新，但使用修正后的目标函数。输出：更新后的策略模型 $\pi_{\theta'}$。

```
问题 q → [组采样] → {o_k, R_k}_{k=1}^G
                ↓
        [GAL: 无偏组优势学习] 
                ↓
        A_k^{unbiased} (消除分层偏差)
                ↓
        [DCR: 动态系数修正]
                ↓
        rectified coefficients per-token
                ↓
        [策略更新] → π_θ' (KL约束)
```

关键设计：GAL与DCR正交——GAL解决"估计什么"（what to estimate），DCR解决"如何稳定更新"（how to update stably）。

## 核心模块与公式推导

### 模块 1: 无偏组优势学习 GAL（对应框架图 GAL框）

**直觉**: GRPO的简单平均基线会在组内混入不同质量响应时产生"稀释效应"——正确答案的优势被错误答案拉低，导致优质信号被抑制。

**Baseline 公式 (GRPO)**:
$$A_k^{\text{GRPO}} = R_k - \bar{R} = R_k - \frac{1}{G}\sum_{i=1}^{G} R_i$$
符号: $R_k \in \{0, 1\}$ 为第 $k$ 个响应的二元正确性奖励, $G$ 为组大小, $\bar{R}$ 为算术平均基线。

**变化点**: 当组内正确率 $p \neq 0.5$ 时，$\bar{R}$ 系统性偏离真实质量阈值。设组内有 $n_+$ 个正确响应、$n_-$ 个错误响应，则正确响应的期望优势为 $1 - \frac{n_+}{G} = \frac{n_-}{G}$，错误响应为 $0 - \frac{n_+}{G} = -\frac{n_+}{G}$——这一相对比例随 $n_+/n_-$ 变化而剧烈波动，导致梯度方差不可控。

**本文公式（推导）**:
$$\text{Step 1}: \quad \tilde{R}_k = \frac{R_k - \mu_{\text{layer}}}{\sigma_{\text{layer}} + \epsilon} \quad \text{按质量层（正确/错误/部分）分别标准化，避免跨层耦合}$$
$$\text{Step 2}: \quad w_k = \text{softmax}\left(\frac{\tilde{R}_k}{\tau_{\text{temp}}}\right) \quad \text{层内注意力权重，突出高质量响应}$$
$$\text{Step 3}: \quad A_k^{\text{GAL}} = \frac{\tilde{R}_k - \sum_{i \in \text{same layer}} w_i \tilde{R}_i}{\sigma_{\text{layer}}} \quad \text{层内加权中心化，消除层间异质性}$$
$$\text{最终}: \quad L_{\text{GAL}} = -\mathbb{E}_{q \sim \mathcal{D}} \mathbb{E}_{\{o_k\}} \left[ \sum_{k=1}^{G} \frac{1}{G} \cdot A_k^{\text{GAL}} \cdot \log \pi_\theta(o_k|q) \right]$$

**对应消融**: Figure 4显示移除GAL导致"slow convergence and a low plateau"，最终准确率下降约3.2%。

---

### 模块 2: 动态系数修正 DCR（对应框架图 DCR框）

**直觉**: 训练过程中某些token位置（如数字、等号）的梯度会出现脉冲式异常，固定裁剪阈值要么过度抑制正常更新，要么无法遏制破坏性梯度。

**Baseline 公式 (GRPO的固定裁剪)**:
$$L_{\text{GRPO}} = -\mathbb{E}\left[ \sum_{k=1}^{G} \frac{1}{G} \min\left( r_k(\theta) A_k, \text{clip}(r_k(\theta), 1-\epsilon, 1+\epsilon) A_k \right) \right]$$
其中 $r_k(\theta) = \frac{\pi_\theta(o_k|q)}{\pi_{\theta_{\text{old}}}(o_k|q)}$，$\epsilon$ 为全局固定超参（通常0.2）。

**变化点**: 固定$\epsilon$假设所有token、所有训练阶段的梯度分布稳定，但实际中：(1) 不同token的梯度尺度差异可达10倍；(2) 训练初期vs后期的最优裁剪阈值不同。

**本文公式（推导）**:
$$\text{Step 1}: \quad g_t^{(i)} = \nabla_\theta \log \pi_\theta(o^{(i)}_t|q, o^{(i)}_{<t}) \quad \text{提取第}i\text{个样本第}t\text{个token的梯度}$$
$$\text{Step 2}: \quad \bar{g}_t = \text{EMA}_{\beta}(|g_t|) = \beta \bar{g}_{t-1} + (1-\beta)|g_t| \quad \text{维护梯度幅度的指数移动平均}$$
$$\text{Step 3}: \quad \epsilon_t^{\text{dynamic}} = \tau \cdot \frac{\bar{g}_t}{\max_{t'} \bar{g}_{t'}} + \epsilon_{\text{base}} \quad \text{基于相对梯度历史动态调整阈值}$$
$$\text{最终}: \quad L_{\text{DCR}} = -\mathbb{E}\left[ \sum_{k=1}^{G} \frac{1}{G} \min\left( r_k A_k, \text{clip}\left(r_k, 1-\epsilon_t^{\text{dynamic}}, 1+\epsilon_t^{\text{dynamic}}\right) A_k \right) + \lambda \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}}) \right]$$

其中 $\tau$ 为全局缩放因子（Figure 6分析其inverted U-shape效应），$\epsilon_{\text{base}}$ 保证最小裁剪保护。

**对应消融**: Figure 4显示移除DCR导致"severe volatility"，训练后期准确率震荡幅度达±8%；Figure 6显示$\tau=0.15$时最优，过大($\tau=0.3$)或过小($\tau=0.05$)均损害性能。

## 实验与分析

**主实验结果**（Qwen2.5-Math-1.5B，训练数据规模相同）：

| Method | Numina-Math | MATH-lighteval | GSM8K | KL散度 (vs base) |
|:---|:---|:---|:---|:---|
| Base Model | 43.0% | 38.5% | 72.1% | 0.00 |
| SFT | 41.2% (-1.8) | 36.2% (-2.3) | 70.5% (-1.6) | 0.89 |
| DPO | 44.5% | 40.1% | 73.8% | 0.67 |
| PPO | 45.8% | 42.3% | 75.2% | 0.45 |
| GRPO | 47.0% | 45.5% | 78.6% | 0.28 |
| **GFT (本文)** | **52.3%** | **47.8%** | **81.4%** | **0.12** |
| Δ vs GRPO | **+5.3%** | **+2.3%** | **+2.8%** | **-57%** |


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d5e430c1-2f94-46a2-bd9b-8f1044c4ace5/figures/Figure_5.png)
*Figure 5: Figure 5 KL divergence quantifies distributional drift from the base model. SFT exhibits the highest divergence, whileGFT maintains a significantly lower level, effectively mitigating catastrophic for*



**核心发现分析**:
- **SFT退化现象获定量验证**: SFT在三项基准上全面倒退，KL散度最高（0.89），确认模仿学习在数学推理上的根本缺陷（Figure 1）。
- **GFT的复合增益**: Numina-Math上+5.3%为最大提升，因该数据集难度分层最明显，GAL的质量分层标准化收益最大；GSM8K相对简单，GRPO已接近天花板，+2.8%仍显著。
- **KL效率优势**: GFT的KL散度0.12仅为GRPO的43%、SFT的13%，说明GAL+DCR在提升能力的同时更好地保持基础分布（Figure 5）。

**消融实验**（Figure 4）:
- 移除DCR（-DCR）: 训练曲线剧烈震荡，最终准确率下降4.1%
- 移除GAL（-GAL）: 收敛缓慢且进入低平台期，最终准确率下降3.2%
- 同时移除两者退化为朴素GRPO变体，性能与标准GRPO持平

**超参敏感性**（Figure 6）:
$\tau$ 控制DCR的动态范围，呈现清晰倒U型：$\tau=0.15$时Numina-Math达峰值52.3%；$\tau=0.05$时修正不足（50.1%）；$\tau=0.30$时过度修正抑制有效更新（49.8%）。

**公平性检查**:
- Baselines包含SFT、DPO、PPO、GRPO及RFT/Rejection Sampling，覆盖主流后训练范式
- 计算成本：GFT无需critic网络，显存开销与GRPO同级；GAL的分层计算增加约5%前向开销
- 局限：当前仅在1.5B模型验证，7B/14B扩展性待验证；验证器依赖规则匹配，开放域推理场景适用性有限

## 方法谱系与知识库定位

**方法家族**: 基于组采样的无critic强化学习（GRPO-lineage）

**父方法**: GRPO (Group Relative Policy Optimization, DeepSeek-Math, 2024)

**改动槽位**:
| 槽位 | 父方法 (GRPO) | 本文改动 |
|:---|:---|:---|
| objective | 组内算术平均基线 | GAL: 质量分层加权标准化 |
| training_recipe | 固定裁剪阈值 $\epsilon$ | DCR: 逐token动态系数修正 |
| inference | 标准组采样解码 | 相同（无改动） |
| data_curation | 纯规则奖励 | 兼容LLM-as-judge扩展 |

**直接对比基线**:
- **GRPO**: GFT保留其无critic架构，但替换核心估计量为无偏版本，并增加动态稳定性机制
- **DPO**: GFT支持从SFT渐进过渡，DPO需预构建偏好对且无法利用组内信号
- **ReFT / RFT**: GFT的GAL类似响应过滤思想，但将硬过滤软化为加权优势，保留更多信息

**后续方向**:
1. **跨模态扩展**: 将GAL的分层标准化思想迁移到视觉推理（几何证明、图表理解），处理多模态响应的异质性
2. **在线DCR自适应**: 当前EMA需预设$\beta$，可探索基于梯度预测的完全在线阈值选择
3. **与过程奖励模型PRM结合**: GAL目前仅利用结果奖励，分层机制可自然扩展至过程级细粒度优势估计

**知识库标签**: 
- modality: text / math-reasoning
- paradigm: RLHF-without-critic / group-sampling-RL
- scenario: post-training / mathematical-reasoning
- mechanism: advantage-estimation-bias-reduction / dynamic-gradient-clipping
- constraint: low-KL-drift / training-stability

