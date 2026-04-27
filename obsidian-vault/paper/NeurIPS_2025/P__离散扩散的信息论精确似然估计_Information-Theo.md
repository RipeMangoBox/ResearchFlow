---
title: Information-Theoretic Discrete Diffusion
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 离散扩散的信息论精确似然估计
- Information-Theo
- Information-Theoretic Discrete Diffusion (InfoDis)
- The I-MDSE and I-MDCE relations pro
acceptance: Poster
cited_by: 1
code_url: https://github.com/Dongjae0324/infodis
method: Information-Theoretic Discrete Diffusion (InfoDis)
modalities:
- Text
paradigm: self-supervised
---

# Information-Theoretic Discrete Diffusion

[Code](https://github.com/Dongjae0324/infodis)

**Topics**: [[T__Text_Generation]], [[T__Self-Supervised_Learning]] | **Method**: [[M__Information-Theoretic_Discrete_Diffusion]] | **Datasets**: Synthetic DNA sequences, HellaSwag, BeaverTails

> [!tip] 核心洞察
> The I-MDSE and I-MDCE relations provide tight, principled estimators of log-likelihood for discrete diffusion and masked diffusion models, enabling practical extensions like time-free formulas and coupled Monte Carlo estimation with reduced variance.

| 中文题名 | 离散扩散的信息论精确似然估计 |
| 英文题名 | Information-Theoretic Discrete Diffusion |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2510.24088) · [Code](https://github.com/Dongjae0324/infodis) · [Project](未提供) |
| 主要任务 | Text Generation, Self-Supervised Learning, OOD Detection, Model Influence Analysis |
| 主要 baseline | RADD (primary), LLaDA (secondary), DSE/DCE (Lou et al., 2024), I-MMSE (Guo-Shamai-Verdú) |

> [!abstract] 因为「离散扩散模型缺乏像高斯扩散 I-MMSE 那样的信息论基础来精确估计似然」，作者在「DSE/DCE 变分下界」基础上改了「将其重新诠释为基于互信息分解的精确估计器（I-MDSE/I-MDCE），并推导出无时间积分公式与耦合蒙特卡洛估计器」，在「合成 DNA 序列与 HellaSwag/ARC-hard/PIQA/BeaverTails」上取得「蒙特卡洛方差显著降低且精确恢复真实似然」的结果

- 无时间似然估计器在 HellaSwag、ARC-hard、PIQA 上相比时间积分基线（Eq. 16）实现一致的蒙特卡洛方差降低
- 耦合似然比估计器在 BeaverTails 数据集上相比解耦独立估计显著降低方差
- 在 4 阶马尔可夫 DNA 序列上，I-MDSE 无条件估计与 I-MDCE 条件估计均精确匹配真实似然（Figure 1a/1b）

## 背景与动机

离散扩散模型（如用于文本生成的 multinomial diffusion 或 masked diffusion）已成为连续扩散之外的重要生成范式，但它们长期面临一个根本问题：如何精确估计数据的似然（log-likelihood）。对于高斯扩散，Guo-Shamai-Verdú 的 I-MMSE 恒等式建立了互信息与最小均方误差之间的精确关系，使得似然估计有坚实的信息论基础。然而，在离散状态空间中，现有方法如 Austin et al. 的 structured denoising diffusion 和 Campbell et al. 的连续时间框架，以及 Lou et al. 的 DSE/DCE 损失，仅能提供变分下界或上界，无法给出精确值。

具体而言，DSE（Denoising Score Entropy）和 DCE（Denoising Cross-Entropy）被用作 log-likelihood 的变分界，这导致两个实际后果：一是推断时需要沿扩散时间步做数值积分（time-integral Monte Carlo），引入额外方差；二是无法确定学到的分数网络在何种意义下给出了"最优"似然估计。例如，在评估语言模型对提示-响应对的概率，或进行 OOD 检测时，不精确的似然估计会直接影响下游任务的可靠性。

本文的核心动机正是填补这一理论空白：为离散扩散建立类似 I-MMSE 的信息论恒等式，将现有的 DSE/DCE 损失从"近似界"提升为"精确估计器"，并据此设计更高效的推断算法。

## 核心创新

核心洞察：离散扩散中的去噪分数熵损失在最优时精确等于数据与扩散数据之间的互信息，因为离散状态空间的结构允许将 DSE/DCE 重新诠释为互信息分解的紧估计器，从而使无时间积分的高效似然推断与方差降低的耦合估计成为可能。

| 维度 | Baseline (DSE/DCE, Lou et al. 2024) | 本文 (InfoDis) |
|:---|:---|:---|
| 理论地位 | 变分下界/上界，近似 log-likelihood | 精确恒等式：最优时 DSE/DCE = 互信息 |
| 似然推断 | 时间积分蒙特卡洛（Eq. 16），需沿时间步数值积分 | 无时间闭式公式（Eq. 15/17），端点直接计算 |
| 比率估计 | 独立蒙特卡洛估计 log p(x⁺) − log p(x⁻)，方差大 | 耦合估计器（Eq. 18），共享噪声路径，方差显著降低 |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/aacc896f-8b7b-45bd-9a34-815a748a2506/figures/Figure_1.png)
*Figure 1 (comparison): Comparison of true and estimated...*



InfoDis 框架包含五个核心模块，数据流如下：

1. **离散扩散前向过程**：输入干净数据 X₀，通过连续时间马尔可夫链（CTMC）在离散状态空间中逐步加噪，输出各时间步的噪声数据 X_t。这是标准离散扩散的前向过程，本文沿用 Austin et al. 和 Campbell et al. 的设定。

2. **分数网络训练**：输入噪声数据 X_t 与时间 t，输出估计的分数函数 s_θ(X_t, t)。训练目标仍为 DSE 或 DCE 损失，但本文的理论保证改变了其诠释——在最优参数 θ* 下，这些损失不再是界而是精确值。

3. **I-MDSE/I-MDCE 似然分解**：输入训练好的分数网络与数据样本，利用新建立的信息论恒等式将 log-likelihood 分解为互信息项。这是框架的理论核心，将现有训练目标重新诠释为精确估计器。

4. **无时间估计器**：输入分数网络与数据样本，直接通过端点分布（t=0 与 t=T）计算似然，消去对中间时间步的数值积分。对应 Eq. 15（无条件）与 Eq. 17（条件）。

5. **耦合比率估计器**：输入两个待比较的响应（如 RLHF 中的 chosen/rejected）、共享提示与分数网络，通过共享噪声路径生成耦合的扩散轨迹，输出方差降低的 log-likelihood ratio。对应 Eq. 18。

```
X₀ ──[CTMC 前向扩散]──→ X_t ──[分数网络 s_θ]──→ 训练完成
                                      ↓
                    ┌─────────────────┼─────────────────┐
                    ↓                 ↓                 ↓
              [I-MDSE/I-MDCE]  [无时间估计器]    [耦合估计器]
              理论恒等式          Eq. 15/17           Eq. 18
                    ↓                 ↓                 ↓
              精确似然分解      无条件/条件似然      低方差似然比
```

## 核心模块与公式推导

### 模块 1: I-MDSE 关系（对应框架图"理论核心"位置）

**直觉**：将高斯信道中 I-MMSE 的优美性质移植到离散空间，证明去噪分数熵的最小值精确等于互信息，而非仅仅是其界。

**Baseline 公式** (I-MMSE for Gaussian, Guo-Shamai-Verdú [11]):
$$I(X_0; Y) = \frac{1}{2} \int_0^{\text{SNR}} \text{mmse}(X_0 | Y_\gamma) \, d\gamma$$
符号: $\text{mmse}$ = 最小均方误差, $\gamma$ = 信噪比参数。该恒等式建立了高斯噪声信道中互信息与估计误差的精确关系。

**变化点**：离散空间没有均方误差的概念，且 DSE 损失此前仅被证明是 log-likelihood 的变分下界。作者发现离散状态空间的结构允许类似的分解，但需用分数熵替代均方误差。

**本文公式（推导）**:
$$\text{Step 1}: \quad \text{ell}_{\text{DSE}}(\theta; X_0, X_t) = \sum_{i} \sum_{x \neq X_{t,i}} Q_t(x | X_{t,i}) \cdot s_\theta(X_t, t)_i[x] + \text{(归一化项)}$$
$$\text{Step 2}: \quad \min_\theta \mathbb{E}_{q_{0,t}}[\text{ell}_{\text{DSE}}(\theta; X_0, X_t)] = I(X_0; X_t) \quad \text{(最优时精确相等，非下界)}$$
$$\text{最终}: \quad I(X_0; X_t) = \min_\theta \mathbb{E}[\text{ell}_{\text{DSE}}(\theta; X_0, X_t)]$$
符号: $X_0$ = 干净数据, $X_t$ = 时间 t 的扩散数据, $Q_t$ = 前向转移速率, $s_\theta$ = 分数网络输出。关键：最优分数 $s^*$ 使得期望 DSE 损失精确等于互信息，而非小于或大于它。

**对应消融**：Table 1a 显示基于 I-MDSE 的无时间估计器相比时间积分基线显著降低方差。

---

### 模块 2: 无时间似然估计器（对应框架图"推断优化"位置）

**直觉**：若 I-MDSE 给出了精确的互信息分解，则 log-likelihood 的时间积分表达式中，中间时间步的贡献应当可以解析求出或 telescoping 消去。

**Baseline 公式** (Lou et al. 2024, Eq. 16):
$$\log p_0(x) = \mathbb{E}_{q_{T|0}}\left[\log \frac{p_T(x_T)}{q_{T|0}(x_T|x)}\right] + \int_0^T \mathbb{E}_{q_{s|0}}[\text{ell}_{\text{DSE}}^*(x, x_s)] \, ds$$
符号: $q_{T|0}$ = 前向条件分布, $p_T$ = 先验分布, $\text{ell}_{\text{DSE}}^*$ = 最优分数损失。该式需要沿时间轴数值积分，引入离散化误差与蒙特卡洛方差。

**变化点**：利用离散扩散前向过程的特定结构（CTMC 转移矩阵的指数形式），作者发现时间积分项可以解析计算或重组为端点项。

**本文公式（推导）**:
$$\text{Step 1}: \quad \log p_0(x) = \mathbb{E}_{q_{T|0}}\left[\log p_T(x_T)\right] - \mathbb{E}_{q_{0|0}}\left[\log q_{T|0}(x_T|x)\right] + \underbrace{\int_0^T \mathbb{E}[\text{ell}_{\text{DSE}}^*] \, ds}_{\text{需消去}}$$
$$\text{Step 2}: \quad \text{利用 } I(X_0; X_T) = \int_0^T \mathbb{E}[\text{ell}_{\text{DSE}}^*] \, ds \text{ 及 CTMC 的端点分布性质，重组得:}$$
$$\text{最终 (Eq. 15)}: \quad \log p_0(x) = \mathbb{E}_{q_{T|0}}\left[\log \frac{p_T(x_T)}{q_{T|0}(x_T|x)}\right] + \text{(time-free endpoint terms)}$$
条件扩展 (Eq. 17):
$$\log p_0(x_{\text{resp}} | x_{\text{prompt}}) = \mathbb{E}\left[\log \frac{p_T(x_T)}{q_{T|0}(x_T|x)} \Big| x_{\text{prompt}}\right] + \cdots$$
关键：所有时间积分被消去，仅需在端点 T 处采样，大幅降低蒙特卡洛方差。

**对应消融**：Table 1a 显示在 HellaSwag/ARC-hard/PIQA 上，时间自由估计器相比时间积分基线实现一致的方差降低。

---

### 模块 3: 耦合蒙特卡洛比率估计器（对应框架图"比率推断"位置）

**直觉**：比较两个样本的似然（如 RLHF 中的 preferred vs dispreferred）时，独立估计各自的 log-likelihood 再相减会累积双方方差；若让两者共享同一随机噪声路径，公共随机性会相消。

**Baseline 公式** (Decoupled independent MC):
$$\log \frac{p_0(x^+)}{p_0(x^-)} = \underbrace{\mathbb{E}_{\text{indep}}[\log p_0(x^+)]}_{\text{var } \sigma^2_+} - \underbrace{\mathbb{E}_{\text{indep}}[\log p_0(x^-)]}_{\text{var } \sigma^2_-}$$
总方差 = $\sigma^2_+ + \sigma^2_-$，因两项独立采样而相加。

**变化点**：引入 Common Random Numbers (CRN) 技术，对 $x^+$ 和 $x^-$ 使用同一组随机种子生成前向噪声路径，使两者的蒙特卡洛估计高度正相关。

**本文公式（推导）**:
$$\text{Step 1}: \quad \text{定义耦合扩散: } X_t^+ \sim q_{t|0}(\cdot|x^+), \quad X_t^- \sim q_{t|0}(\cdot|x^-) \text{ 使用共享 } \omega \sim \mathcal{U}[0,1]$$
$$\text{Step 2}: \quad \log \frac{p_0(x^+)}{p_0(x^-)} = \mathbb{E}_{\text{coupled}}\left[\log \frac{p_T(X_T^+)}{p_T(X_T^-)} + \int_0^T \left(s^*(X_t^+, t) - s^*(X_t^-, t)\right) dt \right]$$
$$\text{最终 (Eq. 18)}: \quad \text{var}_{\text{coupled}} \ll \text{var}_{\text{decoupled}} \text{ 因 } \text{Cov}(\hat{L}^+, \hat{L}^-) > 0$$
符号: $x^+, x^-$ = 两个待比较样本, $\omega$ = 共享随机源, $s^*$ = 最优分数。耦合使积分内的随机项高度相关，相减后残差方差远小于独立估计。

**对应消融**：Table 1b 显示在 BeaverTails 数据集上，耦合估计器相比解耦基线显著降低 log-likelihood ratio 的蒙特卡洛方差。

## 实验与分析



本文在两类验证任务上评估 InfoDis 框架：合成数据上的精确似然恢复，以及真实语言模型上的方差降低与应用验证。

**合成 DNA 序列验证（Section 5.1）**：作者使用 4 阶马尔可夫链生成具有已知真实分布的合成 DNA 序列，训练 RADD 模型后应用 I-MDSE 无时间估计器（Eq. 15）与 I-MDCE 条件估计器（Eq. 17）。Figure 1a 显示无条件对数似然估计值与真实 NLL 呈高度线性一致，验证了无时间估计器的准确性；Figure 1b 显示条件概率估计在复杂依赖结构下仍紧密匹配真实值。这为后续真实数据实验提供了可靠性基础。

**真实数据方差降低（Section 5.2）**：在 LLaDA（8B masked diffusion 语言模型）上，Table 1a 对比了无时间估计器与时间积分基线（Eq. 16）在 HellaSwag、ARC-hard、PIQA 三个常识推理基准上的蒙特卡洛方差。结果显示无时间估计器在所有数据集与采样数设置下均实现一致的方差降低，且无需额外训练成本——仅改变推断时的计算公式。Table 1b 进一步在 BeaverTails 安全对齐数据集上验证耦合比率估计器（Eq. 18）：相比对两个响应独立估计再相减的解耦基线，耦合估计器利用共享噪声路径显著降低 log-likelihood ratio 的方差，这对 RLHF 中的偏好对评分至关重要。



**消融与公平性检查**：核心消融隐含于 Table 1a/1b 中——去掉"无时间"回到时间积分，或去掉"耦合"回到独立估计，均直接导致方差上升。作者坦诚的局限性包括：实验主要验证估计器精度与方差降低，而非下游生成质量（如 perplexity 或 FID 等价指标）；真实世界应用（OOD 检测、模型影响分析）的细节在 Appendix D 中，未在主实验充分展开；理论保证假设最优分数函数，而实际学习的分数是近似的。此外，缺少与连续扩散 I-MMSE 实现的直接对比，以及标准生成基准的竞争性评估，是该工作从"理论工具"走向"实用系统"需补全的环节。

## 方法谱系与知识库定位

**方法家族**：信息论扩散 → 离散状态空间扩散

**父方法**：Kong et al. (2023) "Information-theoretic diffusion" [20] —— 建立连续扩散的信息论框架，本文将其扩展至离散空间。

**改动槽位**：
- **objective**：将 DSE/DCE 从变分界重新诠释为基于互信息的精确估计器（I-MDSE/I-MDCE）
- **inference_strategy**：新增无时间公式与耦合蒙特卡洛估计器，替代时间积分推断
- **state_space**：从连续高斯扩展到离散分类/掩码空间

**直接基线与差异**：
- **RADD** [3]：用于合成数据验证的训练目标与模型架构，本文仅改变似然估计方式
- **LLaDA** [6]：用于真实实验的 masked diffusion 基线，本文为其提供新的条件似然估计工具
- **Lou et al. (2024) DSE/DCE** [26]：核心公式来源，本文将其理论地位从"界"提升为"恒等式"
- **I-MMSE** [11]：灵感来源，本文建立其离散类比

**后续方向**：(1) 将无时间估计器集成到大规模离散扩散的训练动态中，实现训练-推断一体化优化；(2) 探索 I-MDCE 在更复杂掩码策略（如自适应掩码率）下的形式；(3) 利用精确似然估计提升离散扩散在安全对齐、可控生成等下游任务中的竞争力。

**标签**：text / diffusion_model / self-supervised / likelihood_estimation / information_theory / discrete_state_space / variance_reduction

## 引用网络

### 直接 baseline（本文基于）

- Fingerprinting Denoising Diffusion Probabilistic Models _(CVPR 2025, 方法来源, 未深度分析)_: DDPM is the foundational continuous diffusion work; discrete diffusion builds on
- Large Language Diffusion Models _(NeurIPS 2025, 实验对比, 未深度分析)_: Very recent concurrent work on large language diffusion models; cited for compar

