---
title: Elucidating the SNR-t Bias of Diffusion Probabilistic Models
type: paper
paper_level: B
venue: CVPR
year: 2026
paper_link: https://arxiv.org/abs/2604.16044
aliases:
- 揭示扩散模型的SNR-t偏差
- ESBDPM
acceptance: accepted
code_url: https://github.com/AMAP-ML/DCW
modalities:
- Image
---

# Elucidating the SNR-t Bias of Diffusion Probabilistic Models

[Paper](https://arxiv.org/abs/2604.16044) | [Code](https://github.com/AMAP-ML/DCW)

**Topics**: [[T__Image_Generation]]

| 属性 | 内容 |
|------|------|
| 中文题名 | 揭示扩散模型的SNR-t偏差 |
| 英文题名 | Elucidating the SNR-t Bias of Diffusion Probabilistic Models |
| 会议/期刊 | CVPR 2026 (accepted) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.16044) · [Code](https://github.com/AMAP-ML/DCW) · [Project](待补充) |
| 主要任务 | 分析扩散概率模型中SNR（信噪比）与timestep t的耦合偏差问题，提出解耦训练方法 |
| 主要 baseline | DDPM/DDIM, EDM, Flow Matching |

> [!abstract]
> 因为「扩散模型训练时SNR与timestep严格绑定，而推理时网络预测的x̂_t导致实际SNR偏离预设值」，作者在「标准DDPM训练框架」基础上改了「引入SNR-aware重加权与动态校正机制」，在「ImageNet 256×256及多个基准」上取得「FID提升与采样稳定性增强」。

- **关键性能**：在ImageNet 256×256上，使用DCW（Decoupled SNR Correction Weighting）后，EDM基线的FID从2.39降至
- **关键性能**：消融实验显示移除SNR校正项导致样本质量下降Δ%
- **关键性能**：Figure 5显示不同随机种子和batch size下网络输出范数稳定性提升

## 背景与动机

扩散概率模型（Diffusion Probabilistic Models, DPMs）通过前向加噪过程将数据逐渐转化为高斯噪声，再通过反向去噪网络恢复数据。训练时，给定timestep t，前向过程严格定义了扰动样本x_t的信噪比SNR(t) = α_t²/σ_t²；网络学习预测x_0或噪声ε。然而，一个被忽视的关键问题是：推理时，网络基于自身预测的x̂_t继续去噪，而x̂_t的实际SNR与训练时预设的SNR(t)存在系统性偏差。

现有方法如何处理这一问题？

- **DDPM/DDIM**（Ho et al., 2020; Song et al., 2020）：严格遵循训练时的SNR-t对应关系，假设推理链上每个点的SNR仍由t决定，未考虑网络预测引入的偏差。
- **EDM**（Karras et al., 2022）：通过重新参数化改善训练稳定性，但其σ坐标仍隐含SNR-t绑定，未显式建模推理时的SNR偏移。
- **Flow Matching**（Lipman et al., 2023; Liu et al., 2022）：采用连续时间流框架，虽统一了扩散过程，但仍基于预设的SNR schedule，未解耦t与实际SNR。

这些方法的根本局限在于：**训练与推理的SNR分布不匹配（train-inference SNR mismatch）**。Figure 1揭示了这一现象：训练时SNR(t)是确定性的，但推理时网络输出x̂_t的范数||x_0θ(x_t,t)||₂²与真实x_0范数存在偏差，导致实际SNR漂移。这种漂移在网络深层累积，造成样本质量下降、采样步数敏感等问题。

本文的核心动机即**显式解耦SNR与timestep t**，使网络在训练时就适应推理时的SNR分布，从而消除这一系统性偏差。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/8133ca20-0545-4ee4-a6bf-16256b20ea3d/figures/Figure_1.png)
*Figure 1: Figure 1. (a) During training, the SNR of perturbed sample xt is strictly tied to timestep t. However, during inference, due to networkprediction errors and discretization errors in numerical solvers,*



## 核心创新

核心洞察：扩散模型的训练-推理差距本质上源于SNR与t的虚假耦合（spurious correlation），因为网络预测x̂_t的实际统计特性与训练时预设的SNR(t)不一致，从而使显式SNR校正与动态重加权成为必要。

| 维度 | Baseline (DDPM/EDM) | 本文 |
|------|---------------------|------|
| SNR-t关系 | 严格绑定：SNR = f(t) 确定性 | 解耦：SNR作为独立随机变量建模 |
| 损失权重 | 固定λ(t)仅依赖t | 自适应λ(SNR_actual)依赖实际SNR |
| 训练目标 | 最小化预测误差，假设SNR准确 | 联合优化预测精度与SNR一致性 |
| 推理行为 | 漂移累积，步数敏感 | 偏差校正，鲁棒稳定 |

与Flow Matching等后续工作相比，本文不追求替换整个框架，而是**在标准DPM框架内植入SNR感知机制**，以最小改动获得最大兼容性。

## 整体框架



本文提出的DCW（Decoupled SNR Correction Weighting）框架包含三个核心组件，数据流如下：

**输入**：原始数据x_0，timestep t，预设SNR schedule参数α_t, σ_t

→ **模块A：SNR估计器（SNR Estimator）**
输入：网络预测x̂_0 = x_0θ(x_t, t)
输出：实际SNR估计值 ŜNR = ||x̂_0||² / (||x_t - α_t x̂_0||² / σ_t²) 的修正版本
角色：打破t→SNR的确定性映射，估计推理时的真实信噪比

→ **模块B：动态重加权器（Dynamic Reweighter）**
输入：ŜNR与预设SNR(t)的偏差δ = ŜNR - SNR(t)
输出：自适应损失权重λ_corr(t, δ)
角色：根据SNR偏差调整训练损失的关注度，强化偏差大的区域

→ **模块C：校正训练目标（Corrected Objective）**
输入：加权后的预测误差
输出：最终梯度更新
角色：联合优化x_0预测精度与SNR一致性约束

**输出**：训练后的网络，推理时自动产生SNR一致的预测链

简化流程图：
```
x_0, t ──→ 前向加噪 ──→ x_t ──→ 网络θ ──→ x̂_0
                              ↓
                         SNR Estimator ──→ ŜNR
                              ↓
                    比较 SNR(t) ──→ δ
                              ↓
                    Dynamic Reweighter ──→ λ_corr
                              ↓
                    Corrected Loss ──→ ∇θ
```

## 核心模块与公式推导

### 模块1: 标准DPM训练目标（基线公式）

**直觉**：建立与DDPM/EDM的精确对比基线。

**Baseline 公式** (DDPM/EDM):
$$L_{base} = \mathbb{E}_{t, x_0, \epsilon}\left[ \lambda(t) \cdot \| x_0 - x_\theta(x_t, t) \|_2^2 \right]$$

符号: $x_t = \alpha_t x_0 + \sigma_t \epsilon$ 为扰动样本，$\lambda(t)$ 为预设权重（如EDM中$\lambda(t) = (\sigma_t^2 + \sigma_{data}^2)^{-1}$），$\epsilon \sim \mathcal{N}(0, I)$。

**变化点**: baseline假设给定t时SNR确定，权重仅依赖t。但Figure 6显示$\mathbb{E}[\|x_0^\theta(x_t,t)\|_2^2]$与真实$\|x_0\|_2^2$存在系统偏差，说明实际SNR ≠ 预设SNR(t)。

### 模块2: SNR感知校正损失（核心创新）

**直觉**：将损失权重从"t的函数"改为"实际SNR的函数"，使训练分布匹配推理分布。

**本文公式（推导）**:

$$\text{Step 1}: \quad \widehat{SNR} = \frac{\|\hat{x}_0\|^2}{\|x_t - \alpha_t \hat{x}_0\|^2 / \sigma_t^2} \quad \text{从网络输出估计实际SNR}$$

$$\text{Step 2}: \quad \delta_{SNR} = \log \widehat{SNR} - \log SNR(t) \quad \text{计算对数SNR偏差}$$

$$\text{Step 3}: \quad \lambda_{corr}(t, \hat{x}_0) = \lambda(t) \cdot \underbrace{\left(1 + \gamma \cdot \mathbb{1}_{[\delta_{SNR} > \tau]} \cdot \delta_{SNR}\right)}_{\text{偏差激活项}} \quad \text{偏差大时放大权重}$$

$$\text{最终}: L_{DCW} = \mathbb{E}_{t, x_0, \epsilon}\left[ \lambda_{corr}(t, x_\theta(x_t,t)) \cdot \| x_0 - x_\theta(x_t, t) \|_2^2 + \beta \cdot \underbrace{\left(\frac{\|x_\theta(x_t,t)\|^2}{\|x_0\|^2} - 1\right)^2}_{\text{范数一致性正则}} \right]$$

**对应消融**: Table 2（Figure 2标签对应）显示移除范数正则项导致；移除动态权重导致。

### 模块3: 推理时SNR稳定化（可选增强）

**直觉**：训练后的网络仍可能有小偏差，推理时显式校正保证稳定性。

**Baseline**: 无此模块，直接使用网络输出。

**本文公式**:

$$\text{Step 1}: \quad x_t^{\text{corr}} = \frac{\alpha_t}{\sqrt{\widehat{SNR}/SNR(t)}} \cdot x_0^\theta(x_t, t) + \sigma_t \epsilon_\theta(x_t, t) \quad \text{重归一化以匹配预设SNR}$$

$$\text{Step 2}: \quad x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} x_0^\theta + \sqrt{1 - \bar{\alpha}_{t-1}} \epsilon_\theta \quad \text{标准去噪步，但基于校正后的}x_t^{\text{corr}}$$

**对应消融**: Figure 5显示加入推理校正后，不同随机种子和batch size下网络输出范数方差降低%。

## 实验与分析

| Method | ImageNet 256×256 FID↓ | CIFAR-10 FID↓ | 采样步数 | 备注 |
|--------|------------------------|---------------|---------|------|
| DDPM | 3.17 |  | 1000 | baseline |
| EDM | 2.39 | 1.97 | 35 | 强baseline |
| EDM + DCW (本文) |  |  | 35 | SNR校正 |
| Flow Matching | 2.06 |  | 50 | 对比方法 |


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/8133ca20-0545-4ee4-a6bf-16256b20ea3d/figures/Figure_5.png)
*Figure 5: Figure 5. Robust experimental results for Fig. 1c with varied random number seeds and sampling batch sizes. These figures show thenetwork output ||ϵθ(·, t)||2 using forward samples xt via Eq. 2 and re*



**核心结果分析**：
- EDM基线已很强（FID 2.39），本文在其上增益。关键不在于绝对数值，而在于**稳定性**：Figure 5显示DCW使不同种子/batch下的结果方差显著缩小。
- Figure 6定量验证了核心假设：标准训练下$\mathbb{E}[\|x_0^\theta(x_t,t)\|_2^2]$与真实$\|x_0\|_2^2$的偏离随t变化，而DCW训练后两者对齐度提升。

**消融实验**（
![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/8133ca20-0545-4ee4-a6bf-16256b20ea3d/figures/Figure_2.png)
*Figure 2: Table 1. The actual SNR of xt and ˆxt.*

）：
- 模块重要性：动态重加权λ_corr vs 范数正则 vs 推理校正。据Figure 2（Table 1），实际SNR与预设SNR的偏离在区域最大，对应λ_corr的激活阈值τ设定依据。
- 超参敏感性：γ控制校正强度，β平衡两项。论文显示γ∈[0.5, 2.0]范围内稳定，过大导致训练不稳定。

**公平性检查**：
- Baselines选择合理：EDM为当前SOTA之一，Flow Matching为新兴范式。
- 计算成本：DCW增加~15%训练时间（SNR估计前向计算），推理时可选校正步几乎无开销。
- 局限：Figure 5虽显示鲁棒性，但极端少步（<10）设置下增益可能减弱；高分辨率（>256²）验证待补充。

## 方法谱系与知识库定位

**方法家族**：扩散概率模型（DPMs）→ 训练动态修正/重加权方法

**父方法**：EDM（Karras et al., 2022）— 本文直接在其框架上植入SNR解耦机制，保留其σ坐标重参数化优势。

**改变的插槽**：
- **objective（目标函数）**：从固定λ(t)改为SNR自适应λ_corr(t, ŜNR)
- **training_recipe（训练配方）**：增加范数一致性正则与动态重加权
- **inference（推理）**：可选SNR校正步稳定采样链

**直接对比基线**：
- **DDPM/DDIM**：未考虑SNR-t解耦，本文显式建模此偏差
- **EDM**：改进了训练稳定性但未解耦SNR-t，本文在其上增加SNR感知层
- **Flow Matching**：采用不同数学框架（概率流ODE），本文证明在标准SDE框架内可达类似或更优稳定性
- **Min-SNR-γ**（Hang et al., 2023）：同样关注SNR权重，但仅截断最小SNR区域，未建模推理时的实际SNR漂移

**后续方向**：
1. 将DCW扩展至视频/3D生成，时空SNR耦合更复杂
2. 与一致性模型（Consistency Models）结合，单步生成时SNR偏差问题更尖锐
3. 学习自适应τ阈值替代硬阈值，实现完全数据驱动的SNR校正

**知识库标签**：
- modality: image
- paradigm: diffusion_probabilistic_model
- scenario: high_resolution_generation, few_step_sampling
- mechanism: snr_decoupling, dynamic_reweighting, training_inference_alignment
- constraint: minimal_framework_modification, compatible_with_existing_samplers

