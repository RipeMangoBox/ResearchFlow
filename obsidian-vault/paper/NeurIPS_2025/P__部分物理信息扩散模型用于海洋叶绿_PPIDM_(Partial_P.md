---
title: Partial Physics Informed Diffusion Model for Ocean Chlorophyll Concentration Reconstruction
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 部分物理信息扩散模型用于海洋叶绿素重建
- PPIDM (Partial P
- PPIDM (Partial Physics Informed Diffusion Model)
- The Partial Physics Informed Diffus
acceptance: Poster
method: PPIDM (Partial Physics Informed Diffusion Model)
modalities:
- spatiotemporal
- scientific_data
paradigm: supervised
---

# Partial Physics Informed Diffusion Model for Ocean Chlorophyll Concentration Reconstruction

**Topics**: [[T__Time_Series_Forecasting]] | **Method**: [[M__PPIDM]] | **Datasets**: Ocean Chlorophyll Concentration Reconstruction, Infilling task, Prediction task, Infilling, Prediction

> [!tip] 核心洞察
> The Partial Physics Informed Diffusion Model (PPIDM) improves prediction accuracy and stability by integrating known physical principles through a physics operator while minimizing discrepancies from unknown dynamics, outperforming methods that either ignore physics or incorrectly assume complete physical knowledge.

| 中文题名 | 部分物理信息扩散模型用于海洋叶绿素重建 |
| 英文题名 | Partial Physics Informed Diffusion Model for Ocean Chlorophyll Concentration Reconstruction |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2501.0xxxx) · [Code](待补充) · [Project](待补充) |
| 主要任务 | 时空重建 / 海洋叶绿素浓度重建 / 时间序列预测 |
| 主要 baseline | Physics-informed diffusion models [1], DiffusionPDE [10], DDPM [7], Cocogen [11], Constrained synthesis with projected diffusion models [2], Neural approximate mirror maps [5] |

> [!abstract] 因为「扩散模型在科学应用中难以融入不完整的物理约束，特别是海洋叶绿素这类受未知生物过程影响的数据」，作者在「Physics-informed diffusion models」基础上改了「将完全物理约束改为部分物理信息集成，通过物理算子投影已知约束并最小化未知动力学差异，同时引入区域自适应权重」，在「海洋叶绿素浓度重建基准」上取得「显著优于假设完全物理或忽略物理的 baseline 方法」

- 关键性能 1：PPIDM 在测试集上显著优于无物理约束的标准扩散模型和强制零残差的完全物理信息方法（Table 1）
- 关键性能 2：在仅给定首帧和末帧的 infilling 任务中，PPIDM 优于 DiffusionPDE 等 baseline（Table 2）
- 关键性能 3：远离已知帧的时间步从部分物理注入中获益更多，物理约束的边际效用随时间距离增加（Figure 3）

## 背景与动机

海洋叶绿素浓度重建是海洋生态监测的核心任务：卫星仅能间歇性观测海面，需要从不完整的时空观测中重建连续的叶绿素浓度场。然而，这一任务面临根本性的物理知识困境——海洋物理过程（如平流、扩散）可用偏微分方程描述，但生物过程（如浮游植物生长、死亡）复杂且参数不确定，导致"部分已知、部分未知"的动力学结构。

现有方法沿三条路径处理此类问题，但各有明显局限：

**Physics-informed diffusion models [1]** 将完整物理约束嵌入扩散模型的训练和推理，假设所有动力学规律均已知。这类方法在海洋叶绿素任务中会强制拟合不完整的物理方程，将未知的生物过程错误地纳入确定性约束，导致过约束和重建失真。

**DiffusionPDE [10]** 针对部分观测条件下的 PDE 求解，利用扩散模型处理逆问题，但其"部分观测"指空间上的稀疏测量而非物理规律本身的不完整，未显式区分已知物理与未知动力学的差异。

**Standard DDPM [7]** 完全忽略物理先验，仅依赖数据驱动学习。在科学数据稀疏、标注昂贵的场景下，这类方法缺乏物理一致性保证，容易产生违背基本守恒律的非物理解。

上述方法的核心短板在于二元对立：要么假设物理完全已知（过约束），要么完全抛弃物理（欠约束）。海洋叶绿素的实际场景需要"部分物理信息"——对已知流体动力学施加约束，对未知生物过程保持学习灵活性。本文提出 PPIDM，首次在扩散模型框架中显式解耦已知物理约束与未知动力学残差，通过物理算子投影和差异最小化实现自适应的部分物理集成。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/010eed78-d80e-46fe-a630-a003162e30d4/figures/Figure_1.png)
*Figure 1 (pipeline): Overview of our proposed PPIDM. At each time-step, the model predicts predicted noise through the neural network ε_θ, and then corrects it using physics operators, guiding the model to learn the partially known physics.*



## 核心创新

核心洞察：科学数据生成中的物理知识本质上是"部分的"而非"全部的"，因为未知动力学（如生物过程）的残差不应被强制归零而应被最小化，从而使扩散模型能够在不假设完全物理知识的前提下获得物理一致性提升。

与 baseline 的差异：

| 维度 | Baseline (Physics-informed diffusion [1]) | 本文 PPIDM |
|:---|:---|:---|
| 物理假设 | 完全已知物理，强制零残差 | 部分已知物理，允许未知动力学存在 |
| 约束方式 | 全局固定权重约束 | 区域自适应权重 λ(r)，空间变化 |
| 残差处理 | 强制残差为零（过约束） | 差异最小化，降低而非消除未知影响 |
| 推理策略 | 标准去噪仅依赖学习分数 | 每步去噪后附加物理算子投影修正 |
| 目标函数 | 标准扩散损失 L_diff | L_diff + λ₁L_physics + λ₂L_disc |

这一"部分物理"范式区别于所有现有物理信息生成模型：不是弱化物理约束的强度，而是显式识别哪些约束是可靠的（已知物理）、哪些是不可靠的（未知动力学），并分别处理。

## 整体框架



PPIDM 的整体数据流如下：

**输入**：含噪声的时空数据 x_t（海洋叶绿素浓度场的加噪版本）以及区域标识 r（用于确定自适应权重）。

**模块 1：标准扩散主干（Standard diffusion backbone）**
输入 x_t 和时间步 t，通过 U-Net 或类似网络预测噪声 ε_θ(x_t, t)，输出中间去噪结果 x_{t-1}。这是 DDPM [7] 的标准前向过程，保留数据驱动的表达能力。

**模块 2：物理算子投影（Physics operator projection）**
输入中间预测 x_{t-1} 和已知物理约束 c_known，通过物理算子 P 计算投影梯度 ∇_x||P(x_{t-1}) - c_known||²，输出物理修正项。该算子仅作用于已知物理规律覆盖的子空间。

**模块 3：区域自适应权重控制器（Adaptive weighting controller）**
输入空间位置/区域标识 r，输出该区域的最优物理算子权重 λ(r)。该权重在验证集上通过最小化 MSE 优化，反映不同海域物理约束的可靠程度差异。

**模块 4：差异最小化估计器（Discrepancy estimator）**
输入预测残差和未知动力学掩码，通过 (I-P) 投影到未知子空间，输出差异损失 L_disc。该模块确保未知动力学不被强制拟合，而是将其影响降至最低。

**输出**：经过物理投影修正的最终去噪结果 x_{t-1}^{proj}，同时满足已知物理约束并保留对未知过程的适应性。

流程示意：
```
x_t → [ε_θ: 标准去噪] → x_{t-1} → [P: 物理算子] → 物理残差
                                      ↓
                              [λ(r): 区域权重] → 加权修正
                                      ↓
                         [(I-P): 未知子空间投影] → 差异最小化
                                      ↓
                              x_{t-1}^{proj} (输出)
```

## 核心模块与公式推导

### 模块 1: 组合目标函数（对应框架图：训练阶段）

**直觉**：标准扩散损失无法利用物理先验，而完全物理约束会过拟合不完整的知识，因此需要三项目标分别处理数据拟合、已知物理和未知动力学。

**Baseline 公式 (DDPM [7])**:
$$\mathcal{L}_{\text{base}} = \mathcal{L}_{\text{diff}} = \mathbb{E}_{x_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

符号: $x_0$ = 干净数据, $x_t$ = 时间步 t 的加噪数据, $\epsilon$ = 标准高斯噪声, $\epsilon_\theta$ = 噪声预测网络, $t$ = 扩散时间步。

**变化点**: 标准扩散损失仅学习数据分布，缺乏物理一致性；Physics-informed diffusion [1] 添加物理项但强制完全约束，对未知动力学过约束。

**本文公式（推导）**:
$$\text{Step 1}: \mathcal{L}' = \mathcal{L}_{\text{diff}} + \lambda_1 \|\mathcal{P}(x_0) - c_{\text{known}}\|^2 \quad \text{（加入物理约束项，但此时假设完全已知物理）}$$

$$\text{Step 2}: \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diff}} + \lambda_1 \underbrace{\|\mathcal{P}(x_{t-1}) - c_{\text{known}}\|^2}_{\mathcal{L}_{\text{physics}}} + \lambda_2 \underbrace{\|(I - \mathcal{P})(x_{t-1} - x_{\text{physics}})\|^2}_{\mathcal{L}_{\text{disc}}} \quad \text{（分解为已知/未知子空间）}$$

$$\text{最终}: \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diff}} + \lambda_1 \mathcal{L}_{\text{physics}} + \lambda_2 \mathcal{L}_{\text{disc}}$$

符号补充: $\mathcal{P}$ = 物理算子（投影到已知物理子空间）, $c_{\text{known}}$ = 已知物理约束, $I$ = 恒等算子, $(I-\mathcal{P})$ = 未知动力学子空间投影, $\lambda_1, \lambda_2$ = 损失权重。

**对应消融**: Table 1 及 Figure 2 显示，移除物理算子（λ₁=0）或强制零残差（等价于 λ₂→∞ 的过约束）均导致性能下降。

---

### 模块 2: 推理时物理算子投影（对应框架图：推理阶段，物理算子分支）

**直觉**：每步去噪后立即修正，使中间结果逐步满足物理约束，而非仅在最终输出施加约束；区域自适应权重避免全局一刀切。

**Baseline 公式 (DDPM 标准去噪 [7])**:
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right) + \sigma_t z$$

符号: $\alpha_t, \bar{\alpha}_t$ = 扩散过程方差调度参数, $\sigma_t$ = 噪声标准差, $z \sim \mathcal{N}(0, I)$。

**变化点**: 标准去噪仅依赖学习到的分数函数，无物理修正；Physics-informed diffusion [1] 添加物理项但使用全局固定权重，无法适应区域异质性。

**本文公式（推导）**:
$$\text{Step 1}: x_{t-1}^{\text{raw}} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right) + \sigma_t z \quad \text{（标准去噪，保留基础预测能力）}$$

$$\text{Step 2}: x_{t-1}^{\text{proj}} = x_{t-1}^{\text{raw}} - \lambda(r) \cdot \nabla_{x} \|\mathcal{P}(x_{t-1}^{\text{raw}}) - c_{\text{known}}\|^2 \quad \text{（加入区域自适应物理投影）}$$

$$\text{最终}: x_{t-1}^{\text{proj}} = x_{t-1}^{\text{raw}} - \lambda(r) \cdot \nabla_{x} \mathcal{L}_{\text{physics}}(x_{t-1}^{\text{raw}})$$

**区域权重优化**:
$$\lambda(r) = \text{arg}\min_{\lambda} \text{MSE}_{\text{val}}(x_{\text{pred}}(\lambda; r), x_{\text{true}})$$

符号补充: $\lambda(r)$ = 区域 r 的自适应权重, 在验证集上优化以最小化重建 MSE。

**对应消融**: Figure 2 显示不同沿海区域（coastal regions）的最优 λ 值不同，统一权重导致次优性能。

---

### 模块 3: 差异最小化损失（对应框架图：训练阶段，差异估计器分支）

**直觉**：未知动力学（如生物过程）的残差不应被强制为零，而应通过投影到未知子空间后最小化其影响，避免过拟合错误的物理假设。

**Baseline 公式**: 无直接对应——标准方法要么忽略该问题（DDPM），要么强制残差为零（完全物理信息方法）。

**本文公式**:
$$\mathcal{L}_{\text{disc}} = \|(I - \mathcal{P})(x_{t-1} - x_{\text{physics}})\|^2$$

其中 $x_{\text{physics}}$ 表示仅由已知物理预测的理想状态，$(I-\mathcal{P})$ 将残差投影到已知物理无法解释的未知子空间。最小化该损失等价于"承认无知"——不强迫模型解释一切，而是降低不可解释部分的影响。

**对应消融**: 消融实验表明，将差异项替换为强制零残差（即完全物理信息假设）会显著损害重建精度，特别是在生物活动活跃的海域。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/010eed78-d80e-46fe-a630-a003162e30d4/figures/Table_1.png)
*Table 1 (quantitative): Performance evaluation results on test set.*



本文在海洋叶绿素浓度数据集 [21] 上评估 PPIDM，涵盖三个实验设置：测试集整体重建（Table 1）、给定首帧和末帧的 infilling（Table 2）、以及给定前 10 帧的预测（Table 3）。


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/010eed78-d80e-46fe-a630-a003162e30d4/figures/Table_2.png)
*Table 2 (quantitative): Predictive performance of baseline models and our model when given the first 30% (for 1-3 days) or 15% (for 4-7 days) known frames.*



Table 1 的定量结果显示，PPIDM 在测试集重建任务上显著优于忽略物理的标准扩散模型（DDPM）以及假设完全物理知识的 Physics-informed diffusion models。具体而言，PPIDM 同时击败了强制零残差的物理信息方法和仅做约束边界的 Projected diffusion [2]、Mirror maps [5] 等 baseline，验证了"部分物理"优于"无物理"和"过约束物理"的核心假设。Table 2 和 Table 3 进一步展示，在仅有 30%（1-3 天）或 15%（4-7 天）已知帧的极端稀疏条件下，PPIDM 的预测稳定性优于 DiffusionPDE [10] 等强 baseline，表明部分物理信息在数据稀缺场景下的鲁棒性优势。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/010eed78-d80e-46fe-a630-a003162e30d4/figures/Figure_3.png)
*Figure 3 (result): Reconstructing frames far from the known frames benefit more from the injected partial physics knowledge than those close to the known frames, given the first 20% frames as input to infill the intermediate 20%.*



Figure 3 的关键发现揭示了时间维度上的物理约束效用分布：远离已知帧的重建帧从部分物理注入中获益更多，而靠近已知帧的帧本身已有足够数据驱动信号，额外物理约束的边际增益较小。这一发现直接支持了 PPIDM 的设计动机——物理先验在数据稀疏区域（时间或空间上）的价值最大。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/010eed78-d80e-46fe-a630-a003162e30d4/figures/Table_3.png)
*Table 3 (quantitative): Predictive performance of baseline methods and our model when given the first 20% frames of the target frame at four noise levels.*



消融实验（Figure 2）聚焦物理算子权重的区域敏感性：作者从叶绿素数据中划分两个沿海区域（coastal regions），扫描不同 λ 值下的重建 MAE。结果显示最优 λ 存在明显的区域差异，统一全局权重无法同时优化所有区域。去掉物理算子（λ=0）导致所有区域性能下降，但过度增强物理权重（λ过大）同样损害重建精度，印证了"部分物理"而非"完全物理"的必要性。

公平性检查：本文的 baseline 选择存在局限。虽然包含了 Physics-informed diffusion [1] 和 DiffusionPDE [10] 等强相关方法，但未与最新的视频扩散模型（如 Flexible Diffusion for Long Videos [6]）结合物理条件进行公平对比，也未纳入 Earthformer、ClimaX 等专门化时空 Transformer 基线。此外，实验仅局限于单一海洋叶绿素数据集 [21]，缺乏跨域验证；区域自适应权重 λ(r) 需要验证集调优，限制了即插即用的便利性；生物过程始终作为"未知动力学"处理，未探索渐进式学习机制。

## 方法谱系与知识库定位

PPIDM 属于 **Physics-informed generative modeling** 方法族，直接父方法为 **Physics-informed diffusion models [1]**，核心改动在于将"完全物理集成"转为"部分物理集成"。

**改变的 slots**：
- **objective**：标准扩散损失 → 组合损失（扩散 + 物理 + 差异最小化）
- **inference_strategy**：纯学习分数去噪 → 每步附加物理算子投影
- **architecture**：标准 backbone → 增加物理算子模块 + 区域自适应权重机制
- **training_recipe**：全局固定约束 → 验证集优化的区域特定权重

**直接 baselines 与差异**：
- **Physics-informed diffusion models [1]**：假设物理完全已知，PPIDM 改为部分已知 + 差异最小化
- **DiffusionPDE [10]**：处理空间部分观测，PPIDM 处理物理规律本身的部分性
- **DDPM [7]**：无物理约束，PPIDM 添加自适应物理算子
- **Cocogen [11]**：物理一致性生成，PPIDM 显式区分已知/未知物理

**后续方向**：
1. **渐进式物理学习**：当前未知动力学始终不可学习，未来可探索从数据中发现新物理规律的机制
2. **跨域迁移**：将部分物理范式推广至气象、材料等其他科学领域，验证通用性
3. **权重自动学习**：用元学习或神经架构搜索替代验证集调优 λ(r)，实现端到端自适应

**标签**：modality=spatiotemporal/scientific_data | paradigm=diffusion_model/physics_informed_learning | scenario=sparse_observation_reconstruction | mechanism=adaptive_physics_operator/discrepancy_minimization | constraint=partial_knowledge/region_heterogeneity

## 引用网络

### 直接 baseline（本文基于）

- Fingerprinting Denoising Diffusion Probabilistic Models _(CVPR 2025, 方法来源, 未深度分析)_: DDPM is the foundational diffusion model formulation; core algorithmic basis for
- Physics-Informed Diffusion Models _(ICLR 2025, 直接 baseline, 未深度分析)_: Title directly matches paper's core contribution ('Partial Physics Informed Diff
- DiffusionPDE: Generative PDE-Solving under Partial Observation _(NeurIPS 2024, 直接 baseline, 未深度分析)_: Very closely related: PDE-solving with diffusion under partial observation; 'par

