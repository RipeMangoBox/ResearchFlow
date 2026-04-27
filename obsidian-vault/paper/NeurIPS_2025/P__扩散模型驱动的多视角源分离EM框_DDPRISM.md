---
title: 'A Data-Driven Prism: Multi-View Source Separation with Diffusion Model Priors'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 扩散模型驱动的多视角源分离EM框架
- DDPRISM
- Diffusion models can solve multi-vi
acceptance: Poster
cited_by: 1
code_url: https://github.com/swagnercarena/ddprism
method: DDPRISM
modalities:
- Image
paradigm: supervised
---

# A Data-Driven Prism: Multi-View Source Separation with Diffusion Model Priors

[Code](https://github.com/swagnercarena/ddprism)

**Topics**: [[T__Image_Generation]] | **Method**: [[M__DDPRISM]] | **Datasets**: 1D Manifold Contrastive MVSS, Grassy MNIST Full Resolution

> [!tip] 核心洞察
> Diffusion models can solve multi-view source separation without explicit assumptions about source distributions, using only multiple views with different linear transformations of unknown sources.

| 中文题名 | 扩散模型驱动的多视角源分离EM框架 |
| 英文题名 | A Data-Driven Prism: Multi-View Source Separation with Diffusion Model Priors |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2510.05205) · [Code](https://github.com/swagnercarena/ddprism) · [DOI](https://doi.org/10.48550/arxiv.2510.05205) |
| 主要任务 | Multi-View Source Separation（多视角源分离）/ Image Generation |
| 主要 baseline | PCPCA, CLVM-Linear, CLVM-VAE, Removing Structured Noise using Diffusion Models [25] |

> [!abstract] 因为「科学数据中的源分离需要预先知道源分布，形成'需要分离才能约束先验，但需要先验才能分离'的循环困境」，作者在「Removing Structured Noise using Diffusion Models [25]」基础上改了「用扩散模型替代显式源分布假设，以EM框架联合训练各源专属降噪器并进行预测-校正后验采样」，在「1D Manifold Contrastive MVSS 与 Grassy MNIST」上取得「前两个源近乎完美恢复，Grassy MNIST 上 PQM 1.00 / FID 1.57 / PSNR 25.60」

- **训练时间**: 32-68 小时（A100/H100），相比 PCPCA 的 5 秒提升约 23000 倍
- **Grassy MNIST**: PQM 1.00（vs U-Net small 的 1.00，但 FID 1.57 vs 2.47 更优），PSNR 25.60 vs MLP encoder-decoder 的 17.93
- **推理延迟**: 22 ms（1D）至 1.5 s（Galaxy）每样本，baseline 均 < 0.1 ms

## 背景与动机

科学数据中的源分离面临一个根本性的循环困境：要分离出未知源信号，需要知道每个源的分布作为先验；但要确定源的分布，又需要已经分离好的纯净测量。例如在天文学中，一张望远镜图像可能包含多个重叠星系的混合光信号，每个仪器通道观测到的是不同线性组合，但天文学家并不知道单个星系的真实样貌。

现有方法主要从三个方向尝试突破。PCPCA（Probabilistic Contrastive PCA）[27] 采用线性子空间投影，通过对比式主成分分析寻找与背景差异最大的方向，但假设源服从高斯分布且混合关系线性。CLVM-Linear 与 CLVM-VAE [28] 引入变分推断框架，前者使用线性编码器-解码器，后者用 U-Net/CNN 增强表达能力，然而它们仍然需要预设源的参数化形式（如 VAE 的隐空间先验），且 ELBO 优化可能无法准确捕捉复杂的多模态分布。此外，[25] "Removing Structured Noise using Diffusion Models" 首次将扩散模型用于结构化噪声去除，但仅处理单源噪声场景，未扩展到多源联合推断。

这些方法的共同瓶颈在于**显式源分布假设**：无论是高斯、VAE 隐空间还是其他参数化先验，一旦真实源分布复杂（如多模态、非对称、带噪声），模型就会产生系统性偏差。更棘手的是，在 contrastive 多视角设置中（每个新视角引入一个新源），后期源的信号被层层混合，线性方法和简单 VAE 的恢复质量急剧退化。本文的核心动机正是：能否让数据自己说话，用扩散模型的非参数化表达能力替代显式源先验，实现"无先验假设"的多视角源分离？


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ee3d519c-ec22-4967-a863-73868c2c05f3/figures/fig_001.png)
*Figure: Comparison of posterior samples for*



本文提出 DDPRISM，以期望最大化（EM）框架迭代训练各源专属的扩散降噪器，通过预测-校正（Predictor-Corrector）采样联合推断所有源的后验分布——即使没有任何源被单独观测过。

## 核心创新

核心洞察：扩散模型的 score-based 生成能力可以作为**隐式源先验**，因为 score function 无需显式概率密度即可描述任意复杂分布，从而使"不假设源分布形式的多视角源分离"成为可能。

| 维度 | Baseline [25] / PCPCA / CLVM | 本文 DDPRISM |
|:---|:---|:---|
| **源先验** | 显式参数化（高斯/VAE 隐空间/单源扩散先验） | 无显式假设，以各源专属降噪器隐式学习 |
| **推断策略** | 线性投影、变分推断或单源噪声去除 | EM 框架内联合扩散后验采样，预测-校正迭代 |
| **多源关系** | 单源处理或独立优化 | 结构化混合矩阵（contrastive/mixed）约束下的联合推断 |
| **训练方式** | 直接优化线性模型或 VAE ELBO | E-step: PC 采样估计后验；M-step: 用 posterior 样本训练降噪器 |

与 [25] 的关键差异在于：DDPRISM 将扩散模型从"去除已知结构的噪声"推广到"分离未知结构的多个源"，核心是通过 EM 框架协调多个源专属降噪器的联合学习，而非单模型去噪。

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ee3d519c-ec22-4967-a863-73868c2c05f3/figures/fig_002.png)
*Figure: Comparison of the mean Sinkhorn di-*



DDPRISM 的完整数据流遵循经典的 EM 结构，但内外层都围绕扩散模型重新设计：

1. **输入层**：多视角观测 $\{y_\alpha\}$ 与结构化混合矩阵 $\{A_{\alpha\beta}\}$。Contrastive 设置中 $c_{\alpha\beta} = \mathbb{1}[\beta \leq \alpha]$，即第 $\alpha$ 个视角包含前 $\alpha$ 个源的叠加；Mixed 设置中 $c_{\alpha\beta} = \mathbb{1}[\beta=\alpha] + f_{\text{mix}}\mathbb{1}[\beta\neq\alpha]$，所有源以不同比例出现在所有视角。

2. **初始化模块（Gaussian Prior Warm-start）**：在正式 EM 前，运行短 EM 循环优化高斯先验参数，生成初始 posterior 样本。消融显示 0 init laps 时 PSNR 从 25.60 降至 23.35，验证了初始化的必要性。

3. **E-step：联合扩散后验采样（Joint Diffusion Posterior Sampler）**：固定当前降噪器 $\{d_{\theta_\beta}\}$，对每个源 $\beta$ 使用 Predictor-Corrector（PC）算法从 $p(x^\beta | \{y_\alpha\})$ 采样。PC 包含 16,384 个 predictor 步，每步附带 1 个 corrector 步，通过 variance exploding 噪声调度控制扩散强度。

4. **M-step：源专属降噪器训练（Denoiser Training）**：用 E-step 采得的 posterior 样本训练各源独立的降噪器 $d_{\theta_\beta}(x^\beta_t, t)$。时间参数 $t \sim \text{Beta}(\alpha=3, \beta=3)$ 采样，采用 Karras preconditioning 策略。

5. **迭代与输出**：EM 循环（如 8/16/32/64 laps）直至收敛，最终输出各源的后验样本、先验采样及联合后验。

```
多视角观测 y_α ──→ [Gaussian Prior Init] ──→ E-step: PC Sampler 
       ↑                                              ↓
       └──────── M-step: Train d_{θ_β} ←──── 后验样本 {x^β}
                          ↓
                    迭代 EM laps
                          ↓
              输出: 源后验 / 源先验 / 联合后验
```

关键设计选择：通过消融发现**简化 MLP 编码器-解码器**优于复杂 U-Net，这与直觉相反但有效——过度复杂的架构反而损害源分离性能。

## 核心模块与公式推导

### 模块 1: 结构化混合矩阵（对应框架图：输入层 → E-step）

**直觉**：多视角源分离的可识别性完全取决于混合矩阵的结构，本文通过两种确定性设计保证问题非退化。

**Baseline 公式**（标准线性混合）：
$$y_\alpha = \sum_\beta A_{\alpha\beta} x^\beta + \epsilon$$
符号：$y_\alpha$ = 第 $\alpha$ 视角观测，$x^\beta$ = 第 $\beta$ 个源，$A_{\alpha\beta}$ = 混合矩阵，$\epsilon$ = 噪声。

**变化点**：标准线性混合当所有 $A_{\alpha\beta}$ 未知时完全不可识别。本文引入**视角共享的混合结构**，将 $A_{\alpha\beta}$ 分解为观测特定因子与源关系指示函数的乘积。

**本文公式**：
$$A^{i_\alpha}_{\alpha\beta} = A_{i_\alpha} \cdot c_{\alpha\beta}$$

**Step 1**（Contrastive 混合）：
$$c_{\alpha\beta} = \begin{cases} 1 & \text{if } \beta \leq \alpha \\ 0 & \text{otherwise} \end{cases} \quad \text{（第 }\alpha\text{ 视角仅见前 }\alpha\text{ 个源）}$$

**Step 2**（Mixed 混合）：
$$c_{\alpha\beta} = \begin{cases} 1 & \text{if } \beta = \alpha \\ f_{\text{mix}} & \text{otherwise} \end{cases} \quad \text{（主源权重1，其余以 }f_{\text{mix}}\text{ 泄漏）}$$
注意 $f_{\text{mix}} = 1.0$ 时问题完全退化，实际取 $f_{\text{mix}} < 1$。

**对应消融**：Table 6 显示混合矩阵结构直接影响可分离性，contrastive 设置下前两个源近乎完美恢复，第三个源因累积混合而稍模糊。

---

### 模块 2: Variance Exploding 噪声调度与 Karras Preconditioning（对应框架图：M-step 训练）

**直觉**：扩散模型的训练稳定性高度依赖噪声调度，科学数据的高动态范围需要精细的噪声尺度控制。

**Baseline 公式**（标准 DDPM 调度）：
$$\beta(t) \text{ 线性/余弦插值于 } [\beta_{\min}, \beta_{\max}]$$

**变化点**：标准调度在科学数据的高精度要求下易产生数值不稳定；Karras et al. 的 variance exploding 调度配合 preconditioning 更适合连续数据的高分辨率重建。

**本文公式（推导）**：

**Step 1**（噪声水平定义）：
$$\sigma(t) = \exp\left[\log(\sigma_{\min}) + \left(\log(\sigma_{\max}) - \log(\sigma_{\min})\right) \cdot t\right]$$
采用对数线性插值，$t \in [0,1]$，$\sigma_{\min}, \sigma_{\max}$ 为数据依赖的边界。

**Step 2**（时间采样分布）：
$$t \sim \text{Beta}(\alpha=3, \beta=3) \quad \text{（更多采样于中等噪声区域，平衡学习与去噪）}$$

**Step 3**（Karras preconditioning）：
训练目标调整为 preconditioned score matching，网络输入输出经缩放以稳定梯度：
$$\text{input} = \frac{x_t}{\sqrt{\sigma(t)^2 + 1}}, \quad \text{target} = \frac{\sigma(t)\cdot \nabla_{x_t}\log p(x_t)}{\sqrt{\sigma(t)^2 + 1}}$$

**最终训练损失**：
$$\mathcal{L}_{\theta_\beta} = \mathbb{E}_{t\sim\text{Beta}(3,3), x^\beta\sim p_{\text{post}}} \left\| d_{\theta_\beta}(x^\beta_t, t) - \left(-\sigma(t)\nabla_{x^\beta_t}\log p(x^\beta_t)\right) \right\|^2$$

**对应消融**：Table 6 中 MLP encoder-decoder 配置 FID 49.03 vs 默认 1.57，说明架构简化与调度选择的协同至关重要。

---

### 模块 3: EM 框架内的 Predictor-Corrector 联合采样（对应框架图：E-step 核心）

**直觉**：传统 VI 或 Gibbs 采样在高维、多模态后验中混合极慢；扩散模型的 PC 采样可利用 score function 直接引导至高密度区域。

**Baseline 公式**（Gibbs 采样 / VI）：
$$x^\beta \sim q_\phi(x^\beta | y_\alpha) \text{（变分分布）或交替条件采样}$$

**变化点**：Gibbs 采样在本文多源设置中即使给 8 倍计算量仍表现极差（仅运行到第二个视角）；VI 的摊销推断引入表达瓶颈。本文将 PC 采样嵌入 EM，使 E-step 精确（渐近意义上）而非近似。

**本文公式（推导）**：

**Step 1**（Predictor：基于 ODE 的确定性推进）：
$$x^\beta_{t-\Delta t} = x^\beta_t + \frac{\sigma'(t)}{\sigma(t)}\left(x^\beta_t + \sigma(t)^2 \nabla_{x^\beta_t}\log p(x^\beta_t | \{y_\alpha\})\right)\Delta t$$
其中后验 score 通过各源降噪器与混合矩阵联合构造：
$$\nabla_{x^\beta_t}\log p(x^\beta_t | \{y_\alpha\}) \propto \nabla_{x^\beta_t}\log p(x^\beta_t) + \sum_\alpha \nabla_{x^\beta_t}\log p(y_\alpha | x^\beta_t, \{x^{\beta'\}_{\beta'\neq\beta})$$

**Step 2**（Corrector：Langevin 随机校正）：
$$x^\beta_{t} \leftarrow x^\beta_{t} + \epsilon_t \nabla_{x^\beta_t}\log p(x^\beta_t | \{y_\alpha\}) + \sqrt{2\epsilon_t}\, z, \quad z\sim\mathcal{N}(0,I)$$
步长 $\epsilon_t$ 自适应选择以保证收敛。

**Step 3**（联合推断）：
所有源的降噪器 $\{d_{\theta_\beta}\}$ 同时参与，通过混合矩阵耦合，迭代至 16,384 步完成。

**最终输出**：
$$\{x^\beta_{0}\}_{\beta=1}^{K} \sim p(\{x^\beta\} | \{y_\alpha\}_{\alpha=1}^{N})$$

**对应消融**：Table 5 显示 Gibbs 采样替代方案在 8x 计算下仍失败；减少 PC 步数至 16（vs 256）使 PQM 从 1.00 降至 0.87，PSNR 从 25.60 降至 21.01，验证了长程采样的必要性。

## 实验与分析



本文在三个层级验证 DDPRISM：1D 流形合成实验、Grassy MNIST 图像实验、以及真实星系图像分离。核心结果集中在 Table 1（1D 对比）与 Table 6（Grassy MNIST 消融）。

**1D Manifold Contrastive MVSS**（Table 1）：在三个源、三个视角的对比设置中，DDPRISM 的 Sinkhorn divergence 与 PQMass 全面优于 PCPCA、CLVM-Linear、CLVM-VAE。关键发现是**前两个源近乎完美恢复**——这与 contrastive 结构的理论保证一致（源 1 仅在视角 1 出现，源 2 在视角 1-2 中出现）；第三个源因被前两个视角的累积信号"污染"，posterior 稍模糊但仍优于所有 baseline。Gibbs 采样替代方案即使获得 8 倍计算资源，也只能运行到第二个视角且质量低下，证明了 PC 采样的不可替代性。

**Grassy MNIST Full Resolution**（Table 6）：默认配置（简化 MLP + 8 EM laps + 256 PC steps）达到 PQM 1.00、FID 1.57（Posterior）、PSNR 25.60。Prior 采样（无观测条件）FID 为 20.10，说明模型确实学到了有意义的源分布而非简单记忆。与架构变体相比：U-Net small 虽 PQM 同样 1.00，但 FID 2.47 明显更差；完整 MLP encoder-decoder 灾难性失败至 FID 49.03。这一反直觉结果——**更简单的架构更好**——可能是因为复杂网络的归纳偏置与源分离任务冲突。



**关键消融发现**（Table 6 及文中）：
- **数据量缩减至 1/64**：PQM 崩溃至 0.0，PSNR 降至 15.34，说明扩散模型对数据量的敏感性
- **EM laps 从 8 增至 64**：Posterior FID 从 0.04（8 laps）恶化至 1.57（64 laps），存在过拟合；但 Prior FID 在 8 laps 更差，揭示 posterior/prior 优化的张力
- **PC 步数 16 vs 256**：PQM 0.87 vs 1.00，PSNR 21.01 vs 25.60，采样质量对步数高度敏感

**效率与公平性审视**：训练时间 32-68 小时（A100/H100）vs baseline 的 5 秒-18 分钟，推理 22 ms-1.5 s vs <0.1 ms，**计算开销达 1000-23000 倍**。Baseline 选择方面，PCPCA/CLVM 系列是线性/浅层方法，未与同期扩散源分离方法或端到端监督网络（如 Wave-U-Net [22]）对比。特别值得注意的是，直接 lineage 父方法 [25] 在实验中未被比较，可能因任务设定差异（单源去噪 vs 多源分离）。作者披露的限制包括：后期源采样较模糊、Gibbs 替代方案失败、以及计算成本限制实用部署。

## 方法谱系与知识库定位

**方法家族**：Score-based / Diffusion Generative Models → Inverse Problems with Generative Priors

**直接父方法**：Removing Structured Noise using Diffusion Models [25]（2023）。DDPRISM 将 [25] 的单源结构化噪声去除扩展至多源联合推断，核心演进在于：引入 EM 框架协调多降噪器、设计结构化混合矩阵保证可识别性、以联合 PC 采样替代单源去噪。

**改变的 slots**：
| Slot | 父方法 [25] | DDPRISM |
|:---|:---|:---|
| inference_strategy | 单源去噪，直接后验采样 | 多源联合 PC 采样，EM 迭代 |
| architecture | 单一降噪网络 | 每源独立降噪器 + 简化 MLP 编解码 |
| training_recipe | 直接 score matching 训练 | EM：E-step 采样 → M-step 训练 |
| objective | 标准 score matching | 嵌入 EM 的 score matching，Beta(3,3) 时间采样 |
| data_curation | 单观测噪声去除 | 多视角结构化混合（contrastive/mixed） |

**直接 baselines 与差异**：
- **PCPCA [27]**：线性投影，DDPRISM 以非线性扩散替代线性假设
- **CLVM-Linear/VAE [28]**：变分推断，DDPRISM 以扩散采样替代摊销推断，避免 ELBO 偏差
- **Gibbs sampling（本文自建）**：验证 PC 采样的必要性，证明传统 MCMC 在此设置下失效

**后续方向**：
1. **计算效率**：将 16,384 步 PC 采样压缩至可接受范围（如蒸馏、flow matching 替代）
2. **理论保证**：contrastive 结构下源可识别性的严格条件，mixed 设置的 $f_{\text{mix}}$ 阈值分析
3. **跨模态扩展**：从图像推广至音频、神经信号等时序/谱数据，验证扩散先验的通用性

**标签**：`modality:image` · `paradigm:generative_modeling` · `scenario:inverse_problem` · `mechanism:diffusion_model` · `mechanism:expectation_maximization` · `constraint:multi-view` · `constraint:unsupervised`

