---
title: 'D2SA: Dual-Stage Distribution and Slice Adaptation for Efficient Test-Time Adaptation in MRI Reconstruction'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- MRI重建的双阶段分布与切片自适应
- D2SA (Dual-Stage
- D2SA (Dual-Stage Distribution and Slice Adaptation)
- D2SA achieves efficient and accurat
acceptance: Poster
code_url: https://uceclz0.github.io/d2sa-webpage/
method: D2SA (Dual-Stage Distribution and Slice Adaptation)
modalities:
- Image
- medical_imaging
paradigm: self-supervised
---

# D2SA: Dual-Stage Distribution and Slice Adaptation for Efficient Test-Time Adaptation in MRI Reconstruction

[Code](https://uceclz0.github.io/d2sa-webpage/)

**Topics**: [[T__Medical_Imaging]] | **Method**: [[M__D2SA]] | **Datasets**: Cross-domain MRI reconstruction - Anatomy Shift, Cross-domain MRI reconstruction - Dataset Shift, Cross-domain MRI reconstruction - Modality Shift, Cross-domain MRI reconstruction - Acceleration Shift, Cross-domain MRI reconstruction - Sampling Shift

> [!tip] 核心洞察
> D2SA achieves efficient and accurate test-time adaptation for MRI reconstruction through a dual-stage approach combining patient-wise distribution modeling via MRI implicit neural representation with slice-level anisotropic diffusion refinement.

| 中文题名 | MRI重建的双阶段分布与切片自适应 |
| 英文题名 | D2SA: Dual-Stage Distribution and Slice Adaptation for Efficient Test-Time Adaptation in MRI Reconstruction |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://doi.org/10.48550/arxiv.2503.20815) · [Code](https://uceclz0.github.io/d2sa-webpage/) · [Project](https://uceclz0.github.io/d2sa-webpage/) |
| 主要任务 | MRI重建中的测试时自适应（TTA），应对分布偏移 |
| 主要 baseline | DIP-TTT, ZS-SSL, FINE+SST, NR2N+SST, SSDU+SST |

> [!abstract] 因为「MRI重建模型在扫描仪和采集协议变化时遭遇严重性能下降，且现有测试时自适应方法计算效率低下或易过平滑」，作者在「FINE+SST / DIP-TTT」基础上改了「引入MR-INR进行患者级分布建模，再以各向异性扩散进行切片级细化」，在「五种MRI分布偏移（解剖/数据集/模态/加速/采样）」上取得「SSIM 0.824-0.882，推理时间从53-167分钟降至17-45分钟，最高3.5倍加速」。

- **速度提升**：FINE+MRINR+SST 在解剖偏移上将每患者推理时间从 53.5 分钟降至 17.1 分钟（3.1×）
- **采样偏移突破**：SSIM 0.824 vs 最优 baseline DIP-TTT 的 0.771，PSNR 28.49 vs 27.65
- **兼容多种SSL框架**：与 FINE、NR2N、SSDU 集成后均获一致提升，NR2N+MRINR+SST 在加速偏移达 PSNR 28.89

## 背景与动机

MRI重建的核心挑战在于：临床部署时，训练好的深度学习模型面对新扫描仪、新解剖部位、新加速因子或新采样模式时，性能会急剧退化。例如，一个在膝关节2×随机采样数据上训练的模型，直接用于脑部4×均匀采样数据，SSIM可能从0.8跌至0.1以下。这种分布偏移（distribution shift）在医疗场景中尤为致命，因为重新标注或重训练成本极高。

现有方法沿三条路径应对这一问题：

**DIP-TTT** [11] 采用逐切片重复训练（deep image prior），将每个测试切片视为独立优化问题，通过自监督损失微调网络。该方法虽能缓解偏移，但需为每一切片从头优化，导致每患者耗时52-137分钟，临床不可行。

**ZS-SSL** 直接在测试时应用零样本自监督学习，无需训练但依赖大量自监督信号，在解剖偏移上需99.5分钟，且对复杂偏移鲁棒性不足。

**FINE+SST / NR2N+SST / SSDU+SST** 等将自监督学习框架与单切片训练（SST）结合，虽比DIP-TTT快，但仍缺乏显式的患者级分布建模，导致切片间信息无法共享，且标准卷积细化易引入过平滑，丢失细微病理结构。

上述方法的共同瓶颈在于：**未将「患者级分布偏移建模」与「切片级细节保留」解耦**。要么逐切片独立优化（慢），要么全局微调（糙），无法在效率与精度间取得平衡。本文提出将适应过程显式分解为两阶段：先以隐式神经表示（INR）捕获患者整体分布，再以各向异性扩散精细化每一切片，从而实现「一次患者级计算，多次切片级复用」。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/87dfa85f-3c32-445c-82a8-c7ad8c83320d/figures/fig_001.jpeg)
*Figure: Illustration of TTA strategies for MRI reconstruction under distribution shifts. (a) Single-*



## 核心创新

核心洞察：**患者级分布偏移具有跨切片共享的低维结构**，因为同一患者的所有切片受同一扫描仪和生理条件约束；而切片级细节需保留各向异性边缘信息。通过 INR 的连续函数表示能力建模前者，通过方向可控的扩散过程处理后者的分离设计，从而使「分钟级推理」与「像素级精度」得以兼得。

| 维度 | Baseline (FINE+SST / DIP-TTT) | 本文 (D2SA) |
|:---|:---|:---|
| 分布建模 | 无显式分布建模，逐切片独立或全局微调 | MR-INR：患者共享隐编码 + 可学习均值/方差调整 |
| 特征表示 | 标准卷积网络，离散网格表示 | PE+SIREN 隐式神经表示，连续坐标映射 |
| 切片细化 | 端到端卷积微调，易过平滑 | 冻结卷积特征 + 各向异性扩散，方向自适应平滑 |
| 推理策略 | 每切片重训练或长时自监督 | 两阶段分解：患者级预计算 → 切片级快速细化 |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/87dfa85f-3c32-445c-82a8-c7ad8c83320d/figures/fig_002.jpeg)
*Figure: Overview of the proposed two-stage D2SA framework. (a) Functional-Level Patient*



D2SA 是一个即插即用的两阶段框架，可叠加于任意预训练 MRI 重建器（FINE、NR2N、SSDU 等）。数据流如下：

**输入**：单患者欠采样 k 空间数据（多切片）

→ **预训练重建器**（FINE/NR2N/SSDU）：生成初始重建图像，其卷积层权重在后续阶段冻结

→ **Stage 1: MR-INR（患者级分布适应）**：将该患者所有切片坐标输入 PE+SIREN 网络，学习共享隐编码 $\mathbf{z}_p$ 及可学习的均值/方差参数 $(\alpha_p, \beta_p, \mu_p)$，输出患者适配的分布参数

→ **Stage 2: SST+AD（切片级各向异性扩散细化）**：利用 Stage 1 的分布参数，在冻结的卷积特征上执行可学习的各向异性扩散迭代，方向性控制平滑强度以保留边缘

→ **输出**：最终重建图像

```
undersampled k-space → [Pretrained Reconstructor] → initial image
                                                      ↓
                    ┌─────────────────────────────────┘
                    ↓
[MR-INR Stage 1] ← all slices + shared latent z_p
  ├── PE+SIREN coordinate encoding
  ├── learnable affine (α, β) for distribution shift
  └── patient-wise distribution parameters
                    ↓
[SST+AD Stage 2] ← frozen conv features + anisotropic diffusion
  ├── directional smoothing D(∇u)
  └── slice-wise refinement
                    ↓
              final reconstruction
```

## 核心模块与公式推导

### 模块 1: MR-INR 患者级分布适应（对应框架图 Stage 1）

**直觉**：标准 INR 逐场景独立优化，无法利用同一患者多切片的共享信息；通过引入患者级隐编码和分布调整参数，将「偏移建模」显式参数化。

**Baseline 公式** (标准 INR / DIP-TTT): $$f_{\text{INR}}(\mathbf{x}) = \text{SIREN}(\gamma(\mathbf{x}))$$
符号: $\mathbf{x}$ = 像素坐标, $\gamma(\cdot)$ = 位置编码, SIREN = 正弦激活MLP

**变化点**：标准 INR 无患者级条件，每场景需独立优化；且缺乏对分布偏移（均值/方差变化）的显式补偿。

**本文公式（推导）**:
$$\text{Step 1}: f_{\text{MR-INR}}(\mathbf{x}, \mathbf{z}_p) = \text{SIREN}(\gamma(\mathbf{x}), \mathbf{z}_p) \quad \text{加入患者共享隐编码 } \mathbf{z}_p \text{ 以跨切片共享信息}$$
$$\text{Step 2}: + \, \alpha_p \cdot \mu_p + \beta_p \quad \text{可学习仿射变换显式建模患者级均值/方差偏移}$$
$$\text{最终}: f_{\text{MR-INR}}(\mathbf{x}, \mathbf{z}_p) = \text{SIREN}(\gamma(\mathbf{x}), \mathbf{z}_p) + \alpha_p \cdot \mu_p + \beta_p$$

**Stage 1 损失函数**:
$$\mathcal{L}_{\text{Stage1}} = \lambda_{\text{Self}} \mathcal{L}_{\text{Self}} + \lambda_{\text{INR}} \mathcal{L}_{\text{INR}} + \lambda_{\text{Reg}} \mathcal{L}_{\text{Reg}}$$
其中 $\mathcal{L}_{\text{Self}}$ 为自监督损失（如 FINE/NR2N/SSDU 的原始损失），$\mathcal{L}_{\text{INR}}$ 为 INR 重建损失约束隐式表示质量，$\mathcal{L}_{\text{Reg}}$ 为正则化项防止过拟合。

**对应消融**：Table 7 显示仅 Stage 1（PE+SIREN）得 SSIM 0.845/PSNR 26.37/LPIPS 0.346，加入 Stage 2 后提升至 0.876/27.71/0.320，SSIM +0.031，PSNR +1.34，LPIPS -0.026。

---

### 模块 2: 各向异性扩散切片细化（对应框架图 Stage 2）

**直觉**：在冻结卷积特征上直接微调会破坏预训练知识；各向异性扩散仅在梯度小的方向平滑，保留边缘结构，且计算轻量。

**Baseline 公式** (各向同性扩散 / 标准高斯平滑): $$\mathbf{u}^{t+1} = \mathbf{u}^t + \eta \cdot \Delta \mathbf{u}^t$$
符号: $\mathbf{u}^t$ = 第 t 步特征图, $\eta$ = 步长, $\Delta$ = 拉普拉斯算子（各向同性）

**变化点**：各向同性扩散在所有方向均匀平滑，导致边缘模糊；且标准 SST 需端到端训练卷积参数，计算冗余。

**本文公式（推导）**:
$$\text{Step 1}: \mathbf{u}^{t+1} = \mathbf{u}^t + \eta \cdot \text{div}(D(\nabla \mathbf{u}^t) \nabla \mathbf{u}^t) \quad \text{将标量扩散系数替换为梯度依赖的张量 } D(\nabla \mathbf{u})$$
$$\text{Step 2}: D(\nabla \mathbf{u}) = g(\|\nabla \mathbf{u}\|) \cdot \mathbf{n}\mathbf{n}^\text{top} + \mathbf{t}\mathbf{t}^\text{top} \quad \text{沿梯度方向 } \mathbf{n} \text{ 抑制扩散，切向 } \mathbf{t} \text{ 允许扩散}$$
$$\text{最终}: \mathbf{u}^{t+1} = \mathbf{u}^t + \eta \cdot \text{div}\left(g(\|\nabla \mathbf{u}^t\|) \, \text{proj}_{\mathbf{t}}(\nabla \mathbf{u}^t)\right)$$
其中 $g(\cdot)$ 为可学习的边缘停止函数，控制各向异性程度。

**Stage 2 损失函数**（简化，去除正则化以提高效率）:
$$\mathcal{L}_{\text{Stage2}} = \lambda_{\text{Self}} \mathcal{L}_{\text{Self}} + \lambda_{\text{INR}} \mathcal{L}_{\text{INR}}$$

**对应消融**：作者未直接对比各向异性 vs 各向同性扩散的消融（为方法局限性），但 Table 7 中 Stage 1+2 联合优于 Stage 1 单独验证了细化模块的必要性。

---

### 模块 3: 可学习仿射变换（分布补偿层）

**直觉**：INR 输出的特征分布可能与目标域存在系统性偏移，类似 BN 但针对患者级统计量自适应。

**本文公式**:
$$\hat{\mathbf{f}} = \alpha_p \cdot \mathbf{f} + \beta_p$$
或显式均值调整形式：$\text{SIREN输出} + \alpha_p \cdot \mu_p + \beta_p$

符号: $\alpha_p$ = 缩放参数, $\beta_p$ = 偏置参数, $\mu_p$ = 患者级均值估计

该组件无独立消融表，但其效果嵌入 MR-INR 整体性能中。

## 实验与分析



本文在五种 MRI 分布偏移上系统评估 D2SA：解剖偏移（Knee→Brain）、数据集偏移（Stanford→fastMRI）、模态偏移（AXT2→AXT1PRE）、加速偏移（2×→4×）、采样偏移（random→uniform）。评估指标包括 SSIM、PSNR、LPIPS 及每患者推理时间（分钟），在 NVIDIA RTX 3090 上执行。

 展示了跨五种偏移的完整对比。核心发现：D2SA 在保持或提升重建质量的同时，实现 2-4 倍推理加速。以 FINE+MRINR+SST 为例，在解剖偏移上 SSIM 0.882、PSNR 27.68、LPIPS 0.311，与最优 baseline NR2N+SST（0.883/27.72/0.307）基本持平，但时间从 63.9 分钟骤降至 17.1 分钟（3.7× 对 NR2N+SST，3.1× 对 FINE+SST）。在采样偏移上优势最为显著：SSIM 0.824 vs 最优 baseline DIP-TTT 的 0.771，PSNR 28.49 vs 27.65，LPIPS 0.232 vs 0.254，同时时间 22.1 分钟 vs 38.9 分钟（1.8×）。NR2N+MRINR+SST 在数据集偏移（PSNR 28.76）和加速偏移（PSNR 28.89）上取得最高 PSNR，验证框架的通用性。



消融实验揭示关键设计选择。Table 7（INR 骨干选择）显示：PE+SIREN 在 Stage 1 得 SSIM 0.845/PSNR 26.37/LPIPS 0.346，优于 WIRE（0.842/26.25/0.348）、Finer（0.843/26.30/0.347）、RPE+MLP（0.841/26.18/0.349）和 SIREN-only（0.844/26.32/0.347）；加入 Stage 2 后 PE+SIREN 进一步提升至 0.876/27.71/0.320，LPIPS 优势尤为明显（0.320 vs WIRE 0.321 / Finer 0.323）。Table 8（损失权重）验证 $\lambda_{\text{INR}}$ 不能过小：当 $\lambda_{\text{INR}}$ 从 1.0 降至 0.1 时，Stage 1 PSNR 从 24.97 跌至 23.85，Stage 2 LPIPS 从 0.287 恶化至 0.356，说明 INR 损失对维持表示质量至关重要。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/87dfa85f-3c32-445c-82a8-c7ad8c83320d/figures/fig_003.jpeg)
*Figure: (a) Learnable affine transform scales feature maps by α and β before the final layer. (b)*



公平性检查：baseline 覆盖较全，包括 DIP-TTT、ZS-SSL 及三种主流 SSL 框架（FINE/NR2N/SSDU）的 SST 变体，均为该领域代表性方法。但存在以下局限：
- 各向异性扩散的「方向性」优势缺乏直接消融（各向同性 vs 各向异性未对比）
- 推理时间 17-45 分钟对实时临床仍偏长
- 未与 2024-2025 年最新扩散模型 TTA 方法对比
- 采样偏移上性能跃升异常突出，可能存在数据集特定优势

## 方法谱系与知识库定位

**方法家族**：测试时自适应（Test-Time Adaptation / Test-Time Training）→ 压缩感知 MRI 重建

**父方法**：DIP-TTT [11]（"Test-time training can close the natural distribution shift performance gap in deep learning based compressed sensing"）及 FINE+SST 框架。D2SA 将 DIP-TTT 的逐切片重复训练解构为「患者级 INR 预计算 + 切片级高效细化」，继承其自监督损失设计但彻底重构推理流程。

**改动插槽**：
- **架构**：新增 MR-INR（PE+SIREN + 患者隐编码 + 仿射变换）
- **目标函数**：Stage 1 三损失加权组合 → Stage 2 双损失简化
- **训练/推理策略**：单阶段端到端微调 → 显式两阶段分解（患者级共享计算）
- **数据策展**：利用患者内切片相关性作为隐式结构先验

**直接 baseline 差异**：
- **DIP-TTT**：逐切片独立优化，无跨切片共享 → D2SA 通过 INR 隐编码实现患者级参数共享
- **ZS-SSL**：零样本无适应 → D2SA 显式建模分布偏移
- **FINE+SST / NR2N+SST / SSDU+SST**：标准 SST 端到端微调 → D2SA 冻结卷积 + 各向异性扩散，防过平滑且加速

**后续方向**：
1. 将 MR-INR 扩展至 CT、超声等其他医学成像模态
2. 探索 INR 与扩散模型的联合，进一步缩短推理至分钟以内
3. 各向异性扩散的可学习系数与显式边缘检测网络结合

**标签**：modality: medical_imaging / paradigm: self-supervised, test-time_adaptation / scenario: distribution_shift, cross-domain / mechanism: implicit_neural_representation, anisotropic_diffusion, meta-learning / constraint: inference_efficiency, plug-and-play

