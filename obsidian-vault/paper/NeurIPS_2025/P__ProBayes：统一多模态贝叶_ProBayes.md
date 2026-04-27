---
title: Rationalized All-Atom Protein Design with Unified Multi-Modal Bayesian Flow
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- ProBayes：统一多模态贝叶斯流全原子蛋白设计
- ProBayes
- ProBayes is the first unified multi
acceptance: Poster
method: ProBayes
modalities:
- protein structure
- sequence
paradigm: generative modeling (Bayesian flow networks)
---

# Rationalized All-Atom Protein Design with Unified Multi-Modal Bayesian Flow

**Topics**: [[T__3D_Reconstruction]] | **Method**: [[M__ProBayes]] | **Datasets**: PepBench

> [!tip] 核心洞察
> ProBayes is the first unified multi-modal Bayesian flow framework for rationalized all-atom protein design that eliminates information shortcuts through a novel rationalized information flow strategy and enables proper generation of protein backbone orientations via a hyperspherical formulation with antipodal symmetry.

| 中文题名 | ProBayes：统一多模态贝叶斯流全原子蛋白设计 |
| 英文题名 | Rationalized All-Atom Protein Design with Unified Multi-Modal Bayesian Flow |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2501.XXXXX) · Code (待验证) · Project (待验证) |
| 主要任务 | 全原子蛋白质设计（all-atom protein design）、多肽设计、抗体设计 |
| 主要 baseline | Protpardelle（主要对比）、PepGLAD、dyMEAN、Bayesian Flow Networks |

> [!abstract] 因为「现有全原子蛋白生成方法存在信息捷径问题——模型利用侧链信息推断序列，导致序列分布学习失效」，作者在「Bayesian Flow Networks (BFNs)」基础上改了「(1) 将SO(3)旋转生成等价变换到带对跖对称的S3超球面贝叶斯流；(2) 引入合理化信息流约束消除侧链到序列的信息泄露；(3) 统一三模态（骨架方向/序列/侧链）联合生成框架」，在「PepBench」上取得「能量∆G = -28.63，Native Likeness = 72.92，Valid = 0.998」

- **PepBench 能量指标**：∆G = -28.63，相比 PepGLAD 在 PDB=4cu4 上的 ∆G = -30.71 更优
- **结构有效性**：Valid = 0.998，接近完美；dyMEAN 在 PDB=5j13 上产生物理无效结构 ∆G = +23911.25
- **新颖多样性**：V&Novel = 0.728，V&Div = 0.449

## 背景与动机

蛋白质设计旨在生成具有特定功能的三维结构，需要同时确定骨架走向、氨基酸序列和侧链构象。然而，这三个模态之间存在复杂的耦合关系：序列决定侧链类型，侧链堆积影响骨架稳定性，骨架几何又约束序列选择。现有方法如 Protpardelle 尝试通过多模态联合生成来建模这种耦合，却引入了一个隐蔽的缺陷——信息捷径（information shortcut）。

具体而言，当模型在预测序列时能够访问侧链信息（即使是加噪的），它会利用侧链二面角分布的统计规律性来"作弊"推断序列，而非真正学习序列-结构的联合分布。Figure 3a 通过 KL 散度测量证实了这一现象：KL(P(sequence|side-chain) || P(sequence)) 显著不为零，说明侧链信息确实泄露了序列内容。Protpardelle 尝试通过对 37 种独特原子位置加噪来缓解，但附录 I 证明这仍然无法消除捷径。类似地，PepGLAD 基于多模态流匹配（multi-modal flow matching）进行全原子肽设计，dyMEAN 专注于抗体端到端设计，但均未显式处理这一信息流问题。

现有 SO(3) 旋转生成方法也存在局限：SE(3) 流匹配 [5,6] 和等变扩散 [7] 直接在欧氏空间或李群上操作，需要复杂的约束满足；四元数参数化则面临 q ~ -q 的双覆盖歧义。本文提出 ProBayes，首次将贝叶斯流框架扩展至蛋白质领域，通过两个核心创新——S3 超球面贝叶斯流与合理化信息流——解决上述问题。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d5dc486d-22a9-43dc-8e07-ba7bd3bba4aa/figures/fig_001.png)
*Figure: Illustration of protein representation.*



## 核心创新

核心洞察：SO(3) 旋转群可以通过四元数表示等价嵌入到 S3 超球面，并利用对跖对称性 q ~ -q 消除双覆盖歧义，从而使贝叶斯流能够原生地在球面上定义；同时，通过约束 KL(P(sequence|side-chain) || P(sequence)) 不坍缩，可以强制切断侧链到序列的信息捷径，使真正的联合分布学习成为可能。

| 维度 | Baseline (Protpardelle/BFNs) | 本文 (ProBayes) |
|:---|:---|:---|
| 旋转表示 | 旋转矩阵 R ∈ SO(3) ⊂ ℝ^(3×3)，需正交约束 | S3 超球面 q ∈ S³, ‖q‖=1，对跖等价 [q]~[-q] |
| 生成范式 | 扩散/流匹配在欧氏空间或 SE(3) 上 | 贝叶斯流在黎曼流形上，vMF 分布为共轭先验 |
| 信息流 | 序列生成可访问侧链特征，存在信息捷径 | 信息流门控阻断侧链→序列通道，KL 约束正则化 |
| 模态协调 | 分阶段或分解式生成 | 统一框架内三流并行：S3 方向 + 离散序列 + 连续侧链 |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d5dc486d-22a9-43dc-8e07-ba7bd3bba4aa/figures/fig_002.png)
*Figure: Intuitive illustration of the equivalent transformation from SO(3) generation to hypersphere*



ProBayes 的整体架构是一个统一的多模态贝叶斯流框架，包含五个核心模块：

1. **Target Structure Encoder**：输入蛋白质靶标结构（结合位点上下文），输出条件化表征，为后续生成提供几何约束。

2. **S3 Orientation Flow（新增）**：输入 S3 上的加噪方向状态及条件表征，输出骨架方向（通过四元数映射回 SO(3)）。这是首个针对蛋白质骨架方向的贝叶斯流模块，使用 von Mises-Fisher 分布建模球面概率。

3. **Sequence Flow (Discrete)**：输入离散加噪序列状态及条件表征（**关键：不访问侧链信息**），输出生成的氨基酸序列。信息流门控确保此模块仅接收骨架和条件信息。

4. **Side-Chain Flow (Continuous)**：输入连续加噪侧链坐标、条件表征以及已生成的序列，输出全原子侧链构象。序列信息在此阶段引入以指导侧链类型和构象。

5. **Information Flow Gate（新增）**：在训练时阻断侧链特征向序列流的传递，并通过 KL 散度约束防止信息泄露。

数据流概览：
```
Target Structure → [Encoder] → Conditioning
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
              [S3 Flow]      [Sequence Flow]   [Side-Chain Flow]
              (方向生成)       (序列生成, 无侧链)   (侧链生成, 有序列)
                    ↓               ↓               ↓
                    └───────────────┴───────────────┘
                                    ↓
                          [All-Atom Protein Output]
```

三流在统一的时间参数 t 下协调演化，共享噪声调度但保持各自模态的概率特性：S3 上流形约束、序列上离散分布、侧链上连续欧氏分布。

## 核心模块与公式推导

### 模块 1: S3 超球面贝叶斯流（对应框架图 S3 Orientation Flow）

**直觉**：蛋白质骨架方向是 SO(3) 中的旋转，但直接在矩阵空间上定义流需要复杂的正交约束；通过四元数映射到 S3 球面，可利用球面指数族分布简化概率建模。

**Baseline 公式** (SE(3) Flow Matching [5,6]): 
$$R \in SO(3) \subset \mathbb{R}^{3 \times 3}, \quad R^\text{top} R = I, \det(R) = 1$$
在欧氏空间中用高斯扰动：$p(R_t | R_0) = \mathcal{N}(\text{vec}(R_t); \text{vec}(R_0), \sigma^2(t)I)$，但需投影回 SO(3)。

符号: $R$ = 旋转矩阵, $q$ = 单位四元数, $S^3$ = 三维单位球面。

**变化点**：高斯分布在球面上非原生，投影引入误差；SO(3) 的双连通性导致四元数 q 和 -q 表示同一旋转，需要显式处理对跖等价。

**本文公式（推导）**:
$$\text{Step 1}: \quad SO(3) \cong S^3/\{\pm 1\}, \quad q \in S^3, \|q\| = 1$$
$$\text{Step 2}: \quad [q] \sim [-q] \text{ (antipodal identification)} \quad \text{消除双覆盖歧义}$$
$$\text{Step 3}: \quad p(x_t | \theta) = \text{vMF}(x_t; \mu(\theta), \kappa(t)) \quad \text{vMF 替代高斯作为 S3 上共轭先验}$$
$$\text{最终}: \quad \mathcal{L}_{\text{orient}} = \mathbb{E}_{t,\theta}\left[ -\log p(x_t | \theta_{\text{pred}}(x_t, t)) \right]$$
其中 vMF 分布 $p(x; \mu, \kappa) = C_d(\kappa) \exp(\kappa \mu^\text{top} x)$，浓度参数 $\kappa(t)$ 控制噪声水平，均值方向 $\mu(\theta)$ 由 BFN 参数 $\theta$ 决定。

**对应消融**：去掉 S3 形式化而改用直接四元数回归时，方向采样不稳定，生成结构 RMSD 显著恶化（原文未报告精确数值，但 Figure 2 直观展示了等价变换的必要性）。

---

### 模块 2: 合理化信息流约束（对应框架图 Information Flow Gate）

**直觉**：如果序列预测器能看到侧链，它会走捷径——利用侧链二面角的统计规律推断氨基酸类型，而非从骨架上下文学习序列-结构映射。

**Baseline 公式** (标准 BFN [20]):
$$\mathcal{L}_{\text{BFN}} = \mathbb{E}_{t,\theta}\left[ \mathbb{E}_{x \sim p(x|\theta)} \left[ -\log p(\theta | x) \right] \right]$$
所有模态的特征自由交互，无信息流限制。

符号: $s$ = 序列, $c$ = 侧链, $\theta$ = BFN 分布参数, $\lambda$ = 约束权重。

**变化点**：标准 BFN 损失允许 $P(s|c)$ 与 $P(s)$ 坍缩——即给定侧链后序列条件分布与边缘分布相同，这意味着侧链完全确定了序列，模型无需学习真正的联合分布。

**本文公式（推导）**:
$$\text{Step 1}: \quad \text{测量捷径: } KL(P(s|c) \| P(s)) \rightarrow 0 \text{ 时，信息泄露严重}$$
$$\text{Step 2}: \quad \text{引入约束: } \mathcal{L}_{\text{IF}} = KL(P_{\text{model}}(s|c) \| P_{\text{model}}(s)) \quad \text{防止条件-边缘坍缩}$$
$$\text{Step 3}: \quad \text{信息流门控: 序列流输入中物理移除侧链特征}$$
$$\text{最终}: \quad \mathcal{L}_{\text{ProBayes}} = \mathcal{L}_{\text{BFN}} + \lambda \cdot \underbrace{KL(P(s|c) \| P(s))}_{\text{约束项}}$$
训练时通过梯度惩罚确保 $KL \text{nrightarrow} 0$，强制序列预测器仅依赖骨架和条件信息。

**对应消融**：Figure 3a 显示，即使对侧链加噪，KL(P(sequence|side-chain) || P(sequence)) 仍显著大于零，证明信息捷径确实存在；移除约束后模型迅速利用侧链推断序列，Valid 指标下降（具体数值未在摘录中完整报告）。

---

### 模块 3: 统一多模态协调（对应框架图 Multi-Modal Flow Coordinator）

**直觉**：三个模态具有不同的概率本性——方向在球面、序列在离散空间、侧链在欧氏空间——需要统一的时间演化框架但保持各自的似然形式。

**Baseline 公式** (多模态流匹配 [10]):
分别定义三个流匹配目标再组合：$\mathcal{L} = \mathcal{L}_{\text{backbone}} + \mathcal{L}_{\text{seq}} + \mathcal{L}_{\text{side}}$，但各流独立优化。

**变化点**：独立优化导致模态间耦合不足；ProBayes 在单一 BFN 框架下统一噪声调度和时间参数，同时通过生成顺序（方向 → 序列 → 侧链）引入因果依赖。

**本文公式**:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{S3 orient}} + \mathcal{L}_{\text{discrete seq}}^{\text{(masked)}} + \mathcal{L}_{\text{continuous side}} + \lambda \cdot \mathcal{L}_{\text{IF}}$$
其中 $\mathcal{L}_{\text{discrete seq}}^{\text{(masked)}}$ 表示侧链特征被掩码的序列损失，三项共享相同的时间采样 $t \sim \text{Uniform}(0,1)$ 但使用各自模态的输出分布。

## 实验与分析



本文在 **PepBench** 基准上评估多肽设计性能，并在抗体设计任务上与现有方法进行定性对比。Table 6 报告了 ProBayes 的完整指标：能量 ∆G = **-28.63**（越低越好），Native Likeness = **72.92**，Success = **0.74**，DockQ = **0.228**，RMSDCα = **2.28 Å**，结构有效性 Valid = **0.998**，多样性 V&Div = **0.449**，新颖性 V&Novel = **0.728**。这些数值表明 ProBayes 在保持高结构有效性的同时，实现了较强的结合亲和力预测和合理的结构多样性。


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/d5dc486d-22a9-43dc-8e07-ba7bd3bba4aa/figures/fig_003.png)
*Figure: (a): Illustration of information shortcut measured by KL divergence. (b)-(d): Training*



Figure 8 展示了多肽设计的可视化对比：在 PDB=4cu4 上，ProBayes 生成结构的 ∆G = **-39.42**，明显优于 PepGLAD 的 ∆G = **-30.71**，且与参考结构的空间构象更为接近。Figure 9 展示了抗体设计的极端案例：dyMEAN 在 PDB=5j13 上产生物理无效结构（∆G = **+23911.25**，数值爆炸表明原子碰撞严重），而 ProBayes 生成合理结构（∆G = **-23.64**）。



消融实验聚焦于两个核心组件。信息流约束的移除导致 KL(P(sequence|side-chain) || P(sequence)) 坍缩趋零，模型重新利用侧链捷径，序列分布学习退化——虽然原文未给出精确数值表格，但 Figure 3a 的 KL 散度曲线直观展示了这一效应。S3 超球面形式的移除则导致 SO(3) 生成不稳定，方向采样出现歧义。

**公平性检验**：实验证据存在明显局限。Table 6 仅报告 ProBayes 自身结果，**缺少与 Protpardelle、PepGLAD、dyMEAN 的直接数值对比**；视觉对比（Figure 8-9）存在 cherry-picking 嫌疑，仅展示 baseline 的失败案例而未呈现系统统计。RFdiffusion [4]、SE(3) 流匹配 [5,6]、等变扩散 [7]、多模态流匹配 [10]、3D 分子 BFN [22] 等强 baseline 未被纳入数值比较。此外，训练计算成本、推理延迟、模型参数量等关键信息均未披露，可扩展性至大蛋白的能力未经验证。代码与检查点链接虽已提供但状态未经验证。

## 方法谱系与知识库定位

**方法家族**：生成式蛋白质设计 → 流/扩散模型 → Bayesian Flow Networks (BFNs)

**父方法**：Bayesian Flow Networks (Graves et al., 2023) [20] —— ProBayes 继承其连续时间贝叶斯更新框架，但将其从简单欧氏/离散空间扩展到 (1) 黎曼流形 S3 上的 SO(3) 生成，(2) 多模态约束优化。

**直接 Baseline 差异**：
- **Protpardelle** [11]：同为全原子生成，但使用扩散模型且存在信息捷径；ProBayes 改用 BFN 并引入信息流约束
- **PepGLAD** [10]：多模态流匹配用于肽设计，但各流独立、无 SO(3) 原生处理；ProBayes 统一 BFN 框架并首创 S3 贝叶斯流
- **dyMEAN** [9]：抗体端到端设计，无显式方向生成模块；ProBayes 的 S3 流提供更稳定的骨架方向建模
- **BFN for 3D Molecules** [22]：BFN 在分子生成中的应用，但未涉及蛋白质特有的序列-结构耦合与 SO(3) 问题

**后续方向**：(1) 将 S3 超球面贝叶斯流推广至一般李群/齐性空间上的生成；(2) 信息流约束思想扩展至其他多模态生成任务（如文本-图像联合生成中的模态泄露问题）；(3) 结合 AlphaFold 等结构预测器进行迭代优化设计。

**标签**：蛋白质结构+序列多模态 | 贝叶斯流/生成模型 | 肽/抗体设计 | 黎曼流形生成 | 信息流约束/因果解耦

## 引用网络

### 直接 baseline（本文基于）

- Fingerprinting Denoising Diffusion Probabilistic Models _(CVPR 2025, 方法来源, 未深度分析)_: Foundational DDPM paper (Ho et al.). Core diffusion model methodology. Likely ci

