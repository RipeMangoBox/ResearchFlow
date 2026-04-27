---
title: 'Equivariance by Contrast: Identifiable Equivariant Embeddings from Unlabeled Finite Group Actions'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 对比等变学习：无监督有限群等变嵌入
- Equivariance by
- Equivariance by Contrast (EbC)
- Equivariance by Contrast (EbC) is t
acceptance: Poster
code_url: https://github.com/dynamical-inference/ebc
method: Equivariance by Contrast (EbC)
modalities:
- Image
- time series
paradigm: self-supervised
---

# Equivariance by Contrast: Identifiable Equivariant Embeddings from Unlabeled Finite Group Actions

[Code](https://github.com/dynamical-inference/ebc)

**Topics**: [[T__Self-Supervised_Learning]], [[T__Representation_Learning]] | **Method**: [[M__Equivariance_by_Contrast]] | **Datasets**: Infinite dSprites with group actions, Neural time series

> [!tip] 核心洞察
> Equivariance by Contrast (EbC) is the first general-purpose encoder-only method that can learn identifiable equivariant embeddings solely from group action observations, without requiring group-specific architectural choices.

| 中文题名 | 对比等变学习：无监督有限群等变嵌入 |
| 英文题名 | Equivariance by Contrast: Identifiable Equivariant Embeddings from Unlabeled Finite Group Actions |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2510.21706) · [Code](https://github.com/dynamical-inference/ebc) · [Project](未提供) |
| 主要任务 | Self-Supervised Learning, Representation Learning, Equivariant Representation Learning |
| 主要 baseline | E-SSL, EquiMod, SIE, RECL, G-CNN |

> [!abstract] 因为「学习等变表示需要硬编码群结构或监督信号」，作者在「SimCLR/BYOL 对比学习框架」基础上改了「用对比损失在潜在空间学习线性群表示 R(g)，编码器无需群特定结构」，在「Infinite dSprites 多群作用基准」上取得「首个 encoder-only 非阿贝尔群等变学习」

- 关键性能 1：EbC 在 Infinite dSprites 上成功学习 O(n)、GL(n)、R^m × Z_n × Z_n 等群的等变表示，为首个无需群特定架构的 encoder-only 方法
- 关键性能 2：在神经时间序列（海马体记录）上验证实际应用有效性
- 关键性能 3：提供理论可识别性（identifiability）保证，学习到的嵌入可辨识到等价变换

## 背景与动机

等变表示学习（Equivariant Representation Learning）旨在让神经网络的表示空间能够"预测"输入变换的效果：若对图像施加一个旋转，其潜在编码也应按已知规则变换。然而，现有方法面临一个根本困境——要么需要**预先知道群结构并将其硬编码进网络**（如 G-CNN 的 steerable filter），要么需要**监督信号标明每个变换对应的群元素**，这极大限制了方法在未知群或复杂群（如非阿贝尔群）上的适用性。

具体而言，现有三条技术路线各有局限：

**G-CNN 等群等变网络**（Group Equivariant Convolutional Networks）将群结构直接嵌入卷积核，使网络层天然满足 φ(g·y) = g·φ(y)。但这要求设计者预先知道群的结构并推导对应的群卷积公式，无法处理未知群或训练时才显现的群作用。

**E-SSL（Equivariant Self-Supervised Learning）** 等自监督方法虽无需显式标签，但仍依赖已知变换生成增广视图，通过辅助分支强制等变性。其核心局限在于仅适用于简单变换（如旋转、平移），且仍需变换参数作为监督信号。

**SIE（Split Invariant-Equivariant representations）** 与 **RECL（Rotationally Equivariant Contrastive Learning）** 近期尝试将等变性引入对比学习，但前者需要预定义不变/等变分解结构，后者局限于旋转群 SO(2)，均非通用方案。

上述方法的根本瓶颈在于：**群结构要么固化于架构（architecture），要么显式作为监督（supervision）**。对于更一般的有限群——尤其是非阿贝尔群（如 O(n) 中旋转与反射不可交换）或积群（如 R^m × Z_n × Z_n）——既无现成等变层可用，也难以获取完整标注。本文提出 EbC，首次证明仅从未标注的群作用观测对 (y, g·y) 出发，用标准编码器 + 潜在空间学习的线性映射即可实现可识别的等变嵌入。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c098adb0-7050-48f5-b75c-2ab22c6118eb/figures/Figure_3.png)
*Figure 3 (qualitative): Qualitative explanation of the training objective*



## 核心创新

核心洞察：**将群结构从编码器网络权重"解耦"到潜在空间的可学习线性映射**，因为对比学习天然适合约束"变换后编码 ≈ 编码后变换"的配对关系，从而使标准编码器无需任何群特定设计即可学习任意有限群的等变表示。

| 维度 | Baseline (SimCLR/E-SSL/G-CNN) | 本文 (EbC) |
|:---|:---|:---|
| **等变约束位置** | 网络层内硬编码（G-CNN）或辅助分支监督（E-SSL） | 潜在空间可学习线性映射 R̂(g) |
| **编码器架构** | 群特定层（steerable filter / G-convolution）或专用分支 | 标准编码器（如 ResNet），完全群无关 |
| **训练信号** | 实例判别（SimCLR）或已知变换参数（E-SSL/RECL） | 仅观测对 (y, g·y)，无需 g 的标签 |
| **群适用范围** | 预设群（SO(2)、平移等） | 任意有限群，包括非阿贝尔群与积群 |
| **理论保证** | 无显式可识别性证明 | 非线性 ICA 框架下的可识别性定理 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c098adb0-7050-48f5-b75c-2ab22c6118eb/figures/Figure_2.png)
*Figure 2 (pipeline): Overview of the approach*



EbC 的数据流可概括为四阶段流水线：

1. **群作用数据生成**：从原始观测 y 出发，应用群元素 g 得到变换后观测 g·y，构成正样本对 (y, g·y)。负样本 y⁻ 从其他观测随机采样。

2. **标准编码器 φ**：输入为原始观测 y 或变换后观测 g·y，输出潜在编码 x̂ = φ(y) ∈ ℝ^d。关键：φ 为任意标准网络（实验中用 ResNet），**无任何群特定结构**。

3. **可学习群表示 R̂(g)**：对每个群元素 g，学习一个可逆线性映射矩阵 R̂(g) ∈ GL(d)。输入为 φ(y)，输出为 R̂(g)φ(y)，即在潜在空间中"模拟"群作用 g 的效果。

4. **对比损失计算**：将 φ(g·y) 与 R̂(g)φ(y) 作为正样本对拉近距离，将 φ(g·y) 与负样本编码 φ(y⁻) 推远距离，联合训练 φ 与 {R̂(g)}。

整体流程的 ASCII 示意：
```
y ──[群作用 g]──→ g·y
│                 │
▼                 ▼
φ(y)            φ(g·y)
│                 ▲
│                 │ (正样本，应相似)
└─[R̂(g)]─→ R̂(g)φ(y)
              │
              └─ vs ─→ φ(y⁻) (负样本，应远离)
```


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c098adb0-7050-48f5-b75c-2ab22c6118eb/figures/Figure_1.png)
*Figure 1 (architecture): Commutative diagram illustrating the contrastive objective*



## 核心模块与公式推导

### 模块 1: 标准编码器映射（对应框架图左侧）

**直觉**：保持编码器完全"群无知"，将所有群结构后移到潜在空间，这是 EbC 区别于 G-CNN 的关键设计。

**Baseline 公式** (G-CNN): 
$$\text{G-conv}(g \cdot y) = g \cdot \text{G-conv}(y) \quad \text{(hardcoded in filter weights)}$$

符号: $y$ = 输入观测, $g$ = 群元素, G-conv = 群等变卷积层。

**变化点**: G-CNN 将群等变性固化于卷积核参数，导致每换一种群就需重新设计网络。EbC 取消此假设，改用标准前馈网络。

**本文公式**:
$$\hat{x} = \phi(y)$$

符号: $\phi$ = 标准编码器（如 ResNet）, $\hat{x}$ = d 维潜在编码。

**对应消融**: Table 2 显示使用标准编码器即可达到与群特定架构相当或更优的等变学习效果。

---

### 模块 2: 潜在空间群表示 R̂(g)（对应框架图中间）

**直觉**：既然编码器不处理群结构，则必须在潜在空间中"补回"群作用——用最小二乘从数据拟合最佳线性近似。

**Baseline 公式**: 无直接对应；传统方法无此模块（等变性由网络层保证）。

**变化点**: 这是 EbC 的新增核心组件。对每个群元素 g，从 k 个样本对 {$(y_i, g \cdot y_i)$}_{i=1}^k 估计矩阵 R̂(g)。

**本文公式（推导）**:
$$\text{Step 1}: \quad \hat{R}(g) = \text{arg}\min_{R \in GL(d)} \sum_{i=1}^{k} \| \phi(g \cdot y_i) - R \phi(y_i) \|^2$$
加入正交性或可逆约束以保证 R̂(g) 构成有效群表示，避免退化解。

$$\text{Step 2}: \quad \text{若群结构已知，可进一步约束 } \hat{R}(g_1 g_2) = \hat{R}(g_1)\hat{R}(g_2) \text{（同态约束）}$$
重归一化/投影以保证群表示的代数一致性。

$$\text{最终}: \quad \hat{R}(g)\phi(y) \approx \phi(g \cdot y)$$

**对应消融**: Figure 5 显示样本数 k 减少时 R̂(g) 估计质量下降，但 EbC 在 k 较小时仍优于 baseline。

---

### 模块 3: 等变对比损失 L_EbC（对应框架图右侧）

**直觉**：将 SimCLR 的"同实例不同增广"正样本对，替换为"同观测群变换前后"的等变对，并引入 R̂(g) 作为可学习的"潜在增广"。

**Baseline 公式** (SimCLR):
$$\mathcal{L}_{\text{SimCLR}} = -\mathbb{E}_{x, x^+}\left[\log \frac{\exp(\text{sim}(z, z^+)/\tau)}{\sum_{z^-} \exp(\text{sim}(z, z^-)/\tau)}\right]$$

符号: $z, z^+$ = 同实例的两个增广视图编码, $z^-$ = 负样本编码, $\tau$ = 温度参数, sim = 余弦相似度。

**变化点**: SimCLR 的正样本对 $(z, z^+)$ 来自随机增广（裁剪、颜色抖动等），无显式等变约束；EbC 的正样本对要求 $z^+ = \hat{R}(g)\phi(y)$ 主动"预测"群作用效果，且负样本对比的是变换后编码 $\phi(g \cdot y)$ 与其他观测编码。

**本文公式（推导）**:
$$\text{Step 1}: \quad \text{定义等变正样本对 } (\phi(g \cdot y), \hat{R}(g)\phi(y)) \text{，替换 SimCLR 的 } (z, z^+)$$
加入 R̂(g) 变换项以在潜在空间强制执行等变性。

$$\text{Step 2}: \quad \text{负样本保持为其他观测编码 } \phi(y^-) \text{，但查询端改为 } \phi(g \cdot y)$$
重归一化分母以包含所有负样本的对比。

$$\text{最终}: \quad \mathcal{L}_{\text{EbC}} = -\mathbb{E}_{y,g}\left[\log \frac{\exp(\text{sim}(\phi(g \cdot y), \hat{R}(g)\phi(y))/\tau)}{\sum_{y^-} \exp(\text{sim}(\phi(g \cdot y), \phi(y^-))/\tau)}\right]$$

符号: $y$ = 原始观测, $g \cdot y$ = 群变换后观测, $\hat{R}(g)$ = 学习的群表示矩阵, $y^-$ = 负样本观测, $\tau$ = 温度参数。

**对应消融**: Table 2 显示替换为标准对比损失（无 R̂(g) 项）时等变学习失败，验证 R̂(g) 的必要性。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c098adb0-7050-48f5-b75c-2ab22c6118eb/figures/Table_1.png)
*Table 1 (comparison): Overview*




![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c098adb0-7050-48f5-b75c-2ab22c6118eb/figures/Table_2.png)
*Table 2 (comparison): Model comparison across different loss functions, negative samples, and groups*



本文在 **Infinite dSprites** 数据集上构建多群作用基准进行主实验，涵盖旋转群 SO(2)、正交群 O(n)、一般线性群 GL(n)、以及积群 R^m × Z_n × Z_n 等有限群实例。Table 1 汇总显示，EbC 在各类群上均成功学习等变表示，而现有方法或受限于群类型（RECL 仅旋转）、或需要群特定设计（G-CNN）。具体而言，EbC 在 O(n) 与 GL(n) 等非阿贝尔群上首次实现 encoder-only 等变学习，这是 E-SSL、EquiMod 等 baseline 无法完成的——后者在复杂群作用下的等变误差显著上升。


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c098adb0-7050-48f5-b75c-2ab22c6118eb/figures/Figure_4.png)
*Figure 4 (result): E2C learns faithful representations of group actions*



Figure 4 定性展示了 EbC 学习的群表示忠实度：潜在空间中 R̂(g) 的作用轨迹与真实群作用高度一致，验证了线性映射足以捕获有限群结构。在海马体神经时间序列数据上（Figure 12），EbC 成功提取具有等变结构的神经表征，为等变学习在神经科学中的应用提供了概念验证。




![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/c098adb0-7050-48f5-b75c-2ab22c6118eb/figures/Figure_5.png)
*Figure 5 (ablation): Performance on model selection for group representations*



消融实验（Figure 5 与 Table 2）进一步剖析关键设计：
- **损失函数变体**：移除 R̂(g) 项（退化为标准对比学习）导致等变学习完全失效；使用 MSE 替代对比损失则负样本利用不足，收敛速度下降。
- **负样本数量**：减少负样本数时 EbC 仍保持稳健，表明等变约束本身提供了强归纳偏置。
- **样本量 k 估计 R̂(g)**：k 从充足降至 10 以下时，R̂(g) 估计误差上升，但联合训练部分补偿此效应。
- **数据规模**：训练数据从 1M 降至 50k 时，EbC 性能衰减可控，优于需要大量负样本的 SimCLR 变体。

公平性审视：对比的 E-SSL、EquiMod、SIE、RECL 均为该方向代表性方法，且作者未宣称 SOTA 而是强调"首个通用性"。局限包括：评估仅限 dSprites 合成数据与单一神经时间序列域；假设有限群且已知群结构以构造 (y, g·y) 对；连续群（如李群）的扩展性未验证。训练计算细节见附录 B。

## 方法谱系与知识库定位

EbC 属于**对比学习驱动的等变表示学习**方法族，直接继承 **SimCLR/BYOL** 的自监督对比范式，但在三个核心 slot 上完成结构性替换：

| 谱系关系 | 方法 | 差异 |
|:---|:---|:---|
| **父方法** | SimCLR | 将实例判别目标替换为等变对比目标；增广视图替换为群作用对 |
| **父方法** | BYOL | 继承无标签训练哲学，但引入显式负样本与群表示矩阵 |
| **直接 baseline** | E-SSL | EbC 无需已知变换参数，编码器完全标准 |
| **直接 baseline** | RECL | EbC 从旋转群 SO(2) 推广至任意有限群 |
| **直接 baseline** | SIE | EbC 无需预分裂不变/等变子空间 |
| **直接 baseline** | EquiMod | EbC 的 R̂(g) 为显式学习而非模块注入 |
| **架构对比** | G-CNN | EbC 取消群特定层，等变性完全由损失与潜在映射习得 |

后续方向：(1) 向连续群/李群扩展，需将有限群上的矩阵表示推广为算子学习；(2) 与生成模型结合，利用等变嵌入提升扩散模型或流匹配的对称性保持能力；(3) 在更多真实域（分子结构、物理模拟）验证，尤其关注群结构未知时的联合发现。

**标签**: 模态=图像/时间序列 | 范式=自监督对比学习 | 场景=表示学习/等变嵌入 | 机制=潜在空间群表示/可识别性理论 | 约束=有限群/已知群作用对

