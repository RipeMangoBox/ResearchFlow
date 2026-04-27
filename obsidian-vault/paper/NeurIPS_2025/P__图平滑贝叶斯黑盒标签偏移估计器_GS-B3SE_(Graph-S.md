---
title: Graph–Smoothed Bayesian Black-Box Shift Estimator and Its Information Geometry
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 图平滑贝叶斯黑盒标签偏移估计器
- GS-B3SE (Graph-S
- GS-B3SE (Graph-Smoothed Bayesian Black-Box Shift Estimator)
- GS-B3SE is a fully probabilistic bl
acceptance: Spotlight
cited_by: 1
method: GS-B3SE (Graph-Smoothed Bayesian Black-Box Shift Estimator)
modalities:
- Text
- Image
paradigm: supervised
---

# Graph–Smoothed Bayesian Black-Box Shift Estimator and Its Information Geometry

**Topics**: [[T__Domain_Adaptation]] | **Method**: [[M__GS-B3SE]] | **Datasets**: [[D__CIFAR-10]], [[D__CIFAR-100]] (其他: MNIST)

> [!tip] 核心洞察
> GS-B3SE is a fully probabilistic black-box shift estimator that uses Laplacian-Gaussian priors on target log-priors and confusion-matrix columns tied together on a label-similarity graph, achieving superior estimation accuracy and downstream performance with tractable posterior inference.

| 中文题名 | 图平滑贝叶斯黑盒标签偏移估计器 |
| 英文题名 | Graph–Smoothed Bayesian Black-Box Shift Estimator and Its Information Geometry |
| 会议/期刊 | NeurIPS 2025 (Spotlight) |
| 链接 | [arXiv](https://arxiv.org/abs/2505.16251) · [Code](待补充) · [Project](待补充) |
| 主要任务 | Label Shift Estimation / Domain Adaptation |
| 主要 baseline | BBSE, RLLS, MLLS, Saerens EM re-weighting |

> [!abstract] 因为「经典黑盒偏移估计器产生脆弱的点估计、忽略采样噪声与类别间语义相似性，尤其在多类别数据稀缺场景失效」，作者在「BBSE」基础上改了「引入图结构拉普拉斯-高斯先验的贝叶斯层次模型，以 HMC/Newton-CG 进行后验推断」，在「MNIST/CIFAR-10/CIFAR-100 标签偏移基准」上取得「MNIST L1 prior error 0.002 vs MLLS 0.010 (5×提升)；CIFAR-100 accuracy 0.783 vs MLLS 0.734 (+4.9 pp)」

- **MNIST**: L1 prior error 0.002，较最优 baseline MLLS (0.010) 降低 5×；下游准确率 0.986 vs MLLS 0.963 (+2.3 pp)
- **CIFAR-10**: L1 prior error 0.025 vs MLLS 0.052 (2×提升)；准确率 0.844 vs MLLS 0.812 (+3.2 pp)
- **CIFAR-100**: L1 prior error 0.22 vs MLLS 0.71 (3.2×提升)；准确率 0.783 vs MLLS 0.734 (+4.9 pp)

## 背景与动机

标签偏移（Label Shift）是域自适应中的经典问题：假设源域与目标域的类别条件分布 P(X|Y) 保持不变，但类别先验 P(Y) 发生变化，目标是从目标域未标注数据中恢复新的类别先验 q。例如，一个在医院 A 训练的皮肤病变分类器部署到医院 B 时，各类病变的就诊比例可能截然不同，但病变本身的视觉特征分布不变。此时需要利用源域训练的分类器（黑盒）和目标域的预测输出，反推目标域的真实类别比例。

现有方法沿三条技术路线处理该问题：

**BBSE (Black-Box Shift Estimator)** [18] 是最直接的闭式解法，通过经验混淆矩阵的矩阵求逆得到点估计 q̂ = C̃⁻ᵀ · (1/n') Σᵢ ĥ(xᵢ)。该方法计算高效，但完全忽视采样不确定性，且每个类别独立估计，无法利用语义相似类别间的信息共享。

**RLLS (Regularized Learning under Label Shift)** [7] 在 BBSE 基础上添加 ℓ₂ 正则化，通过凸优化缓解矩阵求逆的数值不稳定性，但仍为点估计框架，未建模后验不确定性。

**MLLS (Maximum Likelihood Label Shift)** [18] 采用似然最大化视角，在留出验证集上调参，但同样缺乏对混淆矩阵 C 和目标先验 q 的概率建模，且高类别数场景下 per-class 数据稀缺导致估计方差剧增。

这些方法的共同瓶颈在于：**将标签偏移估计视为确定性反问题，用单个点估计代表整个后验分布**。这导致三重缺陷：(1) 无法量化估计不确定性；(2) 语义相近的类别（如"猫"与"虎"）无法相互借用统计强度；(3) 类别数 K 增大时，经验混淆矩阵的 per-column 样本量不足，估计误差急剧恶化。尤其在 CIFAR-100 (K=100) 等高类别数场景中，现有方法的 L1 prior error 可达 0.7 以上，严重制约下游分类校正效果。

本文的核心动机正是将这一确定性框架彻底概率化：通过图结构先验耦合语义相似类别，以贝叶斯后验推断替代点估计，同时保持黑盒设定下仅需分类器输出的便利性。

## 核心创新

核心洞察：**标签偏移估计的不确定性可以通过类别语义相似图进行结构化传播**，因为在图拉普拉斯先验下，相邻类别的混淆矩阵列与目标先验被迫平滑一致，从而使高类别数、小样本 per-class 场景下的稳定估计成为可能。

| 维度 | Baseline (BBSE) | 本文 (GS-B3SE) |
|:---|:---|:---|
| **估计范式** | 确定性点估计（矩阵求逆） | 完全贝叶斯后验推断 |
| **类别关系** | 各类别完全独立，无信息共享 | 通过 k-NN 相似图耦合，语义相近类别共享统计强度 |
| **不确定性量化** | 无（单点输出） | 完整后验分布，HMC 采样或 Newton-CG 优化 |
| **先验结构** | 无显式先验（或简单 L2 正则） | 层次 Gamma 超先验 + 拉普拉斯-高斯图先验 |
| **退化相容性** | — | 当 L=0 且超先验退化为 delta 质量时，精确恢复 BBSE |

与 baseline 的本质差异在于：BBSE 将混淆矩阵 C 和目标先验 q 视为固定未知量求解；GS-B3SE 则将二者视为随机变量，用图结构精度矩阵 τ·L 编码类别相似性，使后验推断能够"借调"相邻类别的观测信息。这一改变不仅是算法层面的正则化添加，而是**从频率学派反问题到贝叶斯层次模型的范式转换**。

## 整体框架


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/87d9d665-052d-402b-a110-2cededd9187e/figures/Table_1.png)
*Table 1: Information geometry identification of OS-IP-SE*



GS-B3SE 的完整数据流包含五个串联模块：

**① 源域分类器训练**（输入：10K 均匀分布的源域标注样本；输出：ResNet-18 骨干分类器 h(x)）。与常规监督训练无异，使用标准交叉熵损失。

**② 经验混淆矩阵估计**（输入：5K 留出源域验证集；输出：经验混淆矩阵 C̃）。计算分类器在验证集上的预测-标签联合分布，为后续贝叶斯模型提供似然基础。

**③ 标签相似图构建**（输入：类别名称文本或训练特征；输出：k-NN 图的拉普拉斯矩阵 L = D - W）。这是 GS-B3SE 独有的预处理模块：对一般数据集，用冻结 CLIP ViT-B/32 文本编码器将类别名嵌入为 512-d 向量（L2 归一化），取 k 近邻（K=10 时 k=4，K=100 时 k=8），边权重 Wᵢⱼ = exp(-‖eᵢ-eⱼ‖²/(2σ²))，σ 为图中边集内中位数成对距离；对 MNIST 因类别名为单位数字，改用 128-d 倒数第二层特征在训练图像上平均后的 4-NN 图。确保图连通（λ₂(L) > 0）。

**④ 贝叶斯后验推断**（输入：目标域预测 ĥ(x)、C̃、图拉普拉斯 L；输出：q 和 C 的联合后验分布）。替代 BBSE/RLLS/MLLS 的点估计模块，提供两种推断引擎：(a) HMC-NUTS：4 条独立链，500 warmup + 1000 后验迭代，自适应 leap-frog 步长；(b) 快速块 Newton-CG：容差 10⁻⁴，每 Newton 步最多 8 次内迭代，相对变化 < 10⁻³ 停止。

**⑤ Saerens 似然校正**（输入：估计的目标先验 q̂；输出：目标域校正后预测）。沿用经典后处理，根据贝叶斯规则调整分类器输出以匹配新先验。

整体流程可概括为：
```
源数据 → [训练 h(x)] → 源分类器
           ↓
验证集 → [C̃ 估计] ──┐
                    ├──→ [贝叶斯推断: p(q,C | C̃, ĥ(x), L)] → q̂ → [Saerens 校正] → 目标预测
类别名/特征 → [k-NN 图] → L ──┘
```

## 核心模块与公式推导

### 模块 1: 图拉普拉斯先验构造（对应框架图 ③→④）

**直觉**: 语义相似的类别应当具有相似的混淆模式和先验概率，图拉普拉斯是编码这种平滑约束的自然数学工具。

**Baseline 公式 (BBSE)**: 无显式图结构；各类别独立处理。

**本文公式（推导）**:
$$L = D - W \quad \text{(图拉普拉斯：度矩阵减权重矩阵)}$$
$$W_{ij} = \exp\left(-\frac{\|e_i - e_j\|^2}{2\sigma^2}\right) \quad \text{(RBF 边权重，} \sigma = \text{median pairwise distance)}$$
$$E = \{(i,j) \text{mid} e_j \in \text{k-NN}(e_i)\} \quad \text{(k-NN 边集，保证 } \lambda_2(L) > 0\text{)}$$

符号: $e_i \in \mathbb{R}^{512}$ 为 CLIP 文本嵌入（MNIST 时为 128-d 倒数第二层特征平均），$D_{ii} = \sum_j W_{ij}$ 为度矩阵，$\lambda_2(L)$ 为代数连通性（Fiedler 值），决定图平滑强度下界。

### 模块 2: 拉普拉斯-高斯层次先验（对应框架图 ④ 核心）

**直觉**: 将目标先验 q 和混淆矩阵列 C_{:,i} 的 logit 变换后变量赋予高斯先验，但用图拉普拉斯替代标准精度矩阵，使相邻类别在单纯形上平滑耦合。

**Baseline 公式 (BBSE/独立先验)**: 
$$\hat{q} = \tilde{C}^{-T} \cdot \frac{1}{n'}\sum_i \hat{h}(x_i) \quad \text{(无先验的点估计)}$$
或独立 Dirichlet/Normal 先验：各类别精度矩阵为对角阵，无跨类别耦合。

**变化点**: 独立先验无法解决高 K 场景下 per-column 样本不足问题；图拉普拉斯精度矩阵 τ·L 将语义相似类别的后验"绑定"在一起，使数据稀缺的类别能从邻居"借调"统计强度。

**本文公式（推导）**:
$$\text{Step 1 (超先验)}: \tau_q \sim \text{Gamma}(1,1), \quad \tau_C \sim \text{Gamma}(1,1) \quad \text{(Gamma(1,1) 为弱信息先验)}$$
$$\text{Step 2 (对数先验的图耦合)}: \log q \sim \mathcal{N}(0, (\tau_q L)^\text{dagger}) \quad \text{(伪逆处理 L 的零空间)}$$
$$\text{Step 3 (混淆矩阵列的图耦合)}: C_{:,i} \sim \text{Logistic-Normal}(0, (\tau_C L)^\text{dagger}) \quad \text{(单纯形上的 logistic-Normal，} L\neq 0 \text{ 时获图耦合精度)}$$
$$\text{Step 4 (似然)}: \hat{h}(x) \text{mid} q, C \sim \text{Multinomial}(C^T q) \text{ 或等价形式}$$
$$\text{最终 (联合后验)}: p(q, C, \tau_q, \tau_C \text{mid} \{\hat{h}(x_j)\}, \tilde{C}) \propto p(\{\hat{h}(x_j)\} \text{mid} q, C) \cdot p(\tilde{C} \text{mid} C) \cdot p(\log q \text{mid} \tau_q, L) \cdot p(\{C_{:,i}\} \text{mid} \tau_C, L) \cdot p(\tau_q) p(\tau_C)$$

**信息几何相容性关键**: $P_T L = L$，其中 $P_T$ 为单纯形切空间投影算子。因 $\mathbf{1}$ 是 L 的零特征向量，切空间投影保持 L 不变，确保图约束与 Fisher-Rao 度量相容。

**退化验证（理论连续性）**:
$$\text{极限情况}: L = 0, \tau_q, \tau_C \to \infty \text{ (delta 质量)} \Rightarrow \text{后验众数} = \text{确定性 BBSE 解}$$

### 模块 3: 后验推断算法（对应框架图 ④ 内部）

**直觉**: 完整后验无闭式解，需兼顾采样精确性与优化速度，提供双引擎满足不同场景需求。

**Baseline 公式 (BBSE)**: 闭式矩阵求逆 $O(K^3)$。

**本文公式（推导）**:
$$\text{HMC-NUTS}: \text{哈密顿动力学模拟，NUTS 自动终止轨迹，避免手动调参}$$
$$\text{配置}: 4 \text{ 链} \times (500 \text{ warmup} + 1000 \text{ 后验}) \text{，自适应 leap-frog 步长}$$
$$\text{Block Newton-CG}: \theta^{(t+1)} = \theta^{(t)} - [\nabla^2 \log p(\theta)]^{-1} \nabla \log p(\theta) \text{，CG 内迭代解线性系统}$$
$$\text{配置}: \text{容差 } 10^{-4}, \max 8 \text{ 内迭代/Newton 步，停止准则 } \|\Delta\theta\|/\|\theta\| < 10^{-3}$$

**对应消融**: 原文未提供显式消融表对比 HMC vs Newton-CG 的精度-速度权衡，但指出 Newton-CG 作为快速替代方案在实验中后验 mode 估计与 HMC 一致。

## 实验与分析


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/87d9d665-052d-402b-a110-2cededd9187e/figures/Table_2.png)
*Table 2 (comparison): Baseline methods and their key ideas*




![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/87d9d665-052d-402b-a110-2cededd9187e/figures/Table_3.png)
*Table 3 (result): Label corruption and downstream performance*



本文在三个标准视觉基准上评估标签偏移估计与下游分类性能：MNIST (K=10)、CIFAR-10 (K=10)、CIFAR-100 (K=100)。所有实验采用合成标签偏移：从 Dirichlet 分布采样目标先验 q，保持分类器黑盒（仅使用输出预测，不重新训练）。评估指标为 L1 prior error ‖q̂ - q‖₁ 和经 Saerens 校正后的目标域分类准确率。

**核心数值**：Table 3 显示，在 MNIST 上 GS-B3SE 的 L1 prior error 为 0.002（bootstrap 标准误），相比最优 baseline MLLS 的 0.010 降低 5 倍；下游准确率 0.986 相比 MLLS 的 0.963 提升 +2.3 个百分点。CIFAR-10 上 L1 error 0.025 vs MLLS 0.052（2× 提升），准确率 0.844 vs 0.812（+3.2 pp）。最具挑战性的 CIFAR-100 (K=100) 上，GS-B3SE 取得 L1 error 0.22，较 MLLS 的 0.71 降低 3.2 倍（绝对减少 0.49），准确率 0.783 较 MLLS 0.734 提升 +4.9 个百分点。这一增益在高类别数场景尤为显著，验证了图平滑先验对 per-class 数据稀缺问题的缓解作用。

**方法对比视角**：Table 2 展示了 GS-B3SE 与四类 baseline 的核心思想差异。BBSE 直接矩阵求逆，RLLS 添加 ℓ₂ 正则，MLLS 似然最大化，Saerens EM 迭代重加权；GS-B3SE 是唯一采用完全概率建模且引入图结构先验的方法。

**公平性检验**：实验设计保持对称信息访问——所有方法共享相同的 C̃ 和 ĥ(x)。但 GS-B3SE 额外使用 CLIP 构建的标签相似图，这是其核心贡献却也造成信息不对称：baseline 无法利用类别语义关系。作者未显式消融"纯贝叶斯（无图）"vs"图平滑+点估计"vs"完整模型"，使得 5×/2×/3.2× 的提升中，贝叶斯框架与图先验各自的贡献难以分离。此外，实验仅使用 ResNet-18 单一骨干，未测试更强特征提取器（如 CLIP 视觉编码器本身）是否会缩小 gap；且合成标签偏移可能未涵盖真实世界中 P(X|Y) 同时漂移的复杂场景。计算方面，HMC 的 4×1500 迭代显著慢于闭式求逆，但块 Newton-CG 作为快速替代缓解了此问题。GPU 使用 NVIDIA T4。

## 方法谱系与知识库定位

**方法家族**: 标签偏移估计 / 黑盒域自适应 / 贝叶斯层次模型

**父方法**: BBSE (Black-Box Shift Estimator) [18] —— GS-B3SE 在 L=0 且超先验退化的极限下精确恢复 BBSE，构成理论连续性证明。

**改变的插槽**:
- **objective**: 点估计 → 贝叶斯后验推断（图结构拉普拉斯-高斯先验）
- **architecture**: 无显式模型 → Gamma 超先验 + 层次 Laplacian-Gaussian 结构
- **inference_strategy**: 闭式矩阵求逆 → HMC-NUTS 采样 / 块 Newton-CG 优化
- **data_pipeline**: 仅经验混淆矩阵 → 额外构建 CLIP/k-NN 标签相似图

**直接 baseline 与差异**:
- **BBSE** [18]: 确定性矩阵求逆；GS-B3SE 将其扩展为贝叶斯层次模型，退化极限相容
- **RLLS** [7]: ℓ₂ 正则凸优化；GS-B3SE 用图结构非对角精度替代对角正则
- **MLLS** [18]: 似然最大化点估计；GS-B3SE 提供完整后验与不确定性量化
- **Saerens EM** [47]: 迭代重加权；GS-B3SE 一步概率推断，无需 EM 迭代

**后续方向**:
1. **图构建自动化**: 当前 k-NN 图依赖 CLIP 嵌入质量，探索自适应图学习或缺失图场景下的鲁棒推断
2. **P(X|Y) 漂移扩展**: 放松类别条件分布稳定假设，向广义协变量漂移+标签漂移联合建模延伸（类似作者前期工作 [28] 的方向）
3. **大规模 K 优化**: K→1000+ 时拉普拉斯矩阵稀疏性与推断可扩展性的进一步优化

**标签**: #modality:image #modality:text #paradigm:bayesian #paradigm:graph_learning #scenario:domain_adaptation #scenario:label_shift #mechanism:laplacian_prior #mechanism:hmc #mechanism:information_geometry #constraint:black_box

