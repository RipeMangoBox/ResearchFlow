---
title: 'AZ-NAS: Assembling Zero-Cost Proxies for Network Architecture Search'
type: paper
paper_level: C
venue: CVPR
year: 2024
paper_link: null
aliases:
- 组装零成本代理的高效NAS方法
- AZ-NAS
acceptance: Poster
cited_by: 38
code_url: https://github.com/lewbei/Awesome-Neural-Architecture-Search
method: AZ-NAS
followups:
- 基于Fisher信息矩阵特征值熵_VKDNW_(Variance_
- 基于Fisher谱熵的无训练神经_VKDNW_(Variance_
---

# AZ-NAS: Assembling Zero-Cost Proxies for Network Architecture Search

[Code](https://github.com/lewbei/Awesome-Neural-Architecture-Search)

**Topics**: [[T__Neural_Architecture_Search]] | **Method**: [[M__AZ-NAS]] | **Datasets**: [[D__NAS-Bench-201]], [[D__ImageNet-1K]]

| 中文题名 | 组装零成本代理的高效NAS方法 |
| 英文题名 | AZ-NAS: Assembling Zero-Cost Proxies for Network Architecture Search |
| 会议/期刊 | CVPR 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2403.19232) · [Code](https://github.com/lewbei/Awesome-Neural-Architecture-Search) · [Project] |
| 主要任务 | Neural Architecture Search (NAS), 图像分类架构搜索 |
| 主要 baseline | AutoFormer, TF-TAS, TE-NAS, NASWOT, Zen-NAS |

> [!abstract] 因为「单一零成本代理（如仅激活或仅梯度）无法全面评估网络质量」，作者在「TE-NAS 等训练自由NAS」基础上改了「将激活、梯度、FLOPs 三种代理组装融合，并在单次前向-反向传播中同步计算」，在「NAS-Bench-201 / ImageNet (MobileNetV2 & AutoFormer)」上取得「Kendall's τ 与 Spearman's ρ 均为最优，AutoFormer-Tiny 精度 76.1% (+1.4 vs AutoFormer)」

- **搜索成本**: 0.03–0.17 GPU days，相比 AutoFormer 的 24 GPU days 降低 **141–800×**，相比 TF-TAS 的 0.5 GPU days 降低 **2.9–16.7×**
- **AutoFormer-Tiny 精度**: 76.1% Top-1，超越 AutoFormer 74.7% (+1.4) 与 TF-TAS 75.3% (+0.8)
- **NAS-Bench-201 排名一致性**: Kendall's τ 与 Spearman's ρ 在所有数据集上均为最佳，显著优于 TE-NAS、NASWOT、Zen-NAS

## 背景与动机

神经架构搜索（NAS）旨在自动发现最优神经网络结构，但传统方法需要完整训练每个候选架构，计算成本极高。以 AutoFormer 为例，在 ViT 搜索空间上需要 **24 GPU days** 的搜索时间，严重限制了其实用性。为降低成本，**训练自由 NAS（training-free NAS）** 应运而生——通过零成本代理（zero-cost proxy）在随机初始化网络上快速评估架构质量，无需训练即可排名。

现有训练自由方法主要从单一视角评估网络：
- **NASWOT** 仅利用激活信息，需移除 Batch Normalization 等特殊处理；
- **Zen-NAS** 同样基于激活，依赖特定非线性分析；
- **TE-NAS** 采用梯度信息，但需要**多次前向-反向传播**，计算开销仍然较大。

这些方法的核心局限在于：**单一代理无法全面刻画网络质量**。激活类方法忽略梯度流信息，梯度类方法忽略计算效率，且各自需要特殊的网络修改或多轮计算。更关键的是，不同代理在不同架构类型（CNN vs. ViT）上的有效性差异显著，导致泛化能力不足。例如，TE-NAS 的多轮梯度计算在 Transformer 上效率低下，而 NASWOT 的 BN 移除技巧对特定结构有偏。

本文提出 **AZ-NAS**，核心思想是：**从多个互补视角（激活、梯度、FLOPs）同时评估架构，并在单次前向-反向传播中完成所有计算**，实现更全面、更高效的训练自由架构搜索。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a2cbe5e7-c51d-4916-aeff-ab8fb922f968/figures/Figure_1.png)
*Figure 1 (comparison): Comparison of training-free NAS methods on ImageNet. We compare the types of search space in NAS methods (cell-based, chain-based, and macro) and search cost. By assembling our proposed zero-cost proxies, our method (AZ-NAS) achieves the best trade-off between accuracy and search cost.*



## 核心创新

核心洞察：**单一零成本代理如同"盲人摸象"，仅捕捉网络某一局部特性；将激活、梯度、FLOPs 三种代理组装融合，可从多视角互补地评估架构质量，从而在单次前向-反向传播中实现更准确、更通用的排名**。

| 维度 | Baseline (TE-NAS / NASWOT / Zen-NAS) | 本文 (AZ-NAS) |
|:---|:---|:---|
| **评估视角** | 单一代理（仅激活 或 仅梯度） | 多代理组装（激活 + 梯度 + FLOPs） |
| **计算方式** | 多次前向-反向传播 或 特殊网络修改（如移除 BN） | **单次**前向-反向传播同步计算 |
| **架构泛化** | 针对 CNN 或特定结构优化 | CNN (MobileNetV2) 与 ViT (AutoFormer) 通用 |
| **搜索成本** | 0.5 GPU days (TF-TAS) 或更高 | **0.03–0.17 GPU days** |

## 整体框架



AZ-NAS 的完整流程包含四个阶段，数据流如下：

1. **架构采样（Architecture Sampling）**：从搜索空间（MobileNetV2 或 AutoFormer）中生成候选架构集合；
2. **零成本代理计算（Zero-Cost Proxy Computation）**：对每个候选架构执行**单次前向-反向传播**，同步提取三类指标：
   - **激活代理** $\phi_{\text{act}}$：衡量特征空间的丰富度与多样性；
   - **梯度代理** $\phi_{\text{grad}}$：捕捉梯度流的健康程度与训练潜力；
   - **FLOPs 代理** $\phi_{\text{flops}}$：编码计算复杂度约束；
   - **渐进性代理** $\phi_{\text{prog}}$（CNN 专用）：度量特征空间随深度的扩展性；
3. **代理组装聚合（Proxy Assembly）**：通过函数 $f$ 将多代理分数融合为综合评分；
4. **架构选择与输出（Architecture Selection）**：按综合评分排序，选取 Top-K 架构进行最终评估。

```
Search Space (MobileNetV2 / AutoFormer)
    ↓
[Architecture Sampling] → Candidate Architectures
    ↓
[Single Forward-Backward Pass]
    ├── Activation Proxy  (φ_act)
    ├── Gradient Proxy    (φ_grad)
    ├── FLOPs Proxy       (φ_flops)
    └── Progressivity Proxy (φ_prog, CNN only)
    ↓
[Proxy Assembly: f(φ_act, φ_grad, φ_flops)]
    ↓
[Architecture Ranking & Selection]
    ↓
Selected Top Architecture
```

关键设计：所有代理在**单次前向-反向传播中同步计算**，无需像 TE-NAS 那样迭代多轮，也无需像 NASWOT 那样修改网络结构（如移除 BN）。

## 核心模块与公式推导

### 模块 1: 组装函数（对应框架图「Proxy Assembly」层）

**直觉**: 单一代理只能从特定视角评估网络，组合多代理可获得更稳健的架构排名。

**Baseline 公式** (NASWOT / Zen-NAS / TE-NAS):
$$\text{Single-Proxy}(\mathcal{A}) = \phi_{\text{single}}(\mathcal{A})$$
符号: $\mathcal{A}$ = 候选架构, $\phi_{\text{single}}$ = 单一代理函数（如仅激活或仅梯度）

**变化点**: 单一代理在特定搜索空间可能失效（如激活代理对 ViT 注意力机制不敏感）；且不同代理捕捉不同特性——激活反映表达能力，梯度反映可训练性，FLOPs 反映效率。

**本文公式（推导）**:
$$\text{Step 1}: \quad \phi_{\text{act}}(\mathcal{A}) = \text{measure}(\text{activations at random init}) \quad \text{（特征空间多样性）}$$
$$\text{Step 2}: \quad \phi_{\text{grad}}(\mathcal{A}) = \text{measure}(\text{gradients from single backward}) \quad \text{（梯度流健康度）}$$
$$\text{Step 3}: \quad \phi_{\text{flops}}(\mathcal{A}) = \text{compute}(\text{FLOPs of } \mathcal{A}) \quad \text{（计算效率约束）}$$
$$\text{最终}: \quad \text{AZ-NAS}(\mathcal{A}) = f\left(\phi_{\text{act}}(\mathcal{A}), \phi_{\text{grad}}(\mathcal{A}), \phi_{\text{flops}}(\mathcal{A})\right)$$

其中 $f$ 为组装函数，具体实现为各代理的加权组合或归一化聚合。关键优势：三项在**单次前向-反向传播中同步获得**，无需额外计算。

**对应消融**: Table 5 显示将 AZ-NAS 的零成本代理整合到其他方法中的效果，验证组装策略的通用增益。

---

### 模块 2: 渐进性代理（对应框架图可选组件，CNN 专用）

**直觉**: 优质网络应随深度增加逐步扩展特征空间，而非过早饱和或崩溃。

**Baseline**: 现有方法（如 NASWOT）仅关注单层激活统计，忽略深度方向的特征演化。

**变化点**: 引入深度维度的特征空间扩展度量；但发现该代理对 ViT **不适用**——因为随机输入下注意力模块在各 token 上产生相似注意力值，导致渐进性信号失效。

**本文公式**:
$$\text{Progressivity}(\mathcal{A}) = \text{corr}\left(\{d_1, d_2, ..., d_L\}, \{\dim(\mathbf{h}_1), \dim(\mathbf{h}_2), ..., \dim(\mathbf{h}_L)\}\right)$$
符号: $d_l$ = 第 $l$ 层深度, $\mathbf{h}_l$ = 第 $l$ 层特征表示, $\dim(\cdot)$ = 有效特征维度（通过主成分分析或相关矩阵特征值估计）

**设计细节**: 
- 对 CNN：计算各层特征表示的相关矩阵特征值分布，度量有效维度随深度的增长相关性（见 Figure 2 的 1D/2D 示例）；
- 对 ViT：显式**禁用**此代理（脚注说明），避免随机输入下注意力机制的退化行为误导评估。

**对应消融**: Table 5 及 NAS-Bench-201 实验中的消融显示，移除渐进性代理对 CNN 搜索有负面影响，而保留它（或对 ViT 错误启用）会降低排名一致性。

---

### 模块 3: 同步前向-反向计算机制（对应框架图效率核心）

**直觉**: 多代理无需多轮计算——梯度反向传播时自然同时产生激活与梯度信息。

**Baseline 公式** (TE-NAS):
$$\text{TE-NAS}(\mathcal{A}) = \phi_{\text{grad}}^{(1)}(\mathcal{A}) \circ \phi_{\text{grad}}^{(2)}(\mathcal{A}) \circ ... \circ \phi_{\text{grad}}^{(K)}(\mathcal{A})$$
需要 $K$ 次独立的前向-反向传播，计算成本随代理数量线性增长。

**变化点**: 利用单次前向-反向传播中**激活值（前向缓存）与梯度值（反向传播）同时可得**的特性，将多代理计算压缩到一轮。

**本文公式**:
$$\text{Step 1}: \quad \text{Forward}(\mathbf{x}; \mathcal{A}) \rightarrow \text{cache } \{\mathbf{a}^{(l)}\}_{l=1}^{L} \quad \text{（缓存各层激活）}$$
$$\text{Step 2}: \quad \text{Backward}(\mathcal{L}; \mathcal{A}) \rightarrow \{\mathbf{g}^{(l)}\}_{l=1}^{L} \quad \text{（计算各层梯度）}$$
$$\text{Step 3}: \quad \phi_{\text{act}} = f_{\text{act}}(\{\mathbf{a}^{(l)}\}), \quad \phi_{\text{grad}} = f_{\text{grad}}(\{\mathbf{g}^{(l)}\}), \quad \phi_{\text{flops}} = f_{\text{flops}}(\mathcal{A})$$
$$\text{最终}: \quad \text{All proxies computed in } \mathcal{O}(1) \text{ forward-backward passes}$$

**效率增益**: 从 TE-NAS 的多轮迭代降至 **1 轮**，搜索成本从 TE-NAS 的较高开销降至 **0.03–0.17 GPU days**。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a2cbe5e7-c51d-4916-aeff-ab8fb922f968/figures/Table_1.png)
*Table 1 (comparison): Quantitative comparison of the training-free NAS methods on NAS-Bench-201.*



本文在三个层面验证 AZ-NAS：**排名一致性**（NAS-Bench-201）、**CNN 搜索质量**（ImageNet-MobileNetV2）、**ViT 搜索质量**（ImageNet-AutoFormer）。

**NAS-Bench-201 排名一致性**：Table 1 显示，AZ-NAS 在 CIFAR-10、CIFAR-100、ImageNet-16-120 三个数据集上的 Kendall's τ 与 Spearman's ρ 均为所有训练自由方法中最高，显著优于 TE-NAS、NASWOT、Zen-NAS。这表明**多代理组装在架构排名任务上具有系统性优势**——单一代理方法（无论激活还是梯度）均存在盲角，而 AZ-NAS 的多视角互补有效覆盖了这些盲区。

**ImageNet 实际搜索性能**：Table 2（MobileNetV2 搜索空间）显示，AZ-NAS 在各 FLOPs 约束下均取得最优 Top-1 精度，甚至超越部分训练依赖的方法。Table 3（AutoFormer ViT 搜索空间）展示更细粒度的对比：AZ-NAS-Tiny 达到 **76.1%**，相比 AutoFormer 的 74.7% 提升 **+1.4**，相比 TF-TAS 的 75.3% 提升 **+0.8**；Small 规模 82.0% 略超 AutoFormer 81.7%（+0.3）与 TF-TAS 81.9%（+0.1）。搜索成本方面，AZ-NAS 仅需 **0.03–0.17 GPU days**，相比 AutoFormer 的 24 GPU days 降低两个数量级，相比 TF-TAS 的 0.5 GPU days 仍有 **2.9–16.7×** 优势。


![Table 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a2cbe5e7-c51d-4916-aeff-ab8fb922f968/figures/Table_5.png)
*Table 5 (ablation): Quantitative comparison of incorporating zero-cost proxies.*



**消融实验**：Table 5 验证各代理组件的贡献。将 AZ-NAS 的零成本代理整合到其他方法中可一致提升其性能，证明组装策略的**模块通用性**。Figure 3 的相关矩阵显示各代理间存在适度互补性（非完全相关），支撑了"多视角必要"的核心假设。

**公平性核查**：
- **基线强度**：主要对比 AutoFormer（训练依赖，24 GPU days）与 TF-TAS（训练自由，0.5 GPU days），但未包含 2022 年后的更新零样本 NAS 方法（如基于流匹配的搜索），也未对比 DARTS、ENAS 等经典方法在 CNN 空间的表现。
- **结果稳健性**：Table 3 中 **Base 规模 AZ-NAS 为 82.1%，低于 AutoFormer 82.4% 与 TF-TAS 82.2%**，与作者"多数情况下更优"的声称存在矛盾；且 Table 2、Table 3 未报告标准差（仅注明 Table 2 为三次随机运行平均），单次运行结果的统计显著性未知。
- **架构限制**：渐进性代理对 ViT 显式禁用，限制了"完全通用"的声称；注意力机制在随机输入下的退化行为是经验发现，缺乏理论解释。

## 方法谱系与知识库定位

**方法族**: 训练自由神经架构搜索（Training-Free NAS）→ **TE-NAS 谱系**

**父方法**: **TE-NAS**（Lee et al., 理论启发的训练自由 NAS，需多轮前向-反向传播）。AZ-NAS 继承其"无需训练即可评估架构"的核心思想，但在四个关键 slot 上进行了替换/修改：

| 变更 Slot | TE-NAS / 传统方法 | AZ-NAS |
|:---|:---|:---|
| **Objective** | 单一梯度代理 | 激活+梯度+FLOPs 多代理组装 |
| **Inference Strategy** | 多次前向-反向传播 | **单次**前向-反向同步计算 |
| **Architecture** | 主要针对 CNN | CNN + ViT 通用（渐进性代理自适应启停）|
| **Training Recipe** | 较高计算开销 | **0.03–0.17 GPU days** |

**直接基线差异**：
- **vs. TE-NAS**: 从单代理扩展到多代理组装，从多轮计算压缩到单次传播；
- **vs. NASWOT/Zen-NAS**: 从纯激活视角扩展到激活+梯度+FLOPs 三视角，且无需移除 BN 等特殊修改；
- **vs. TF-TAS**: 同为单次计算，但 AZ-NAS 多代理组装 vs. TF-TAS 单代理，搜索成本更低（0.03–0.17 vs. 0.5 GPU days）；
- **vs. AutoFormer**: 训练自由 vs. 训练依赖，搜索成本降低 **141–800×**。

**后续方向**：
1. **代理动态权重学习**：当前组装函数 $f$ 为固定加权，可探索基于搜索空间特性的自适应权重；
2. **更多架构类型的自适应代理**：针对 Mamba、状态空间模型等新架构设计专用代理，扩展"渐进性代理"的自适应启停机制；
3. **理论解释缺失补齐**：为注意力机制在随机输入下的退化行为建立理论分析，指导更鲁棒的 ViT 代理设计。

**标签**: 模态=视觉 | 范式=训练自由NAS/零样本架构搜索 | 场景=ImageNet分类 | 机制=多代理组装/单次前反向同步计算 | 约束=低计算预算（<0.2 GPU days）

## 引用网络

### 后续工作（建立在本文之上）

- [[P__基于Fisher信息矩阵特征值熵_VKDNW_(Variance_]]: Zero-cost NAS proxy method likely compared against as a baseline approach
- [[P__基于Fisher谱熵的无训练神经_VKDNW_(Variance_]]: Zero-cost proxy assembly method; directly comparable approach in same space

