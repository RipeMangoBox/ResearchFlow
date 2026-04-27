---
title: Condensing Action Segmentation Datasets via Generative Network Inversion
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- 生成网络反演实现动作分割数据集凝练
- CAS (Condensing
- CAS (Condensing Action Segmentation datasets via generative network inversion)
acceptance: poster
cited_by: 3
code_url: https://www.comp.nus.edu.sg/~dinggd/projects/cas/cas.html
method: CAS (Condensing Action Segmentation datasets via generative network inversion)
---

# Condensing Action Segmentation Datasets via Generative Network Inversion

[Code](https://www.comp.nus.edu.sg/~dinggd/projects/cas/cas.html)

**Topics**: [[T__Segmentation]], [[T__Continual_Learning]] (其他: Dataset Distillation) | **Method**: [[M__CAS]] | **Datasets**: [[D__50Salads]], [[D__Breakfast]] (其他: GTEA)

| 中文题名 | 生成网络反演实现动作分割数据集凝练 |
| 英文题名 | Condensing Action Segmentation Datasets via Generative Network Inversion |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2503.14112) · [Code](https://www.comp.nus.edu.sg/~dinggd/projects/cas/cas.html) · [DOI](https://doi.org/10.1109/CVPR52734.2025.01652) |
| 主要任务 | 时序动作分割 (Temporal Action Segmentation, TAS) 的数据集凝练 (Dataset Condensation) |
| 主要 baseline | Mean, Coreset [38], Encoded (TCA reconstruction), TCA [11], MSTCN [12], ASFormer [39] |

> [!abstract] 因为「时序动作分割数据集存储开销巨大（245MB-28GB）且缺乏专门的凝练方法」，作者在「TCA [11] 生成模型」基础上改了「生成网络反演优化隐变量 + 多样性序列采样」，在「GTEA / 50Salads / Breakfast」上取得「以 0.09%-1.5% 存储达到原始数据 82.6%-95.2% 性能」

- **GTEA**: Acc 75.2%，仅用 1.5% 存储达到原始数据 95.2% 的准确率（原始 79.0%）
- **50Salads**: Acc 74.4%，仅用 0.09% 存储达到原始数据 92.3% 的准确率（原始 80.6%）
- **Breakfast**: Acc 55.5%，仅用 0.16% 存储达到原始数据 82.6% 的准确率（原始 67.2%）

## 背景与动机

时序动作分割（Temporal Action Segmentation, TAS）需要将未修剪的长视频逐帧标注为动作类别，是视频理解的核心任务。然而，TAS 数据集（如 Breakfast、50Salads、GTEA）通常包含大量高维 I3D 特征（2048 维/帧），原始存储可达数 GB，在边缘设备部署、增量学习或隐私敏感场景下难以承受。

现有方法如何处理这一存储问题？**Mean baseline** 直接存储每段的平均特征，简单但丢失时序动态；**Coreset [38]** 用 Herding 选择最接近均值的代表帧，仍无法生成连续时序特征；**TCA [11]** 作为生成模型可编码-解码特征，但仅做重建而不优化隐空间表示，压缩率与质量均受限。图像领域的数据集凝练方法（如 DM [21]、DSA [26]、MTT [18]）依赖梯度匹配或轨迹匹配，尚未扩展到视频时序数据。

这些方法的共同短板在于：**无法将长视频段压缩为可优化的紧凑隐表示，同时保留生成连续时序特征的能力**。Mean/Coreset 丢弃了生成能力；TCA 虽有生成模型但未针对凝练优化隐变量；图像凝练方法无法处理时序结构的特殊性。因此，TAS 领域亟需一种专门的数据集凝练范式——这正是 CAS 的出发点：用生成网络反演将段级特征压缩为隐变量，以极小的存储开销重建完整训练数据。

## 核心创新

核心洞察：**动作段的可压缩性不在于单帧特征本身，而在于段内特征沿时间轴的连续变化规律可被低维隐变量捕获**，因为 TCA 解码器已学习到动作类别与时间坐标的条件生成分布，从而使通过优化隐变量 z* 来重建整条段特征成为可能。

| 维度 | Baseline (TCA / Mean / Coreset) | 本文 (CAS) |
|:---|:---|:---|
| **压缩表示** | 存储原始特征 / 均值向量 / 单帧索引 | 优化后的隐变量 z*（256-d） |
| **序列选择** | 随机采样或全量使用 | 基于编辑距离的多样性贪心采样 |
| **特征生成** | 无生成能力 / 重复均值 / 单帧复制 | 解码器 D(z*, a, c) 生成变长连续特征 |
| **优化目标** | 无 / 无 / 无 | 网络反演：最小化重构损失 L_inv (Eq. 8) |
| **存储比例** | 100% / ~0.01% / ~0.01% | 0.09% - 1.5% |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b682f9fe-ff2c-4b17-adce-7917136b92bb/figures/fig_001.png)
*Figure: Comparison of action segmentation performance with*



CAS 采用**三阶段流水线**，数据流如下：

1. **TCA 生成模型训练**（输入：原始 I3D 特征 x、动作标签 a、时间坐标 c；输出：训练好的编码器 q_φ(z|x,a,c) 与解码器 p_θ(x|z,a,c)）。在原始数据上训练 7.5K epochs，学习动作-时间条件化的特征生成分布。

2. **多样性序列采样 + 网络反演**（输入：原始数据集 D 的 N 条序列；输出：采样子集 S 及每段的优化隐变量 z*）。先用贪心策略基于归一化编辑距离选择多样序列，再对每段执行 Eq. (8) 优化 10K  iterations。

3. **合成数据生成与 TAS 训练**（输入：z*、a、c；输出：解码特征 x̂* 及训练好的 MSTCN/ASFormer）。解码器重建合成特征，替代原始特征训练下游分割模型。

```
原始视频 → I3D提取 → [x, a, c]
                ↓
         ┌─────────────┐
         │ TCA训练阶段  │ → 编码器 q_φ + 解码器 p_θ
         └─────────────┘
                ↓
    [多样性采样] → 选序列 S（最大化编辑距离）
                ↓
    [网络反演] → 优化 z* = argmin ||D(z,a,c) - x||²
                ↓
         存储 {z*, a, c}（~0.1-1.5% 原始大小）
                ↓
    [解码生成] → x̂* = D(z*, a, c)
                ↓
         ┌─────────────┐
         │ TAS骨干训练  │ → MSTCN / ASFormer
         └─────────────┘
```

## 核心模块与公式推导

### 模块 1: 时间坐标条件化（对应框架图"解码器生成"环节）

**直觉**: 动作段长度各异，需让解码器知道"当前生成的是段内哪个位置"，才能输出连贯的时序特征。

**Baseline 公式**: TCA [11] 原始实现未显式使用归一化时间坐标，仅依赖隐变量与动作标签生成特征，难以精确控制段内位置。

**本文公式**:
$$c_i = \frac{i-1}{\text{ell}-1}, \quad c_i \in [0,1]$$
符号: $i$ = 段内帧索引, $\text{ell}$ = 段长度, $c_i$ = 归一化时间坐标。将帧位置线性映射到单位区间，使解码器获得显式的时间先验。

**条件生成**:
$$\hat{x}_i = p_\theta(x \text{mid} z, a, c_i), \quad i \in [1, ..., \text{ell}]$$
通过遍历 $i$ 即可生成完整段的变长特征序列。

---

### 模块 2: 生成网络反演（对应框架图"网络反演"核心环节）

**直觉**: 直接存储高维特征浪费空间；若解码器足够表达，只需存储"指向正确输出的隐变量指针"。

**Baseline 公式**:
- Mean: $\bar{x} = \frac{1}{\text{ell}}\sum_{i=1}^{\text{ell}} x_i$（丢失时序结构）
- Coreset: $x^* = \argmin_{x \in \text{segment}} ||x - \bar{x}||$（仅保留单帧）
- TCA Encoded: 编码器输出 $z = q_\phi(x,a,c)$ 但不优化，重建质量受限

**变化点**: Baseline 要么不生成（Mean/Coreset），要么只做前向编码不优化（Encoded）。本文将网络反演引入 TAS：固定训练好的解码器 D，以 L2 重构为目标优化隐变量。

**本文公式（推导）**:
$$\text{Step 1}: \quad z^* = \argmin_{z \in \mathbb{R}^d} \underbrace{||\mathbf{D}(\mathbf{z}, \mathbf{a}, \mathbf{c}) - \mathbf{x}||_2^2}_{\mathcal{L}_{\text{inv}}} \quad \text{（加入重构项以精确恢复原始特征）}$$
$$\text{Step 2}: \quad \hat{\mathbf{x}}^* = \mathbf{D}(\mathbf{z}^*, \mathbf{a}, \mathbf{c}) \quad \text{（用优化后的 } z^* \text{ 重建合成数据，保证训练可用性）}$$
符号: $\mathbf{x} \in \mathbb{R}^{\text{ell} \times D}$ = 原始段特征 ($D=2048$), $\mathbf{z} \in \mathbb{R}^d$ = 隐变量 ($d=256$), $\mathbf{a}$ = 动作标签, $\mathbf{c}$ = 时间坐标向量, $\mathbf{D}$ = 训练好的 TCA 解码器。

**对应消融**: 去掉网络反演（改用 Encoded 前向编码）后，GTEA Acc 从 75.2% 降至 70.4%（Δ -4.8%），50Salads 从 74.4% 降至 69.0%（Δ -5.4%），Breakfast 从 55.5% 降至 37.9%（Δ -17.6%）。

---

### 模块 3: 多样性序列采样（对应框架图"多样性采样"环节）

**直觉**: 预算有限时，随机采样可能选中高度相似的序列，浪费存储配额；需显式最大化采样集合的多样性。

**Baseline 公式**: 随机采样 $P(s_i) = 1/|\mathcal{D}|$，无多样性保证。

**变化点**: 引入归一化编辑距离衡量序列差异，贪心选择最远序列。

**本文公式**:
$$\text{Edit}(\mathbf{s}_i, \mathbf{s}_j) = \frac{e[|\mathbf{s}_i|, |\mathbf{s}_j|]}{\max(|\mathbf{s}_i|, |\mathbf{s}_j|)}$$
$$\mathbf{s}^* = \argmax_{\mathbf{s}_i \in \mathcal{D} \text{setminus} \mathcal{S}} \min_{\mathbf{s}_j \in \mathcal{S}} \text{Edit}(\mathbf{s}_i, \mathbf{s}_j)$$
其中 $e[m,n]$ 为 Levenshtein 编辑距离，分子为原始编辑距离，分母归一化到最长序列长度。

**对应消融**: 将多样性采样替换为随机采样，50Salads Acc 从 74.4% 降至 71.9%（Δ -2.5%）；采样比例 $\gamma$ 从 0.3 提升到 0.4 时，GTEA Acc 提升 +16.7%，Edit 提升 +20.6%。

## 实验与分析



本文在 GTEA、50Salads、Breakfast 三个标准 TAS 数据集上评估 CAS，使用 MSTCN [12] 和 ASFormer [39] 作为分割骨干。核心结果如 Table 1 所示：在 GTEA 上，CAS 达到 Acc 75.2% / Edit 71.9% / F1@10 78.3%，以仅 1.5% 的存储达到原始数据 95.2% 的准确率（原始 Acc 79.0%）；在 50Salads 上，Acc 74.4% 达到原始的 92.3%，存储仅 0.09%；在最大的 Breakfast 上，Acc 55.5% 达到原始的 82.6%，存储 0.16%。相比最直接的竞争对手 Mean（存储相近），CAS 在三个数据集上分别提升 Acc +3.9%、+5.4%、+7.9%；相比 Encoded baseline（TCA 前向编码无优化），提升 +4.8%、+5.4%、+17.6%。



消融实验（Table 3/4 及相关图表）揭示了关键组件的贡献。网络反演优化是最大增益来源：去掉它（改用 Encoded）在 Breakfast 上造成 -17.6% 的灾难性下降。多样性采样同样关键：随机采样导致 50Salads Acc -2.5%。每段实例数 K 的影响呈现明显边际递减：Breakfast 上 Edit 从 K=1 的 32.2% 跃升至 K=8 的 52.3%（Δ +20.1%），但 K=16 与 K=8 差异微小，暗示 TCA 解码器的表达能力是瓶颈而非凝练粒度。值得注意的是，作者尝试了每帧独立优化（K=†，每帧一个隐变量），发现与 K=16 段级优化几乎持平，进一步证实生成模型容量限制。



增量学习扩展（Table 5）显示 CAS 在 Breakfast 10-task 增量场景下的适用性：MSTCN 上平均指标 31.0，优于 TCA baseline 的 28.3（+2.7）但低于原始数据的 38.2（-7.2）；ASFormer 上 39.7 vs TCA 30.5（+9.2）vs 原始 44.6（-4.9），更强骨干从凝练数据中获益更多。

公平性检查：本文是首个 TAS 数据集凝练工作，自设了合理的 Mean、Coreset、Encoded baseline，但未与图像领域更强的梯度匹配方法（DM、DSA、MTT）或近期 SRe2L、G-VBSM 对比——这些方法适配到视频时序数据的效果未知。作者也坦承 TCA 作为生成模型可能非最优（K=16 即饱和），且增量学习结果与原始数据差距较大，实际部署价值待验证。存储比较仅含特征/隐变量，未计入解码器模型本身。

## 方法谱系与知识库定位

CAS 属于**数据集凝练 (Dataset Condensation/Distillation)** 方法族，直接继承自 **TCA [11]**（Temporal Context Aggregation）——借用其两层 MLP 编码器-解码器作为生成骨干，但将用途从"特征重建"转向"数据压缩"。

**改动槽位**: data_pipeline（网络反演替代直接存储）、inference_strategy（解码器生成替代特征加载）、training_recipe（两阶段：生成模型训练 → 隐变量优化 → 合成数据训练）。

**直接 baseline 差异**:
- **TCA [11]**: 仅做 VAE 式编码-解码，无隐变量优化，不用于凝练
- **Mean/Coreset**: 非生成式，无法恢复时序动态，存储效率相近但质量远逊
- **Encoded**: TCA 前向编码，无反演优化，重建误差累积

**后续方向**: (1) 用更强生成模型（扩散模型、流匹配）替代 TCA 以突破 K=16 瓶颈；(2) 将梯度匹配/轨迹匹配从图像领域适配到视频时序数据；(3) 探索端到端联合优化（生成模型 + 反演 + TAS 训练）而非三阶段分离。

**标签**: 模态=视频时序特征 | 范式=数据集凝练 + 生成网络反演 | 场景=时序动作分割、增量学习、边缘部署 | 机制=隐空间优化、多样性采样、条件生成 | 约束=极低存储预算、保持下游任务性能

