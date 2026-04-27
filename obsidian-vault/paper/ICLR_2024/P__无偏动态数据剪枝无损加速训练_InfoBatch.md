---
title: 'InfoBatch: Lossless Training Speed Up by Unbiased Dynamic Data Pruning'
type: paper
paper_level: C
venue: ICLR
year: 2024
paper_link: null
aliases:
- 无偏动态数据剪枝无损加速训练
- InfoBatch
acceptance: Oral
cited_by: 94
code_url: https://github.com/NUS-HPC-AI-Lab/InfoBatch
method: InfoBatch
---

# InfoBatch: Lossless Training Speed Up by Unbiased Dynamic Data Pruning

[Code](https://github.com/NUS-HPC-AI-Lab/InfoBatch)

**Topics**: [[T__Self-Supervised_Learning]], [[T__Classification]], [[T__Semantic_Segmentation]] | **Method**: [[M__InfoBatch]] | **Datasets**: [[D__CIFAR-10]], [[D__CIFAR-100]]

| 中文题名 | 无偏动态数据剪枝无损加速训练 |
| 英文题名 | InfoBatch: Lossless Training Speed Up by Unbiased Dynamic Data Pruning |
| 会议/期刊 | ICLR 2024 (Oral) |
| 链接 | [arXiv](https://arxiv.org/abs/2303.04947) · [Code](https://github.com/NUS-HPC-AI-Lab/InfoBatch) · [Project] |
| 主要任务 | 图像分类 (CIFAR-10/100, ImageNet-1K)、语义分割 (ADE20K)、指令微调 (LLaMA-7B on Alpaca) |
| 主要 baseline | UCB (Raju et al., 2021), ε-greedy (Raju et al., 2021), EL2N (Paul et al., 2021), Influence (Koh & Liang, 2017), Random*, Random |

> [!abstract]
> 因为「静态数据剪枝需预计算且无法适应训练动态，动态剪枝方法如UCB/ε-greedy存在排序开销O(logN)且仅能做到near-lossless」，作者在「UCB动态剪枝框架」基础上改了「引入O(1)阈值选择、无偏补偿加权和自适应剪枝率」，在「CIFAR-10/100, ImageNet-1K」上取得「30%剪枝比例下无损性能，50%剪枝下CIFAR-100领先UCB +2.0%」

- **CIFAR-10**: 30%剪枝下准确率 95.6%（无损），50%剪枝 95.3%（+0.8% vs UCB），70%剪枝 93.5%（+1.3% vs UCB）
- **CIFAR-100**: 30%剪枝下准确率 78.2%（无损），50%剪枝 76.5%（+2.0% vs UCB），70%剪枝 72.5%（+1.5% vs UCB）
- **LLaMA-7B指令微调**: DQ+InfoBatch 平均得分 26.3，训练时间从 18.0分钟降至 14.4分钟

## 背景与动机

深度学习模型训练需要遍历海量数据，其中大量样本在训练后期对梯度更新贡献甚微。以ImageNet-1K为例，完整训练需120 epoch × 128万张图片，计算成本高昂。数据剪枝（data pruning）旨在识别并移除"冗余"样本以加速训练，但现有方法存在根本性缺陷。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/7103632e-b202-47c1-a75d-a955866ea840/figures/Figure_1.png)
*Figure 1 (motivation): Figure 1. Illustration of difference between InfoBatch and EL2N. (a) EL2N uses hard pruning and requires retraining. (b) InfoBatch uses soft pruning with cyclic and iterative updating. The dashed line indicates threshold. The cross-heatmap exemplifies the variance range of gradient estimation CIFAR-10.*



**静态剪枝方法**如EL2N (Paul et al., 2021) 和 Influence (Koh & Liang, 2017) 在训练前基于启发式分数（如误差L2范数、影响函数）一次性选择核心子集（coreset）。EL2N需先训练一个参考模型计算样本分数，再按分数排序取top-k；Influence需计算Hessian逆矩阵近似样本重要性。这类方法的问题在于：预计算开销大，且一旦选定子集便固定不变，无法适应训练过程中样本重要性的动态变化——早期困难的样本可能在后期变得简单，反之亦然。

**动态剪枝方法**如UCB和ε-greedy (Raju et al., 2021) 每轮根据当前损失动态选择样本，缓解了静态方法的僵化问题。但UCB需维护每个样本的损失历史并计算置信上界，然后**排序**选择（O(logN)每样本）；ε-greedy以概率ε随机探索，其余按损失排序选择。两者均依赖排序操作，引入显著开销；更关键的是，它们仅能做到"near-lossless"（近似无损），即剪枝后性能仍有可察觉下降。

核心矛盾由此凸显：动态剪枝虽灵活，但排序开销侵蚀加速收益，且缺乏理论保证实现真正无损。本文提出InfoBatch，通过**无偏估计框架**将动态剪枝从"近似无损"推进到"严格无损"，同时以**O(1)阈值选择**消除排序瓶颈。

## 核心创新

核心洞察：动态剪枝可以等价于重要性采样问题，因为只要对选中样本施加逆概率权重补偿，就能保持梯度期望不变，从而使严格无损的动态数据剪枝成为可能。

传统动态剪枝（UCB/ε-greedy）将样本选择视为探索-利用权衡的强化学习问题，忽略了剪枝引入的选择偏差；静态剪枝（EL2N）虽试图找到"最优"子集，但放弃了完整数据分布信息。InfoBatch的关键突破在于：**将数据剪枝重新建模为带补偿的重要性采样**——不是寻找哪个子集"最好"，而是确保任何子集的加权梯度都无偏于完整数据集。

| 维度 | Baseline (UCB/ε-greedy) | 本文 (InfoBatch) |
|:---|:---|:---|
| 选择机制 | 排序-based，O(logN)每样本 | 阈值-based，O(1)每样本 |
| 偏差处理 | 无显式补偿，near-lossless | 逆概率加权，严格无损 |
| 剪枝率 | 固定超参数 | 自适应：低损失样本更激进 (r_small=0.75) |
| 理论保证 | 经验性探索-利用平衡 | 无偏梯度估计，保持ERM一致性 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/7103632e-b202-47c1-a75d-a955866ea840/figures/Figure_2.png)
*Figure 2 (pipeline): Figure 2. Illustration of the proposed InfoBatch framework. InfoBatch mainly consists of two components: (i) an exponential score updating and (ii) an adaptive dynamic pruning. N denotes the score of samples. Soft pruning randomly prunes some samples from Λ with relatively much lower scores than current global mean. Λ is the dynamic data pool. Retain epoch denotes outer epoch progression in the original dataset.*



InfoBatch框架包含四个连续阶段，形成闭环动态更新：

1. **前向传播与损失计算**：对当前batch所有样本计算前向损失 $\text{ell}_i = \mathcal{L}(f_\theta(x_i), y_i)$，输出每样本标量损失值。

2. **分数更新与阈值选择（核心创新）**：基于历史损失指数移动平均更新样本分数，无需排序。对每个样本，若其损失高于阈值$\delta$（高困难度），以标准率$r$剪枝；若低于$\delta$（低困难度/已学会），以更激进率$r_{\text{small}}=0.75$剪枝。此步骤输出二元选择掩码。

3. **逆概率加权**：对被选中样本施加补偿权重 $w_i = 1/(1-r_i)$，其中$r_i$为该样本实际剪枝率（$r$或$r_{\text{small}}$）。被剪枝样本权重为零，不进入反向传播。

4. **加权梯度聚合与模型更新**：使用加权损失计算梯度并更新参数。由于权重设计保证 $\mathbb{E}[w_i \cdot \mathbb{1}[\text{selected}]] = 1$，梯度期望严格等于完整数据集梯度。

```
完整数据集 D
    ↓
[前向传播] → 每样本损失 {ℓ_i}
    ↓
[动态选择模块] ──高损失(ℓ_i > δ): 保留率 1-r ──→ 选中集 S_t
              └──低损失(ℓ_i ≤ δ): 保留率 1-r_small ─┘
    ↓
[逆概率加权] w_i = 1/(1-r_i) for i ∈ S_t
    ↓
[反向传播] ∇_θ Σ_{i∈S_t} w_i · ℓ_i
    ↓
模型更新 θ_{t+1}
    ↓
(循环至下一epoch，分数指数衰减更新)
```

## 核心模块与公式推导

### 模块 1: 自适应样本选择分数（对应框架图"Score-based threshold selection"）

**直觉**: 训练过程中样本的"困难度"动态变化，需用指数移动平均平滑瞬时损失波动，避免高频抖动导致的选择不稳定。

**Baseline 公式** (UCB): $$s_i^{\text{UCB}} = \bar{\text{ell}}_i + \sqrt{\frac{2\ln n}{n_i}} \quad \text{(需对所有样本排序)}$$
符号: $\bar{\text{ell}}_i$ = 样本$i$的历史平均损失, $n_i$ = 被选中次数, $n$ = 总轮数。UCB按此分数排序取top-k，复杂度O(N log N)。

**变化点**: UCB的排序操作每epoch开销O(N log N)，且置信项引入额外超参。本文改为**阈值分割**：直接比较当前平滑损失与固定阈值$\delta$，实现O(1)决策。

**本文公式（推导）**:
$$\text{Step 1: } \tilde{\text{ell}}_i^{(t)} = \alpha \cdot \text{ell}_i^{(t)} + (1-\alpha) \cdot \tilde{\text{ell}}_i^{(t-1)} \quad \text{指数移动平均平滑损失}$$
$$\text{Step 2: } s_i = \mathbb{1}[\tilde{\text{ell}}_i > \delta] \cdot r + \mathbb{1}[\tilde{\text{ell}}_i \leq \delta] \cdot r_{\text{small}} \quad \text{阈值分割确定剪枝率}$$
$$\text{最终: } \mathbb{1}[i \in S_t] \sim \text{Bernoulli}(1-s_i) \quad \text{按保留率随机采样}$$

**对应消融**: Table 3 显示固定比例 vs 自适应比例的影响。

---

### 模块 2: 无偏补偿加权（对应框架图"Inverse probability weighting"）

**直觉**: 直接丢弃样本会改变梯度期望，相当于用有偏分布近似原始分布；重要性采样的标准技巧是对选中样本放大权重以抵消选择概率的影响。

**Baseline 公式** (标准ERM / 动态剪枝无补偿):
$$\mathcal{L}_{\text{base}} = \mathbb{E}_{(x,y)\sim D_{\text{selected}}}[\nabla_\theta \mathcal{L}(f_\theta(x), y)] \approx \mathbb{E}_{(x,y)\sim D}[\nabla_\theta \mathcal{L}(f_\theta(x), y)]$$
符号: $D_{\text{selected}}$ = 剪枝后的子集分布。此近似仅在$\mathbb{E}[D_{\text{selected}}] = \mathbb{E}[D]$时成立，但剪枝引入选择偏差，故为近似（near-lossless根源）。

**变化点**: Baseline假设选中样本的梯度期望自然近似完整期望，这在非均匀选择下不成立。本文显式引入**逆概率权重**，将选择过程纳入期望计算，使等式严格成立。

**本文公式（推导）**:
$$\text{Step 1: } \mathcal{L}_{\text{full}} = \mathbb{E}_{(x,y)\sim D}[\text{ell}(f_\theta(x), y)] = \sum_{i \in D} \frac{1}{|D|} \text{ell}_i \quad \text{完整数据期望}$$
$$\text{Step 2: } \mathcal{L}_{\text{pruned}} = \mathbb{E}_{(x,y)\sim D_{\text{selected}}}\left[\frac{1}{P(\text{selected}|x)} \cdot \text{ell}(f_\theta(x), y)\right] \quad \text{加入逆概率权重}$$
$$\text{Step 3: } P(\text{selected}_i) = 1 - s_i = \begin{cases} 1-r & \text{if } \tilde{\text{ell}}_i > \delta \\ 1-r_{\text{small}} & \text{if } \tilde{\text{ell}}_i \leq \delta \end{cases} \quad \text{自适应选择概率}$$
$$\text{最终: } \mathcal{L}_{\text{InfoBatch}} = \sum_{i \in S_t} \frac{1}{1-s_i} \cdot \text{ell}_i = \sum_{i \in S_t} \frac{1}{1-r_i} \cdot \text{ell}_i$$

验证无偏性：$\mathbb{E}_{S_t}[\sum_{i \in S_t} \frac{\text{ell}_i}{1-r_i}] = \sum_{i \in D} P(i \in S_t) \cdot \frac{\text{ell}_i}{1-r_i} = \sum_{i \in D} (1-r_i) \cdot \frac{\text{ell}_i}{1-r_i} = \sum_{i \in D} \text{ell}_i = \mathcal{L}_{\text{full}}$

**对应消融**: Table 3 中"去掉无偏加权"导致性能下降，验证该组件必要性（具体Δ值。

---

### 模块 3: 有效训练损失与梯度聚合（对应框架图"Backward pass & update"）

**直觉**: 将被剪枝样本的"零贡献"与选中样本的"放大贡献"结合，形成等价于完整遍历的梯度流。

**Baseline 公式** (标准SGD):
$$\mathcal{L}_{\text{standard}} = \sum_{i \in \mathcal{B}} \text{ell}(f_\theta(x_i), y_i), \quad \nabla_\theta \mathcal{L}_{\text{standard}} = \sum_{i \in \mathcal{B}} \nabla_\theta \text{ell}_i$$
符号: $\mathcal{B}$ = 当前mini-batch，所有样本均匀权重1。

**变化点**: 标准SGD假设batch从完整数据均匀采样；InfoBatch的batch来自非均匀选择分布，需用加权梯度替代。

**本文公式（推导）**:
$$\text{Step 1: } \mathcal{L}_{\text{InfoBatch}}^{(t)} = \sum_{i \in \mathcal{B}_t \cap S_t} \frac{1}{1-r_i} \cdot \text{ell}_i + \sum_{j \in \mathcal{B}_t \text{setminus} S_t} 0 \cdot \text{ell}_j \quad \text{剪枝样本零贡献，选中样本逆概率加权}$$
$$\text{Step 2: } \nabla_\theta \mathcal{L}_{\text{InfoBatch}}^{(t)} = \sum_{i \in \mathcal{B}_t \cap S_t} \frac{1}{1-r_i} \cdot \nabla_\theta \text{ell}_i \quad \text{加权反向传播}$$
$$\text{最终: } \theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}_{\text{InfoBatch}}^{(t)}$$

注意：实际实现中，$1/(1-r_i)$为常数因子（$r=0.5$时为2.0，$r_{\text{small}}=0.75$时为4.0），可与学习率$\eta$合并，不增加计算开销。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/7103632e-b202-47c1-a75d-a955866ea840/figures/Table_1.png)
*Table 1 (comparison): Table 1. The accuracy (%) comparison on state-of-the-art methods. All methods are trained with standard training settings and without extra data. Bold indicates the best performance. Underline indicates the dynamic score-based pruning. Details are available in Appendix A.*



本文在CIFAR-10/100、ImageNet-1K、ADE20K及LLaMA-7B指令微调任务上验证InfoBatch。核心结果见Table 1与Table 2：在CIFAR-10上，InfoBatch于30%剪枝比例达到95.6%准确率，与完整数据集训练无损持平，而同期最优动态方法UCB仅95.0%（-0.6%）、静态方法EL2N为95.2%（-0.4%）。CIFAR-100上差距拉大：30%剪枝下InfoBatch 78.2% vs UCB 76.8%（+1.4%），50%剪枝下76.5% vs UCB 74.5%（+2.0%），70%剪枝下72.5% vs UCB 71.0%（+1.5%）。值得注意的是，**剪枝比例越高，InfoBatch优势越显著**——高剪枝场景下基线方法的近似误差累积，而无偏设计保证了InfoBatch的稳定性。

ImageNet-1K上（Table 2），InfoBatch采用更激进的$r_{\text{small}}=0.75$处理低损失样本（20%分位），在ResNet-50和ViT-Base上均实现无损或接近无损。跨架构鲁棒性见Table 5：从ResNet-18到Swin-Tiny，InfoBatch保持一致优势。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/7103632e-b202-47c1-a75d-a955866ea840/figures/Table_3.png)
*Table 3 (ablation): Table 3. Ablation of component contributions. The experiments are conducted on CIFAR-10/100.*



消融实验（Table 3）验证各组件贡献。去掉自适应比例机制（改用固定$r$）导致CIFAR-10/100性能下降（具体Δ值；去掉无偏补偿加权则偏差引入显著，性能逼近普通动态剪枝。Figure 3进一步展示超参敏感性：平滑系数$\alpha$在0.9附近最优，阈值$\delta=0.875$为经验稳定点，剪枝比例与准确率呈预期 trade-off 但InfoBatch曲线始终位于最上方。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/7103632e-b202-47c1-a75d-a955866ea840/figures/Figure_3.png)
*Figure 3 (ablation): Figure 3. (a) Acc. vs pruning ratio. (b) Acc. vs smoothing coef. (c) Acc. vs Inf. γ on CIFAR10. (d) Overhead and savings. The experiments are conducted on CIFAR-10 using ResNet-18. Default hyper-parameter: pruning ratio=0.5, smoothing coefficient=0.9, information region ratio γ=0.87.*



效率方面，Table 4显示InfoBatch在CIFAR-10/100上实现约50%前向计算削减（$r=0.5$），且每样本O(1)开销使总加速比接近理论上限。LLaMA-7B指令微调（Table 6相关）中，DQ+InfoBatch将训练时间从18.0分钟压缩至14.4分钟，BBH/DROP/MMLU/Human-Eval平均得分保持26.3不变，验证无损加速扩展至大语言模型场景。

**公平性检验**：基线选择合理，涵盖静态（EL2N, Influence, Herding）与动态（UCB, ε-greedy, Random*）主流方法。但Table 1未显式列出"完整数据集"准确率数字（仅通过InfoBatch 30%剪枝"无损"推断），且LLaMA实验与Dataset Quantization (DQ)耦合，难以孤立量化InfoBatch贡献。作者披露自适应比例使精确epoch匹配复杂化（†标记控制比较）。缺失基线包括Forgetting (Toneva et al., 2018)动态方法及近期静态方法如Moderate的动态扩展。

## 方法谱系与知识库定位

InfoBatch属于**动态数据剪枝**谱系，直接parent为Raju et al. (2021) 的UCB/ε-greedy框架。核心继承：每epoch基于当前损失动态选择样本；关键突破：将探索-利用范式替换为无偏估计范式，消除排序开销并达到严格无损。

**直接基线与差异**：
- **UCB (Raju et al., 2021)**: 同谱系动态剪枝，使用置信上界排序选择 → InfoBatch改为阈值O(1)选择，添加逆概率加权
- **ε-greedy (Raju et al., 2021)**: 同谱系，ε概率随机探索其余按损失排序 → InfoBatch完全确定性的自适应阈值，理论保证无损
- **EL2N (Paul et al., 2021)**: 静态剪枝代表，预计算误差分数取coreset → InfoBatch动态适应，无需预训练参考模型
- **Influence (Koh & Liang, 2017)**: 静态剪枝，Hessian-based影响函数 → InfoBatch避免昂贵二阶计算

**改变的slots**：data_pipeline（排序→阈值O(1)）、objective（均匀加权→逆概率补偿加权）、training_recipe（固定比例→自适应$r$/$r_{\text{small}}$）。

**后续方向**：(1) 将无偏框架扩展至其他样本选择场景如主动学习、课程学习；(2) 结合数据增强/混合策略，探索剪枝与增强的联合优化；(3) 理论分析自适应比例$r(t)$的收敛速率，超越当前经验设计。

**标签**：modality: 图像/文本通用 | paradigm: 动态数据剪枝/重要性采样 | scenario: 大规模训练加速 | mechanism: 无偏估计/逆概率加权/自适应阈值 | constraint: 无损性能/计算开销敏感

