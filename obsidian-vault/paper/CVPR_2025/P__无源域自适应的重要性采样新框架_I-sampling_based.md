---
title: 'Revisiting Source-Free Domain Adaptation: Insights into Representativeness, Generalization, and Variety'
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- 无源域自适应的重要性采样新框架
- I-sampling based
- I-sampling based SFDA (unnamed in provided text)
acceptance: poster
cited_by: 6
method: I-sampling based SFDA (unnamed in provided text)
---

# Revisiting Source-Free Domain Adaptation: Insights into Representativeness, Generalization, and Variety

**Topics**: [[T__Domain_Adaptation]], [[T__Self-Supervised_Learning]] | **Method**: [[M__I-sampling_based_SFDA]] | **Datasets**: [[D__Office-Home]] (其他: DomainNet126, VisDA-C)

| 中文题名 | 无源域自适应的重要性采样新框架 |
| 英文题名 | Revisiting Source-Free Domain Adaptation: Insights into Representativeness, Generalization, and Variety |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [DOI](https://doi.org/10.1109/cvpr52734.2025.02392) · Code  · Project  |
| 主要任务 | Source-Free Domain Adaptation (SFDA), 图像分类域自适应 |
| 主要 baseline | SHOT+DPC, U-SFAN, SDE, GPUE, CST (UDA), I-SFDA, ERM |

> [!abstract] 因为「SFDA 中伪标签数据选择策略（C-sampling/T-sampling）无法保证选中数据的代表性与可靠性，且单一损失函数易导致 posterior collapse」，作者在「SHOT 框架」基础上改了「I-sampling 重要性采样策略 + LSA+LCE+LIM 三模块损失 + 多轮迭代渐进训练」，在「Office-Home 12-task average」上取得「73.8% vs U-SFAN 71.6% (+2.2%)」，在「DomainNet-126」上取得「+3.6% over GPUE」

- **Office-Home (ResNet-50)**: 平均准确率 73.8%，超越 U-SFAN 2.2%、SDE 1.2%、SHOT+DPC 0.9%
- **DomainNet-126 (ResNet-50)**: 超越第二名 GPUE 3.6%
- **消融关键发现**: LSA 单独使用导致准确率暴跌至 28.2%，三模块组合不可或缺

## 背景与动机

无源域自适应（Source-Free Domain Adaptation, SFDA）解决一个实际难题：企业部署了预训练模型后，由于隐私或存储限制，无法获取原始训练数据，但需让模型适应新场景。例如，用产品图片训练的分类器需适配艺术画作，却无权限回溯原始产品图库。

现有方法沿两条路径处理此问题。**SHOT** 通过信息最大化（Information Maximization）和伪标签自训练实现适应，但采用统一阈值筛选伪标签，易引入噪声。**U-SFAN** 与 **SDE** 改进了特征对齐策略，却在数据选择上依赖置信度采样（C-sampling）或均匀采样（T-sampling），未考虑训练动态变化。**GPUE** 引入不确定性估计，但计算开销较大且未解决渐进式选择问题。

这些方法的共同短板在于**数据选择的静态性**：它们一次性或按固定规则选择伪标签数据，忽视了关键事实——模型在早期训练阶段对易样本预测更准确，而随着适应进行，未选中样本的质量会提升。C-sampling 按置信度阈值选取，高置信度样本未必代表整体分布；T-sampling 均匀选取则混入大量错误标签。更深层的问题是，现有方法多用单一损失（如仅 L_CE 或仅 L_IM），导致选中数据过少时模型崩溃（posterior collapse），或仅 LSA 时因伪标签噪声严重退化。

本文提出**重要性采样（I-sampling）驱动的渐进式框架**，将数据选择建模为动态过程，配合三模块损失函数协同训练，从根本上解决代表性与稳定性之间的矛盾。
![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/afc9ead7-3b70-4ee4-9c89-06fcfc3e6f15/figures/Figure_2.png)
*Figure 2: Table 3. Ablation study of sub-modules in our proposed method on Office-Home and DomainNet126 datasets with six domain tasks, i.e.,A→C, A→P, A→R, Re→Cl, Re→Pa, and Re→Sk. LCE is the cross-entropy loss*



## 核心创新

核心洞察：**伪标签数据的可靠性具有时序结构性**，因为模型在不同训练阶段对不同难度样本的预测准确度动态变化，从而使**渐进式重要性采样 + 多损失协同**成为稳定高效 SFDA 的关键。

| 维度 | Baseline (SHOT/U-SFAN/SDE) | 本文 |
|:---|:---|:---|
| **数据选择** | C-sampling（置信度阈值）或 T-sampling（均匀采样），静态单次选择 | I-sampling（重要性采样），多轮渐进动态选择，早期选易样本、后期利用提升后的难样本 |
| **损失函数** | 单一损失：L_CE 或 L_IM 或 L_SA 单独使用 | 三模块组合：L_SA（语义域对齐）+ L_CE（交叉熵监督）+ L_IM（信息最大化），分别作用于 D_t,u 和 D_t,l |
| **训练流程** | 单轮或简单迭代，无累积机制 | 多轮迭代（R rounds），每轮选择 D'^r_{t,l} 并累积至 D^r_{t,l}，形成课程式学习 |

与 baseline 的本质差异在于：本文将 SFDA 重新定义为**部分标签学习 + 无标签学习的动态混合问题**，而非静态伪标签自训练。

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/afc9ead7-3b70-4ee4-9c89-06fcfc3e6f15/figures/Figure_1.png)
*Figure 1: Figure 1. The approach follows a progressive training process (described in Sec. 4.3) conducted over R rounds, with each round comprisingQ epochs. Each round consists of two main stages: sample select*



整体流程遵循**渐进式多轮训练**，共 R 轮迭代，每轮包含 Q 个 epoch。数据流如下：

1. **输入**: 源域预训练模型 θ^0（仅模型参数，无源数据）+ 目标域全部无标签数据 D_t
2. **I-sampling 选择模块**: 基于当前模型 θ^{r-1} 计算各目标样本的重要性分数，选取最可靠的子集 D'^r_{t,l} 赋予伪标签
3. **累积模块**: 将本轮新选中的 D'^r_{t,l} 合并至历史累积集 D^r_{t,l} = D^{r-1}_{t,l} ∪ D'^r_{t,l}，剩余数据记为 D^r_{t,u} = D_t \\ D^r_{t,l}
4. **三模块训练模块**: 同时优化
   - L_CE 于 D^r_{t,l}（可靠伪标签的监督学习）
   - L_SA 于 D^r_{t,u}（未选中数据的语义域对齐，防 collapse）
   - L_IM 于 D^r_{t,u}（信息最大化，保持预测多样性）
5. **输出**: 更新后的模型 θ^r，进入下一轮迭代

```
源预训练模型 ──→ I-sampling ──→ 累积 D^r_{t,l} ──┐
     ↑                    └─→ D^r_{t,u} ──────────┤
     └──────────────── 三模块训练 ← L_CE/L_SA/L_IM ←┘
              ↓
         最终模型 θ^R
```

关键设计：I-sampling 在早期轮次选中高准确度样本，随着 D^r_{t,u} 的准确率提升（Figure 2(d)），后续轮次可安全纳入更多样本，形成**自增强循环**。

## 核心模块与公式推导

### 模块 1: I-sampling（重要性采样）—— 对应框架图步骤 2

**直觉**: 伪标签的可靠性不仅取决于当前模型置信度，更取决于样本在训练历史中的稳定性与对模型更新的敏感度，需动态评估而非静态阈值。

**Baseline 形式 (C-sampling/T-sampling)**: 按固定置信度阈值 τ 选取：D_{t,l} = {x ∈ D_t | max_y p(y|x) > τ}，或均匀随机选取固定比例。

**变化点**: C-sampling 的阈值 τ 全局固定，无法适应模型进化；T-sampling 忽略可靠性差异。本文提出基于**重要性分数**的排序选择，考虑样本对模型参数更新的影响权重，使早期轮次优先选择"模型已掌握"的样本，避免噪声干扰。

**本文公式**: 
$$s_i = \text{Importance}(x_i; \theta^{r-1}) \quad \text{（综合置信度历史稳定性与梯度影响）}$$
$$D'^r_{t,l} = \text{TopK}_{s_i}(D_t, k_r) \quad \text{（每轮选取 top-k_r 重要样本，k_r 可随轮次调整）}$$

**对应消融**: Table 4 显示，在 A→C 任务上，I-sampling 61.2% vs C-sampling 59.9% (+1.3%)，vs T-sampling 59.1% (+2.1%)；在 Re→Cl 上 77.7% vs C-sampling 76.5% (+1.2%)，vs T-sampling 76.2% (+1.5%)。

---

### 模块 2: 三模块组合损失 L = L_SA + L_CE + L_IM —— 对应框架图步骤 4

**直觉**: 单一损失无法同时处理"有标签数据的学习"和"无标签数据的防 collapse"，需分工明确的三模块协同。

**Baseline 公式 (SHOT 等)**: 
$$L_{base} = L_{IM} = -\sum_{x \in D_t} H(p(y|x)) + H(\bar{p}(y))$$
或单纯 $L_{CE}$ 于伪标签子集。符号：$H(\cdot)$ 为熵，$\bar{p}(y) = \frac{1}{|D_t|}\sum_x p(y|x)$ 为边际预测分布。

**变化点**: 
- 仅 L_CE 时，D_{t,l} 过小或含噪则过拟合（65.8%）
- 仅 L_IM 时，无监督信号弱（67.1%）
- 仅 L_SA 时，小样本错误标签导致严重 posterior collapse（28.2%）
- 三者组合实现互补：L_CE 提供可靠监督，L_SA 对齐语义空间，L_IM 保持预测多样性

**本文公式（推导）**:

$$\text{Step 1 (L_CE)}: L_{CE} = -\sum_{(x,\hat{y}) \in D^r_{t,l}} \hat{y} \log p(y|x; \theta) \quad \text{（对累积可靠伪标签的标准监督）}$$

$$\text{Step 2 (L_IM)}: L_{IM} = -\sum_{x \in D^r_{t,u}} \sum_y p(y|x) \log p(y|x) + \sum_y \bar{p}(y) \log \bar{p}(y) \quad \text{（熵最小化 + 边际熵最大化，防 collapse）}$$

$$\text{Step 3 (L_SA)}: L_{SA} = \text{Align}(f(x; \theta), \text{semantic prototypes}) \text{ for } x \in D^r_{t,u} \quad \text{（语义域对齐，利用类别原型约束未选中数据）}$$

$$\text{最终}: L = L_{SA} + L_{CE} + L_{IM}$$

符号说明：$\hat{y}$ 为 I-sampling 赋予的伪标签；$f(x;\theta)$ 为特征表示；semantic prototypes 通过 D^r_{t,l} 的类中心动态计算。

**对应消融**: Table 3 显示，完整组合 73.8%；去掉 L_CE（仅 L_SA+L_IM）降至 68.9% (-4.9%)；去掉 L_IM（L_SA+L_CE）降至 71.3% (-2.5%)；去掉 L_SA（L_CE+L_IM）降至 69.0% (-4.8%)。最关键的是，L_SA 单独使用仅 28.2%（-45.6%），证明三模块缺一不可的协同效应。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/afc9ead7-3b70-4ee4-9c89-06fcfc3e6f15/figures/Table_1.png)
*Table 1: Table 1. Classification Accuracy (%) on Office-Home (ResNet-50) and VisDA-C (ResNet-101). The best results under SFDA setting arehighlighted in bold. Note that “SF” means whether a method belongs to S*



本文在三个标准 SFDA benchmark 上评估：Office-Home（4 域 65 类 12 任务，ResNet-50）、DomainNet-126（大规模 126 类 7 任务，ResNet-50）、VisDA-C（合成到真实，ResNet-101）。核心结果来自 Table 1 与 Table 2。

在 **Office-Home 平均 12 任务**上，本文方法达到 **73.8%**，相比当前 SFDA SOTA U-SFAN（71.6%）提升 **+2.2%**，相比 SDE（72.6%）提升 **+1.2%**，相比 SHOT+DPC（72.9%）提升 **+0.9%**。值得注意的是，即使与使用源数据的 UDA 强 baseline CST（72.7%）相比，本文仍领先 **+1.1%**，凸显了无源设置下超越有源方法的潜力。在 **DomainNet-126** 这一更具挑战性的 benchmark 上，本文以 **+3.6%** 的大幅优势超越第二名 GPUE，验证了 I-sampling 在大规模复杂域迁移中的稳健性。VisDA-C 上同样呈现一致优势趋势（具体数值。


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/afc9ead7-3b70-4ee4-9c89-06fcfc3e6f15/figures/Table_2.png)
*Table 2: Table 2. Classification Accuracy (%) on DomainNet-126 (ResNet-50). The best results under SFDA setting are highlighted in bold. Notethat “SF” means whether a method belongs to SFDA method.*



消融实验（Table 3 与 Table 4）揭示了设计选择的敏感性。Table 3 的子模块消融显示，**L_CE 的缺失代价最大**：完整组合 73.8% → L_SA+L_IM 68.9%（-4.9%），说明可靠伪标签的监督信号是性能基石。**L_SA 单独使用导致灾难性崩溃至 28.2%**，印证了"小样本错误标签 + 无约束对齐 = posterior collapse"的理论分析；但 L_SA 作为辅助模块与 L_CE、L_IM 共存时不可或缺，去掉后（L_CE+L_IM）降至 69.0%（-4.8%）。Table 4 的采样策略消融中，I-sampling 在 A→C 任务上 61.2%，替换为 C-sampling 降 1.3%、T-sampling 降 2.1%；在 Re→Cl 上分别降 1.2% 与 1.5%，一致性验证了动态重要性评估优于静态阈值。



Figure 2 的四幅子图提供了训练动态的深层洞察：(a) 整体准确率显示 I-sampling 全程优于 C/T-sampling；(b) 每轮新选中数据 D'^r_{t,l} 的准确率逐轮递减，证实早期轮次" cherry-picking "最易样本的策略合理性；(c) 累积集 D^r_{t,l} 的准确率因混入更多样本而略降；(d) 关键发现——**剩余未选中数据 D^r_{t,u} 的准确率在前几轮急剧提升**，这正是后续轮次能选出更多可靠样本的前提，构成自增强闭环。

公平性检验：比较基本合理——SFDA 方法间均无源数据，公平对比；CST 作为 UDA 方法使用源数据，理论上占优却落后，反衬本文方法强度。潜在局限：多轮迭代（5 rounds）带来额外训练时间，但未报告具体 GPU 小时数；12 任务中 8 个最优但非全部，存在一定方差；LSA 单独崩溃的极端现象（28.2%）缺乏更深层理论解释。

## 方法谱系与知识库定位

本文属于 **SHOT 谱系** 的 SFDA 方法，直接继承 SHOT 的信息最大化与伪标签自训练框架，但在三个关键 slot 上做了结构性改造：

| 改动 Slot | 父方法 SHOT | 本文 |
|:---|:---|:---|
| data_pipeline | 置信度阈值伪标签选取 | I-sampling 渐进式重要性采样 |
| objective | 单一 L_IM（或 L_CE 辅助） | L_SA + L_CE + L_IM 三模块协同 |
| training_recipe | 单轮或简单多轮 | 多轮迭代 + 累积机制 + 课程式学习 |

**直接 Baseline 差异**：
- **SHOT+DPC (+0.9%)**: 添加 DPC 去噪但未改变静态选择本质
- **U-SFAN (+2.2%)**: 改进特征对齐，数据选择仍粗糙
- **SDE (+1.2%)**: 优化熵正则化，缺乏渐进式选择
- **GPUE (+3.6%)**: 引入不确定性估计，计算更复杂且未利用训练动态

**后续方向**：(1) 将 I-sampling 推广至其他噪声标签场景（如半监督学习、联邦学习）；(2) 理论分析 LSA 的 collapse 条件与三模块的协同收敛保证；(3) 结合 foundation model 的预训练特征，进一步降低对源模型质量的依赖。

**标签**: 模态=图像分类 / 范式=自训练 + 课程学习 / 场景=域自适应（无源数据） / 机制=重要性采样 + 多任务损失协同 / 约束=零源数据访问、隐私保护部署

