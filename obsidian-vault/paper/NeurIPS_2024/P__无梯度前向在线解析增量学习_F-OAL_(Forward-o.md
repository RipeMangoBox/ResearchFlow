---
title: 'F-OAL: Forward-only Online Analytic Learning with Fast Training and Low Memory Footprint in Class Incremental Learning'
type: paper
paper_level: C
venue: NeurIPS
year: 2024
paper_link: null
aliases:
- 无梯度前向在线解析增量学习
- F-OAL (Forward-o
- F-OAL (Forward-only Online Analytic Learning)
acceptance: Poster
cited_by: 12
code_url: https://github.com/liuyuchen-cz/F-OAL
method: F-OAL (Forward-only Online Analytic Learning)
---

# F-OAL: Forward-only Online Analytic Learning with Fast Training and Low Memory Footprint in Class Incremental Learning

[Code](https://github.com/liuyuchen-cz/F-OAL)

**Topics**: [[T__Continual_Learning]], [[T__Few-Shot_Learning]], [[T__Compression]] | **Method**: [[M__F-OAL]] | **Datasets**: [[D__CIFAR-100]], [[D__DTD]], [[D__Tiny-ImageNet]] (其他: CORe50, FGVCAircraft)

| 中文题名 | 无梯度前向在线解析增量学习 |
| 英文题名 | F-OAL: Forward-only Online Analytic Learning with Fast Training and Low Memory Footprint in Class Incremental Learning |
| 会议/期刊 | NeurIPS 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2403.15751) · [Code](https://github.com/liuyuchen-cz/F-OAL) · [DOI](https://doi.org/10.52202/079017-1314) |
| 主要任务 | Class-Incremental Learning (类增量学习) |
| 主要 baseline | LwF, EWC, iCaRL, ER, ASER, SCR, DVC, PCR, EASE, LAE, SLCA |

> [!abstract] 因为「反向传播在增量学习中导致高GPU内存占用和慢速训练」，作者在「OAL (Online Analytic Learning)」基础上改了「将梯度下降替换为前向-only递归最小二乘更新，并增加Feature Fusion与Smooth Projection模块」，在「CIFAR-100/CORe50/FGVCAircraft/DTD/Country211」上取得「91.1%/96.3%/66.2%/82.8%/24.4%平均准确率，其中4项达到SOTA」

- **CIFAR-100**: 91.1% Aavg，超过最强exemplar-free方法EASE (+0.3%) 和最强replay-based方法DVC (+0.6%)
- **FGVCAircraft**: 66.2% Aavg，超过DVC达 **+10.6%**，为最大提升幅度
- **GPU内存**: 峰值仅 **1.9GB** (batch size 256)，远低于其他方法最高9.8GB
- **训练速度**: CIFAR-100仅 **261秒**（含特征提取），为竞争方法中最快

## 背景与动机

类增量学习（Class-Incremental Learning, CIL）要求模型按顺序学习新类别，同时不遗忘旧类别。一个典型场景是：先学"猫""狗"，再学"鸟""鱼"，模型必须能正确分类所有四类，而非仅最新两类。现有方法面临根本性张力——replay-based方法（如iCaRL、DVC）存储旧样本缓冲，引发隐私与存储问题；exemplar-free方法（如LwF、EWC、EASE）依赖反向传播和梯度计算，导致GPU内存占用高、训练慢。

具体而言，**LwF** 通过知识蒸馏保留旧模型输出作为软标签，但蒸馏温度敏感且对新任务容量受限；**EWC** 用Fisher信息矩阵约束重要参数，却需二阶近似且对复杂任务扩展性差；**EASE** 作为当前最强exemplar-free方法，虽用能量分类器避免存储旧样本，仍依赖完整的反向传播管线。这些方法的共同瓶颈在于：**梯度计算需要维护计算图、激活值和优化器状态**，在增量设置中逐任务累积，内存开销随任务数增长。

更深层的问题是**近期偏差（recency bias）**：标准反向传播天然偏向最新批次，导致旧类别权重被抑制。Figure 3显示传统方法分类器权重呈现明显的时间衰减模式。OAL (Online Analytic Learning) 曾尝试用解析闭式解替代梯度下降，但仍需存储完整数据矩阵进行批处理求逆，无法真正在线更新。

本文的核心动机是：**能否彻底消除反向传播，仅通过前向计算实现真正的在线增量学习？** F-OAL通过递归最小二乘（Recursive Least Squares）将训练转化为纯前向过程，同时保持甚至提升准确率。



## 核心创新

核心洞察：**解析分类器的递归最小二乘更新天然等价于无偏的在线学习**，因为特征自相关矩阵 $\mathbf{R}$ 累积所有历史样本的外积和，不区分新旧任务，从而使**消除反向传播的同时避免近期偏差**成为可能。

与 baseline 的差异：

| 维度 | Baseline (OAL/EASE/DVC) | 本文 F-OAL |
|:---|:---|:---|
| **训练范式** | 反向传播梯度下降，需计算图与激活存储 | 前向-only递归最小二乘，无梯度计算 |
| **分类器更新** | 迭代优化或批处理矩阵求逆 | 增量更新 $\mathbf{R}_t = \mathbf{R}_{t-1} + \phi(\mathbf{x}_t)\phi(\mathbf{x}_t)^\text{top}$，在线闭式解 |
| **特征处理** | 直接使用编码器输出或简单投影 | Smooth Projection (随机投影+正则化) + Feature Fusion (原始+投影特征融合) |
| **内存机制** | 依赖replay buffer或完整历史数据 | 仅维护$\mathbf{R}$矩阵和互相关向量，$O(d^2)$固定开销 |
| **时间复杂度** | 多epoch迭代，随任务数增长 | 单遍前向，CIFAR-100仅261秒 |

关键突破在于：F-OAL不是简单加速OAL，而是将"训练即反向传播"的深度学习范式替换为"训练即前向计算"的解析信号处理范式，同时通过两个新模块弥补随机投影带来的信息损失。

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5166a9b2-e8f4-498e-a40f-3c67af74b0fa/figures/Figure_1.png)
*Figure 1 (pipeline): The diagram illustrates the learning agenda of F-OAL, including the encoder and forecaster pipelines with recursive least squares classification.*



F-OAL的完整数据流如下：

1. **Frozen ViT-B Encoder**: 输入原始图像，输出768维预训练特征。编码器完全冻结，不参与任何训练，确保特征提取的稳定性。

2. **Feature Fusion (FF)**: 接收原始ViT特征，将其与后续Smooth Projection的输出进行融合（拼接或加权组合）。该模块增强表示丰富度，尤其对细粒度数据集的判别性提升关键。

3. **Smooth Projection (SP)**: 将融合后的特征通过随机投影矩阵 $\mathbf{P}$ 映射到固定1000维空间，并施加正则化。投影维度远小于原始特征空间，降低$\mathbf{R}$矩阵计算开销。

4. **Analytic Classifier (AC)**: 接收1000维投影特征，通过递归最小二乘在线更新权重。核心维护正则化特征自相关矩阵 $\mathbf{R}$ 和特征-标签互相关向量 $\mathbf{H}$，输出类别logits。

```
Raw Image → [Frozen ViT-B] → Feature (768-d)
                                ↓
                    ┌─────────────────────┐
                    ↓                     ↓
              [Smooth Projection]    [Raw Feature]
                    ↓ (1000-d)          ↓
                    └──────→ [Feature Fusion] ←──────┘
                                ↓
                         [Analytic Classifier]
                           (RLS update, no BP)
                                ↓
                          Class Logits
```

整个训练过程**无需反向传播**：Encoder冻结，AC通过RLS解析更新，FF和SP为前向变换。这是F-OAL实现低内存和快速训练的结构基础。

## 核心模块与公式推导

### 模块 1: Analytic Classifier (AC)（对应框架图最右侧）

**直觉**: 线性分类器的最优权重应有闭式解，无需迭代梯度下降；在线场景下，该解应能增量维护而非重算。

**Baseline 公式** (标准反向传播分类器):
$$\mathbf{W}^{(t+1)} = \mathbf{W}^{(t)} - \eta \nabla_{\mathbf{W}} \mathcal{L}_{\text{CE}}$$
符号: $\mathbf{W} \in \mathbb{R}^{d \times C}$ 为分类器权重，$\eta$ 为学习率，$\mathcal{L}_{\text{CE}}$ 为交叉熵损失，需维护计算图与激活值用于反向传播。

**变化点**: 梯度下降存在近期偏差——最新批次的梯度主导更新方向，旧任务梯度被覆盖；且需$O(batch \times d)$激活存储。本文改用**岭回归闭式解**，并通过递归最小二乘实现在线更新。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathbf{W}^* = \text{arg}\min_{\mathbf{W}} \|\mathbf{\Phi}\mathbf{W} - \mathbf{Y}\|_F^2 + \gamma\|\mathbf{W}\|_F^2 = (\mathbf{\Phi}^\text{top}\mathbf{\Phi} + \gamma\mathbf{I})^{-1}\mathbf{\Phi}^\text{top}\mathbf{Y}$$
加入了Tikhonov正则化项 $\gamma\|\mathbf{W}\|_F^2$ 以解决矩阵奇异和过拟合问题，$\gamma=1$为默认设置。

$$\text{Step 2}: \quad \mathbf{R}_t = \mathbf{R}_{t-1} + \phi(\mathbf{x}_t)\phi(\mathbf{x}_t)^\text{top}, \quad \mathbf{H}_t = \mathbf{H}_{t-1} + \phi(\mathbf{x}_t)\mathbf{y}_t^\text{top}$$
将批处理求逆转化为增量更新，重归一化以保证所有样本被平等对待（无近期偏差）。

$$\text{最终}: \quad \mathbf{W}_t = \mathbf{R}_t^{-1}\mathbf{H}_t$$
其中 $\mathbf{R}_t = \sum_{n=1}^{t} \phi(\mathbf{x}_n)\phi(\mathbf{x}_n)^\text{top} + \gamma\mathbf{I}$ 为正则化特征自相关矩阵，$\mathbf{H}_t$ 为累积的互相关矩阵。

**对应消融**: Table 5显示将AC替换为同结构但反向传播的FCC，CIFAR-100平均准确率从91.1%暴跌至32.4%，**Δ = -58.7%**，证明解析学习是避免灾难性遗忘的核心机制。

---

### 模块 2: Smooth Projection (SP)（对应框架图中间）

**直觉**: 高维特征直接求逆计算代价高，随机投影可降维；但朴素投影会损失判别信息，需正则化稳定。

**Baseline 公式** (无投影或固定线性投影):
$$\mathbf{z} = \mathbf{P}\phi(\mathbf{x}), \quad \mathbf{P} \in \mathbb{R}^{m \times d} \text{ (固定或学习得到)}$$
标准方法中投影矩阵可学习（需反向传播）或为预训练PCA（需存储统计量）。

**变化点**: 可学习投影违反前向-only约束；PCA需存储均值/方差且对增量数据敏感。本文采用**随机高斯投影+显式正则化**，投影矩阵采样后固定，不更新。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathbf{P}_{ij} \sim \mathcal{N}(0, \frac{1}{m}), \quad m = 1000$$
随机投影矩阵元素从高斯分布采样，满足Johnson-Lindenstrauss引理的保距性质。

$$\text{Step 2}: \quad \mathbf{z} = \frac{1}{\sqrt{m}}\mathbf{P}\phi(\mathbf{x}), \quad \tilde{\mathbf{z}} = [\mathbf{z}; \|\mathbf{z}\|_2] \text{ (可选归一化)}$$
缩放保证方差稳定，维度$m=1000$为平衡计算效率与信息保留的经验选择。

$$\text{最终}: \quad \mathbf{R} = \sum_n \mathbf{z}_n\mathbf{z}_n^\text{top} + \gamma\mathbf{I}, \quad \gamma=1$$
正则化项$\gamma\mathbf{I}$在投影后仍保留，防止$\mathbf{R}$在增量初期样本不足时奇异。

**对应消融**: Table 4显示去掉SP后，FGVCAircraft下降3.5%（66.2%→62.7%），DTD下降3.5%（82.8%→79.3%），细粒度数据集对投影降维更敏感。

---

### 模块 3: Feature Fusion (FF)（对应框架图左侧融合节点）

**直觉**: 随机投影虽降维但损失部分原始特征信息，融合原始特征与投影特征可兼得效率与判别力。

**Baseline 公式** (单一特征流):
$$\mathbf{f}_{\text{in}} = \phi(\mathbf{x}) \text{ 或 } \mathbf{f}_{\text{in}} = \mathbf{z}$$
直接使用原始特征或仅用投影特征。

**变化点**: 原始特征维度高（768-d）不利于快速求逆；纯投影特征信息有损失。本文将两者拼接或加权组合。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathbf{f}_{\text{fusion}} = [\phi(\mathbf{x}); \mathbf{z}] = [\phi(\mathbf{x}); \frac{1}{\sqrt{m}}\mathbf{P}\phi(\mathbf{x})]$$
拼接原始特征与投影特征，形成增强表示。

$$\text{Step 2}: \quad \mathbf{R}^{\text{fusion}} = \sum_n \mathbf{f}_{\text{fusion},n}\mathbf{f}_{\text{fusion},n}^\text{top} + \gamma\mathbf{I}$$
自相关矩阵在增广空间构建，保留双路信息。

$$\text{最终}: \quad \mathbf{W}^{\text{fusion}} = (\mathbf{R}^{\text{fusion}})^{-1}\mathbf{H}^{\text{fusion}}$$
解析解在融合空间直接求得。

**对应消融**: Table 4显示去掉FF后，Country211下降3.1%（24.4%→21.3%），DTD下降2.3%（82.8%→80.5%）；同时去掉FF和SP导致DTD暴跌11.6%（82.8%→71.2%），证明两模块的协同必要性。

## 实验与分析



F-OAL在6个标准增量学习基准上评估：CIFAR-100（粗粒度，100类）、CORe50（物体识别，50类）、FGVCAircraft（细粒度飞机，100类）、DTD（纹理，47类）、Tiny-ImageNet（200类）、Country211（地理细粒度，211类）。所有方法统一使用预训练ViT-B/16编码器，保证公平比较。

核心结果来自Table 2：F-OAL在**CORe50达96.3%**（超DVC +0.5%，SOTA）、**FGVCAircraft达66.2%**（超DVC +10.6%，SOTA）、**DTD达82.8%**（超DVC +6.8%，SOTA）、**Country211达24.4%**（超DVC/iCaRL +6.6%，SOTA）。CIFAR-100上91.1%超过EASE 90.8%（+0.3%）和DVC 90.5%（+0.6%）。唯一未达SOTA的是Tiny-ImageNet（91.2%），落后EASE 92.0%约0.8%，但仍超过DVC 90.5%。细粒度数据集（FGVCAircraft、DTD、Country211）的巨大提升表明，FF和SP模块对判别性相似类别尤为关键。





消融实验（Table 4、Table 5）揭示各组件贡献：Analytic Classifier是根基——替换为同结构反向传播分类器导致CIFAR-100从91.1%崩溃至32.4%（-58.7%），验证"前向-only"不仅是效率优化更是精度保障。Smooth Projection在FGVCAircraft贡献3.5%（66.2%→62.7%），Feature Fusion在Country211贡献3.1%（24.4%→21.3%）。两模块联合移除在DTD造成-11.6%的灾难性下降，说明原始特征与投影特征的互补性。

效率方面，Figure 2显示F-OAL峰值GPU内存仅**1.9GB**（batch size 256），对比EASE 3.2GB、DVC 5.7GB、SLCA 9.8GB。训练时间CIFAR-100仅261秒（含特征提取），CORe50 570秒，对比其他方法通常需数千秒。

公平性检查：比较基本公平——所有方法共享相同ViT-B预训练backbone，但这也意味方法优势部分源于"仅训练分类器"的设定。Replay方法缓冲限制为5000，更大缓冲可能提升其表现。未与DER++、MEMO、FOSTER、SimpleCIL等更新方法比较。作者承认对预训练backbone的强依赖是局限，自监督预训练或更强backbone可能进一步改变格局。Figure 3的权重可视化显示F-OAL无近期偏差，当前任务与历史任务的分类器权重L2范数处于平均水平，从机制层面解释了低遗忘率。

## 方法谱系与知识库定位

F-OAL属于**解析学习（Analytic Learning）**方法家族，直接继承自 **OAL (Online Analytic Learning)**。谱系演进路径为：OAL提出用闭式解析解替代梯度下降 → F-OAL将批处理求逆升级为在线递归最小二乘，彻底消除反向传播，并新增Feature Fusion与Smooth Projection模块适配深度特征。

**直接基线与差异**：
- **OAL**: 父方法，F-OAL将其批处理矩阵求逆改为增量RLS更新，实现真正在线学习
- **EASE**: 最强exemplar-free竞争者，F-OAL在CIFAR-100超过其0.3%，但Tiny-ImageNet落后0.8%；核心差异是EASE仍用反向传播能量分类器
- **DVC**: 最强replay-based竞争者，F-OAL在4/6数据集超越；差异在于DVC需5000样本缓冲和梯度训练
- **SLCA**: 另一exemplar-free方法，F-OAL训练速度快约10倍且内存为其1/5

**改动槽位**（slots changed）：
- **training_recipe**: 反向传播 → 前向-only递归最小二乘
- **architecture**: 标准线性分类器 → Analytic Classifier with $\mathbf{R}$矩阵
- **data_pipeline**: 直接特征 → Smooth Projection随机投影
- **inference_strategy**: 单特征流 → Feature Fusion双路融合
- **objective**: 交叉熵最小化 → 无显式损失，解析闭式解

**后续方向**：（1）将前向-only范式扩展至编码器微调，打破冻结backbone限制；（2）结合自监督预训练提升特征质量；（3）探索更大规模类别增量（如ImageNet-1K分1000阶段）的$\mathbf{R}$矩阵可扩展性。

**标签**: 视觉(vision) / 增量学习(class-incremental learning) / 在线学习(online learning) / 解析学习(analytic learning) / 无梯度(gradient-free) / 低内存(low-memory) / 细粒度识别(fine-grained)

