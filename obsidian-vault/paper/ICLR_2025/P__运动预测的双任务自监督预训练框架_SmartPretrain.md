---
title: 'SmartPretrain: Model-Agnostic and Dataset-Agnostic Representation Learning for Motion Prediction'
type: paper
paper_level: C
venue: ICLR
year: 2025
paper_link: null
aliases:
- 运动预测的双任务自监督预训练框架
- SmartPretrain
acceptance: Poster
cited_by: 1
code_url: https://github.com/youngzhou1999/SmartPretrain
method: SmartPretrain
---

# SmartPretrain: Model-Agnostic and Dataset-Agnostic Representation Learning for Motion Prediction

[Code](https://github.com/youngzhou1999/SmartPretrain)

**Topics**: [[T__Self-Supervised_Learning]], [[T__Autonomous_Driving]], [[T__Contrastive_Learning]] | **Method**: [[M__SmartPretrain]] | **Datasets**: Argoverse

| 中文题名 | 运动预测的双任务自监督预训练框架 |
| 英文题名 | SmartPretrain: Model-Agnostic and Dataset-Agnostic Representation Learning for Motion Prediction |
| 会议/期刊 | ICLR 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2410.08669) · [Code](https://github.com/youngzhou1999/SmartPretrain) · [Project](待补充) |
| 主要任务 | 自动驾驶场景下的运动预测（motion prediction）自监督预训练 |
| 主要 baseline | HiVT, HPNet, QCNet, Forecast-MAE |

> [!abstract] 因为「运动预测模型预训练受限于单数据集和特定架构，无法充分利用多源数据」，作者在「Forecast-MAE 的 MAE 预训练范式」基础上改了「引入对比学习+重建的双任务目标、双分支动量架构、以及跨数据集预训练能力」，在「Argoverse / Argoverse 2」上取得「HiVT minFDE 从 0.969 降至 0.929（-4.1%），QCNet minFDE 从 1.253 降至 1.191（-4.9%）」

- **HiVT + SmartPretrain** 在 Argoverse 验证集：minADE 0.661→0.644 (-2.6%), minFDE 0.969→0.929 (-4.1%), MR 0.092→0.086 (-6.5%)
- **QCNet + SmartPretrain** 在 Argoverse 2 验证集：minADE 0.72→0.696 (-3.3%), minFDE 1.253→1.191 (-4.9%), MR 0.157→0.145 (-7.6%)
- **Forecast-MAE backbone + SmartPretrain** vs Forecast-MAE 自有预训练：minFDE 1.409→1.372 (-2.6%)

## 背景与动机

运动预测是自动驾驶系统的核心模块，其任务是根据历史轨迹和场景地图预测周围车辆、行人等交通参与者的未来轨迹。当前主流方法（如 HiVT、QCNet）通常直接在目标数据集上从头训练，未能充分利用大规模异构驾驶数据的表征潜力。自监督预训练已在计算机视觉和自然语言处理领域取得巨大成功，但在运动预测领域仍处于早期阶段。

现有方法如何处理这一问题？**Forecast-MAE** 首次将 Masked Autoencoder（MAE）引入运动预测预训练，通过随机掩码历史轨迹帧并重建原始信号来学习表征。然而，Forecast-MAE 仅依赖单一重建任务，且其设计紧密耦合于特定数据集格式，难以扩展至多源异构数据。**HiVT、QCNet、HPNet** 等预测模型虽架构各异，但均遵循「单数据集监督训练」范式，缺乏系统的预训练机制。

这些方法的根本局限在于三方面：**第一**，预训练目标单一——仅 MAE 重建无法同时学习判别性表征；**第二**，架构绑定——预训练流程与特定模型结构或数据格式强耦合，无法即插即用；**第三**，数据孤岛——无法聚合多个数据集的互补信息（如 Argoverse 的高速公路场景与 Argoverse 2 的复杂交互场景）。具体而言，当自动驾驶公司积累大量异构驾驶日志时，现有方法无法跨数据集预训练，导致数据价值浪费。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b8f4e207-296d-44e8-aa92-352eaac06999/figures/Figure_1.png)
*Figure 1 (motivation): Illustration comparing existing motion prediction pre-training paradigm with ours.*



本文提出 SmartPretrain，通过「对比学习 + 掩码重建」的双任务框架和模型无关、数据集无关的设计，解决上述局限，使任意运动预测模型都能从多源数据中获益。

## 核心创新

核心洞察：**运动预测的预训练应当同时学习「判别性场景表征」和「生成性轨迹重建」**，因为单一重建任务仅能捕捉数据分布的局部结构，而对比学习能显式建模场景间的语义差异，从而使跨数据集、跨模型的通用预训练成为可能。

| 维度 | Baseline (Forecast-MAE) | 本文 (SmartPretrain) |
|:---|:---|:---|
| 预训练目标 | 仅掩码重建损失 $L_{MAE}$ | 对比损失 + 重建损失 $L_c + \lambda L_r$ |
| 网络架构 | 单分支编码器-解码器 | 双分支：online 分支 + momentum 分支（EMA 更新） |
| 数据兼容性 | 单数据集，格式绑定 | 多数据集联合预训练，dataset-agnostic |
| 模型兼容性 | 特定 backbone | 任意运动预测编码器即插即用 |
| 投影头 | 无 | Projector + Contrastive Predictor（2-layer MLP） |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b8f4e207-296d-44e8-aa92-352eaac06999/figures/Figure_2.png)
*Figure 2 (pipeline): Overview of our model-agnostic and dataset-agnostic pre-training pipeline.*



SmartPretrain 的整体数据流如下：

**输入**：原始轨迹序列（历史观测）+ 高清地图数据，来自一个或多个异构数据集（如 Argoverse、Argoverse 2、nuScenes 等）。

**模块 1: Motion Prediction Encoder（在线编码器）**
- 输入：原始轨迹与地图数据
- 输出：编码后的场景表征向量
- 角色：可替换为任意现有预测模型的编码器（HiVT/QCNet/HPNet/Forecast-MAE），实现 model-agnostic

**模块 2: Projector（在线投影头）**
- 输入：编码器输出的场景表征
- 输出：投影后的潜在特征
- 角色：2-layer MLP + BatchNorm，将表征映射到适合对比学习的空间

**模块 3: Contrastive Predictor（对比预测器）**
- 输入：在线投影特征
- 输出：预测表征（用于与 momentum 分支对比）
- 角色：2-layer MLP + BatchNorm，实现 BYOL 风格的非对称预测

**模块 4: Momentum Branch（动量分支）**
- 输入：同批次数据（无梯度）
- 输出：EMA 更新的目标表征
- 角色：编码器 + Projector 的指数移动平均副本，为对比学习提供稳定目标

**模块 5: Trajectory Decoder（轨迹解码器）**
- 输入：在线投影特征
- 输出：重建的完整轨迹序列
- 角色：2-layer MLP + LayerNorm，执行掩码重建预文本任务

**输出**：预训练完成后，在线编码器作为初始化权重，接入下游任务的预测头进行微调。

```
[多源数据集] → [Encoder_θ] → [Projector_θ] ─┬→ [Contrastive Predictor] ─┐
      ↑                                       │                           ├──→ L_c (对比)
      └─ EMA ── [Encoder_ξ] → [Projector_ξ] ─┘←──────────────────────────┘
                                              │
                                              └→ [Trajectory Decoder] → L_r (重建)
                                                                     
θ: 在线参数（梯度更新）    ξ: 动量参数（EMA: ξ ← m·ξ + (1-m)·θ）
```

## 核心模块与公式推导

### 模块 1: 双任务联合损失函数（对应框架图中心）

**直觉**：单一预文本任务无法同时获得判别性和重建性表征，需显式组合两种学习信号。

**Baseline 公式** (Forecast-MAE): 
$$L_{MAE} = \mathbb{E}_{x \sim \mathcal{D}} \|x - \text{Decoder}(\text{Mask}(\text{Encoder}(x)))\|^2$$
符号: $x$ = 完整轨迹序列, $\text{Mask}(\cdot)$ = 随机时间帧掩码, $\mathcal{D}$ = 单一数据集。

**变化点**：Forecast-MAE 仅优化重建误差，缺乏显式建模不同场景间关系的机制；且仅适用于单数据集 $\mathcal{D}$。

**本文公式（推导）**:
$$\text{Step 1}: \quad L_c = -\mathbb{E}_{x_i, x_j \sim \mathcal{D}_{mix}} \left[ \frac{\exp(\text{sim}(z_i^{pred}, z_j^{mom}) / \tau)}{\sum_{k} \exp(\text{sim}(z_i^{pred}, z_k^{mom}) / \tau)} \right]$$
加入了对比损失项，其中 $z_i^{pred} = \text{Predictor}(\text{Projector}(\text{Encoder}_\theta(x_i)))$ 为在线分支预测，$z_j^{mom} = \text{Projector}(\text{Encoder}_\xi(x_j))$ 为动量分支目标，$\tau$ 为温度系数。该设计使模型学习场景间的相对相似性。

$$\text{Step 2}: \quad L_r = \mathbb{E}_{x \sim \mathcal{D}_{mix}} \|x - \text{Decoder}(\text{Projector}(\text{Encoder}_\theta(\tilde{x})))\|^2$$
保留重建任务但作用于掩码输入 $\tilde{x}$，确保模型保留生成完整轨迹的能力。

$$\text{最终}: \quad L = L_c + \lambda L_r$$
其中 $\lambda$ 为平衡超参数，$\mathcal{D}_{mix} = \text{bigcup}_m \mathcal{D}_m$ 为多个异构数据集的混合分布。

**对应消融**：Table 4 显示仅使用 $L_c$ 或仅使用 $L_r$ 时性能均低于联合训练，验证双任务的互补性。（具体 Δ 数值待补充）

---

### 模块 2: 动量分支与 EMA 更新（对应框架图右侧）

**直觉**：对比学习需要稳定的目标表征以避免模型坍塌，但直接共享梯度会导致表示退化。

**Baseline 公式** (SimSiam/BYOL 风格):
$$\theta \leftarrow \theta - \eta \nabla_\theta L_c, \quad \xi = \theta \text{ (直接共享，不稳定)}$$

**变化点**：直接参数共享或停止梯度策略在运动预测的高维结构化输出上效果不佳；需引入缓慢更新的动量编码器提供一致目标。

**本文公式（推导）**:
$$\text{Step 1}: \quad \xi^{(t)}_{enc} = m \cdot \xi^{(t-1)}_{enc} + (1-m) \cdot \theta^{(t)}_{enc}$$
编码器动量更新，$m \in [0.996, 0.999]$ 为动量系数，保证目标表征的时序一致性。

$$\text{Step 2}: \quad \xi^{(t)}_{proj} = m \cdot \xi^{(t-1)}_{proj} + (1-m) \cdot \theta^{(t)}_{proj}$$
投影头同步 EMA 更新，确保对比空间的对齐。

$$\text{最终目标}: \quad z^{mom} = \text{Projector}_\xi(\text{Encoder}_\xi(x)), \quad \nabla_\xi L_c \equiv 0$$
动量分支不参与梯度反向传播，仅作为对比学习的锚点目标。

**对应消融**：

---

### 模块 3: 数据集无关的多源混合策略（对应框架图输入层）

**直觉**：不同数据集的坐标系、采样频率、场景分布各异，直接拼接会导致域冲突。

**Baseline 公式** (标准迁移学习):
$$\mathcal{D}_{eff} = \mathcal{D}_{source} \text{xrightarrow}{\text{预训练}} \mathcal{D}_{target} \text{xrightarrow}{\text{微调}}$$
符号: 源域与目标域分离，需顺序执行。

**变化点**：传统迁移学习存在负迁移风险，且无法同时利用多个数据集的互补信息；需设计 dataset-agnostic 的混合采样。

**本文公式（推导）**:
$$\text{Step 1}: \quad p(\mathcal{D}_m) = \frac{|\mathcal{D}_m|^{\alpha}}{\sum_k |\mathcal{D}_k|^{\alpha}}$$
采用大小相关的采样概率，$\alpha \in [0,1]$ 控制平衡程度，$\alpha=0$ 为均匀采样，$\alpha=1$ 为按大小比例采样。

$$\text{Step 2}: \quad x_{batch} \sim \text{bigotimes}_{m} \left( p(\mathcal{D}_m) \cdot \mathcal{U}(\mathcal{D}_m) \right)$$
每批次按混合概率从各数据集独立采样，通过统一的数据预处理接口（标准化坐标、统一时间分辨率）实现 dataset-agnostic。

$$\text{最终}: \quad \mathcal{D}_{mix} = \{(x, y, map) \text{mid} \text{unified format}\}$$

**对应消融**：Figure 3 显示随着预训练数据规模扩大（单数据集 → 跨数据集迁移 → 多数据集扩展），HiVT 在 Argoverse 验证集上的 minFDE/minADE/MR 持续下降，验证数据扩展的有效性。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b8f4e207-296d-44e8-aa92-352eaac06999/figures/Table_1.png)
*Table 1 (comparison): Performance comparison with multiple baselines on Argoverse and Argoverse 2 (Format: minADE / minFDE↓).*



本文在 Argoverse 和 Argoverse 2 的验证集与测试集上评估 SmartPretrain，覆盖四个主流 backbone：HiVT、HPNet、QCNet、Forecast-MAE。如 Table 1 所示，SmartPretrain 在所有设置下均带来一致提升。以 Argoverse 验证集为例，HiVT 基线的 minFDE 为 0.969，经 SmartPretrain 预训练后降至 0.929，相对改善 -4.1%；MR 从 0.092 降至 0.086（-6.5%）。在 Argoverse 2 验证集上，QCNet 的 minFDE 从 1.253 降至 1.191（-4.9%），MR 从 0.157 降至 0.145（-7.6%），显示该方法对更复杂、更长时域的预测场景同样有效。

特别地，当使用 Forecast-MAE 作为 backbone 时（Table 2），SmartPretrain 的 minFDE 为 1.372，相比 Forecast-MAE 从头训练的 1.436 降低 -4.5%，相比 Forecast-MAE 自有 MAE 预训练的 1.409 仍降低 -2.6%。这一直接对比证明，SmartPretrain 的双任务框架优于单一 MAE 预训练策略。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/b8f4e207-296d-44e8-aa92-352eaac06999/figures/Figure_3.png)
*Figure 3 (ablation): We evaluate the performance of SmartPretrain with multiple sizes of pre-training data. We observe consistent performance boosts as pre-training data scales, while pre-training on more diverse tasks further improves performance.*



消融实验方面，Figure 3 展示了预训练数据规模的 Scaling 效应：随着从「无预训练」到「单数据集」到「跨数据集迁移」再到「多数据集扩展」，性能单调提升。Table 4 的双任务消融表明，去掉对比学习 $L_c$ 或重建损失 $L_r$ 任一任务均导致性能下降（具体 Δ 数值待补充）。Table 5 进一步探索了不同重建目标的选择（如重建原始坐标 vs. 速度 vs. 加速度），发现重建归一化坐标最优。

**公平性检查**：作者坦诚由于投稿时仅有 Forecast-MAE 开源，未能与更多自监督预训练基线（如 SimCLR、MoCo 的运动预测适配版本）对比；HPNet 和 Forecast-MAE 因计算限制未进行多数据集扩展实验。训练成本方面，单数据集预训练需 8 张 A100 40GB GPU，多数据集扩展需 32 张 A100，预训练 128 epochs。未报告推理延迟开销，但预训练仅在微调前执行一次，不影响在线推理。

## 方法谱系与知识库定位

**方法谱系**：SmartPretrain 属于「运动预测自监督预训练」谱系，直接继承自 **Forecast-MAE**（MAE-based 预训练先驱），并融合 BYOL/SimSiam 的动量对比学习机制。从 Forecast-MAE 到 SmartPretrain 的演化涉及四个关键槽位变更：

| 变更槽位 | Forecast-MAE | SmartPretrain |
|:---|:---|:---|
| objective | 监督预测损失 / 纯 MAE 重建 | 对比损失 + 重建损失 |
| architecture | 单分支编码器-解码器 | 双分支 online+momentum + Projector + Predictor |
| training_recipe | 单阶段预训练 | 双任务预训练 → 下游微调 |
| data_curation | 单数据集 | 多数据集混合采样，dataset-agnostic |

**直接 baseline 差异**：
- **vs. HiVT/QCNet/HPNet（无预训练）**：SmartPretrain 提供即插即用的预训练权重，不改变原有架构
- **vs. Forecast-MAE**：增加对比学习任务、动量分支、多数据集能力，从单任务扩展为双任务框架

**后续方向**：(1) 将预训练扩展至端到端自动驾驶（联合感知-预测-规划）；(2) 引入时序掩码策略的动态自适应，替代固定掩码率；(3) 探索更大规模的多模态数据（含图像、LiDAR）的 dataset-agnostic 预训练。

**知识库标签**：
- **modality**: 轨迹序列 + 矢量化地图
- **paradigm**: 自监督预训练 / 对比学习 + 掩码重建
- **scenario**: 自动驾驶运动预测
- **mechanism**: 动量编码器 / EMA / 双任务学习
- **constraint**: model-agnostic / dataset-agnostic / 即插即用

