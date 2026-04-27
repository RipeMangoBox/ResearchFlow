---
title: Towards Generalizable Trajectory Prediction using Dual-Level Representation Learning and Adaptive Prompting
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- 自适应提示的双层轨迹预测表示学习
- PerReg / PerReg+
acceptance: poster
cited_by: 5
method: PerReg / PerReg+
baselines:
- 视觉傅里叶提示微调VFPT_Visual_Fourier_P
---

# Towards Generalizable Trajectory Prediction using Dual-Level Representation Learning and Adaptive Prompting

**Topics**: [[T__Autonomous_Driving]], [[T__Self-Supervised_Learning]], [[T__Domain_Adaptation]] | **Method**: [[M__PerReg_-_PerReg+]] | **Datasets**: nuScenes, nuScenes Multi-Dataset, AV2, WOMD

| 中文题名 | 自适应提示的双层轨迹预测表示学习 |
| 英文题名 | Towards Generalizable Trajectory Prediction using Dual-Level Representation Learning and Adaptive Prompting |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2501.04815) · [Code] · [Project] |
| 主要任务 | Motion Forecasting / 运动预测（自动驾驶场景下的多智能体轨迹预测） |
| 主要 baseline | Forecast-MAE [6]（直接对比，primary baseline）、AutoBot [24]、MTR [14]、Forecast-PEFT [2]（最相关PEFT工作） |

> [!abstract] 因为「运动预测预训练模型在跨数据集迁移时需要全参数微调，计算成本高且易破坏预训练表示」，作者在「Forecast-MAE」基础上改了「引入Perceiver-based双层表示学习与自适应提示调优（Adaptive Prompt Tuning），以冻结主干+优化提示参数实现参数高效迁移」，在「WOMD→nuScenes迁移学习」上取得「B-FDE 2.27 vs PT(WOMD) 2.75，降低0.48」

- **WOMD单数据集训练**：PerReg+ B-FDE 2.09，优于Forecast-MAE+（2.29）0.20，优于MTR（2.13）0.04
- **nuScenes迁移学习（WOMD→nuScenes, Full fine-tuning）**：B-FDE 2.27，优于Prompt Tuning迁移（2.53）0.26
- **nuScenes多数据集预训练**：minADE 0.84，与Forecast-MAE+持平，优于MTR（0.85）0.01

## 背景与动机

自动驾驶中的运动预测（Motion Forecasting）任务要求模型根据历史观测推断周围车辆、行人等交通参与者的未来轨迹。当前主流范式是先在大规模数据上进行自监督预训练，再在目标数据集上微调。然而，一个核心痛点是：当预训练数据与目标场景存在域差异（如不同城市的道路拓扑、交通密度、传感器配置）时，标准做法需要全参数微调（full fine-tuning）——这不仅计算开销大，还容易覆盖预训练学到的通用表示，导致灾难性遗忘。

现有方法如何应对此问题？**Forecast-MAE** [6] 提出用Masked Autoencoder进行运动预测预训练，通过随机掩码历史轨迹帧并重建来学习通用表示，但其微调阶段仍需更新全部参数；**MTR** [14] 采用Motion Transformer架构，结合全局意图定位与局部运动细化，在单数据集上表现强劲，却未系统探索跨数据集迁移；**AutoBot** [24] 聚焦多智能体信息融合与轨迹聚合，同样依赖全参数训练。这些方法的根本局限在于：预训练与微调之间缺乏参数高效的适配机制——一旦面对新域，要么全量调整破坏预训练收益，要么从头训练浪费数据。

更关键的是，视觉与NLP领域已广泛验证的**Prompt Tuning**（如Visual Prompt Tuning）在运动预测中尚未有效落地。Forecast-PEFT [2]虽专门针对运动预测提出参数高效微调，但据作者所述，其未在本文可见对比中出现。这留下一个明确缺口：如何设计适合时空轨迹数据的提示机制，实现"冻结主干、轻量适配"的跨域泛化？本文正是填补这一缺口——提出基于Perceiver的双层表示架构，配合自适应提示调优，使预训练模型无需改动核心参数即可适配新数据集。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/67d6b84c-f0d0-454d-bd5e-49ba666cac94/figures/Figure_1.png)
*Figure 1 (pipeline): Comparison between Masked Autoencoder (MAE) and our pretraining strategy with Prefixes.*



## 核心创新

核心洞察：运动预测中的跨域迁移可以通过**双层粒度表示（场景级+智能体级）与可学习提示令牌（learnable prompt tokens）**的协同实现参数高效适配，因为轨迹数据天然具有层次化结构（全局场景上下文与局部智能体动力学），而提示调优能在不修改预训练表示的前提下注入域特定信息，从而使大规模多数据集预训练后的模型以极低成本迁移到任意目标域成为可能。

| 维度 | Baseline (Forecast-MAE) | 本文 (PerReg/PerReg+) |
|:---|:---|:---|
| **架构** | 标准Transformer encoder，单层级表示 | Perceiver-based双层表示学习（场景级+智能体级） |
| **微调策略** | 全参数微调（所有θ可优化） | 自适应提示调优：冻结主干，仅优化提示参数p_k与预测头 |
| **训练范式** | 单数据集预训练或直接预训练 | 多数据集联合预训练 + 提示调优迁移 + 可选全微调 |
| **跨域适配成本** | 高（需更新全部参数） | 极低（仅提示参数<1%总参数量） |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/67d6b84c-f0d0-454d-bd5e-49ba666cac94/figures/Figure_2.png)
*Figure 2 (architecture): Proposed Perceiver architecture for trajectory prediction with Dual-Level Representation Learning and Adaptive Prompting.*



PerReg的整体数据流可分为四个阶段，形成"预训练→提示选择→冻结推理→可选精调"的闭环：

1. **Dual-Level Representation Learner（双层表示学习器）**：输入原始场景特征与智能体历史轨迹，并行构建**场景级表示**（全局道路拓扑、交通流上下文）与**智能体级表示**（局部运动模式、交互关系），输出层次化多粒度特征用于预训练。

2. **PerReg Encoder（预训练主干）**：以Perceiver架构处理上述双层表示，通过cross-attention机制压缩高维时空输入为紧凑隐变量。该模块在预训练阶段通过重建任务优化，在下游适配时**完全冻结**。

3. **Adaptive Prompt Selection（自适应提示选择）**：根据目标域标识（如nuScenes/AV2/WOMD），从提示库中选择对应可学习提示令牌p_k，或初始化新域提示。提示与输入特征拼接后送入冻结encoder——这是参数高效适配的关键阀门。

4. **Prediction Head（预测头）**：接收冻结encoder输出的条件化表示，输出未来轨迹分布Ŷ。在提示调优阶段仅优化p_k与预测头；在PerReg+的完整迁移策略中，可先提示调优再执行轻量全微调。

```
Raw Scene & Agent Features
        ↓
[Dual-Level Representation Learner] ──→ Scene-level reps
        ↓                              └── Agent-level reps
[PerReg Encoder] (frozen in adaptation)
        ↑
[Adaptive Prompt p_k] (selectable per domain)
        ↓
[Prediction Head] ──→ Predicted Trajectories Ŷ
        ↓
{Optional: Full Fine-tuning for PerReg+}
```

与Forecast-MAE的核心差异在于：本文将"表示学习"显式拆分为双层，并将"域适配"从全局参数优化降级为提示令牌的选择与优化，大幅降低迁移门槛。

## 核心模块与公式推导

### 模块 1: 自适应提示调优（Adaptive Prompt Tuning）——对应框架图右侧适配分支

**直觉**：视觉与语言模型中，提示调优通过在输入前添加可学习令牌来引导冻结模型行为；本文将其扩展至时空轨迹域，使同一预训练主干通过不同提示"切换"到不同数据集特性。

**Baseline 公式** (Forecast-MAE 全参数微调):
$$\theta^* = \text{arg}\min_{\theta} \mathcal{L}(f_\theta(X), Y)$$
符号: $\theta$ = 模型全部可训练参数, $X$ = 输入历史轨迹与场景特征, $Y$ = 未来轨迹真值, $\mathcal{L}$ = 预测损失（如负对数似然或L2回归）。

**变化点**：全参数微调在跨域时更新所有表示，易破坏预训练通用知识；且每新域需存储完整模型副本，存储成本随数据集线性增长。

**本文公式（推导）**:
$$\text{Step 1: 提示注入} \quad Z = \text{Concat}[p_k; X] \quad \text{将域特定提示} p_k \text{ 与输入拼接}$$
$$\text{Step 2: 冻结前向} \quad H = f_{\text{frozen}}(Z) = f_{\text{frozen}}(p_k, X) \quad \text{主干参数完全固定}$$
$$\text{Step 3: 仅优化提示与头部} \quad p_k^*, \phi^* = \text{arg}\min_{p_k, \phi} \mathcal{L}(g_\phi(H), Y)$$
$$\text{最终推理形式}: \hat{Y} = g_\phi(f_{\text{frozen}}(X, p_k^*))$$
其中 $\phi$ 为预测头参数，$p_k \in \mathbb{R}^{L_p \times d}$ 为第k个域的提示令牌（通常 $L_p \ll$ 序列长度，参数量占比<1%）。

**对应消融**：Table 4（Ablation Study）显示移除自适应提示或改用固定提示的性能下降（具体Δ值。

---

### 模块 2: 双层表示学习（Dual-Level Representation Learning）——对应框架图左侧编码器

**直觉**：轨迹预测既需理解"场景允许什么"（道路结构、交通规则），也需建模"智能体会做什么"（运动学约束、社交交互）；单层表示难以同时捕获这两种不同时间尺度的模式。

**Baseline 公式** (Forecast-MAE 单层MAE):
$$H = \text{Encoder}(\text{Mask}(X_{\text{flat}}))$$
$$\mathcal{L}_{\text{MAE}} = \| \text{Decoder}(H) \odot M - X_{\text{target}} \odot M \|^2$$
符号: $X_{\text{flat}}$ = 展平后的统一序列表示, $M$ = 掩码矩阵, $\odot$ = 逐元素乘法。Forecast-MAE将所有输入（场景+智能体）视为同质令牌序列处理。

**变化点**：单层表示对场景上下文与智能体动态使用相同抽象级别，导致：（1）场景级慢变特征被强制与高频运动特征竞争注意力；（2）跨智能体交互与场景-智能体关系纠缠不清。

**本文公式（推导）**:
$$\text{Step 1: 场景级编码} \quad H^{\text{scene}} = \text{Perceiver}_{\text{scene}}(X^{\text{map}}, X^{\text{static}}) \quad \text{聚合道路拓扑、静态障碍物}$$
$$\text{Step 2: 智能体级编码} \quad H^{\text{agent}}_i = \text{Perceiver}_{\text{agent}}(X^{\text{hist}}_i, \{X^{\text{hist}}_j\}_{j \neq i}) \quad \text{编码第}i\text{个智能体及其交互}$$
$$\text{Step 3: 双层融合} \quad H^{\text{fused}} = \text{CrossAttn}(H^{\text{agent}}, H^{\text{scene}}) \quad \text{智能体查询场景上下文}$$
$$\text{Step 4: 预训练重建目标}: \mathcal{L}_{\text{PerReg}} = \lambda_1 \| \hat{X}^{\text{scene}} - X^{\text{scene}} \|^2 + \lambda_2 \sum_i \| \hat{X}^{\text{agent}}_i - X^{\text{agent}}_i \|^2$$
**重归一化/正则化**：Perceiver架构通过latent bottleneck压缩输入，天然起到正则化作用；双层损失权重 $\lambda_1, \lambda_2$ 平衡场景重建与轨迹重建的重要性。

**对应消融**：Figure 4（Impact of Reconstruction Query Drop Rate）显示不同查询丢弃率对验证集的影响，暗示双层查询机制中的采样策略对泛化至关重要。

---

### 模块 3: 迁移学习策略（PerReg+ 完整流程）——对应Table 6策略对比

**直觉**：提示调优虽高效，但复杂域迁移可能仍需微调主干；本文提出"先提示、后精调"的两阶段策略，兼顾效率与性能上限。

**Baseline 策略** (直接预训练 / 标准迁移):
$$\text{Option A: 源域预训练} \rightarrow \text{目标域全微调} \quad \text{（高成本，易遗忘）}$$
$$\text{Option B: 源域预训练} \rightarrow \text{目标域提示调优} \quad \text{（低成本，可能欠拟合）}$$

**本文策略（推导）**:
$$\text{Phase 1: 多数据集预训练} \quad \theta_{\text{pre}} = \text{arg}\min_\theta \sum_{d \in \mathcal{D}} \mathcal{L}_{\text{PerReg}}(f_\theta(X_d))$$
$$\text{Phase 2: 提示调优迁移} \quad p_k^* = \text{arg}\min_{p_k} \mathcal{L}(f_{\theta_{\text{pre}}, \text{frozen}}(X_{\text{target}}, p_k), Y_{\text{target}})$$
$$\text{Phase 3 (PerReg+): 轻量全微调} \quad \theta^*, \phi^* = \text{arg}\min_{\theta, \phi} \mathcal{L}(f_{\theta}(X_{\text{target}}, p_k^*), Y_{\text{target}}) \quad \text{可选，解冻主干}$$

**对应消融**：Table 6显示PT(WOMD→nuScenes) + Full fine-tuning在B-FDE上达2.27，优于仅PT(WOMD→nuScenes)的2.53（Δ=0.26）与直接PT(nuScenes)的2.62，证明两阶段策略的有效性。

## 实验与分析


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/67d6b84c-f0d0-454d-bd5e-49ba666cac94/figures/Table_2.png)
*Table 2 (quantitative): Quantitative results. Performance metrics are presented for Argoverse 2 motion forecasting and nuScenes.*



本文在三个主流自动驾驶运动预测基准上评估：nuScenes、Argoverse 2 (AV2)、Waymo Open Motion Dataset (WOMD)。核心评估指标为minADE（平均位移误差）与B-FDE（最终位移误差的best-of-K）。如Table 2所示，PerReg/PerReg+在单数据集与多数据集设置下与Forecast-MAE、AutoBot、MTR进行全面对比。

**单数据集与多数据集预训练**：在WOMD上，PerReg+取得最强表现——B-FDE 2.09，较Forecast-MAE+（2.29）降低0.20，较MTR（2.13）降低0.04；minADE 0.65，较Forecast-MAE+（0.74）降低0.09，为所有对比方法中最佳。然而在nuScenes与AV2上，PerReg+并非全面领先：nuScenes的B-FDE 2.32仍逊于MTR的2.27；AV2的B-FDE 2.12与minADE 0.76均落后于Forecast-MAE+（2.04/0.74）与MTR（1.99/0.82未直接优于）。这表明PerReg+的优势在WOMD域更为显著，可能与WOMD的数据规模或场景特性更契合双层表示设计有关。


![Table 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/67d6b84c-f0d0-454d-bd5e-49ba666cac94/figures/Table_4.png)
*Table 4 (ablation): Ablation Study. Quantitative analysis of the impact of individual components.*



**迁移学习**：Table 6（Transfer Pre-training vs. Direct Pre-training）揭示本文核心卖点。以WOMD为源域、nuScenes为目标域：直接提示调优迁移（PT WOMD→nuScenes）B-FDE为2.53；在此基础上执行Full fine-tuning（即PerReg+完整流程）后降至2.27，改善0.26。对比之下，仅在WOMD上提示调优后直接评估（PT WOMD）B-FDE高达2.75，证明跨域迁移的必要性；而直接在nuScenes上预训练+提示调优（PT nuScenes）为2.62，说明多数据集预训练带来的初始化优势。


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/67d6b84c-f0d0-454d-bd5e-49ba666cac94/figures/Figure_4.png)
*Figure 4 (ablation): Impact of Reconstruction Query Drop Rate on the validation set.*



**关键消融**：Figure 4展示Reconstruction Query Drop Rate对验证集性能的影响，表明适当的查询丢弃可防止过拟合、促进泛化。Table 4的消融研究（具体数值。

**公平性审视**：本文存在若干比较局限。最直接相关的baseline **Forecast-PEFT** [2]——同为运动预测参数高效微调——未在可见表格中出现对比；SSL-Lanes [18]、PreTraM [21]、Traj-MAE [14]等预训练方法亦缺失。MTR在nuScenes与AV2部分指标上仍优于PerReg+，说明本文并非全面SOTA。此外，作者未报告计算成本（FLOPs、训练时间）、推理延迟、或提示参数量占比的精确数字，其"高效"声称缺乏定量支撑。失败模式方面，作者未明确讨论双层表示在稀疏场景（如高速公路简单跟车）是否可能过度复杂化。

## 方法谱系与知识库定位

PerReg属于**运动预测预训练谱系**，直接parent为**Forecast-MAE** [6]——后者建立MAE自监督预训练范式，PerReg继承其重建预训练目标但进行三处关键改造：

| 改动槽位 | 父方法 (Forecast-MAE) | 本文修改 |
|:---|:---|:---|
| **架构** | 标准Transformer encoder | Perceiver-based + 双层表示学习 |
| **微调策略** | 全参数微调 | 自适应提示调优（冻结主干） |
| **训练配方** | 单数据集/直接预训练 | 多数据集联合预训练 + 两阶段迁移 |

**直接关联工作差异**：
- **Forecast-PEFT** [2]：最相关PEFT baseline，专为运动预测设计，但本文未直接对比；PerReg差异在于引入"双层表示"而非仅微调策略优化。
- **Visual Prompt Tuning** [13]：NLP/ViT提示调优的起源；PerReg将其适配至时空轨迹域，提示与运动特征拼接而非图像patch。
- **Point-PEFT** [22]：3D点云PEFT方法；PerReg借鉴其参数高效思想，但应用于轨迹序列而非点云表示。

**后续方向**：
1. **提示组合与复用**：当前每域独立提示，未来可探索跨域提示插值或组合，实现零样本新域适配；
2. **更细粒度层次**：双层（场景/智能体）或可扩展至三层（道路-场景-智能体-交互），捕捉更复杂交通层级；
3. **计算效率实证**：补充提示参数量、推理延迟、存储节省的精确测量，强化"高效"声称的可信度。

**标签**：
- **模态 (modality)**：时空序列 / 自动驾驶传感器融合
- **范式 (paradigm)**：自监督预训练 → 参数高效微调（PEFT）
- **场景 (scenario)**：多智能体运动预测 / 跨域泛化
- **机制 (mechanism)**：提示调优 (Prompt Tuning) / 双层表示学习 / Perceiver架构
- **约束 (constraint)**：参数高效 / 冻结主干 / 多数据集联合训练

## 引用网络

### 直接 baseline（本文基于）

- [[P__视觉傅里叶提示微调VFPT_Visual_Fourier_P]] _(方法来源)_: Visual Prompt Tuning is foundational parameter-efficient fine-tuning method for 

