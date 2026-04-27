---
title: Availability-aware Sensor Fusion via Unified Canonical Space
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 统一规范空间的可用性感知传感器融合
- ASF (Availabilit
- ASF (Availability-aware Sensor Fusion)
- Availability-aware sensor fusion (A
acceptance: Poster
cited_by: 3
code_url: https://github.com/kaist-avelab/K-Radar
method: ASF (Availability-aware Sensor Fusion)
modalities:
- Image
- point cloud
- radar
paradigm: supervised
---

# Availability-aware Sensor Fusion via Unified Canonical Space

[Code](https://github.com/kaist-avelab/K-Radar)

**Topics**: [[T__Object_Detection]], [[T__Autonomous_Driving]] | **Method**: [[M__ASF]] | **Datasets**: K-Radar v1.0

> [!tip] 核心洞察
> Availability-aware sensor fusion (ASF) with unified canonical projection (UCP) and cross-attention across sensors along patches (CASAP) achieves superior robustness to sensor degradation and failure while maintaining low computational cost.

| 中文题名 | 统一规范空间的可用性感知传感器融合 |
| 英文题名 | Availability-aware Sensor Fusion via Unified Canonical Space |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2503.07029) · [Code](https://github.com/kaist-avelab/K-Radar) · [Project](https://arxiv.org/abs/2503.07029) |
| 主要任务 | 3D Object Detection, Autonomous Driving |
| 主要 baseline | L4DR, 3D-LRF, DPFT, RTNH, BEVFusion |

> [!abstract] 因为「现有传感器融合方法假设所有传感器持续可用，无法应对传感器失效或恶劣天气导致的性能骤降」，作者在「BEVFusion 统一鸟瞰图表示」基础上改了「Unified Canonical Projection (UCP) + Cross-Attention across Sensors Along Patches (CASAP) + Simultaneous Configuration Loss (SCL)」，在「K-Radar benchmark」上取得「APBEV 87.2% vs L4DR 77.5% (+9.7%)，AP3D 73.6% vs 53.5% (+20.1%)」

- **关键性能 1**: K-Radar v1.0 上 AP3D@IoU=0.5 达到 73.6%，相比 SOTA L4DR 提升 +20.1% 绝对值
- **关键性能 2**: Heavy snow 条件下 AP3D 66.4% vs L4DR 37.0%，差距扩大至 +29.4%
- **关键性能 3**: 统一权重模型 L+R 配置 APBEV 87.0%，与完整 C+L+R 的 87.2% 几乎持平，无需重新训练

## 背景与动机

自动驾驶系统依赖相机、LiDAR、4D Radar 等多传感器融合以实现全天候 3D 目标检测。然而实际部署中，传感器随时可能因恶劣天气（暴雨、大雪、雾霾）或硬件故障而失效——例如相机镜头被积雪遮挡、LiDAR 在浓雾中散射严重。此时，传统融合方法面临严峻挑战。

现有方法主要分为两类。**Deeply Coupled Fusion (DCF)** 直接将各传感器的特征图拼接后送入检测头，假设所有传感器始终可用；一旦某个传感器缺失，特征维度不匹配导致模型崩溃。**Sensor-wise Cross-attention Fusion (SCF)** 如 DPFT 引入跨传感器交叉注意力，允许不同传感器组合，但注意力在 token 级别进行，计算开销大且特征尺寸随传感器组合变化，难以保证检测头输入的一致性。BEVFusion 系列虽提出统一 BEV 表示，但未针对传感器可用性进行训练，部署时遇到传感器子集仍需重新训练或性能急剧下降。

核心痛点在于：**没有一种方法能在训练时覆盖所有传感器组合，同时保持统一的网络权重和一致的 feature map 尺寸，实现「一次训练、任意子集部署」**。本文提出的 ASF 正是为解决这一缺口，通过统一规范投影确保无论哪些传感器可用，融合特征尺寸始终一致，并借助同时配置损失让所有传感器组合在训练中得到优化。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/032fa6a1-79b1-4e9a-9b1a-f66299f869de/figures/Figure_1.png)
*Figure 1 (pipeline): Comparison of sensor fusion methods: DCF, SCP, and the proposed ASF, illustrating the unified canonical space and sensor dependency.*



## 核心创新

核心洞察：**将「传感器可用性」本身作为训练维度**，因为 BEV 空间的统一规范表示使得任意传感器子集都能投影到固定尺寸的特征图，从而使「单模型覆盖 7 种传感器组合、无需重新训练」成为可能。

| 维度 | Baseline (DCF/SCF/BEVFusion) | 本文 (ASF) |
|:---|:---|:---|
| 特征对齐 | 直接拼接或变长 token，尺寸随组合变化 | **UCP**: 所有传感器投影到固定尺寸 BEV feature map |
| 跨传感器交互 | Token 级全交叉注意力，计算量大 | **CASAP**: Patch 级交叉注意力，降低计算成本 |
| 训练目标 | 仅优化完整传感器配置 (C+L+R) | **SCL**: 同时优化全部 7 种传感器组合的损失和 |
| 部署灵活性 | 传感器变化需重新训练或专用模型 | **统一权重**: 任意子集直接推理，性能 graceful degradation |

## 整体框架



ASF 的完整数据流如下：

1. **Sensor-specific encoders**: 分别处理相机前视图像、LiDAR 点云、4D Radar 张量，输出各传感器的原始特征表示。
2. **Unified Canonical Projection (UCP)**: 将异构传感器特征统一投影到 BEV 空间，生成固定尺寸的特征图 $FM_{fused}$。**关键特性**: 无论输入是 C+L+R、L+R、仅 R 等任何组合，输出尺寸完全一致。
3. **Cross-Attention across Sensors Along Patches (CASAP)**: 在 patch 级别执行跨传感器交叉注意力，融合可用传感器的信息，输出增强后的 BEV 特征。
4. **Detection head**: 接收尺寸始终一致的融合特征，输出 3D 目标检测的分类与回归结果。

```
Raw Sensors (C/L/R) → Encoders → UCP → FMfused (fixed size) → CASAP → Fused BEV → Detection Head → 3D Predictions
         ↑___________________________↓
              任意传感器子集，统一尺寸
```

UCP 解决了 DCF/SCF 的维度不一致问题；CASAP 在保持融合能力的同时降低计算量；检测头因输入尺寸固定而无需任何修改即可适配所有传感器组合。

## 核心模块与公式推导

### 模块 1: Unified Canonical Projection (UCP)（对应框架图 UCP 模块）

**直觉**: 不同传感器的原始特征空间异构（相机 front-view、LiDAR 点云、Radar 张量），必须对齐到统一空间才能灵活融合，且尺寸不能随传感器组合变化。

**Baseline (DCF/BEVFusion)**: DCF 直接拼接各传感器特征图，尺寸为 $\sum_i H_i \times W_i \times C_i$，随组合变化；BEVFusion 统一投影到 BEV，但未处理传感器缺失时的填充/掩码一致性。

**变化点**: 本文要求 $FM_{fused}$ 的尺寸严格固定，与可用传感器集合无关。UCP 为每个传感器设计独立的投影分支，缺失传感器的分支输出零填充或学习到的默认特征，保证拼接后总尺寸恒定。

**本文公式**:
$$FM_{fused} = \text{Concat}\left[\text{UCP}_C(\mathbf{F}_C), \text{UCP}_L(\mathbf{F}_L), \text{UCP}_R(\mathbf{F}_R)\right] \in \mathbb{R}^{H \times W \times C_{fixed}}$$
其中 $\mathbf{F}_C, \mathbf{F}_L, \mathbf{F}_R$ 为各传感器编码器输出，当传感器 $s$ 不可用时，$\text{UCP}_s(\cdot) = \mathbf{0}$（或可学习的 placeholder）。输出 $H \times W \times C_{fixed}$ 对所有 7 种组合相同。

**对应消融**: Table 4 显示不同 fusion strategy 的对比，UCP 的固定尺寸设计是 CASAP 和统一检测头的前提。

---

### 模块 2: Cross-Attention across Sensors Along Patches (CASAP)（对应框架图 CASAP 模块）

**直觉**: Token 级交叉注意力（如 SCF/DPFT）计算复杂度高，且对 BEV 特征图的大尺寸不友好；patch 级注意力在保持跨传感器交互的同时显著降低计算量。

**Baseline (SCF/DPFT)**: 标准 cross-attention 在 token 级别进行：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
其中 $Q, K, V$ 为所有传感器 token 的拼接，长度为 $\sum_s N_s$，随传感器组合变化。

**变化点**: 将 BEV 特征图划分为空间 patch，在 patch 级别聚合跨传感器信息，而非逐 token。假设 BEV 空间划分为 $P \times P$ 的 patch grid，每个 patch 内聚合可用传感器的特征。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathbf{P}_{s}^{(i,j)} = \text{Pool}\left(FM_{fused}^{(i,j)} \text{ for sensor } s\right) \in \mathbb{R}^{d_{patch}}$$
对每个传感器 $s$ 和空间位置 $(i,j)$ 提取 patch 级特征。

$$\text{Step 2}: \quad \mathbf{Q}_{s}^{(i,j)} = W_Q \mathbf{P}_{s}^{(i,j)}, \quad \mathbf{K}_{s'}^{(i,j)} = W_K \mathbf{P}_{s'}^{(i,j)}, \quad \mathbf{V}_{s'}^{(i,j)} = W_V \mathbf{P}_{s'}^{(i,j)}$$
同位置不同传感器间计算注意力。

$$\text{Step 3}: \quad \text{Attn}_{s}^{(i,j)} = \sum_{s' \in \mathcal{S}_{avail}} \text{softmax}\left(\frac{\mathbf{Q}_{s}^{(i,j)} \mathbf{K}_{s'}^{(i,j)T}}{\sqrt{d_k}}\right) \mathbf{V}_{s'}^{(i,j)}$$
仅对可用传感器集合 $\mathcal{S}_{avail}$ 求和，缺失传感器自动不参与。

$$\text{最终}: \quad FM_{fused}^{out} = \text{Concat}\left[\text{FFN}(\text{Attn}_{s}^{(i,j)})\right]_{i,j} + FM_{fused}^{in}$$
带残差连接，输出与输入同尺寸。

**对应消融**: Table 2 显示 ASF (C+L+R) VRAM 1.6GB、FPS 13.5，相比 DPFT (C+R) 的 4.0GB/11.5FPS，内存降低 60% 且速度更快，验证 CASAP 的计算效率优势。

---

### 模块 3: Simultaneous Configuration Loss (SCL)（对应框架图 Training 部分）

**直觉**: 若仅在完整配置 C+L+R 上训练，模型从未见过传感器缺失场景，部署时必然失效；需让所有可能的传感器组合都参与训练优化。

**Baseline (标准训练)**: 
$$L_{base} = L_{cls}(\mathbf{y}, \hat{\mathbf{y}}_{C+L+R}) + L_{reg}(\mathbf{b}, \hat{\mathbf{b}}_{C+L+R})$$
仅对完整传感器配置计算损失。

**变化点**: 定义传感器组合集合 $SC = \{C, L, R, C+L, C+R, L+R, C+L+R\}$（共 7 种），对每种组合分别前向传播并求和损失。

**本文公式（推导）**:
$$\text{Step 1}: \quad L_s = L_{cls,s} + L_{reg,s} \quad \text{（对组合 } s \in SC \text{ 分别计算）}$$
每种组合使用 UCP 的对应输入（缺失传感器置零），通过共享权重网络。

$$\text{Step 2}: \quad L_{SCL} = \sum_{s \in SC} (L_{cls,s} + L_{reg,s})$$
所有 7 种组合损失直接相加，无额外权重调节。

$$\text{最终}: \quad \theta^* = \text{arg}\min_\theta L_{SCL}$$
单次优化覆盖全部可用性场景。

**对应消融**: Table 4 显示 ASF without SCL 性能明显低于完整 ASF，SCL 增强了对传感器缺失的鲁棒性；Table 3 验证所有 10 个 ASF 模型共享相同权重，C*+L+R（相机失效模拟）Sedan AP3D 仅下降 1.7% (79.3→77.6)，而 C+L*+R（LiDAR 失效）下降 20.4% (79.3→58.9)，说明 SCL 有效但 LiDAR 仍是最关键传感器。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/032fa6a1-79b1-4e9a-9b1a-f66299f869de/figures/Table_1.png)
*Table 1 (comparison): Performance comparison of 3D object detectors on the K-Radar benchmark under various weather conditions.*



本文在 **K-Radar v1.0** 上进行 SOTA 对比，在 **K-Radar v2.0** 上进行消融与传感器组合研究。K-Radar 是包含多种恶劣天气（晴、雨、雪、雾、冻雨）的 4D Radar 自动驾驶数据集，对传感器鲁棒性评估尤为关键。

**核心结果**: Table 1 显示，ASF (C+L+R) 在 K-Radar v1.0 上达到 APBEV@IoU=0.5 **87.2%**、AP3D@IoU=0.5 **73.6%**，相比前 SOTA **L4DR** 的 77.5%/53.5% 分别提升 **+9.7%** 和 **+20.1%**。这一优势在恶劣天气下进一步放大：Sleet 条件下 AP3D 67.5% vs 46.2% (+21.3%)，Heavy snow 条件下 66.4% vs 37.0% (**+29.4%**)，表明 ASF 的可用性感知设计对传感器退化具有显著鲁棒性。值得注意的是，ASF 的 L+R 配置（无相机）使用与 C+L+R **完全相同的网络权重**，APBEV 仍达 87.0%，几乎无损失，验证了「一次训练、任意部署」的核心能力。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/032fa6a1-79b1-4e9a-9b1a-f66299f869de/figures/Table_3.png)
*Table 3 (ablation): Ablation study of ASF under various sensor combinations and weather conditions on the K-Radar validation set.*



**消融实验**: Table 3 (K-Radar v2.0) 系统评估了 7 种传感器组合。完整配置 C+L+R 的 Sedan AP3D@IoU=0.3 为 79.3%，而 L-only 73.0%、R-only 47.3%、C-only 仅 14.8%——相机单独使用性能极差，说明 ASF 仍高度依赖 LiDAR/Radar。模拟传感器失效时，C*+L+R（相机遮挡）降至 77.6%（-1.7%），graceful degradation 成立；但 C+L*+R（LiDAR 失效）骤降至 58.9%（-20.4%），表明 LiDAR 是不可替代的骨干传感器。Table 4 进一步验证 SCL 的必要性：去掉 SCL 后性能明显下降（具体数值见表），证明多配置联合训练对部署鲁棒性至关重要。Table 5 显示不同天气下 CASAP 的注意力分配比例，辅助理解融合机制的行为。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/032fa6a1-79b1-4e9a-9b1a-f66299f869de/figures/Figure_3.png)
*Figure 3 (qualitative): Qualitative results of ASF for various sensor combinations under normal and adverse weather conditions.*



**效率对比**: Table 2 中 ASF (C+L+R) 在单张 RTX 3090 上占用 **1.6 GB VRAM**、运行 **13.5 FPS**，相比 DPFT (C+R) 的 4.0 GB / 11.5 FPS，内存效率提升 2.5 倍且速度更快；L+R 配置更达 **20.5 FPS**。这验证了 CASAP patch 级注意力的计算优势。

**公平性检查**: 主要对比的 L4DR、3D-LRF 是 K-Radar 上的近期 SOTA，比较公平。但 CMT、SimpleBEV、RCFusion 等 baseline 在 shallow_extract 中被提及却未在提供的实验细节中出现直接对比；此外 K-Radar v1.0（SOTA 对比）与 v2.0（消融实验）使用不同评估区域，跨版本比较需谨慎。作者坦诚相机单独性能仅 14.8%，方法对 LiDAR 依赖较重；且 SCL 的 7 倍前向传播在训练时间上的实际开销（虽声称 11 epochs）未充分澄清是否为 wall-clock 时间。

## 方法谱系与知识库定位

**方法家族**: Multi-sensor Fusion for 3D Object Detection in Autonomous Driving

**Parent method**: **BEVFusion** (Liu et al.; Liang et al.) — ASF 的 UCP 直接继承其「统一 BEV 表示」思想，但将「统一」从「空间对齐」推进到「可用性对齐」，并新增 CASAP 与 SCL 实现任意子集部署。

**直接 Baseline 差异**:
- **L4DR** [12]: 同期最强 LiDAR+4D Radar 融合，针对恶劣天气优化，但需固定传感器配置，无可用性感知机制。ASF 在其基础上 +9.7% APBEV / +20.1% AP3D。
- **DPFT** [?]: 开源 SCF 方法，token 级交叉注意力。ASF 的 CASAP 改为 patch 级，VRAM 从 4.0GB 降至 1.6GB。
- **3D-LRF** [?]: 另一 SOTA 融合方法，同样假设完整传感器可用。
- **DCF/SCF**: 概念性 baseline，ASF 同时解决了两者的核心缺陷（维度不一致、计算冗余）。

**改动槽位**: data_pipeline (UCP 替换拼接/变长 token) → architecture (CASAP 替换 token 注意力) → training_recipe (SCL 新增多配置联合训练) → inference_strategy (统一权重任意子集部署)。

**后续方向**:
1. 向 nuScenes、KITTI 等更多数据集泛化，验证 UCP 的跨数据集迁移性（当前仅限 K-Radar）。
2. 探索动态注意力权重或传感器可靠性估计，进一步降低对 LiDAR 的强依赖（相机单独 14.8% 仍有巨大提升空间）。
3. 将 SCL 思想扩展到时序融合或端到端自动驾驶规划，实现「可用性感知」在更大系统中的贯穿。

**标签**: 模态=image+point_cloud+radar | 范式=supervised_multi_sensor_fusion | 场景=autonomous_driving_adverse_weather | 机制=unified_BEV_projection+patch_cross_attention+multi_configuration_training | 约束=single_model_arbitrary_sensor_subset_no_retraining

