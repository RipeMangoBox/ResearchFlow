---
title: State Space Models for Event Cameras
type: paper
paper_level: C
venue: CVPR
year: 2024
paper_link: null
aliases:
- 事件相机的状态空间模型检测器
- S5-ViT / S4D-ViT
acceptance: Poster
cited_by: 31
code_url: https://github.com/uzh-rpg/ssms_event_cameras
method: S5-ViT / S4D-ViT
---

# State Space Models for Event Cameras

[Code](https://github.com/uzh-rpg/ssms_event_cameras)

**Topics**: [[T__Object_Detection]] | **Method**: [[M__S5-ViT_-_S4D-ViT]] | **Datasets**: Gen1 Automotive, 1 Mpx, Inference speed Gen1, Inference speed 1 Mpx

| 中文题名 | 事件相机的状态空间模型检测器 |
| 英文题名 | State Space Models for Event Cameras |
| 会议/期刊 | CVPR 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2402.15584) · [Code](https://github.com/uzh-rpg/ssms_event_cameras) · [DOI](https://doi.org/10.1109/CVPR52733.2024.00556) |
| 主要任务 | 基于事件相机的高频目标检测（event-based object detection） |
| 主要 baseline | RVT、GET-T、ERGO-12、Swin-T v2、Nested-T |

> [!abstract] 因为「RNN-based 事件检测器在跨频率推理时性能急剧下降（RVT 平均掉 21.25 mAP）」，作者在「RVT」基础上改了「将 ConvLSTM 替换为 S4D/S5 状态空间模型，并引入带限机制（bandlimiting）与 H2 范数正则」，在「Gen1 / 1 Mpx 检测基准」上取得「47.7/47.8 mAP，推理速度 8.16/9.57 ms，跨频率平均掉点仅 3.76 mAP」

- **速度优势**：S5-ViT-B 在 Gen1 上推理 8.16 ms，比 RVT-B（10.2 ms）快 20%，比 GET-T（16.8 ms）快 2 倍以上
- **跨频率鲁棒性**：20-200Hz 范围内平均性能下降 3.76 mAP，相比 RVT 的 21.25 mAP 提升 5.7 倍
- **精度持平**：Gen1 上 47.7 mAP（vs RVT-B 47.2, GET-T 47.9），1 Mpx 上 47.8 mAP（vs RVT-B 47.4, GET-T 48.4）

## 背景与动机

事件相机（event camera）以微秒级时间分辨率异步输出像素亮度变化，相比传统帧相机更适合高速运动场景。然而，这种异步、稀疏的数据形式使得标准 CNN 难以直接处理。现有方法通常将事件流分箱（binning）为固定时间窗口的体素网格，再用神经网络处理——但这引入了**时间分辨率与计算成本之间的根本张力**：窗口太短则信噪比低，窗口太长则丢失精细时序信息。

当前主流方案采用 RNN 架构进行时序建模。**RVT**（Recurrent Vision Transformer）将 Transformer 块与 ConvLSTM 结合，用 YOLOX 头输出检测结果，在 Gen1 和 1 Mpx 基准上建立了 strong baseline。**GET-T** 同样采用 Transformer + RNN + YOLOX 的范式，取得了更高的 mAP（Gen1 上 47.9，1 Mpx 上 48.4）。这些方法的核心假设是：训练时的固定频率（如 20Hz 或 50Hz）可以泛化到部署时的任意频率。

**但这一假设在真实部署中失效**：当推理频率与训练频率不一致时，RVT 的平均性能下降高达 **21.25 mAP**，GET-T 更达到 **24.53 mAP**（Table 2）。根本原因在于 RNN/LSTM 的离散时间递推对步长敏感，频率变化导致隐藏状态动态特性失配——模型学到了特定频率的"捷径"而非真正的时间连续表示。这一**频率脆弱性（frequency fragility）**严重限制了事件检测器在机器人、自动驾驶等需要自适应频率场景中的实用价值。

本文提出用**连续时间状态空间模型（SSM）**替代离散 RNN，并通过**带限机制**显式约束模型的频率响应，使网络学习平滑的时序核函数，从而在 20-200Hz 范围内保持稳定性能。

## 核心创新

核心洞察：**状态空间模型的连续时间本质天然适合事件相机的异步时序特性**，因为 SSM 通过双线性变换将连续时间动态离散化为任意步长，其核函数（impulse response）由学习到的参数决定，从而允许显式的频率控制；而 RNN 的离散递推隐含固定采样假设，无法适应频率变化。

这使得**带限正则（bandlimiting regularization）**成为可能：通过在损失中加入 H2 范数约束，或直接掩码高频输出矩阵分量，迫使 SSM 学习平滑、低频的时序聚合核，避免对训练频率的过拟合。

| 维度 | Baseline (RVT/GET-T) | 本文 (S5-ViT / S4D-ViT) |
|:---|:---|:---|
| 时序机制 | ConvLSTM / RNN 离散递推 | S4D/S5 状态空间模型，连续时间离散化 |
| 频率适应性 | 固定训练频率，跨频掉点 >20 mAP | 带限约束，20-200Hz 掉点 3.76 mAP |
| 训练目标 | YOLOX 检测损失 | YOLOX 损失 + H2 范数正则项 |
| 并行训练 | TBPTT 截断梯度 | 混合 BPTT/TBPTT，SSM 可并行扫描 |

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/7d7bd51e-fcc8-46c7-b793-c09b8cb238eb/figures/fig_001.png)
*Figure: Top-Left Previous works [13, 34] use RNN architectures with convolutional or attention mechanisms to train models that have*



整体数据流遵循"事件分箱 → ViT-SSM 骨干 → 带限 SSM 层 → YOLOX 检测头"的四级流水线：

1. **事件表示（Event representation）**：输入原始事件相机数据，将 50ms 时间窗口划分为 T=10 个离散分箱（bins），生成体素化的时空张量作为网络输入。

2. **ViT-SSM 骨干（ViT backbone with SSM layers）**：采用与 RVT 类似的分层 Transformer 结构，但将原有的 ConvLSTM 时序模块替换为 S4D 或 S5 状态空间层。SSM 层分布在 S1-S4 四个阶段，逐步聚合多尺度时空特征。

3. **带限 SSM 模块（Bandlimited SSM module）**：核心计算单元。S4D 采用对角化状态矩阵简化计算；S5 支持多输入并行扫描。输出矩阵 C 经过带限掩码处理，或通过 H2 范数正则约束频率响应。

4. **YOLOX 检测头（YOLOX detection head）**：接收骨干输出的时空特征图，输出边界框坐标、类别分数和目标置信度，采用与 RVT/GET-T 相同的 anchor-free 检测范式以保证公平比较。

```
Raw Events → [50ms/10 bins] → Binned Tensor → ViT-SSM (S1-S4 stages)
                                                    ↓
                                        S4D/S5 SSM layers + Bandlimiting
                                                    ↓
                                        Spatiotemporal Features → YOLOX Head
                                                                    ↓
                                                        Bounding Boxes + Classes
```

## 核心模块与公式推导

### 模块 1: 双线性离散化与 SSM 核心计算（对应框架图 S1-S4 阶段）

**直觉**：事件相机的异步特性要求模型具备连续时间建模能力，双线性变换（Tustin 方法）可将连续时间 SSM 稳定地离散化为任意步长，避免 RNN 对固定采样率的依赖。

**Baseline 公式** (连续时间 SSM)：
$$\dot{x}(t) = Ax(t) + Bu(t), \quad y(t) = Cx(t) + Du(t)$$
符号：$x(t) \in \mathbb{R}^N$ 为隐状态，$u(t)$ 为输入事件特征，$y(t)$ 为输出，$A,B,C,D$ 为可学习参数。

**变化点**：RVT 的 ConvLSTM 直接离散化递推 $h_t = f(h_{t-1}, x_t)$，隐含固定 $\Delta t$；本文通过双线性变换显式处理步长 $\Delta$，使同一组连续参数适配不同推理频率。

**本文公式（推导）**：
$$\text{Step 1}: \quad BL = \left(I - \frac{\Delta}{2} \Lambda\right)^{-1} \quad \text{（计算离散化分母，保持数值稳定性）}$$
$$\text{Step 2}: \quad \bar{\Lambda} = BL \cdot \left(I + \frac{\Delta}{2} \Lambda\right) \quad \text{（完整双线性变换，得到离散状态矩阵）}$$
$$\text{Step 3}: \quad \bar{B} = BL \cdot \Delta B, \quad x_k = \bar{\Lambda} x_{k-1} + \bar{B} u_k \quad \text{（离散递推，步长 } \Delta \text{ 可任意调整）}$$
$$\text{最终}: \quad y_k = C x_k + D u_k$$

其中 $\Lambda$ 为对角化后的连续状态矩阵（S4D）或一般形式（S5）。S5 进一步采用并行扫描算法（parallel scan）替代顺序递推，实现高效训练。

**对应消融**：Table 3 显示，将 S5 替换为 S4D（对角化简化）在 1 Mpx 上导致 47.8 → 46.8 mAP（-1.0 mAP），验证了对角近似的精度损失。

---

### 模块 2: 带限机制与 H2 范数正则（对应框架图 Bandlimiting Mask）

**直觉**：SSM 的学习核函数可能包含高频分量，导致对训练频率过拟合；通过显式约束频率响应的带宽，可使模型学习更平滑、泛化性更强的时序聚合模式。

**Baseline 公式** (RVT/GET-T 训练目标)：
$$\mathcal{L}_{\text{YOLOX}} = \frac{1}{B \cdot T} \sum_{b,t} \left(\mathcal{L}_{\text{iou}} + \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{reg}}\right)$$
符号：$B$ 为 batch size，$T$ 为时序长度，三项分别为 IoU 损失、分类损失和回归损失。

**变化点**：Baseline 仅优化检测精度，无任何频率约束；本文发现 SSM 的传递函数 $H(z)$ 的 H2 范数直接反映频率响应能量，加入正则项可惩罚高频增益。

**本文公式（推导）**：
$$\text{Step 1}: \quad \|H\|_2^2 = \text{tr}(C P C^T) \quad \text{其中 } P \text{ 满足 Lyapunov 方程，衡量状态能量}$$
$$\text{（等价实现：对输出矩阵 } C \text{ 施加频率相关的掩码 } mask(\omega, \alpha)\text{）}$$
$$\text{Step 2}: \quad C_{\text{masked}} = C \cdot mask(\omega, \alpha), \quad \alpha \in [0, 0.5] \text{ 控制带宽截止频率}$$
$$\text{最终}: \quad \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{YOLOX}} + \lambda \cdot \|H\|_2$$

**对应消融**：Table 3 显示，S5-legS 配置下去掉带限（$\alpha=0$ vs $\alpha=0.5$）导致 48.33 → 48.48 mAP（+0.15，即带限提升 0.15）；S4D-legS 上效果更明显：46.93 → 47.33 mAP（+0.40）。legS 初始化本身至关重要：替换为 inv 初始化峰值掉 1.05 mAP，lin 初始化掉 1.93 mAP。

## 实验与分析



本文在 **Gen1 Automotive Detection**（228k 边界框，汽车场景）和 **1 Mpx Detection**（高分辨率事件相机）两个主流基准上进行评估。 汇总了与 RVT-B、GET-T、ERGO-12 等方法的对比。

**精度与速度权衡**：S5-ViT-B 在 Gen1 上达到 **47.7 mAP**，相比直接 baseline RVT-B（47.2）提升 +0.5，略低于 GET-T（47.9，差距 -0.2）；在 1 Mpx 上达到 **47.8 mAP**，同样优于 RVT-B（47.4，+0.4）但低于 GET-T（48.4，-0.6）。值得注意的是，ERGO-12 以 50.4 mAP（Gen1）和 40.6 mAP（1 Mpx）取得最高精度，但其推理速度为 69.9 ms，无法与实时方法竞争。S5-ViT-B 的 **8.16 ms**（Gen1）和 **9.57 ms**（1 Mpx）在所有 competitive 方法中最快，比 RVT-B 快 20%，比 GET-T 快 2 倍以上，实现了精度-速度帕累托前沿的显著推进。



**跨频率鲁棒性**（ 对应 Table 2）是本文最核心的优势：在 20-200Hz 范围内评估，S5-ViT 平均性能下降仅 **3.76 mAP**，而 RVT 为 **21.25 mAP**（5.7× 差距），GET-T 为 **24.53 mAP**（6.5× 差距）。这一差距在极端频率偏移时更为显著——例如从 50Hz 训练到 200Hz 推理时，RNN-based 方法几乎失效，而 SSM 的连续时间离散化保持了稳定的隐藏状态动态。

**消融分析**（Table 3/4）：SSM 组件本身不可或缺——完全移除时序递归导致严重性能崩溃；仅在 S4 阶段保留 SSM（而非 S1-S4 全阶段）虽能运行但次优，验证了早期时序信息融合的重要性。带限参数 $\alpha$ 和 legS 初始化共同作用：S5-legS 带限增益 +0.15 mAP，S4D-legS 增益 +0.40 mAP；而错误的初始化（lin）可导致 -1.93 mAP 的灾难性下降。

**公平性检查**：比较基本公平——所有方法使用相同 YOLOX 头、相同分箱表示（T=10, 50ms）。潜在问题：（1）推理时间测量在 T4 GPU 上进行，但部分 baseline（标记 *）为估计值或基于不同 GPU（Titan Xp）；（2）与 GET-T 的 mAP 差距很小（0.2-0.6），但速度优势巨大（2×），实际部署中 S5-ViT 更具吸引力；（3）缺少与纯 SSM（无 Transformer 骨干）的直接对比，无法完全解耦 SSM 与 Attention 的各自贡献。

## 方法谱系与知识库定位

**方法家族**：事件相机目标检测中的 **RNN-to-SSM 演进谱系**。直接父方法为 **RVT**（Recurrent Vision Transformer, 2022）——本文继承其 ViT 分层结构、YOLOX 检测头和混合 BPTT/TBPTT 训练配方，但将时序骨干从 ConvLSTM 替换为 S4D/S5 状态空间模型，并新增带限机制。

**改动槽位**：
- **架构（architecture）**：ConvLSTM → S4D/S5 SSM，保持 Transformer + 检测头范式
- **目标函数（objective）**：YOLOX 损失 → YOLOX 损失 + H2 范数正则
- **推理策略（inference_strategy）**：固定频率假设 → 带限 SSM，显式控制频率响应带宽
- **训练配方（training_recipe）**：继承 RVT 的混合 batching（半 BPTT + 半 TBPTT），无改动
- **数据策划（data_curation）**：无改动，沿用相同分箱预处理

**直接 baselines 差异**：
- **RVT**：本文直接替换其 RNN 核心，解决其频率脆弱性（21.25 → 3.76 mAP 掉点）
- **GET-T**：同样 Transformer+RNN 路线，本文以微小 mAP 代价（-0.2~-0.6）换取 2× 以上速度提升和跨频率鲁棒性
- **ERGO-12**：精度更高但慢 8×，本文定位实时应用

**后续方向**：（1）将带限机制推广至其他连续时间模型（如神经 ODE）；（2）探索 SSM 与事件相机原生异步表示（无需分箱）的直接结合；（3）在视频理解、音频处理等其他高频时序任务中验证 SSM 的频率鲁棒性优势。

**标签**：modality: 事件相机 / paradigm: 状态空间模型+Transformer / scenario: 实时目标检测 / mechanism: 连续时间离散化+带限正则 / constraint: 低延迟+跨频率泛化

