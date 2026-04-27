---
title: 'NTIRE 2026 Challenge on Video Saliency Prediction: Methods and Results'
type: paper
paper_level: C
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.14816
aliases:
- NTIRE 2026视频显著性预测挑战赛综述
- NCVSPM
- 本挑战赛的核心发现是：在视频显著性预测任务上
cited_by: 4
code_url: https://github.com/iLearn-Lab/CVPRW26-ViSAGE
modalities:
- Image
---

# NTIRE 2026 Challenge on Video Saliency Prediction: Methods and Results

[Paper](https://arxiv.org/abs/2604.14816) | [Code](https://github.com/iLearn-Lab/CVPRW26-ViSAGE)

**Topics**: [[T__Benchmark_-_Evaluation]], [[T__Video_Understanding]]

> [!tip] 核心洞察
> 本挑战赛的核心发现是：在视频显著性预测任务上，「大型预训练视频骨干（InternVideo2/V-JEPA2）+ 多尺度时空解码」的组合显著优于此前的专用小模型方案。iLearn 进一步表明，在同一骨干上并行运行归纳偏置互补的双解码器（一个依赖显式空间先验与乘法门控，另一个依赖数据驱动的多尺度学习），并在 logit 空间集成，可以在不增加骨干参数的前提下覆盖更广泛的场景类型，从而获得边际但决定性的性能提升。

| 属性 | 内容 |
|------|------|
| 中文题名 | NTIRE 2026视频显著性预测挑战赛综述 |
| 英文题名 | NTIRE 2026 Challenge on Video Saliency Prediction: Methods and Results |
| 会议/期刊 | arXiv (Cornell University) (技术报告) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.14816) · [Code](https://github.com/iLearn-Lab/CVPRW26-ViSAGE) · [Project](待补充) |
| 主要任务 | 视频显著性预测（Video Saliency Prediction），即模拟人类视觉注意力分布，逐帧生成显著性图 |
| 主要 baseline | InternVideo2-Stage2 6B、V-JEPA2、TMFI、TranSalNet、TASED-Net |

> [!abstract] 因为「高质量眼动标注数据稀缺且小规模，难以支撑大模型训练；同时缺乏将大规模预训练视频基础模型迁移至显著性预测的有效范式」，作者「以 NTIRE 2026 挑战赛为平台，汇总分析 7 支决赛队伍方案」，在「2000 视频、5000+ 众包评估者的新数据集」上发现「iLearn 以 CC 指标优先规则获得第一名」，验证了「大骨干 + 多尺度时空解码」的主流范式。

- **iLearn 第一名**：CC 指标决胜（与另一队伍平均排名相同，CC 优先规则判定第一）
- **数据规模**：2000 个视频，5000+ 名众包鼠标追踪评估者，远超 Hollywood-2（1707 视频、19 观看者）和 DHF1K（1000 视频、17 观看者）
- **参赛规模**：20+ 队伍参赛，7 支队伍通过代码审查进入决赛

## 背景与动机

视频显著性预测旨在模拟人类视觉系统（HVS）对动态场景中不同区域的注意力分配，输出每帧的像素级显著性概率图。该任务直接服务于感知引导视频压缩、自适应内容重定向、视觉质量评估等多媒体应用——例如，在带宽受限场景下，编码器可根据显著性图对注视区域分配更多码率，从而优化主观感知质量。

现有方法沿三条技术路线发展。**TranSalNet** 等专用小模型针对显著性预测任务从零设计轻量架构，但受限于参数规模，难以捕捉复杂时空动态；**TASED-Net** 引入 3D 卷积进行时序建模，却在长程依赖与语义理解上表现不足；**TMFI** 等方法尝试多尺度特征融合，但缺乏大规模预训练带来的通用视频表征能力。这些方案的共同瓶颈在于：视频显著性标注需昂贵眼动仪设备，现有最大公开数据集 Hollywood-2 仅 1707 个视频、19 名观看者，DHF1K 仅 1000 个视频、17 名观看者，数据稀缺严重制约模型容量扩展。

更深层的矛盾在于：计算机视觉领域已涌现 InternVideo2、V-JEPA2 等大规模预训练视频基础模型，其时空表征能力远超任务专用模型，但如何将这些「大骨干」有效适配至显著性预测这一密集预测任务，尚无成熟范式——直接微调易导致过拟合，冻结特征则任务适配不足。NTIRE 2026 挑战赛正是为破解这一困局而设立：构建 2000 视频、5000+ 众包评估者的大规模新数据集，并以竞赛形式系统性探索「大骨干迁移」的最优实践。

本文作为竞赛综述，核心贡献在于汇总并对比 7 支决赛队伍的技术方案，揭示当前最优范式的关键设计选择。

## 核心创新

核心洞察：在视频显著性预测任务上，「大型预训练视频骨干 + 多尺度时空解码」的组合显著优于专用小模型方案，而同一骨干上并行运行归纳偏置互补的双解码器并在 logit 空间集成，可以在不增加骨干参数的前提下覆盖更广泛的场景类型，从而获得边际但决定性的性能提升。

| 维度 | Baseline（单解码器适配） | 本文 iLearn 方案 |
|------|------------------------|----------------|
| 解码器结构 | 单一解码路径，固定融合策略 | 双 Expert 并行：Expert 1 乘法时序门控 + 空间先验；Expert 2 拼接融合 + 多级深度监督 |
| 时序建模 | 简单 3D 卷积或单向时序聚合 | Expert 1 自顶向下乘法门控（最深层→浅层）；Expert 2 最深层时序门控 + 多中间层独立预测头 |
| 空间先验 | 无显式中心偏置或手工设计 | Expert 1 引入可学习中心偏置先验（利用人眼注视中心统计规律）+ FiLM 全局条件注入 |
| 集成方式 | 输出空间平均或单模型推理 | **logit 空间集成**：逆 sigmoid 变换后平均再映射回概率空间 |
| 骨干适配 | 全参数微调或完全冻结 | 两阶段：Stage 1 冻结骨干训练解码器；Stage 2 LoRA 微调骨干第 11/23/35/47 层 |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ba8b61e3-a444-4976-b5f8-970064b2d789/figures/Figure_1.png)
*Figure 1: Figure 1. iLearn Video Saliency Prediction Pipeline.*



iLearn 视频显著性预测框架遵循「共享大骨干 + 双专家解码 + logit 空间集成」的三级流水线。输入为视频帧序列，经共享编码器提取多级时空特征后，分流至两个结构互补的解码器，最终融合输出显著性图。

**模块 1：共享编码器（InternVideo2-Stage2 6B）**
- 输入：原始视频帧序列
- 输出：第 11、23、35、47 层的多级时空特征图
- 角色：提供大规模预训练的通用视频表征，通过 LoRA 进行轻量任务适配

**模块 2：Expert 1 解码器（乘法时序门控路径）**
- 输入：编码器多级特征（重点关注最深层语义特征）
- 输出：显著性概率图 $s_1$
- 角色：利用乘法门控实现自顶向下的时序注意力调制，注入中心偏置先验与全局 FiLM 条件，强调空间显式约束

**模块 3：Expert 2 解码器（拼接融合 + 深度监督路径）**
- 输入：编码器多级特征
- 输出：显著性概率图 $s_2$
- 角色：通过拼接融合保留多尺度信息，在多个中间层附加独立预测头提供密集梯度信号，强调数据驱动的多尺度学习

**模块 4：Logit 空间集成模块**
- 输入：双 Expert 输出 $s_1, s_2$
- 输出：最终显著性图 $\hat{s}$
- 角色：在 logit 空间取平均，平衡双专家的互补偏置

**数据流示意：**
```
视频帧 → [InternVideo2-Stage2 6B] → {第11层, 第23层, 第35层, 第47层特征}
                                    ↓
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
            [Expert 1: 乘法门控]              [Expert 2: 拼接融合]
            · 自顶向下时序调制                  · 多级深度监督
            · 中心偏置先验                      · 中间层独立预测头
            · FiLM 全局条件                     · 最深层时序门控
                    ↓                               ↓
                   s₁                              s₂
                    └───────────────┬───────────────┘
                                    ↓
                    [Logit 空间集成: σ((σ⁻¹(s₁)+σ⁻¹(s₂))/2)]
                                    ↓
                                最终显著性图 ŝ
```

## 核心模块与公式推导

### 模块 1: Logit 空间双专家集成（对应框架图最末端）

**直觉**：直接对概率空间取平均会压缩高置信度响应、放大低置信度噪声，而在 logit 空间平均能保留双专家的极端判断，更适合互补模型的融合。

**Baseline 公式**（标准概率空间集成）：
$$\hat{s}_{\text{prob}} = \frac{s_1 + s_2}{2}$$
符号: $s_1, s_2 \in [0,1]$ 为双 Expert 输出的显著性概率图。

**变化点**：概率空间平均对饱和区域（$s \approx 0$ 或 $s \approx 1$）梯度消失，且假设双专家不确定性同质；实际上 Expert 1 依赖强空间先验、Expert 2 依赖数据驱动多尺度，不确定性结构不同，需在对数几率空间对齐。

**本文公式（推导）**：
$$\text{Step 1: 逆 sigmoid 变换} \quad z_i = \sigma^{-1}(s_i) = \log\frac{s_i}{1-s_i}, \quad i \in \{1,2\}$$
$$\text{Step 2: Logit 空间平均} \quad \bar{z} = \frac{z_1 + z_2}{2} = \frac{1}{2}\left(\sigma^{-1}(s_1) + \sigma^{-1}(s_2)\right)$$
$$\text{最终: 概率映射} \quad \hat{s} = \sigma(\bar{z}) = \sigma\left(\frac{1}{2}\left(\sigma^{-1}(s_1) + \sigma^{-1}(s_2)\right)\right)$$

**对应消融**：

---

### 模块 2: Expert 1 乘法时序门控与空间先验（对应框架图左侧分支）

**直觉**：人类视觉存在显著的中心偏置效应（注视点倾向于图像中心），且时序注意力应从语义最丰富的高层向低层自顶向下调制，而非平等处理所有尺度。

**Baseline 公式**（标准拼接 + 卷积融合，如 TMFI/FPN 范式）：
$$F_{\text{out}}^{l} = \text{Conv}([F_{\text{enc}}^{l}; F_{\text{up}}^{l+1}])$$
符号: $F_{\text{enc}}^{l}$ 为编码器第 $l$ 层特征，$F_{\text{up}}^{l+1}$ 为上层上采样特征，$[\cdot;\cdot]$ 表示通道拼接。

**变化点**：Baseline 拼接融合对各级特征同等对待，缺乏显式时序动态调制与空间先验；Expert 1 引入**乘法门控机制**，将最深层生成的时序注意力图 $A^{L}$ 自顶向下逐层调制浅层特征，同时注入可学习中心偏置 $B$ 和 FiLM 全局条件 $\gamma, \beta$。

**本文公式（推导）**：
$$\text{Step 1: 最深层时序注意力生成} \quad A^{L} = \text{TemporalAttn}(F_{\text{enc}}^{L}) \in \mathbb{R}^{T \times H \times W}$$
$$\text{Step 2: 自顶向下乘法门控} \quad \tilde{F}^{l} = F_{\text{enc}}^{l} \odot \text{Upsample}(A^{l+1}), \quad l = L-1, \ldots, 1$$
$$\text{Step 3: 3D 残差精炼} \quad \hat{F}^{l} = \tilde{F}^{l} + \text{Res3D}(\tilde{F}^{l})$$
$$\text{Step 4: FiLM 全局条件注入} \quad F_{\text{FiLM}}^{l} = \gamma^{l} \odot \hat{F}^{l} + \beta^{l}, \quad (\gamma^{l}, \beta^{l}) = \text{MLP}(\text{GAP}(F_{\text{enc}}^{L}))$$
$$\text{Step 5: 中心偏置先验叠加} \quad s_1 = \sigma\left(\text{Conv}(F_{\text{FiLM}}^{1}) + \alpha \cdot B\right), \quad B \in \mathbb{R}^{H \times W} \text{ 可学习}$$

**对应消融**：

---

### 模块 3: Expert 2 多级深度监督解码（对应框架图右侧分支）

**直觉**：密集预测任务中，仅在最终层监督会导致中间层梯度稀疏、多尺度信息丢失；在多个层级附加独立预测头可提供密集梯度信号，增强浅层细节保留能力。

**Baseline 公式**（标准单头监督，如 TASED-Net）：
$$\mathcal{L} = \text{Loss}(s^{\text{final}}, s^{\text{gt}})$$

**变化点**：单一最终监督对深层网络梯度传播效率低；Expert 2 采用**拼接融合**替代乘法门控（与 Expert 1 形成结构互补），并在多个中间层附加独立显著性预测头，以深度监督（deep supervision）机制提供多层级梯度。

**本文公式（推导）**：
$$\text{Step 1: 拼接融合} \quad F_{\text{fuse}}^{l} = \text{Conv}([F_{\text{enc}}^{l}; \text{Upsample}(F_{\text{dec}}^{l+1})])$$
$$\text{Step 2: 最深层时序门控} \quad F_{\text{dec}}^{L} = F_{\text{enc}}^{l} \odot A^{L} \quad \text{（仅在最深层施加，与 Expert 1 的全层级不同）}$$
$$\text{Step 3: 多级独立预测头} \quad s^{l} = \sigma(\text{Conv}_{l}(F_{\text{fuse}}^{l})), \quad l \in \{l_1, l_2, \ldots, L\}$$
$$\text{Step 4: 深度监督损失} \quad \mathcal{L}_{\text{Expert2}} = \sum_{l} \lambda_l \cdot \text{Loss}(s^{l}, \text{Resize}(s^{\text{gt}}, H_l, W_l))$$
$$\text{最终输出} \quad s_2 = s^{L} \text{（经训练后，最深预测头输出）}$$

**对应消融**：

## 实验与分析

挑战赛最终排名依据多个指标的综合平均排名，CC（Correlation Coefficient）作为平局决胜规则。Table 1 显示 iLearn 与另一队伍平均排名相同，凭 CC 指标优先规则获第一名。

| 排名 | 队伍 | 核心方法 | 骨干网络 | 关键设计 |
|:--:|:---|:---|:---|:---|
| 1 | **iLearn** | 双 Expert + logit 集成 | InternVideo2-Stage2 6B | 乘法门控/拼接融合双路径，LoRA 微调 |
| 2 | CVSP | V-JEPA2 适配 | V-JEPA2 | 自监督预测性时空表征 |
| 3 | ARK MMLAB | 层次化 FPN 解码 | InternVideo2 | 多级时空特征上采样对齐 |
| - | Vertex | TMFI 扩展 | （待补充） | 自底向上聚合路径 |
| - | AAM | 多模态融合 | （待补充） | 音视频 + 双曲解码器 |
| - | SHU-MIIPLab | 扩散模型 | （待补充） | 光流引导扩散 |
| - | NTR | 双流设计 | （待补充） | 运动线索与空间细节分离 |


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ba8b61e3-a444-4976-b5f8-970064b2d789/figures/Figure_2.png)
*Figure 2: Figure 2. An overview of PredJSal. The framework repurposesV-JEPA2 backbone to extract rich spatiotemporal representations,decoded into per-frame saliency maps via a 3D convolutional de-coder with mul*



**核心发现分析**：

1. **「大骨干」效应显著**：前三名均基于 InternVideo2 或 V-JEPA2（6B 级别参数），远超传统 TranSalNet/TASED-Net 的专用小模型，验证大规模预训练视频表征对显著性预测的决定性价值。

2. **iLearn 的边际优势**：与第二名 CVSP 平均排名相同，仅靠 CC 决胜规则区分，说明顶级方案性能差距极小。iLearn 的双 Expert 设计贡献的是**覆盖互补场景类型**的稳健性提升，而非单一指标的大幅跃进。

3. **解码器多样性**：ARK MMLAB 的标准 FPN 式解码同样进入前三，表明「大骨干」基础上解码器设计存在多条有效路径，iLearn 的双 Expert 并非唯一最优解。

**消融与公平性检查**：
- **消融缺失**：iLearn 未披露 Expert 1/2 单独性能、移除中心偏置/深度监督的 ΔCC，LoRA 秩与层数选择依据不明
- **基线强度**：未与 DHF1K 标准基准上的已发表 SOTA（TranSalNet、TASED-Net）直接对比，无法判断绝对提升幅度
- **数据成本**：2000 视频、5000+ 众包评估者虽规模可观，但鼠标追踪与眼动仪数据质量的等效性未经量化验证
- **计算成本**：6B 参数骨干 + 双解码器推理，实时性未评估；V-JEPA2 的自监督预训练成本未披露
- **样本局限**：仅 7 支队伍进入决赛，「大骨干 + 多尺度建模」为最优范式的结论普适性受限

## 方法谱系与知识库定位

**方法家族**：视频密集预测任务的大模型迁移学习（Large Model Transfer for Dense Video Prediction）

**父方法**：InternVideo2（通用视频理解预训练）+ TMFI/FPN（多尺度时序特征融合）

**改动槽位**：
| 槽位 | 父方法状态 | 本文改动 |
|:---|:---|:---|
| 架构 | 单解码器路径 | 双 Expert 并行（乘法门控 vs. 拼接融合） |
| 目标函数 | 单头最终监督 | 深度监督 + logit 空间集成损失 |
| 训练配方 | 全参数微调或完全冻结 | 两阶段：冻结解码器预训练 → LoRA 骨干微调 |
| 数据策划 | 眼动仪小规模标注 | 众包鼠标追踪大规模扩展（2000 视频/5000+ 评估者） |
| 推理 | 单模型输出 | 双模型 logit 空间集成 |

**直接基线对比**：
- **TranSalNet / TASED-Net**：专用小模型，无大预训练骨干 → iLearn 以 InternVideo2 6B 替代，参数规模提升约 100×
- **TMFI**：单一路径多尺度融合 → iLearn 扩展为双路径互补，并引入 logit 空间集成
- **CVSP (V-JEPA2)**：同为大骨干路线，但采用自监督预测性表征而非监督预训练 → iLearn 以监督预训练 + 显式空间先验形成差异化
- **ARK MMLAB**：同 InternVideo2 骨干的标准 FPN 解码 → iLearn 以双 Expert 结构复杂度和归纳偏置注入形成差异化

**后续方向**：
1. **数据质量 bridging**：量化众包鼠标追踪与眼动仪标注的分布差异，设计 domain adaptation 或校准模块
2. **高效适配**：探索参数更少的适配方式（如 prompt tuning、adapter）替代 LoRA，降低 6B 骨干部署成本
3. **统一多任务框架**：将显著性预测与视频质量评估、压缩等下游任务联合训练，验证表征复用性

**知识库标签**：
- **模态**（Modality）：视频（RGB）
- **范式**（Paradigm）：密集预测、大模型迁移学习、集成学习
- **场景**（Scenario）：视觉注意力建模、感知引导处理
- **机制**（Mechanism）：时序门控、多尺度融合、logit 空间集成、LoRA 微调
- **约束**（Constraint）：数据稀缺、计算资源密集、实时性未验证

