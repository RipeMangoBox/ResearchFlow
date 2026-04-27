---
title: 'MARCO: Navigating the Unseen Space of Semantic Correspondence'
type: paper
paper_level: B
venue: CVPR
year: 2026
paper_link: https://arxiv.org/abs/2604.18267
aliases:
- MARCO：探索语义对应的未知空间
- MARCO
- MARCO在单DINOv2骨干（ViT-L/14）基础上引入两个核心训
acceptance: oral
code_url: https://github.com/visinf/MARCO
method: MARCO
modalities:
- Image
---

# MARCO: Navigating the Unseen Space of Semantic Correspondence

[Paper](https://arxiv.org/abs/2604.18267) | [Code](https://github.com/visinf/MARCO)

**Topics**: [[T__Cross-Modal_Matching]], [[T__Semantic_Segmentation]], [[T__Self-Supervised_Learning]] | **Method**: [[M__MARCO]]

> [!tip] 核心洞察
> MARCO在单DINOv2骨干（ViT-L/14）基础上引入两个核心训练机制，分别针对精细定位和语义泛化两个目标。

**第一：轻量级架构适配**。在DINOv2最后12个Transformer层中插入AdaptFormer适配器，并添加上采样头以恢复子patch级别的空间结构。骨干参数冻结，仅训练适配器和上采样头，总参数量323M。消融显示适配器贡献+8.1@0.01/+23.0@0.10，上采样头贡献+12.6@0.01/+10.3@0.10，两者共同将DINOv2转化为精确对应骨干。

**第二：粗到细监督调度**。使用高斯目标分布监督soft-argmax预测，训练初期使用宽目标（σ=3，偏向粗粒度语义对齐），逐步收窄至σ=1（偏向精细定位）。固定σ=1最大化精细精度（27.8@0.01）但损害粗粒度准确率；固定σ=3则相反。调度策略在两者之间取得平衡（27.0@0.01，87.2@0.10），避免各自的退化模式。

**第三：稠密自蒸馏框架（Dense Self-Distillation via Flow Anchoring）**。这是泛化能力的核心来源。具体流程：(1) 教

| 中文题名 | MARCO：探索语义对应的未知空间 |
| 英文题名 | MARCO: Navigating the Unseen Space of Semantic Correspondence |
| 会议/期刊 | CVPR 2026 (oral) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.18267) · [Code](https://github.com/visinf/MARCO) · [Project](https://github.com/visinf/MARCO) |
| 主要任务 | Semantic Correspondence（语义对应/关键点匹配），评估未见关键点（unseen landmarks）和未见类别（unseen categories）的泛化能力 |
| 主要 baseline | DIFT, SD+DINOv2双编码器方法, DistillDIFT, GECO, Jamais Vu |

> [!abstract] 因为「现有语义对应方法依赖双编码器导致参数量大（~950M）、推理慢（~0.85 FPS），且在训练未见过的关键点/类别上泛化能力差」，作者在「单DINOv2骨干」基础上改了「轻量级AdaptFormer适配器 + 粗到细监督调度 + 稠密自蒸馏（Flow Anchoring）」，在「SPair-71k/SPair-U/MP-100」上取得「PCK@0.01 +8.9，未见关键点泛化+5.1，速度提升10×（8.3 vs 0.85 FPS）」

- **精度**：SPair-71k PCK@0.01 达 27.0，较最强单骨干 baseline DistillDIFT 提升 +8.9
- **泛化**：SPair-U 未见关键点 PCK@0.10 达 67.5，较 Jamais Vu 提升 +5.1；MP-100 未见类别提升 +4.7
- **效率**：参数量 323M（3×压缩），推理速度 8.3 FPS（10×加速），硬件条件 RTX 4090 @ 840p

## 背景与动机

语义对应（Semantic Correspondence）旨在找到不同图像中同一语义位置的关键点匹配——例如，给定两张不同姿态的猫图像，模型需要准确找到"左眼内角"对应"左眼内角"，而非粗略的"头部对头部"。这一能力对图像编辑、3D重建、姿态迁移等下游任务至关重要。

现有方法主要分为两条路线。**双编码器方法**（如 SD+DINOv2 组合）将 Stable Diffusion 的语义先验与 DINOv2 的视觉特征结合，在标准基准上表现优异，但参数量高达约 950M，推理速度仅 0.85 FPS，难以实际部署。**单骨干替代方案**（DistillDIFT、GECO）通过知识蒸馏压缩为单一 DINOv2 网络，效率大幅提升，但在精细定位精度上显著落后——尤其在严格阈值 PCK@0.01（允许误差 ≤1% 图像尺寸）下，与双编码器的差距暴露无遗。

更深层的缺陷在于**评估协议的遮蔽效应**：现有主流基准 SPair-71k 的测试协议只评估训练时见过的关键点类型，模型在"训练时标注了猫耳朵但未标注猫胡须"的场景中表现如何，完全未被检验。这导致领域对"泛化能力"的认知存在系统性盲区——模型可能在标准测试上高分，却在真实世界的未见关键点/未见类别上失败。此外，稀疏关键点监督本身构成瓶颈：每张训练图像仅约 20 个标注点，无法覆盖物体表面的完整语义结构，监督区域之外的泛化能力天然受限。

MARCO 的核心动机正是打破这一困局：在**单骨干框架内**同时实现**双编码器级别的精细定位精度**和**跨关键点/跨类别的真实泛化能力**，并通过更严格的评估协议暴露和解决现有方法的隐性失败。
![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/cf92e722-c220-46f4-9f69-26bafb5b30a6/figures/Figure_2.png)
*Figure 2: Figure 2. Flow consistency in DINOv2. Semantic flow (in HSVspace) from raw feature matches between two objects. Fine-tuningon sparse keypoints improves only the landmarks’ representation,reducing geom*



## 核心创新

核心洞察：**稠密自蒸馏可以通过光流锚定（Flow Anchoring）将稀疏关键点监督扩展为结构化稠密对应，从而突破稀疏标注的覆盖瓶颈**，因为分段仿射变形场能够编码物体表面的连续语义结构，从而使模型在监督区域之外获得可靠的自监督信号成为可能。

| 维度 | Baseline（DistillDIFT/GECO） | 本文（MARCO） |
|:---|:---|:---|
| 架构 | 单 DINOv2，无适配器 | DINOv2 + 12层 AdaptFormer 适配器 + 子patch上采样头 |
| 监督密度 | 仅 ~20 个稀疏关键点/GT | 稀疏GT + 稠密自蒸馏伪标签（Flow Anchering扩展） |
| 监督调度 | 固定高斯目标分布（通常 σ=1） | 粗到细调度：σ=3 → σ=1，平衡语义对齐与精细定位 |
| 泛化机制 | 依赖 DINOv2 预训练先验 | EMA教师网络 + MNN挖掘 + Delaunay三角剖分 + k-means聚类过滤 |
| 评估协议 | SPair-71k 仅测已见关键点 | 新增 SPair-U + MP-100，测未见关键点与未见类别 |
| 效率-精度权衡 | 高效但精度低（尤其@0.01） | 精度逼近双编码器，速度10×、参数3×优于双编码器 |

## 整体框架


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/cf92e722-c220-46f4-9f69-26bafb5b30a6/figures/Figure_3.png)
*Figure 3: Figure 3. Overview of MARCO. We insert lightweight adapters into DINOv2 and add a compact upsampling layer (red). At training time,we propose a coarse-to-fine Gaussian RBF loss that progressively shar*



MARCO 的整体数据流遵循"特征提取 → 适配增强 → 粗到细监督 → 稠密自蒸馏"的四阶段管线，全部建立在冻结的 DINOv2 ViT-L/14 骨干之上：

**输入**：源图像 $I_s$ 与目标图像 $I_t$，可为同类别或跨类别图像对。

**模块 A：AdaptFormer 适配器（DINOv2 后12层）**。在冻结的 ViT 最后12个 Transformer 层中插入轻量级瓶颈适配器，仅引入少量可训练参数。输入为 DINOv2 patch token，输出为增强后的特征表示，保留预训练语义能力的同时注入任务特异性。

**模块 B：子patch 上采样头（红色标注）**。将 ViT 的粗粒度 patch 特征（14×14 网格）通过紧凑上采样层恢复至更高分辨率，实现子patch 级别的精细空间定位——这是 PCK@0.01 精度突破的关键。

**模块 C：粗到细监督调度**。训练初期以宽高斯目标（σ=3）监督 soft-argmax 预测，偏向粗粒度语义对齐；随训练推进逐步收窄至 σ=1，聚焦精细定位精度。动态平衡避免固定 σ 的退化模式。

**模块 D：稠密自蒸馏（Flow Anchoring）**。EMA 维护的教师网络提取特征，通过互最近邻（MNN）挖掘可靠稀疏匹配，与 GT 关键点合并为种子点；Delaunay 三角剖分估计分段仿射变形场，扩展为稠密光流；k-means 聚类（BIC 合并）过滤对称歧义与不合理运动，生成伪对应点集 $P_{self}$；最终通过回归损失监督学生网络。

**输出**：源图像到目标图像的稠密语义对应场，支持关键点匹配与稠密光流估计。

```
I_s, I_t → [DINOv2 骨干(冻结)] → patch tokens
                ↓
        [AdaptFormer ×12] → 增强特征
                ↓
        [上采样头] → 高分辨率特征图
                ↓
        ├─→ [粗到细监督: σ=3→1] ← 稀疏 GT 关键点
        └─→ [稠密自蒸馏: Flow Anchoring] ← EMA教师 + MNN + Delaunay + k-means
                ↓
        稠密对应预测 D(u) + 关键点匹配
```

## 核心模块与公式推导

### 模块 1：粗到细高斯监督调度（对应框架图"训练监督"分支）

**直觉**：固定窄高斯（σ=1）强迫模型过早关注精细定位，导致粗粒度语义对齐崩溃；固定宽高斯（σ=3）则相反。通过动态调度，模型先建立可靠的粗对应关系，再逐步精化至子patch 级别。

**Baseline 公式**（标准 soft-argmax 回归）：
$$L_{base} = \sum_{i} \left\| \hat{p}_i - p_i^{GT} \right\|^2, \quad \hat{p}_i = \text{soft-argmax}(M_i)$$
符号：$\hat{p}_i$ 为预测关键点位置，$p_i^{GT}$ 为真值，$M_i$ 为对应热力图。标准变体以高斯分布 $G(p_i^{GT}, \sigma^2 I)$ 作为监督目标，通常固定 $\sigma=1$。

**变化点**：固定 $\sigma$ 导致"精细-粗粒度"权衡僵化。MARCO 引入时变高斯目标，使监督信号随训练epoch 自适应调整。

**本文公式**：
$$\text{Step 1}: \sigma(t) = \sigma_{max} - (\sigma_{max} - \sigma_{min}) \cdot \frac{t}{T_{sched}} \quad \text{线性退火，} \sigma_{max}=3, \sigma_{min}=1$$
$$\text{Step 2}: L_{gauss}(t) = -\sum_{i} \sum_{u \in \Omega} G(u; p_i^{GT}, \sigma(t)^2 I) \cdot \log \frac{\exp(M_i(u)/\tau)}{\sum_{v}\exp(M_i(v)/\tau)}$$
$$\text{最终}: L_{sched} = L_{gauss}(t) + \lambda_{coord} \cdot L_{coord}, \quad L_{coord} = \sum_i \|\hat{p}_i - p_i^{GT}\|^2$$
其中 $\tau$ 为温度系数，$L_{coord}$ 为直接坐标回归辅助损失，$\lambda_{coord}$ 为平衡权重。退火调度保证训练初期 $\sigma(t)\approx 3$ 时模型学习鲁棒语义对应，后期 $\sigma(t)\rightarrow 1$ 时聚焦精细定位。

**对应消融**：固定 $\sigma=1$ 时 PCK@0.01=27.8 但 PCK@0.10 下降；固定 $\sigma=3$ 时 PCK@0.10 高但 PCK@0.01 显著降低。调度策略取得 27.0@0.01 / 87.2@0.10 的平衡最优。

---

### 模块 2：稠密自蒸馏 via Flow Anchoring（对应框架图右侧循环）

**直觉**：稀疏 GT 仅覆盖 ~20 点/图，无法描述完整语义表面。通过教师-学生框架将稀疏匹配扩展为稠密光流，再用几何一致性过滤噪声，可突破标注密度的物理限制。

**Baseline 公式**（标准 EMA 自蒸馏）：
$$\theta_{teacher} \leftarrow \alpha \theta_{teacher} + (1-\alpha) \theta_{student}, \quad L_{distill} = \sum_{i \in \mathcal{G}} \|f_{student}(i) - f_{teacher}(i)\|^2$$
仅在有 GT 的关键点集合 $\mathcal{G}$ 上计算蒸馏损失，监督密度受限。

**变化点**：Baseline 的蒸馏受限于 GT 覆盖范围。MARCO 通过**光流锚定**将稀疏种子点扩展为稠密对应场，再用**聚类过滤**剔除歧义区域，实现"有选择的全图自监督"。

**本文公式推导**：
$$\text{Step 1 (种子挖掘)}: \mathcal{S} = \mathcal{G} \cup \text{MNN}(F_s, F_t), \quad F_s, F_t \in \mathbb{R}^{H\times W \times C}$$
互最近邻（MNN）筛选双向一致的高置信匹配，与 GT 合并为种子集 $\mathcal{S}$。

$$\text{Step 2 (Delaunay 三角剖分 + 分段仿射)}: \mathcal{T} = \text{Delaunay}(\{p_s\}_{s \in \mathcal{S}}), \quad D(u) = A_k \cdot u + b_k, \forall u \in \text{triangle}_k$$
对源图像种子点做 Delaunay 三角剖分 $\mathcal{T}$，每个三角形 $\text{triangle}_k$ 内估计仿射参数 $(A_k, b_k)$，生成稠密位移场 $D(u)$。

$$\text{Step 3 (光流向量聚类与过滤)}: \{\mathcal{C}_j\} = \text{k-means}(\{D(u)\}_{u \in \Omega}), \text{ BIC 合并}$$
$$\mathcal{P}_{self} = \text{bigcup}_{j: \mathcal{C}_j \cap \mathcal{G} \neq \emptyset} \{(u, u+D(u)) : u \in \mathcal{C}_j\}$$
对光流向量聚类，**仅保留包含至少一个 GT 关键点的聚类**，过滤对称歧义（如物体左右翻转导致的错误匹配）和非刚性区域的失真变形。

$$\text{Step 4 (自蒸馏回归)}: L_{self} = \sum_{(u,v) \in \mathcal{P}_{self}} \|D_{student}(u) - (v-u)\|^2$$

$$\text{Step 5 (EMA 更新)}: \theta_{teacher} \leftarrow \alpha \theta_{teacher} + (1-\alpha) \theta_{student}$$

$$\text{最终总损失}: L_{total} = L_{sched} + \lambda_{self} \cdot L_{self}$$

**对应消融**：移除稠密自蒸馏（$L_{self}=0$）导致 SPair-U 泛化崩溃：PCK@0.10 从 67.5 降至 41.8（$\Delta$ -25.7）；GT 锚定聚类过滤相比无过滤版本提升 +2.8@0.10（64.7→67.5），且不损害域内精度。

## 实验与分析

| Method | SPair-71k PCK@0.01 | SPair-71k PCK@0.10 | SPair-U PCK@0.10 | MP-100 PCK@0.10 | 参数量 | FPS (RTX4090@840p) |
|:---|:---|:---|:---|:---|:---|:---|
| SD+DINOv2 (双编码器) | — | ~86 | — | — | ~950M | 0.85 |
| DIFT | — | ~75 | — | — | 304M | — |
| DistillDIFT | 18.1 | — | — | — | 304M | — |
| GECO | — | ~80 | — | — | — | — |
| Jamais Vu | — | ~62.4 | — | — | — | — |
| **MARCO** | **27.0** | **87.2** | **67.5** | **+4.7 vs best** | **323M** | **8.3** |


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/cf92e722-c220-46f4-9f69-26bafb5b30a6/figures/Figure_1.png)
*Figure 1: Figure 1. MARCO, a model for generalizable correspondences. Built on DINOv2, MARCO explores the unseen space of semanticcorrespondence by inferring structure that lies beyond the sparsity of keypoint*



**核心数字解读**：SPair-71k PCK@0.01 = 27.0 是单骨干方法首次逼近双编码器水平的标志性结果，较 DistillDIFT 的 18.1 提升 +8.9（相对 +49%），直接支撑"单骨干可实现精细定位"的核心 claim。PCK@0.10 = 87.2 超越部分双编码器变体，证明粗到细调度未牺牲整体对应质量。

**泛化能力**：SPair-U（未见关键点）PCK@0.10 = 67.5 较 Jamais Vu 提升 +5.1，较无自蒸馏版本提升 +25.7；MP-100（未见类别）+4.7 的优势表明 Flow Anchoring 的聚类过滤机制有效抑制了跨类别转移时的语义漂移。这两个数字是 MARCO 区别于"在标准测试上刷点"方法的关键证据。

**效率验证**：323M 参数量 vs 双编码器 ~950M（3×压缩），8.3 FPS vs 0.85 FPS（10×加速），在 RTX 4090 @ 840p 条件下测量。需注意速度测试依赖特定批处理设置，实际部署可能有波动。

**消融分析**（
![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/cf92e722-c220-46f4-9f69-26bafb5b30a6/figures/Figure_5.png)
*Figure 5: Figure 4. Correspondence mining via flow anchoring. Given asource and target image, the dense displacement field is estimatedfrom sparse matches using piece-wise affine motion on a Delau-nay triangula*

）：AdaptFormer 适配器贡献 +8.1@0.01 / +23.0@0.10；上采样头贡献 +12.6@0.01 / +10.3@0.10；两者协同实现精细定位突破。稠密自蒸馏的 -25.7 崩溃实验（SPair-U）是最强因果证据，证明泛化提升并非来自适配器 alone。

**公平性检查**：双编码器 baseline 的精确数字来自原文 Figure 1 气泡图而非独立结果表，存在读取误差风险；MP-100 为作者自行设计的评估协议，存在协议偏向的潜在风险。分段仿射假设在大位移/非刚性形变区域可能失效，但原文未报告具体失败案例。粗到细调度的超参数（$\sigma_{max}, \sigma_{min}, T_{sched}$）敏感性未充分探索。

## 方法谱系与知识库定位

**方法家族**：DINOv2-based Semantic Correspondence → 单骨干高效化分支

**父方法**：DINOv2（ViT-L/14 预训练特征）+ DistillDIFT（单骨干蒸馏范式）。MARCO 继承 DistillDIFT 的"冻结 DINOv2 + 轻量微调"效率哲学，但彻底重构了监督机制与架构适配策略。

**改动槽位**：
- **架构**：新增 AdaptFormer 适配器（12层）+ 子patch 上采样头，突破 DINOv2 的 patch 粒度限制
- **目标函数**：粗到细高斯调度 + 稠密自蒸馏回归，替代纯 GT 监督
- **训练配方**：EMA 教师-学生框架 + Flow Anchering 伪标签生成
- **数据策划**：无外部数据，但重新利用内部特征流构建自监督信号
- **推理**：与 DistillDIFT 一致，单前向传播，无额外计算

**直接 Baselines 与差异**：
- **DistillDIFT**：同为单 DINOv2 骨干，但无适配器、无上采样、无自蒸馏，精度显著落后
- **GECO**：引入几何编码，但仍是稀疏 GT 监督，无稠密扩展机制
- **Jamais Vu**：专注泛化性，但采用不同技术路线（未见具体架构），MARCO 以 Flow Anchoring 实现更优泛化-效率权衡
- **SD+DINOv2 双编码器**：精度标杆但效率极低，MARCO 以 1/3 参数、10× 速度逼近其性能

**后续方向**：(1) 将 Flow Anchoring 扩展至视频时序对应，利用帧间连续性进一步稠密化；(2) 结合扩散模型的注意力图作为额外教师信号，弥补纯 DINOv2 特征的粒度限制；(3) 探索无 $\sigma$ 调度的自适应不确定性估计，替代启发式高斯宽度退火。

**知识库标签**：`modality: 图像-图像对应` / `paradigm: 自蒸馏 + 适配器微调` / `scenario: 跨实例语义匹配，强调未见关键点/类别的真实泛化` / `mechanism: EMA教师-学生、Delaunay三角剖分光流、k-means聚类过滤` / `constraint: 单骨干效率约束、稀疏标注瓶颈`

