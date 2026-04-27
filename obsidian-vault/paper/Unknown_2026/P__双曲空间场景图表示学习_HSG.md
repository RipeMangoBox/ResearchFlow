---
title: 'HSG: Hyperbolic Scene Graph'
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.17454
aliases:
- 双曲空间场景图表示学习
- HSG
- 真实场景的place-object关系本质上是树状层级结构
code_url: https://github.com/AIGeeksGroup/HSG
method: HSG
modalities:
- Image
---

# HSG: Hyperbolic Scene Graph

[Paper](https://arxiv.org/abs/2604.17454) | [Code](https://github.com/AIGeeksGroup/HSG)

**Topics**: [[T__Retrieval]], [[T__3D_Reconstruction]], [[T__Contrastive_Learning]] | **Method**: [[M__HSG]]

> [!tip] 核心洞察
> 真实场景的place-object关系本质上是树状层级结构，而双曲空间的体积随半径指数增长，天然与树状结构的节点数增长规律匹配。将嵌入空间从欧氏平坦空间替换为双曲流形，使得place（靠近原点，更抽象）对object（远离原点，更具体）的层级蕴含关系可以通过几何距离直接编码，而无需额外的结构约束。蕴含锥损失进一步将这种几何直觉转化为可优化的训练信号，强制object嵌入落在place定义的蕴含锥内，从而将隐式的几何偏置转化为显式的结构约束。

| 中文题名 | 双曲空间场景图表示学习 |
| 英文题名 | HSG: Hyperbolic Scene Graph |
| 会议/期刊 | arXiv 2026 (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.17454) · [Code](https://github.com/AIGeeksGroup/HSG) · [Project](https://arxiv.org/abs/2604.17454) |
| 主要任务 | 多视图场景图构建、place-object层级关系编码、跨视图场景图检索 |
| 主要 baseline | SepMSG, AoMSG (含 AoMSG-B-4 等变体) |

> [!abstract] 因为「欧氏空间无法自然表达place-object层级蕴含关系，导致图级结构指标低下（最佳AoMSG Graph IoU仅25.37）」，作者在「MSG框架」基础上改了「将嵌入空间替换为Lorentz双曲流形并引入蕴含损失」，在「多视图场景图基准」上取得「Graph IoU 33.51（+8.14），PP IoU 33.17，Recall@1 98.39」

- **Graph IoU**: 33.51 vs. AoMSG最佳 25.37，绝对提升 **+8.14**
- **PP IoU**: 33.17 vs. AoMSG最佳 25.37，绝对提升 **+7.80**
- **Recall@1**: 98.39，与最强基线 AoMSG-B-4 (98.61) 差距仅 **0.22**，层级提升未牺牲检索性能

## 背景与动机

真实世界的场景理解天然具有层级结构：一个"厨房"（place）语义上蕴含"冰箱"、"灶台"、"橱柜"（objects）的存在，而objects之间也存在"家具"→"橱柜"→"抽屉"这样的细粒度层级。这种层级关系是非对称的、树状的——place是更抽象的上位概念，object是更具体的下位实例。然而，现有场景图方法并未在几何层面显式编码这一结构。

现有方法以MSG（Multi-view Scene Graph）系列为代表，包括SepMSG和AoMSG两个主要变体。SepMSG采用分离式编码器分别处理place和object，通过对比学习拉近匹配对；AoMSG引入注意力机制聚合多视图特征，在欧氏空间中以L2归一化和余弦相似度度量嵌入距离。这些方法在place检索（Recall@1）上表现优异，但其嵌入空间是"平坦"的——欧氏空间中任意两点的距离具有平移不变性，无法区分"抽象-具体"的层级远近。例如，"厨房"到"冰箱"的距离与"厨房"到"客厅"的距离在几何上无本质区别，都需要模型从数据中学习而非空间结构自然承载。

这一几何不匹配导致严重缺陷：MSG系列在衡量层级结构质量的图级指标上表现有限。具体而言，最佳AoMSG变体的Graph IoU仅为25.37，PP IoU同样为25.37，说明嵌入空间虽能区分不同场景，却未能将place-object的层级关系编码为可度量的几何结构。换言之，模型学会了"这是厨房"，却没学会"厨房包含冰箱"。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/31740a5c-da04-4e20-a4c2-8ed482e4e7ec/figures/Figure_1.png)
*Figure 1: Fig. 1: Hyperbolic place-object representations. Left: A place-centric visual-semantic hierarchy where a central scene representation (black dot) entails multipleobject-level observations (surrounding*



核心洞察在于：欧氏空间的体积随半径多项式增长，而树状结构的节点数随深度指数增长——两者存在根本性的尺度不匹配。双曲空间恰好相反，其体积随半径指数增长，天然与层级结构的膨胀规律契合。本文将MSG框架的嵌入空间从欧氏空间迁移至双曲空间，使层级关系成为几何的内禀属性。

## 核心创新

核心洞察：双曲空间的体积随半径指数增长，与树状层级结构的节点数增长规律天然匹配，从而使place（近原点、抽象）对object（远原点、具体）的层级蕴含关系可以通过几何距离直接编码，无需额外的结构约束。

与baseline的差异：

| 维度 | Baseline (AoMSG/SepMSG) | 本文 (HSG) |
|:---|:---|:---|
| 嵌入空间 | 欧氏空间，L2归一化 + 余弦相似度 | Lorentz双曲流形，双曲距离度量 |
| 层级关系 | 隐式，依赖对比学习从数据拟合 | 显式，蕴含锥损失强制几何约束 |
| 曲率 | 无（平坦空间） | 可学习曲率参数c（初始80，最优区间30~250） |
| 投影机制 | 线性/MLP projector直接输出 | Exp_o指数映射→双曲流形，Log_o对数映射回切空间 |
| 图构建流水线 | 完整保留 | 完全兼容，无需修改 |

关键设计选择：HSG未改变MSG的backbone（DINOv2-Base）、数据集、评估协议或图构建流程，仅替换嵌入空间并扩展损失函数，属于最小侵入式的表示空间迁移。

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/31740a5c-da04-4e20-a4c2-8ed482e4e7ec/figures/Figure_2.png)
*Figure 2: Fig. 2: Scene graph construction and cross-view consistency. Left: Multi-viewobservations are aggregated into a hierarchical scene graph grounded in the recon-structed 3D scene. Right: The same object*



HSG的整体数据流遵循MSG标准流水线，在三个关键节点注入双曲几何：

**输入**: 多视图图像序列 {I_1, I_2, ..., I_n}，经预训练视觉编码器（DINOv2-Base）提取patch级特征。

**模块A - 特征编码（保留）**: 与MSG相同，采用ViT backbone输出图像特征，通过标准projector（维度1024最优）降维。此处不做任何修改以保持特征提取能力。

**模块B - 双曲投影（新增）**: 将欧氏特征通过**指数映射** Exp_o: T_oH_c^d → H_c^d 投影到Lorentz双曲流形H_c^d。这是核心几何替换：原L2归一化向量变为双曲流形上的点，距离度量从余弦相似度切换为双曲距离。

**模块C - 场景图构建（兼容）**: 在双曲空间中执行与MSG相同的图构建操作——聚合多视图特征为节点、建立place-object边关系。双曲距离自然编码"近原点=抽象place，远原点=具体object"的层级语义。

**模块D - 训练目标（扩展）**: 原InfoNCE对比损失保留，新增**蕴含损失** L_entailment，通过双曲蕴含锥强制object嵌入落在place嵌入定义的锥形区域内。

**模块E - 推理回退（兼容）**: 训练完成后通过**对数映射** Log_o: H_c^d → T_oH_c^d 将双曲嵌入映射回切空间，供下游任务使用。此设计确保与现有MSG评估工具链的零修改兼容。

```
多视图图像 → [DINOv2-Base] → 欧氏特征 → [Projector] → d维向量
                                                    ↓
                                            [Exp_o指数映射]
                                                    ↓
                                         Lorentz双曲流形 H_c^d
                                                    ↓
                              [场景图构建: 双曲距离聚合 + 蕴含锥约束]
                                                    ↓
                                         [Log_o对数映射] ← 推理时
                                                    ↓
                                              下游任务
```


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/31740a5c-da04-4e20-a4c2-8ed482e4e7ec/figures/Figure_3.png)
*Figure 3: Fig. 3: HSG model design: HSG adopts a similar architecture to MSG, but replacesL2-normalized hyperspherical embeddings and cosine similarity with Lorentz hyper-boloid embeddings via the exponential m*



## 核心模块与公式推导

### 模块1: 双曲嵌入投影（对应框架图 模块B）

**直觉**: 欧氏空间的平坦几何无法承载层级结构的指数膨胀，需将特征映射到体积指数增长的双曲流形，使原点远近自然对应抽象-具体层级。

**Baseline公式 (MSG系列)**: 欧氏空间中采用L2归一化与余弦相似度：
$$\mathbf{e}_{\text{euclid}} = \frac{\text{MLP}(f)}{\|\text{MLP}(f)\|_2}, \quad s(a,b) = \frac{\mathbf{e}_a \cdot \mathbf{e}_b}{\|\mathbf{e}_a\| \|\mathbf{e}_b\|}$$
符号: $f$ = backbone特征, MLP = projector, $s$ = 相似度得分（越高越匹配）。

**变化点**: 余弦相似度具有平移不变性，"厨房-冰箱"与"厨房-客厅"的高相似度在几何上等价，无法编码蕴含方向；L2归一化将所有嵌入压到球面，丢失层级远近信息。

**本文公式（推导）**:
$$\text{Step 1}: \mathbf{x} = \text{Projector}(f) \in \mathbb{R}^d \quad \text{(保持与MSG相同的projector结构)}$$
$$\text{Step 2}: \mathbf{h} = \text{Exp}_o(\mathbf{x}) = \left(\sqrt{\|\mathbf{x}\|^2 + \frac{1}{c}}, \mathbf{x}\right) \in \mathbb{H}_c^d \quad \text{(指数映射到Lorentz流形)}$$
$$\text{Step 3}: \text{训练时直接在} \mathbb{H}_c^d \text{上计算双曲距离} \quad d_{\mathcal{H}}(\mathbf{h}_i, \mathbf{h}_j) = \frac{1}{\sqrt{c}} \text{arcosh}\left(-c \langle \mathbf{h}_i, \mathbf{h}_j \rangle_{\mathcal{L}}\right)$$
$$\text{推理时}: \mathbf{x}' = \text{Log}_o(\mathbf{h}) \quad \text{(对数映射回切空间，兼容下游)}$$
符号: $c$ = 可学习曲率（初始80）, $\langle \cdot, \cdot \rangle_{\mathcal{L}}$ = Lorentz内积, $\mathbb{H}_c^d$ = d维双曲流形。

**对应消融**: 固定$c=1$时PP IoU从33.2骤降至15.5，说明可学习曲率是层级结构有效性的关键；projector维度1024最优，欧氏基线对维度不敏感。

---

### 模块2: 蕴含损失函数（对应框架图 模块D）

**直觉**: 双曲空间的几何偏置是"隐式"的——近原点更抽象——需通过显式约束将这一偏置转化为可优化的训练信号，强制object嵌入落在place定义的锥形区域内。

**Baseline公式 (AoMSG)**: 仅InfoNCE对比损失：
$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(s(p, o^+)/\tau)}{\sum_{o \in \mathcal{N}} \exp(s(p, o)/\tau)}$$
符号: $p$ = place嵌入, $o^+$ = 正样本object, $\mathcal{N}$ = 负样本集合, $\tau$ = 温度系数, $s$ = 余弦相似度。

**变化点**: InfoNCE仅约束"匹配的place-object应相近"，不约束"object应在place的层级下方"。即使双曲嵌入有层级偏置，无显式监督时该偏置可能不足以形成清晰结构（消融验证：移除蕴含损失后PP IoU从33.2降至21.5）。

**本文公式（推导）**:
$$\text{Step 1}: \text{定义孔径函数} \quad \omega(\mathbf{h}_p) = 2 \cdot \arcsin\left(\frac{2\eta}{\|\mathbf{h}_p\|_{\mathcal{L}}}\right) \quad \text{(place嵌入范数决定锥的开口大小)}$$
$$\text{Step 2}: \text{蕴含锥约束} \quad \text{angle}(\mathbf{h}_p, \mathbf{h}_o) \leq \omega(\mathbf{h}_p) \quad \text{(object必须落在place的锥内)}$$
$$\text{Step 3}: \text{转化为可优化损失} \quad \mathcal{L}_{\text{entailment}} = \sum_{(p,o^+)} \max\left(0, \text{angle}(\mathbf{h}_p, \mathbf{h}_o) - \omega(\mathbf{h}_p)\right)$$
$$\text{最终}: \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{InfoNCE}} + \lambda \cdot \mathcal{L}_{\text{entailment}}$$
符号: $\eta$ = 固定超参数（控制锥的紧致程度）, $\text{angle}(\cdot, \cdot)$ = 双曲角度, $\lambda$ = 损失权重。

**对应消融**: Table 2显示移除蕴含损失（仅双曲空间+InfoNCE）PP IoU = 21.5，较完整HSG（33.2）下降11.7；固定$c=1$+无蕴含损失时PP IoU = 15.5，两者结合缺一不可。

## 实验与分析

**主实验结果**:

| Method | Recall@1 ↑ | PO IoU | PP IoU ↑ | Graph IoU ↑ |
|:---|:---|:---|:---|:---|
| SepMSG-Linear | — | — | — | — |
| SepMSG-MLP | — | 58.63 | — | — |
| AoMSG (最佳) | 98.61 | — | 25.37 | 25.37 |
| **HSG (本文)** | **98.39** | **45.52** | **33.17** | **33.51** |
| Δ vs. AoMSG | -0.22 | — | **+7.80** | **+8.14** |


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/31740a5c-da04-4e20-a4c2-8ed482e4e7ec/figures/Figure_5.png)
*Figure 5: Fig. 5: Distribution of embedding distances from [ROOT]: We embed train-ing images using trained HSG, AoMSG and SepMSG, where for HSG we calculateLorentzian norm d(v) = ||vspace|| and L2 norm for AoMS*



**核心发现分析**:

1. **层级结构质量跃升**: Graph IoU 33.51 vs. AoMSG 25.37（+8.14）是本文最核心的证据。Graph IoU衡量预测场景图与GT场景图的边结构重叠，直接反映层级编码质量。双曲几何+蕴含损失的联合作用使这一指标产生质的飞跃，验证了"几何结构即语义结构"的核心假设。

2. **检索性能无损**: Recall@1 98.39与AoMSG-B-4的98.61差距仅0.22，在误差范围内。这说明双曲嵌入的层级约束未压缩判别性特征空间，解决了"结构-判别"权衡的常见顾虑。

3. **PO IoU的trade-off**: HSG的PO IoU为45.52，明显低于SepMSG-Linear和SepMSG-MLP（55.67~58.63）。PO IoU衡量place-object边的精确关联，双曲表示在此指标上不占优势。论文将此解释为固有trade-off——双曲空间优化层级结构时，对object级别的精确空间关联约束减弱——但未提供改进方案。



**消融实验深度分析**:

| 配置 | PP IoU | 说明 |
|:---|:---|:---|
| 完整HSG | 33.2 | 双曲空间 + 可学习c + 蕴含损失 |
| 移除蕴含损失 | 21.5 | 仅双曲空间+InfoNCE，层级结构大幅退化 |
| 固定c=1 | 15.5 | 蕴含损失退化为零，几乎失效 |
| 两者皆移除 |  | 退化为最简双曲baseline |

蕴含损失是层级质量的**主要来源**（贡献+11.7），可学习曲率是**必要前提**（固定c=1时损失失效）。两者具有强耦合性：曲率决定了双曲空间的"弯曲程度"，进而决定蕴含锥的几何有效性。

**公平性检查与局限**:
- **Baseline强度**: 仅与MSG系列对比，未与MERU、HypSG等其他双曲场景图方法直接比较，竞争范围偏窄
- **数据异常**: Table 3中ConvNeXt-Base与DINOv2-Large的PP IoU=32.94、Graph IoU=33.22完全相同，疑似录入错误
- **超参数敏感**: 曲率初始化最优区间30~250，峰值约80，偏离时性能下降显著
- **计算成本**: 双曲运算（arcosh, arcsinh）较欧氏运算有额外开销，但论文未报告具体训练/推理时间

## 方法谱系与知识库定位

**方法家族**: 多视图场景图表示学习 → MSG系列（SepMSG/AoMSG）→ **HSG（双曲化扩展）**

**父方法**: AoMSG（注意力聚合多视图场景图）。HSG继承其完整的图构建流水线、backbone架构（DINOv2-Base）、数据集与评估协议，属于**表示空间替换+损失函数扩展**的最小侵入式改进。

**改动插槽**:
- **架构**: 保留（Projector后新增Exp_o/Log_o映射层，属轻量适配）
- **目标/损失**: 修改（InfoNCE + 新增蕴含损失）
- **训练配方**: 扩展（新增曲率参数c的优化，需特殊初始化策略）
- **数据策划**: 保留
- **推理**: 兼容（Log_o回退至切空间）

**直接基线对比**:
- **SepMSG**: 分离式编码器，无注意力聚合。HSG与之共享双曲替换思路但架构更先进（基于AoMSG）
- **AoMSG**: 注意力聚合+欧氏空间。HSG替换为双曲空间+蕴含损失，Graph IoU +8.14
- **MERU/HypSG**: 其他双曲视觉表示方法，HSG未直接对比，存在方法学交叉

**后续方向**:
1. **PO IoU改进**: 探索双曲空间中的object精确定位机制，缓解当前trade-off
2. **动态曲率调度**: 当前固定初始化后学习，可借鉴学习率预热设计曲率退火策略
3. **跨模态扩展**: 将双曲层级结构迁移至视觉-语言预训练（如CLIP风格），编码更广泛的语义层级

**知识库标签**:
- **模态**: 视觉（多视图图像）
- **范式**: 对比学习 + 几何约束优化
- **场景**: 室内场景理解 / 场景图构建 / 跨视图检索
- **机制**: 双曲几何表示 / 蕴含锥约束 / 可学习曲率
- **约束**: 保持下游兼容性 / 最小架构修改 / 层级结构可解释性

