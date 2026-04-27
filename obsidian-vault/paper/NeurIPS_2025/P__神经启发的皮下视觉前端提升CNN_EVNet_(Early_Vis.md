---
title: Explicitly Modeling Subcortical Vision with a Neuro-Inspired Front-End Improves CNN Robustness
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 神经启发的皮下视觉前端提升CNN鲁棒性
- EVNet (Early Vis
- EVNet (Early Vision Network)
- Explicitly modeling subcortical vis
acceptance: Poster
cited_by: 1
code_url: https://github.com/lucaspiper99/evnet
method: EVNet (Early Vision Network)
modalities:
- Image
paradigm: supervised
---

# Explicitly Modeling Subcortical Vision with a Neuro-Inspired Front-End Improves CNN Robustness

[Code](https://github.com/lucaspiper99/evnet)

**Topics**: [[T__Classification]], [[T__Adversarial_Robustness]] | **Method**: [[M__EVNet]] | **Datasets**: [[D__ImageNet-1K]] (其他: Aggregate robustness benchmark, Aggregate robustness with PRIME data augmentation, V1 response property alignment, Domain shift datasets, Aggregate Robustness Benchmark)

> [!tip] 核心洞察
> Explicitly modeling subcortical visual processing with a novel SubcorticalBlock front-end, combined with V1 modeling, improves CNN robustness and neural alignment beyond VOneNets alone.

| 中文题名 | 神经启发的皮下视觉前端提升CNN鲁棒性 |
| 英文题名 | Explicitly Modeling Subcortical Vision with a Neuro-Inspired Front-End Improves CNN Robustness |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2506.03089) · [Code](https://github.com/lucaspiper99/evnet) · [Project](未提供) |
| 主要任务 | ImageNet-1k 图像分类、对抗鲁棒性、常见图像损坏鲁棒性、域迁移鲁棒性 |
| 主要 baseline | VOneNet（直接基线）、ResNet50（CNN后端基线）、PRIME（数据增强基线） |

> [!abstract] 因为「CNN对视觉扰动脆弱且现有神经启发方法仅建模V1皮层而忽略视网膜/LGN皮下处理」，作者在「VOneNet」基础上改了「添加SubcorticalBlock（并行P/M通路含光适应、DoG卷积、对比度归一化、神经噪声）并修改VOneBlock」，在「综合鲁棒性基准（ImageNet-C + 域迁移 + 对抗扰动）」上取得「相比基线CNN提升+9.3%，相比PRIME单独使用提升+6.2%」

- **+9.3%**：EVNet在综合鲁棒性基准上相比基线ResNet50的提升
- **+6.2%**：EVNet相比单独PRIME数据增强的额外增益
- **V1对齐**：EVNet在mean V1 response property alignment上优于VOneResNet50和ResNet50

## 背景与动机

尽管CNN在ImageNet等标准基准上达到了超人类精度，它们对轻微图像扰动——如高斯噪声、运动模糊、对抗补丁——却极度脆弱。这种脆弱性与生物视觉系统形成鲜明对比：人类和灵长类动物能在恶劣天气、光照变化或部分遮挡下稳健识别物体。一个具体例子是，给一张猫的图片添加轻微雾气和对比度降低，ResNet50可能将其误分类为"狐狸"，而人类几乎不受影响。

现有方法如何尝试解决这一问题？**VOneNet** [12] 在CNN前端添加了VOneBlock，用Gabor滤波器组模拟V1简单/复杂细胞的感受野特性，并注入神经噪声，显著提升了损坏鲁棒性。**PRIME** 等数据增强方法则通过在训练时合成多样化损坏样本来增强泛化。**Divisive Normalization** [13] 将V1的除法归一化机制嵌入CNN层，改善常见损坏鲁棒性。

然而，这些方法存在根本性局限：VOneNet直接从像素输入模拟V1皮层，完全跳过了视网膜和外侧膝状体（LGN）的皮下处理阶段；而数据增强和归一化方法缺乏对早期视觉通路的结构性建模。生物视觉中，视网膜神经节细胞通过中心-周边拮抗（center-surround antagonism）和并行P/M通路（parvocellular/magnocellular pathways）对光照、对比度进行适应性预处理，这些机制对后续皮层的稳健表征至关重要。VOneNet的缺失导致其无法复现皮下视觉的关键现象，如光适应、对比度饱和和surround抑制。

本文的核心动机正是填补这一空白：显式建模皮下视觉处理是否能像生物系统一样，为CNN提供更鲁棒的早期表征？

## 核心创新

**核心洞察**：生物视觉的鲁棒性不仅源于V1皮层，更源于视网膜和LGN的并行预处理——因为皮下P/M通路的光适应、中心-周边拮抗和对比度增益控制协同塑造了稳定输入，从而使在固定权重前端中显式嵌入这些机制、无需端到端训练即可提升下游鲁棒性成为可能。

| 维度 | Baseline (VOneNet) | 本文 (EVNet) |
|:---|:---|:---|
| **前端范围** | 仅V1皮层（Gabor滤波器组 + 简单/复杂细胞非线性 + 神经噪声） | V1皮层 **+** 视网膜/LGN皮下处理（SubcorticalBlock） |
| **并行通路** | 无 | 分离的P通路（颜色拮抗：红-绿、绿-红、蓝-黄）和M通路（无色差） |
| **预处理机制** | 标准ImageNet归一化 [0.5,0.5,0.5] | 光适应驱动的全局亮度归一化（无标准归一化），含ε=0.05阈值修正 |
| **感受野建模** | V1经典RF（8deg FoV，原始空间频率范围） | 扩展至更高空间频率（7deg FoV）+ 皮下DoG中心-周边RF |
| **跨层连接** | 标准前馈 | 可选LGN-V2跳跃连接（5×5 max-pool, stride 4，64通道拼接） |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/3b15d126-8d15-4295-91d5-8e8ed4aac5f2/figures/Figure_1.png)
*Figure 1 (pipeline): Simulating primate early visual processing as CNN front-end blocks*



EVNet的完整数据流遵循灵长类早期视觉通路的层级结构，从视网膜经LGN到达V1皮层：

**Input Image** (224×224 RGB) → 原始像素值，对含光适应的模型跳过标准[0.5,0.5,0.5]归一化。

**SubcorticalBlock**（核心创新模块）→ 并行处理P通路（3个颜色拮抗DoG通道）和M通路（1个无色差DoG通道），每个通路内部包含四个级联操作：
- *Light Adaptation*：全局亮度归一化，计算(x - x̄)/x̄，其中x̄忽略低于ε=0.05的像素；
- *DoG Convolution*：中心-周边拮抗的高斯差分卷积；
- *Contrast Normalization*：基于局部对比度估计的除法归一化；
- *Neural Noise*：模拟皮下噪声统计特性的随机注入。

**Modified VOneBlock** → 接收SubcorticalBlock输出（或经LGN-V2跳跃连接的64通道拼接张量），使用7deg FoV和扩展至高空间频率的Gabor滤波器组模拟V1处理。

**Standard CNN Back-end** → ResNet50、EfficientNet-B0或CORnet-Z等标准架构，输出1000类分类logits。

```
[RGB Image] → [Light Adaptation] → [DoG Conv] → [Contrast Norm] → [Neural Noise] → 
                                                                    ↓ (optional skip: 5×5 max-pool, stride 4)
                                                                    ↓
[Modified VOneBlock: 7deg FoV, extended SF] ───────────────────────→[concat 60+4=64 ch]→ [CNN Back-end] → logits
```

## 核心模块与公式推导

### 模块 1: Light Adaptation（光适应）——对应框架图 SubcorticalBlock第一层

**直觉**：视网膜神经节细胞通过全局亮度归一化适应环境光照变化，使系统对绝对亮度不敏感而对相对对比度敏感。

**Baseline 公式** (无，此机制为EVNet新增):
标准ImageNet预处理仅做静态归一化：$x_{\text{norm}} = (x - 0.5) / 0.5$

**本文公式（推导）**:
$$\text{Step 1}: x_{\text{LA}} = \frac{x - \bar{x}}{\bar{x}} \quad \text{全局均值归一化，模拟视网膜光适应}$$
$$\text{Step 2}: \bar{x}_{\text{modified}} = \text{mean}(x \text{mid} x \geq \epsilon), \quad \epsilon = 0.05 \quad \text{忽略近零像素防止暗区激活爆炸}$$
$$\text{最终}: x_{\text{LA}} = \frac{x - \bar{x}_{\text{modified}}}{\bar{x}_{\text{modified}}}$$

符号：$x$为输入像素值，$\bar{x}$为全局空间-通道均值，$\epsilon=0.05$为消融实验引入的阈值修正。

**变化点**：原始光适应在暗背景含小亮区时，极低均值会导致归一化后激活爆炸；移除对比度归一化后该问题暴露，故引入ε阈值修正均值计算。

**对应消融**：Table 6显示各组件贡献差异显著；光适应无对比度归一化时训练不稳定。

---

### 模块 2: DoG Convolution（高斯差分卷积）——对应框架图 SubcorticalBlock第二层

**直觉**：视网膜神经节细胞和LGN神经元具有中心-周边拮抗的空间感受野，用两个高斯函数之差可精确建模此特性。

**Baseline 公式** (VOneNet无此机制，直接从像素到Gabor):
VOneBlock直接使用Gabor滤波器：$w_{\text{Gabor}}(x,y; \lambda, \theta, \psi, \sigma, \gamma)$

**本文公式（推导）**:
$$\text{Step 1}: w_{\text{center}}(x,y) = \exp\left(-\frac{x^2 + y^2}{r_c^2}\right) \quad \text{中心高斯（兴奋性）}$$
$$\text{Step 2}: w_{\text{surround}}(x,y) = \frac{k_s}{k_c}\exp\left(-\frac{x^2 + y^2}{r_s^2}\right) \quad \text{周边高斯（抑制性）}$$
$$\text{最终}: w_{\text{DoG}}(x, y) = w_{\text{center}} - w_{\text{surround}} = \exp\left(-\frac{x^2 + y^2}{r_c^2}\right) - \frac{k_s}{k_c} \exp\left(-\frac{x^2 + y^2}{r_s^2}\right)$$

符号：$r_c$为中心半径，$r_s$为周边半径（$r_s > r_c$），$k_s/k_c$为峰值对比敏感度比值，控制surround抑制强度。

**变化点**：VOneNet直接对原始像素应用Gabor滤波模拟V1，忽略了皮下阶段的空间加和特性；EVNet插入DoG卷积，使P通路获得颜色拮抗中心-周边RF（红-绿、绿-红、蓝-黄），M通路获得无色差RF，更符合Kuffler [17]和Rodieck [18]的经典视网膜生理学。

**对应消融**：Table 6显示SubcorticalBlock各组件对鲁棒性贡献差异显著。

---

### 模块 3: Contrast Normalization（对比度归一化）——对应框架图 SubcorticalBlock第三层

**直觉**：早期视觉系统通过divisive normalization实现自适应增益控制，防止高对比度刺激导致饱和，同时增强低对比度信号的相对表征。

**Baseline 公式** (VOneNet无显式皮下对比度归一化):
VOneBlock内部有简单/复杂细胞非线性但未分离皮下阶段的归一化机制。

**本文公式（推导）**:
$$\text{Step 1}: \text{局部对比度估计} = \sqrt{x_{\text{DoG}}^2 * w_{\text{CN}}^n} \quad \text{高斯池化窗口} w_{\text{CN}} \text{估计局部对比度}$$
$$\text{Step 2}: \text{半饱和控制} = c_{50} + \text{局部对比度估计} \quad c_{50} \text{控制敏感度}$$
$$\text{最终}: x_{\text{CN}} = \frac{x_{\text{DoG}}}{c_{50} + \sqrt{x_{\text{DoG}}^2 * w_{\text{CN}}^n}}$$

符号：$x_{\text{DoG}}$为DoG卷积输出，$w_{\text{CN}}$为定义对比度整合池化窗口的高斯核，$c_{50}$为半饱和常数，$n$为非线性指数。

**变化点**：无此层时，光适应后的DoG响应在局部高对比度区域会爆炸性增长，导致训练不稳定；除法归一化将响应压缩至与局部对比度成比例的范围，实现自适应增益控制，模拟Shapley & Victor [19]的视网膜对比度响应特性。

**对应消融**：Table 6显示组件贡献差异显著；移除对比度归一化需配合光适应的ε=0.05修正才能稳定训练。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/3b15d126-8d15-4295-91d5-8e8ed4aac5f2/figures/Table_1.png)
*Table 1 (comparison): EVNetS0 outperforms baselines on mean VI response property alignment, and ImageNet-C*



本文在ImageNet-1k上训练，于多维度鲁棒性基准进行评估。Table 1显示EVNetS0（SubcorticalBlock + VOneBlock + ResNet50后端）在mean V1 response property alignment上优于VOneResNet50和基线ResNet50，验证了皮下处理对神经对齐的改善。Table 2-4分别覆盖ImageNet-C常见损坏、域迁移数据集和对抗扰动：EVNetS0在大多数损坏类型、大多数域迁移数据集以及大多数对抗扰动上均优于基线，且在各项均值（mean）上保持领先。Table 5的Robust Score综合指标进一步确认EVNetS0高于ResNet50和VOneResNet50。


![Table 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/3b15d126-8d15-4295-91d5-8e8ed4aac5f2/figures/Table_6.png)
*Table 6 (ablation): Model components contribution to robustness vary greatly*



核心数值方面，EVNet相比基线CNN在**综合鲁棒性基准上提升+9.3%**，这一增益涵盖对抗扰动、常见损坏和域迁移三大类威胁模型。更具实践意义的是，当EVNet与PRIME数据增强结合时，相比**单独PRIME再提升+6.2%**，证明神经启发的架构修改与训练时增强策略具有互补性而非替代关系。

Table 6的消融实验揭示了组件贡献的异质性：各模块对鲁棒性的贡献"vary greatly"。特别地，光适应与对比度归一化存在强耦合——移除对比度归一化后，光适应的原始公式会导致激活爆炸，必须引入ε=0.05阈值修正才能恢复训练稳定性。这一发现暗示SubcorticalBlock的内部级联结构具有非平凡的非线性交互，而非简单线性叠加。

公平性检验：基线选择基本合理，VOneNet作为最直接的可比方法，ResNet50作为标准CNN参照。但存在几点局限：其一，固定权重前端限制了与全可训练基线在参数量上的公平比较；其二，仅报告3个随机种子，方差估计可能不足；其三，与PushPull-Net [14]、Biological convolutions [15]、Divisive normalization [13]等同期神经启发方法的详细对比在现有上下文中不够清晰；其四，ε=0.05阈值本为消融实验的补救措施，可能混淆组件重要性的纯净估计。训练成本方面，使用48GB NVIDIA A40 GPU，训练约需3天。

## 方法谱系与知识库定位

EVNet属于**Neuro-inspired Robust Vision（神经启发鲁棒视觉）**方法谱系，直接继承自**VOneNet** [12]——后者首次将V1皮层模拟作为CNN固定权重前端以提升扰动鲁棒性。EVNet在此 lineage 中定位为"向下游扩展"：将神经模拟从V1延伸至视网膜和LGN皮下阶段。

**直接基线差异**：
- **VOneNet**：EVNet prepend了SubcorticalBlock，修改VOneBlock FoV（8deg→7deg）并扩展GFB空间频率，新增可选LGN-V2跳跃连接
- **ResNet50**：EVNet替换标准预处理为光适应驱动归一化，前端全部改为固定权重神经模拟模块
- **PRIME**：训练时数据增强 vs. EVNet的架构级修改，二者互补结合可叠加增益

**改变的slots**：architecture（新增SubcorticalBlock、修改VOneBlock）、data_pipeline（光适应替代标准归一化）、inference_strategy（可选LGN-V2跳跃连接）。未改变training_recipe（仍为标准监督训练）和核心data_curation（仍为ImageNet-1k）。

**后续方向**：(1) 解除前端固定权重约束，探索端到端可学习的神经启发模块；(2) 将SubcorticalBlock扩展至视频/时序输入，建模皮下通路的时序动态（当前spike-count公式仅抽象空间编码）；(3) 在更大规模数据集（如ImageNet-21k）和更多后端架构（如Vision Transformer）上验证可扩展性。

**标签**：modality=图像 | paradigm=监督学习+固定权重神经前端 | scenario=鲁棒图像分类 | mechanism=皮下视觉模拟（P/M通路、DoG、divisive normalization） | constraint=生物约束参数、前端不可学习

