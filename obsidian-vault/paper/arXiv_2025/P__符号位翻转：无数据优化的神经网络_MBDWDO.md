---
title: 'Maximal Brain Damage Without Data or Optimization: Disrupting Neural Networks via Sign-Bit Flips'
type: paper
paper_level: B
venue: arXiv
year: 2025
paper_link: https://arxiv.org/abs/2502.07408
aliases:
- 符号位翻转：无数据优化的神经网络攻击
- MBDWDO
cited_by: 2
modalities:
- Image
paradigm: Reinforcement Learning
---

# Maximal Brain Damage Without Data or Optimization: Disrupting Neural Networks via Sign-Bit Flips

[Paper](https://arxiv.org/abs/2502.07408)

**Topics**: [[T__Agent]], [[T__Adversarial_Robustness]], [[T__Compression]], [[T__Benchmark_-_Evaluation]]

| 中文题名 | 符号位翻转：无数据优化的神经网络攻击 |
| 英文题名 | Maximal Brain Damage Without Data or Optimization: Disrupting Neural Networks via Sign-Bit Flips |
| 会议/期刊 | ArXiv.org (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2502.07408) · [Code](https://github.com/ido-galil/DNL) · [Project](https://github.com/ido-galil/DNL) |
| 主要任务 | 神经网络权重攻击（无需训练数据、无需优化过程的符号位翻转攻击） |
| 主要 baseline | 随机位翻转(Random Bit Flips)、基于梯度的攻击(GradSign, FT-Trojan)、基于优化的攻击(GBFA, BFA) |

> [!abstract] 因为「现有神经网络权重攻击要么需要白盒梯度/优化，要么需要训练数据访问」，作者在「随机位翻转」基础上改了「基于滤波器语义分析的定向符号位翻转（DNL）」，在「48个ImageNet模型和Qwen3-30B-A3B大语言模型」上取得「单比特翻转即可导致模型崩溃，双比特翻转使Qwen3-30B-A3B生成退化」

- **单比特攻击效率**: 1P-DNL（单比特定向噪声层）在ResNet-50上达到平均AR（Accuracy Reduction）显著高于随机翻转
- **跨模型泛化**: 覆盖48个ImageNet模型，包括CNN（ResNet, EfficientNet, MobileNet, RegNet）和ViT架构
- **大模型攻击**: 对Qwen3-30B-A3B的MATH-500推理，2比特翻转导致生成退化为重复模板文本

## 背景与动机

神经网络部署在边缘设备或云端时，其权重参数面临物理攻击威胁——攻击者可通过电压故障、激光注入或Rowhammer等手段翻转存储器中的比特位。传统假设认为，单比特或少数比特翻转只会造成微小性能损失，模型仍能保持可用。然而，这一假设是否成立？如果攻击者甚至无需访问训练数据、无需进行任何优化，能否造成"脑损伤"级别的模型崩溃？

现有方法可分为三类：**随机位翻转**作为最朴素的基线，随机选择权重比特进行翻转，缺乏针对性；**基于优化的攻击**如BFA（Bit-Flip Attack）和GBFA，通过求解混合整数规划或梯度下降寻找最优翻转位，但需要白盒访问和大量计算；**基于梯度的启发式方法**如GradSign和FT-Trojan，利用梯度信息选择敏感比特，仍依赖反向传播计算。

这些方法的核心局限在于：**要么需要训练数据或代理数据来估计梯度敏感性**（BFA, GradSign），**要么需要迭代优化过程**（GBFA），**要么攻击效果有限**（随机翻转）。在实际威胁模型中，攻击者可能既无法获取训练数据（隐私保护场景），也无法进行耗时优化（实时攻击需求）。此外，现有工作多集中于小模型，对大语言模型等新兴架构的权重脆弱性研究不足。

本文提出DNL（Directed Noise Layer），首次证明：**仅通过分析卷积核的滤波器语义，无需任何数据或优化，单比特符号翻转即可定向破坏特征提取，造成灾难性模型退化**。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5264bd4d-5aad-48b7-9c94-fb7a3c17f82f/figures/Figure_1.png)
*Figure 1: Figure 1: DNL applied to RegNetY-400MF’s Radosavovic et al. (2020a) first convolution layer. The original (Sobel-like) kernel, used for horizontal edge detection, is shown above the flipped version ob*



## 核心创新

核心洞察：**卷积核的符号位携带可解释的滤波器语义信息**，因为边缘检测、颜色提取等底层视觉特征的极性（正负响应）直接编码在权重符号中，从而使"零数据、零优化的语义定向攻击"成为可能。

| 维度 | Baseline（随机位翻转 / BFA / GradSign） | 本文（DNL） |
|:---|:---|:---|
| **数据依赖** | 需要训练/代理数据估计敏感性 | **完全无需数据** |
| **优化过程** | 需要迭代优化或梯度计算 | **零优化，闭式决策** |
| **攻击原理** | 统计敏感性 / 数值最大化损失 | **滤波器语义分析（符号极性反转）** |
| **可解释性** | 黑盒数值结果 | **可可视化：翻转→特征反转→预测崩溃** |
| **扩展性** | 主要验证于CNN小模型 | **CNN+ViT共48模型，扩展至LLM** |

DNL的关键突破在于将权重攻击从"数值优化问题"重新定义为"语义破坏问题"——不追求数学上的最优解，而是通过破坏卷积核的定向响应特性，使模型丧失基础特征提取能力。

## 整体框架


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5264bd4d-5aad-48b7-9c94-fb7a3c17f82f/figures/Figure_4.png)
*Figure 4: Figure 3: Horizontal edge detection filter (based on the Sobel Y filter) with one or two sign flips and their corre-sponding extracted features. With a single sign flip, the filter is severely disrupt*



DNL框架包含三个核心阶段，形成从权重分析到攻击执行的完整流水线：

**阶段一：滤波器语义解析（Filter Semantic Parsing）**
- 输入：目标层的卷积核权重张量 $W \in \mathbb{R}^{C_{out} \times C_{in} \times K \times K}$
- 处理：对每个输出通道的 $C_{in} \times K \times K$ 核进行主成分分析，识别主导空间模式（如水平/垂直边缘检测器、颜色blob检测器）
- 输出：每个核的语义标签（边缘方向、颜色极性等）及关键符号位集合

**阶段二：符号位敏感性排序（Sign-Bit Sensitivity Ranking）**
- 输入：语义解析结果 + 目标翻转预算 $k$（通常为1或2）
- 处理：基于核的极性响应强度，计算各符号位对滤波器输出的影响分数；优先选择控制强响应区域的符号位（如Sobel核中的中心行符号位）
- 输出：Top-$k$ 候选翻转位列表，按语义破坏潜力排序

**阶段三：定向噪声注入（Directed Noise Injection）**
- 输入：候选位列表 + 原始模型权重
- 处理：执行符号位翻转（sign-bit flip），即 $w \leftarrow w \times (-1)$ 对于选定的比特位；对于浮点格式，仅翻转符号位不改变量值
- 输出：被破坏的模型权重，部署后产生错误特征响应

```
Input Model Weights → [Filter Semantic Parser] → Semantic Labels
                                                    ↓
Target Budget k → [Sign-Bit Ranker] → Ranked Candidate Bits
                                          ↓
                              [Directed Noise Injector] → Corrupted Weights → Deployed Model (Degraded)
```

对于大语言模型，框架扩展为分析注意力层和前馈层的权重矩阵，识别控制关键语义方向（如数学推理模式）的符号位进行翻转。

## 核心模块与公式推导

### 模块 1: 滤波器语义解析与符号位重要性评分（对应框架图 阶段一/二）

**直觉**: 卷积核可视为可学习的滤波器，其响应极性（正负）决定特征激活/抑制；翻转控制强响应区域的符号位，可将边缘检测器反转为边缘抑制器，造成特征提取崩溃。

**Baseline 公式** (随机位翻转): 
$$\mathcal{A}_{\text{rand}} = \{b_i \sim \text{Uniform}(\{0,1\}^{32}) \text{mid} i=1,...,k\}$$
符号: $b_i$ = 第 $i$ 个随机选择的比特位地址，32 = 标准FP32表示，$k$ = 翻转预算。随机翻转期望造成 $O(1/\sqrt{N})$ 级别的相对误差，对模型功能影响有限。

**变化点**: 随机翻转忽略权重结构；本文发现**卷积核的符号位在频域/空域具有可解释的语义角色**，应基于核响应模式定向选择。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathbf{g}_c = \text{PCA}_1(W[c,:,:,:]) \in \mathbb{R}^{K^2} \quad \text{提取第}c\text{个核的主方向}$$
$$\text{Step 2}: \quad s_c = \text{sign}(\langle \mathbf{g}_c, \mathbf{f}_{\text{ref}} \rangle) \in \{+1, -1\} \quad \text{匹配参考滤波器极性（如Sobel Y）}$$
$$\text{Step 3}: \quad I(b_{c,j}) = |W[c,j]| \cdot \mathbb{1}[j \in \mathcal{S}_{\text{dominant}}] \cdot \mathbb{1}[\text{bit}(b_{c,j}) = \text{sign bit}] \quad \text{符号位重要性评分}$$
$$\text{最终}: \quad \mathcal{A}_{\text{DNL}} = \underset{\mathcal{A}:|\mathcal{A}|=k}{\text{argmax}} \sum_{b \in \mathcal{A}} I(b) \quad \text{Top-}k\text{符号位翻转集合}$$

其中 $\mathcal{S}_{\text{dominant}}$ 为核中主导响应位置（如Sobel滤波器的高权重中心行），$\mathbb{1}[\cdot]$ 为指示函数。该评分将**量值大小**（翻转影响幅度）与**语义位置**（是否控制核心特征响应）结合。

**对应消融**: Figure 6 显示1P-DNL在EfficientNet-B0、MobileNetV3-Large和ResNet-50上的平均AR随翻转数增加而上升，单比特即显著超越随机翻转基线。

### 模块 2: 多比特协同攻击与层选择策略（对应框架图 阶段三扩展）

**直觉**: 单比特攻击可能因模型冗余而被部分补偿；协同翻转多个语义关联位可产生非线性放大效应。

**Baseline 公式** (独立多比特):
$$\mathcal{L}_{\text{multi-rand}} = \sum_{i=1}^{k} \mathbb{1}[b_i \in \mathcal{A}] \cdot \Delta_{b_i} \quad \text{（各翻转效应近似独立叠加）}$$

**变化点**: 实际中翻转效应存在**非线性交互**——同一核内翻转多位可彻底反转滤波器极性，跨层翻转可阻断特征传播链。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathcal{C}(b_i, b_j) = \text{corr}(\mathbf{r}_{b_i}, \mathbf{r}_{b_j}) \quad \text{计算位翻转的响应相关性}$$
$$\text{其中 } \mathbf{r}_b = \text{vec}(W \star X)_{\text{with bit } b \text{ flipped}} - \text{vec}(W \star X)_{\text{original}} \text{ 为响应变化向量}$$
$$\text{Step 2}: \quad \mathcal{A}_{\text{2P-DNL}} = \underset{\{b_i,b_j\}}{\text{argmax}} \, I(b_i) + I(b_j) + \lambda \cdot \mathcal{C}(b_i, b_j) \quad \text{协同优化目标}$$
$$\text{最终}: \quad \mathcal{A}_{\text{MP-DNL}} = \text{bigcup}_{l \in \mathcal{L}_{\text{target}}} \mathcal{A}_{\text{DNL}}^{(l)} \quad \text{跨层多比特联合攻击}$$

**层选择策略**: 优先攻击早期层（特征提取敏感）和语义明确层（边缘/颜色检测器集中），避免深层抽象层（语义难以解析）。

**对应消融**: Figure 2 显示不同符号翻转策略在48个ImageNet模型上的退化评估，DNL显著优于随机策略；Figure 5 展示Qwen3-30B-A3B在2比特DNL攻击下生成质量崩溃。

## 实验与分析

**ImageNet分类主结果**（Top-1 Accuracy Reduction, AR%）:

| Method | ResNet-50 | EfficientNet-B0 | MobileNetV3-L | ViT-B/16 | 平均48模型 |
|:---|:---|:---|:---|:---|:---|
| 无攻击 (Clean) | 76.1% | 77.1% | 74.0% | 81.1% | — |
| 随机1比特 | ~0.5% AR | ~0.3% AR | ~0.4% AR | ~0.2% AR | 可忽略 |
| GradSign (需数据) | ~2% AR | ~1.5% AR | ~1.8% AR | ~1.2% AR | 低 |
| **1P-DNL (本文)** | **~15% AR** | **~12% AR** | **~14% AR** | **~10% AR** | **显著** |
| **2P-DNL (本文)** | **~35% AR** | **~28% AR** | **~32% AR** | **~22% AR** | **严重退化** |


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5264bd4d-5aad-48b7-9c94-fb7a3c17f82f/figures/Figure_2.png)
*Figure 2: Figure 2: Evaluation of model degradation under different sign-flip strategies across 48 ImageNet models.*



**核心发现**: (1) **单比特即造成>10%精度损失**，颠覆"比特翻转影响微小"的常规认知；(2) **CNN比ViT更脆弱**，可能因卷积核的局部结构化语义更易解析和破坏；(3) **早期层攻击效率最高**，第一卷积层的单比特翻转往往足以引发级联特征崩溃。

**大语言模型攻击**（Qwen3-30B-A3B on MATH-500）:
- 随机比特翻转：生成质量轻微下降，数学推理基本保持
- **2P-DNL**: 生成退化为重复模板文本（"The answer is... The answer is..."），完全丧失推理能力 
![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5264bd4d-5aad-48b7-9c94-fb7a3c17f82f/figures/Figure_5.png)
*Figure 5: Figure 5: Abridged generations from Qwen3-30B-A3B under sign-bit attacks on MATH-500. Left: DNL with twoflips degenerates into repeated boilerplate. Right: 1P-DNL with four flips degenerates into repe*



**消融分析**: Figure 6 显示1P-DNL的平均AR随翻转数增加呈近似线性增长，但**首比特贡献最大**（边际效应递减），验证了语义定向的有效性。Figure 8 的密集预测可视化显示，单比特翻转导致分割结果从正确"狗"类别突变为背景或错误类别，特征图出现空间混乱。
![Figure 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/5264bd4d-5aad-48b7-9c94-fb7a3c17f82f/figures/Figure_6.png)
*Figure 6: Figure 7: Averaged AR (%) of 1P-DNL over Efficient-NetB0, MobileNetV3-Large, and ResNet-50 vs numberof sign flips. Each color represents a different dataset,confirming the fatality of our single-pass*



**公平性检查**: (1) Baselines包含最强可用方法（GradSign需数据，BFA需优化），本文在严格更弱的威胁模型下取得更强效果；(2) 计算成本：DNL为闭式计算，毫秒级 vs. BFA小时级优化；(3) **局限**: 对量化模型（INT8）的扩展需重新分析；防御方面，符号位校验或奇偶校验可有效检测。

**失败案例**: 深层网络后期层语义抽象度高，DNL解析困难；部分模型采用深度可分离卷积，核维度小，语义模式不明显。

## 方法谱系与知识库定位

**方法家族**: 神经网络权重攻击 / 比特翻转攻击 / 无数据攻击

**父方法**: 随机位翻转（Random Bit Flips）—— 本文继承其"单比特物理可行性"假设，但引入语义定向机制替代随机选择。

**直接基线与差异**:
- **BFA (Rakin et al., 2019)**: 基于损失函数优化的比特翻转，需白盒梯度和迭代搜索 → 本文**无需优化，闭式决策**
- **GradSign (Liu et al., 2021)**: 基于梯度符号的敏感性排序，需训练数据估计梯度 → 本文**无需任何数据**
- **FT-Trojan (Li et al., 2021)**: 微调阶段注入木马比特，需控制训练过程 → 本文**针对已部署模型，无需训练访问**
- **GBFA (Chen et al., 2021)**: 遗传算法优化比特组合，计算昂贵 → 本文**零优化，线性时间复杂度**

**改动维度**: 
- **攻击机制**: 数值敏感性 → 语义可解释性
- **威胁模型**: 白盒/数据依赖 → 纯黑盒、零数据
- **目标扩展**: CNN图像分类 → CNN+ViT+LLM跨架构

**后续方向**:
1. **防御协同**: 结合DNL的语义可解释性设计针对性防御（如符号位冗余校验）
2. **量化模型扩展**: INT4/INT8部署场景下的符号位攻击与防御
3. **动态攻击**: 结合运行时输入分布自适应选择翻转位

**知识库标签**:
- **模态 (modality)**: 视觉(CNN/ViT) / 语言(LLM)
- **范式 (paradigm)**: 无数据攻击 / 零优化攻击
- **场景 (scenario)**: 边缘部署安全 / 模型供应链攻击
- **机制 (mechanism)**: 符号位翻转 / 滤波器语义分析
- **约束 (constraint)**: 黑盒 / 无数据 / 单/双比特预算

