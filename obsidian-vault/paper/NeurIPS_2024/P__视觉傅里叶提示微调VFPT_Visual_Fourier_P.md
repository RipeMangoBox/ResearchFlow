---
title: Visual Fourier Prompt Tuning
type: paper
paper_level: C
venue: NeurIPS
year: 2024
paper_link: null
aliases:
- 视觉傅里叶提示微调VFPT
- Visual Fourier P
- Visual Fourier Prompt Tuning (VFPT)
acceptance: Poster
cited_by: 42
code_url: https://runjia.tech/vfpt_page/
method: Visual Fourier Prompt Tuning (VFPT)
followups:
- 自适应提示的双层轨迹预测表示学习_PerReg_-_PerReg+
---

# Visual Fourier Prompt Tuning

[Code](https://runjia.tech/vfpt_page/)

**Topics**: [[T__Classification]], [[T__Self-Supervised_Learning]], [[T__Domain_Adaptation]] | **Method**: [[M__Visual_Fourier_Prompt_Tuning]] | **Datasets**: [[D__FGVC]] (其他: VTAB-1k, VTAB-1k Natural)

| 中文题名 | 视觉傅里叶提示微调VFPT |
| 英文题名 | Visual Fourier Prompt Tuning |
| 会议/期刊 | NeurIPS 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2411.01327) · [Code](https://runjia.tech/vfpt_page/) · [Project](https://runjia.tech/vfpt_page/) |
| 主要任务 | 视觉参数高效微调（Visual Parameter-Efficient Fine-Tuning） |
| 主要 baseline | VPT, Full Fine-tuning, Linear probing, Adapter, Bias tuning, GPT |

> [!abstract] 因为「纯空间域视觉提示无法捕捉图像的频率域信息，限制了跨任务迁移能力」，作者在「VPT (Visual Prompt Tuning)」基础上改了「将提示分解为空间域与频率域双分支，通过傅里叶变换引入频率域可学习提示」，在「VTAB-1k 24项任务」上取得「23/24项超越VPT且参数量更低，22/24项超越Full Fine-tuning」

- **VTAB-1k平均精度**: 65.57%（ViT-Base/16, supervised IN-21k），仅用0.38%可训练参数
- **对比VPT-S**: 在23/24项任务上更优，VPT-S仅52.94%平均精度
- **MAE预训练场景**: VTAB-1k Natural组53.59% vs. VPT 36.02%，绝对提升+17.57%

## 背景与动机

大规模视觉Transformer（ViT、Swin等）在下游任务适配时面临一个核心矛盾：Full Fine-tuning需要更新全部参数，计算与存储开销巨大；而参数高效微调（PEFT）方法虽只训练少量参数，却常因表达能力不足而性能受损。以VPT（Visual Prompt Tuning）为例，该方法在输入层插入可学习的空间域提示token，冻结主干网络，实现了0.05%-0.31%参数量的高效微调。然而，图像信息天然包含空间与频率两个维度——边缘、纹理等视觉特征在频率域有紧凑表示，纯空间域提示难以捕捉这些频率结构信息。

现有方法的处理方式各有局限：**VPT**仅在空间域学习提示，忽略了频率域的互补知识；**Adapter**在Transformer层间插入瓶颈MLP，增加了推理延迟；**Bias tuning（BitFit）**仅训练偏置项，表达能力受限；**Linear probing**完全冻结特征提取，仅训练分类头，在任务与预训练数据差异较大时性能骤降。具体而言，当预训练数据（如ImageNet-21k）与下游任务（如结构化视觉任务）存在显著domain gap时，VPT的空间提示难以有效迁移，导致VTAB-1k Structured组精度仅26.84%。

本文提出Visual Fourier Prompt Tuning（VFPT），核心动机在于：**将可学习提示分解到空间域与频率域，利用傅里叶变换捕获图像的频率结构，与空间域提示形成互补**，从而在极低参数量下实现超越Full Fine-tuning的迁移性能。

## 核心创新

核心洞察：**图像的频率域表示包含空间域难以编码的全局结构与纹理统计信息**，因为傅里叶变换能将局部空间特征解耦为不同频率分量，从而使少量频率域提示即可补偿空间域提示在跨域迁移中的表达能力缺陷。

| 维度 | VPT (Baseline) | 本文 VFPT |
|:---|:---|:---|
| 提示域 | 仅空间域（spatial-only） | 空间域 + 频率域（dual-domain） |
| 提示生成 | 直接学习空间token嵌入 | 部分参数经DFT→频率域→IDFT转回空间域 |
| 任务适应性 | 固定结构，无显式domain gap适配机制 | 可调Fourier Percentage α，按任务差异动态分配双域比例 |
| 训练稳定性 | 需特定设计的大学习率 | 无需特殊学习率设计 |

与VPT的关键差异在于：VFPT不是替换空间提示，而是**将同一组可学习参数按比例α切分**，一部分保持空间域，另一部分经傅里叶变换引入频率信息，最终融合为双域提示输入Transformer。

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/52f79984-f01a-4225-affd-5e988f096929/figures/Figure_1.png)
*Figure 1 (pipeline): Overview of VPT, VPT-S, and our proposed VFPT frameworks.*



VFPT的完整数据流如下：

1. **Input Image Patch Embedding**：输入图像分割为N个patch，经线性投影得到patch token嵌入序列 $[x_1, x_2, ..., x_N]$；
2. **Spatial Prompt Generation**：从总提示参数 $P_{total}$ 中取 $(1-\alpha)$ 比例，直接生成空间域可学习提示 $P_{spatial}$；
3. **Fourier Prompt Generation**：从 $P_{total}$ 中取 $\alpha$ 比例，先经离散傅里叶变换（DFT）得到频率域表示 $\hat{P}_{fourier}$，再经逆傅里叶变换（IDFT）转回空间域 $P_{fourier}^{(spatial)}$；
4. **Prompt Fusion**：将 $P_{spatial}$ 与 $P_{fourier}^{(spatial)}$ 拼接/融合，形成双域提示序列；
5. **Frozen Transformer Backbone**：融合后的提示与patch嵌入拼接，输入冻结的ViT/Swin Transformer；
6. **Classification Head**：取[CLS] token或全局特征，经分类头输出预测。

```
Image Patches → [Patch Embedding] → Patch Tokens
                                      ↓
Learnable Params ──→ [Spatial Gen] ──→ Spatial Prompts ──┐
              └──α──→ [DFT] → [IDFT] → Fourier Prompts ──┼→ [Fusion] → [Frozen ViT] → [CLS Head] → Output
              (1-α)                                      │
```

其中α为Fourier Percentage，是控制双域分配的核心超参数。

## 核心模块与公式推导

### 模块 1: 双域提示融合（Prompt Fusion）——对应框架图"Prompt Fusion"节点

**直觉**：空间域提示擅长局部细节，频率域提示擅长全局结构，二者拼接可形成互补表示。

**Baseline 公式 (VPT)**:
$$v = [x_1, x_2, ..., x_N, p_1, p_2, ..., p_M]$$
符号: $x_i$ = 第i个图像patch嵌入, $p_j$ = 第j个可学习空间提示, $M$ = 提示总数

**变化点**：VPT的提示纯为空间域，缺乏频率结构信息；VFPT将提示分解为双域，通过傅里叶变换引入频率分量。

**本文公式（推导）**:
$$\text{Step 1 (参数分解)}: P_{total} \rightarrow P_{spatial} \cup P_{fourier}, \quad |P_{fourier}| = \alpha M, \; |P_{spatial}| = (1-\alpha)M$$
$$\text{Step 2 (频率域变换)}: \hat{P}_{fourier} = \mathcal{F}(P_{fourier}^{(learnable)}) \quad \text{对可学习参数做DFT}$$
$$\text{Step 3 (逆变换回空间)}: P_{fourier}^{(spatial)} = \mathcal{F}^{-1}(\hat{P}_{fourier}) \quad \text{IDFT保证与patch同域可拼接}$$
$$\text{最终融合}: v_{final} = [x_1, ..., x_N, \underbrace{p_1^{(s)}, ..., p_{M(1-\alpha)}^{(s)}}_{\text{空间提示}}, \underbrace{p_1^{(f)}, ..., p_{M\alpha}^{(f)}}_{\text{频率提示}}]$$

**对应消融**：Figure 2显示，Natural组在α≈50%时精度最高，Specialized/Structured组需更高α。

---

### 模块 2: Fourier Percentage 分解控制——对应框架图"Fourier Prompt Generator"内部

**直觉**：不同下游任务与预训练数据的domain gap程度不同，需自适应分配频率/空间提示比例。

**Baseline 公式 (VPT)**:
$$\alpha = 0 \quad \text{(无频率分量，全部参数用于空间提示)}$$

**变化点**：VPT的固定结构无法适应任务差异；VFPT引入可调的α，使同一框架能根据任务特性优化双域分配。

**本文公式（推导）**:
$$\text{Step 1 (比例定义)}: P_{fourier} = \alpha \cdot P_{total}, \quad P_{spatial} = (1-\alpha) \cdot P_{total}$$
$$\text{Step 2 (约束保证)}: \alpha \in [0, 1], \quad \text{特别地，} \alpha=0 \text{ 退化为VPT，} \alpha=1 \text{ 为纯频率提示}$$
$$\text{最终}: L_{task}(\theta_{frozen}; P_{total}, \alpha) = \text{CrossEntropy}(\text{Transformer}([x; P_{spatial}; \mathcal{F}^{-1}(\mathcal{F}(\alpha \cdot P_{total}))]))$$

**对应消融**：Table 4（变换类型与维度消融）显示，DFT/IDFT组合优于其他正交变换。

## 实验与分析


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/52f79984-f01a-4225-affd-5e988f096929/figures/Table_2.png)
*Table 2 (quantitative): Image classification accuracy for different backbones.*



本文在VTAB-1k（24项任务，分Natural/Specialized/Structured三组）、FGVC（细粒度视觉分类）及不同预训练目标（Supervised IN-21k / MAE / MoCo v3）上展开系统评估。核心结果表明：VFPT以仅0.38%的可训练参数，在VTAB-1k上达到65.57%平均精度，与Full Fine-tuning（65.57%）持平，但参数量仅为后者的约1/260；更关键的是，VFPT在22/24项任务上超越Full Fine-tuning，在23/24项任务上超越VPT-S（52.94%）且参数量更低。在FGVC上，VFPT取得88.54%，为所有PEFT方法中最高。对于MAE自监督预训练模型，VFPT在VTAB-1k Natural组达到53.59%，较VPT的36.02%提升+17.57%绝对精度，验证了频率域提示对弥补自监督预训练与下游任务差距的有效性。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/52f79984-f01a-4225-affd-5e988f096929/figures/Table_3.png)
*Table 3 (quantitative): Image classification accuracy for different pretrained objectives.*



消融实验聚焦于Fourier Percentage α的作用：Figure 2显示，VTAB-1k Natural组（任务与预训练数据差异小）在α≈50%时达到峰值，而Specialized/Structured组（domain gap大）需要更高的α比例。这表明**频率域提示对跨域迁移的贡献与任务差异正相关**。Table 4进一步验证了DFT/IDFT作为变换核的有效性，以及Fourier维度（保留低频分量比例）对精度的影响——过度截断高频信息会损失细节表达能力。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/52f79984-f01a-4225-affd-5e988f096929/figures/Figure_3.png)
*Figure 3 (qualitative): Visualization of loss landscape and the joint map of Hessian.*



优化景观分析（Figure 3）提供了更深层的证据：VFPT的损失landscape比VPT更平坦，Hessian矩阵的最大特征值比值更小，说明双域提示使优化过程更稳定、泛化边界更紧。这与VFPT无需VPT所依赖的特殊大学习率的设计相印证。

公平性检视：本文对比的baselines覆盖全面（VPT-S/VPT-D、Full Fine-tuning、Linear、Adapter、Bias tuning、Partial-1、MLP-3、Sidetune、GPT），且均为该领域代表性方法。但需注意，**未与LoRA、Prefix Tuning、IA³、DoRA等更近期PEFT方法对比**，这些NLP领域流行的低秩适配方法在视觉迁移中的效果尚不明确；此外，实验仅报告100 epoch训练结果，更长训练或更大batch的影响未讨论。

## 方法谱系与知识库定位

**方法族系**：VFPT属于**视觉提示微调（Visual Prompt Tuning）**谱系，直接继承自VPT（Jia et al., 2022）。

**父方法**：VPT —— 在输入层插入纯空间域可学习提示token，冻结Transformer主干。VFPT将其推广为双域提示框架。

**改动槽位**：
- **Architecture（架构）**：空间域单分支 → 空间域+频率域双分支，新增Fourier Prompt Module；
- **Data pipeline（数据流）**：固定提示插入 → 可调α比例的双域提示融合；
- **Training recipe（训练配置）**：去除VPT依赖的特殊大学习率设计，训练更稳定。

**直接对比方法**：
- **VPT-S/VPT-D**：VFPT在23/24项任务上更优，参数量0.38% vs. VPT-S 0.05%/VPT-D 0.31%；
- **Full Fine-tuning**：VFPT以1/260参数在22/24项任务上超越；
- **Adapter/Bias tuning/GPT**：VFPT在FGVC等任务上取得更高精度，且无需修改层间结构。

**后续方向**：(1) 将Fourier Prompt扩展至层级提示（deep prompting），而非仅输入层；(2) 结合LoRA等低秩适配与频率域提示的混合PEFT框架；(3) 探索小波变换等其他频域分解在视觉提示中的适用性。

**知识库标签**：
- Modality: 视觉（图像分类）
- Paradigm: 参数高效微调 / 提示学习
- Scenario: 跨域迁移 / 预训练模型适配
- Mechanism: 傅里叶变换 / 双域表示学习 / 频率域建模
- Constraint: 极低可训练参数量（<0.5%）

## 引用网络

### 后续工作（建立在本文之上）

- [[P__自适应提示的双层轨迹预测表示学习_PerReg_-_PerReg+]]: Visual Prompt Tuning is foundational parameter-efficient fine-tuning method for 

