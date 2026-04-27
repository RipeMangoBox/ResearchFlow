---
title: 'C-TPT: Calibrated Test-Time Prompt Tuning for Vision-Language Models via Text Feature Dispersion'
type: paper
paper_level: C
venue: ICLR
year: 2024
paper_link: null
aliases:
- 通过文本特征分散校准测试时提示微调
- C-TPT (Calibrate
- C-TPT (Calibrated Test-Time Prompt Tuning)
acceptance: Poster
cited_by: 82
method: C-TPT (Calibrated Test-Time Prompt Tuning)
---

# C-TPT: Calibrated Test-Time Prompt Tuning for Vision-Language Models via Text Feature Dispersion

**Topics**: [[T__Self-Supervised_Learning]], [[T__Domain_Adaptation]] | **Method**: [[M__C-TPT]] | **Datasets**: Natural Distribution Shifts, Fine-Grained

| 中文题名 | 通过文本特征分散校准测试时提示微调 |
| 英文题名 | C-TPT: Calibrated Test-Time Prompt Tuning for Vision-Language Models via Text Feature Dispersion |
| 会议/期刊 | ICLR 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2403.14119) · [Code](https://github.com/hee-suk-yoon/C-TPT) · [Project] |
| 主要任务 | 零样本图像分类、视觉语言模型校准、测试时自适应 |
| 主要 baseline | TPT, CoOp, CoCoOp, PromptAlign |

> [!abstract] 因为「测试时提示微调(TPT)虽然提升准确率但导致模型过度自信、校准误差恶化」，作者在「TPT」基础上改了「联合优化目标：保留熵最小化同时加入ATFD最大化和ECE最小化」，在「ImageNet自然分布偏移和11个细粒度分类数据集」上取得「ECE降低14%-52%，准确率基本持平」

- **ECE降低**: ImageNet变体上，Hard Prompt设置下CLIP-RN50的ECE降低28%，CLIP-ViT-B/16降低52%
- **ATFD提升**: 11个细粒度数据集平均ATFD从CoOp的0.54提升至0.71，相对提升31.5%
- **准确率保持**: 细粒度分类平均准确率56.5%，相比CoOp的56.8%仅微降0.3个百分点

## 背景与动机

视觉语言模型（如CLIP）在零样本图像分类中表现出色，但其预测置信度往往不可靠——模型频繁对错误预测赋予过高置信度，即"过度自信"问题。例如，在CIFAR-10和StanfordCars数据集上，即使使用TPT进行测试时优化，模型的可靠性图（Reliability Diagram）仍显示明显的过度自信偏差，期望校准误差（ECE）居高不下。

现有方法如何应对这一问题？**TPT (Test-Time Prompt Tuning)** 通过熵最小化优化可学习提示，使模型对同一图像的多个增强视图预测一致，从而提升准确率。然而，TPT仅关注准确性，未显式约束模型校准性，导致优化后的提示往往使文本特征聚集，加剧过度自信。**CoOp/CoCoOp** 系列方法在训练阶段学习连续提示向量，但同样需要额外的校准后处理。**PromptAlign** 专注于持续测试时自适应的特征对齐，也未直接解决校准问题。

这些方法的共同短板在于：**提示优化目标与模型校准目标脱节**。具体而言，TPT的熵最小化会促使文本编码器生成高度集中的文本特征，使得不同类别的文本表示过于相似（低分散度），进而导致相似度计算的置信度失真。作者通过实证发现（见图2），ECE与ATFD（平均文本特征分散度）存在显著负相关：文本特征越分散，模型校准越好。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/1802ac51-088e-482d-a615-38fc861ec47b/figures/Figure_1.png)
*Figure 1 (motivation): Observations. (The plots are based on the CIFAR-10 and StanfordCars dataset.) (a) Reliability Diagram. (b) Overconfident predictions when distribution shift exists. (c) Varying confidence over different classes. (d) Correlation between calibration and feature dispersion.*



因此，本文提出C-TPT，在TPT框架中引入校准感知的正则化，通过显式最大化文本特征分散度并直接优化ECE，实现准确率与校准性的联合提升。

## 核心创新

核心洞察：**文本特征的空间分散程度直接决定视觉语言模型的校准质量**，因为当类别文本特征在嵌入空间中彼此分离时，图像-文本相似度分布更不易产生虚假的峰值置信度，从而使测试时提示微调同时具备高准确率和低校准误差成为可能。

| 维度 | Baseline (TPT) | 本文 (C-TPT) |
|:---|:---|:---|
| 优化目标 | 仅熵最小化 $\mathcal{L}_{\text{TPT}} = -H(p)$ | 联合目标：熵最小化 + ATFD最大化 + ECE最小化 |
| 文本特征约束 | 无显式约束，易聚集 | 最大化成对欧氏距离，强制分散 |
| 校准处理 | 无，依赖事后温度缩放 | 测试时直接优化ECE作为目标项 |
| 准确-校准权衡 | 牺牲校准换准确 | 两者联合优化，准确率基本持平 |

与TPT的核心差异在于：C-TPT将"校准"从后处理步骤提升为与"准确性"同等地位的优化目标，通过可微分的ATFD和ECE项实现端到端的校准感知提示学习。

## 整体框架


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/1802ac51-088e-482d-a615-38fc861ec47b/figures/Figure_3.png)
*Figure 3 (pipeline): Illustration of the Calibrated Test-time Prompt Tuning (C-TPT) for zero-shot image recognition. We propose to calibrate CLIP during test-time prompt tuning by minimizing the Average Text Feature Dispersion (ATFD) during test-time prompt tuning.*



C-TPT的完整数据流如下：

**输入**：单张测试图像 + 类别名称集合（零样本设定，无训练标签）

**模块1：图像增强与编码**（冻结CLIP图像编码器）
- 输入：测试图像 → 生成M个随机增强视图
- 输出：M组图像特征向量
- 角色：提供多视图一致性约束的基础

**模块2：可学习提示与文本编码**（冻结CLIP文本编码器）
- 输入：可学习提示令牌 + 类别名称模板（如"a photo of a [CLASS]"）
- 输出：K个类别的文本特征矩阵 $\mathbf{T} \in \mathbb{R}^{K \times d}$
- 角色：生成待优化的类别语义表示

**模块3：TPT熵最小化**
- 输入：M个图像特征与K个文本特征的相似度矩阵
- 输出：基于视图一致性的预测分布 $p(y_i | \mathbf{x}_i^{(1)}, \ldots, \mathbf{x}_i^{(M)}; \theta)$
- 角色：基础测试时优化，确保增强视图预测一致以提升准确率

**模块4：C-TPT校准模块（新增）**
- 输入：文本特征矩阵 $\mathbf{T}$、预测置信度分布
- 输出：校准后的预测 + 分散的文本特征
- 角色：通过ATFD计算和ECE估计，生成校准正则化梯度

**输出**：校准后的类别预测概率，兼具高准确率和可靠置信度

```
测试图像 → [数据增强] → 冻结CLIP图像编码器 → 图像特征
                                      ↓
可学习提示 + 类别名 → 冻结CLIP文本编码器 → 文本特征 → 相似度计算
                                      ↓
                    ┌─────────────────┴─────────────────┐
                    ↓                                   ↓
              TPT熵最小化 (准确率)              C-TPT校准正则化
              -H(p) over aug views              + ATFD最大化
                                                + ECE最小化
                    └─────────────────┬─────────────────┘
                                      ↓
                              联合优化提示 → 校准预测输出
```

## 核心模块与公式推导

### 模块1: TPT熵最小化（基线目标）

**直觉**：同一图像的不同增强视图应预测一致，通过最小化预测熵来优化提示。

**Baseline公式 (TPT)**:
$$\mathcal{L}_{\text{TPT}} = -\sum_{i=1}^{N} p(y_i | \mathbf{x}_i^{(1)}, \ldots, \mathbf{x}_i^{(M)}; \theta) \log p(y_i | \mathbf{x}_i^{(1)}, \ldots, \mathbf{x}_i^{(M)}; \theta)$$

符号: $\theta$ = 可学习提示参数, $M$ = 增强视图数, $p(y_i | \cdot)$ = 第$i$个样本在$M$个视图上的平均预测概率

**变化点**：TPT仅优化视图一致性，但熵最小化会驱使模型"更确定"，当文本特征聚集时，这种确定性往往是虚假的过度自信。

---

### 模块2: ATFD（平均文本特征分散度）

**直觉**：类别文本特征在嵌入空间中越分散，决策边界越清晰，置信度越能反映真实不确定性。

**本文公式**：
$$\text{ATFD} = \frac{1}{K(K-1)} \sum_{i=1}^{K} \sum_{j \neq i} \|\mathbf{t}_i - \mathbf{t}_j\|_2$$

符号: $K$ = 类别数, $\mathbf{t}_i \in \mathbb{R}^d$ = 第$i$类的文本特征向量, $\|\cdot\|_2$ = L2范数

**推导说明**：
- 计算所有类别对之间的成对欧氏距离
- 取平均得到整体分散度指标
- **C-TPT中作为最大化目标**：$-\text{ATFD}$（即最小化负ATFD）

**对应消融**：Table 1/6/7/8/9显示，移除ATFD项后ECE显著上升，校准性能下降。

---

### 模块3: C-TPT联合优化目标（核心创新）

**直觉**：将校准从后处理提升为可微分优化目标，与准确率目标联合求解。

**Step 1**: 引入ATFD最大化以替代单一熵最小化
$$\mathcal{L} = \mathcal{L}_{\text{TPT}} - \lambda_1 \cdot \text{ATFD} \quad \text{（加入ATFD项以防止文本特征聚集导致的过度自信）}$$

**Step 2**: 进一步引入ECE直接优化，形成完整联合目标
$$\mathcal{L}_{\text{C-TPT}} = \mathcal{L}_{\text{TPT}} + \lambda_1 \cdot (-\text{ATFD}) + \lambda_2 \cdot \text{ECE} \quad \text{（重归一化多目标权重以保证准确-校准平衡）}$$

**ECE计算**：
$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$

符号: $B_m$ = 第$m$个置信度区间(bin), $|B_m|$ = 该区间样本数, $\text{acc}(B_m)$ = 区间准确率, $\text{conf}(B_m)$ = 区间平均置信度

**最终**：
$$\text{boxed}{\mathcal{L}_{\text{C-TPT}} = \underbrace{-H(p)}_{\text{TPT准确率}} + \underbrace{\lambda_1 \cdot (-\text{ATFD})}_{\text{特征分散}} + \underbrace{\lambda_2 \cdot \text{ECE}}_{\text{直接校准}}}$$

**对应消融**：Table 2显示，相比单独TPT，联合C-TPT在CLIP-ViT-B/16 Hard Prompt设置下ECE降低52%（从TPT基线降至C-TPT），而准确率变化在±1%以内。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/1802ac51-088e-482d-a615-38fc861ec47b/figures/Table_1.png)
*Table 1 (quantitative): Fine-Grained Classification. We report the results of CLIP-RN50 and CLIP-ViT-B/16 on 10 fine-grained visual classification datasets. The best results after test-time prompt tuning are reported.*



本文在两类基准上评估C-TPT：**11个细粒度分类数据集**（包括OxfordPets、StanfordCars、Food101等）和**ImageNet自然分布偏移**（ImageNet-V2、ImageNet-Sketch、ImageNet-A、ImageNet-R）。核心发现是：C-TPT在保持TPT准确率增益的同时，显著改善模型校准性。

在**自然分布偏移**任务上（Table 2），C-TPT联合TPT后取得显著校准提升：CLIP-RN50 Hard Prompt设置ECE降低28%，CLIP-ViT-B/16 Hard Prompt设置ECE降低52%；Ensemble设置下分别降低14%和24%。这些数字表明，文本特征分散度正则化对更大模型（ViT-B/16）的校准改善更为显著，可能因其更强的表征空间容纳分散特征的能力。

在**细粒度分类**任务上（Table 1），C-TPT在CoOp初始化基础上平均ATFD达到0.71，相比CoOp的0.54提升31.5%；平均ECE从6.98%降至5.83%，相对降低16.5%。关键观察是：准确率从56.8%微降至56.5%（-0.3），验证了C-TPT在准确-校准权衡中的有效性——以极小的准确率代价换取大幅校准改善。


![Figure 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/1802ac51-088e-482d-a615-38fc861ec47b/figures/Figure_4.png)
*Figure 4 (comparison): Comparison of calibration errors between TPT (Test-time Prompt Tuning) and C-TPT. (Left) Comparison of expected calibration error (ECE) and (Right) Adaptive ECE (AdaECE) across ImageNet variants.*



**消融分析**：
![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/1802ac51-088e-482d-a615-38fc861ec47b/figures/Table_2.png)
*Table 2 (quantitative): Natural Distribution Shifts. We report Top-1 acc (%), ECE (%) for the evaluation under natural distribution shifts. The best results after test-time prompt tuning are highlighted.*

 核心消融对比TPT单独使用与TPT+C-TPT联合使用。去掉C-TPT校准模块（即仅用TPT）导致ECE显著恶化：在ImageNet-R上，TPT+C-TPT相比TPT单独使用ECE降低幅度最大。ATFD项的移除对校准影响最为关键，因为文本特征聚集是过度自信的根源。

**公平性检查**：本文比较基准集中于TPT和CoOp家族（CoOp、CoCoOp、PromptAlign），未包含更新的方法如MaPLe、KgCoOp、ProGrad、Tip-Adapter、CLIP-Adapter，可能低估竞争强度。实验仅覆盖CLIP-RN50和CLIP-ViT-B/16，缺乏更大模型（如ViT-L/14）的验证。作者未明确报告校准计算带来的推理开销，且主表标准差置于附录，影响统计显著性判断。此外，部分设置下准确率存在轻微下降（如CoOp+C-TPT vs CoOp），需权衡应用场景对准确率与校准的不同需求。

## 方法谱系与知识库定位

**方法家族**：测试时自适应（Test-Time Adaptation）→ 测试时提示微调（Test-Time Prompt Tuning）

**父方法**：**TPT (Test-Time Prompt Tuning, Shu et al., 2022)** —— C-TPT直接继承TPT的熵最小化框架，修改其优化目标与训练配方。

**变更槽位**：
- **目标函数（objective）**：TPT的纯熵最小化 → 联合目标（熵最小化 + ATFD最大化 + ECE最小化）
- **训练配方（training_recipe）**：单目标测试时优化 → 多目标联合优化，显式约束校准性

**直接基线对比**：
| 基线 | 与C-TPT的核心差异 |
|:---|:---|
| TPT | C-TPT增加ATFD和ECE两项校准正则化，TPT仅优化视图一致性 |
| CoOp | C-TPT在测试时优化，CoOp在训练时学习；C-TPT可与CoOp初始化结合（CoOp+C-TPT） |
| CoCoOp | CoCoOp生成输入条件提示，C-TPT不修改提示结构，仅修改优化目标 |
| PromptAlign | PromptAlign针对持续测试时自适应，C-TPT针对单次测试时校准 |

**后续方向**：
1. **多模态扩展**：将ATFD思想扩展至图像-音频、图像-视频等其他模态组合的校准
2. **自适应权重**：动态调整$\lambda_1, \lambda_2$以适应不同分布偏移强度，避免手动调参
3. **与更强基线结合**：验证C-TPT对MaPLe、Tip-Adapter等新方法的兼容性

**知识标签**：
- **模态**：视觉-语言（Vision-Language）
- **范式**：测试时自适应 / 提示学习（Prompt Learning）
- **场景**：零样本图像分类、分布偏移鲁棒性
- **机制**：特征分散正则化、期望校准误差直接优化、多目标联合优化
- **约束**：冻结预训练CLIP参数、仅优化提示令牌、无标签测试数据

