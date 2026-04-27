---
title: Dual Energy-Based Model with Open-World Uncertainty Estimation for Out-of-distribution Detection
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- 双能量基开放世界OOD检测
- DEBMOU
acceptance: poster
cited_by: 4
---

# Dual Energy-Based Model with Open-World Uncertainty Estimation for Out-of-distribution Detection

**Topics**: [[T__OOD_Detection]], [[T__Classification]] | **Datasets**: [[D__CIFAR-10]], [[D__CIFAR-100]], [[D__ImageNet-200]]

| 中文题名 | 双能量基开放世界OOD检测 |
| 英文题名 | Dual Energy-Based Model with Open-World Uncertainty Estimation for Out-of-distribution Detection |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [DOI](https://doi.org/10.1109/cvpr52734.2025.02396) |
| 主要任务 | Out-of-Distribution (OOD) Detection |
| 主要 baseline | MSP, EBO, EBO (w. Daux), OE, CIDER, UDG, MCD, MIXOE |

> [!abstract] 因为「现有能量基OOD检测方法（如EBO）需要预训练模型估计均值边际能量并引入额外超参数」，作者在「EBO」基础上改了「用双能量基模型结构替代单头结构，设计自然收敛的边际能量损失项C''_x替代超参数依赖的均值能量估计，并提出分解式检测分数S_dec」，在「CIFAR-10/100、ImageNet-200」上取得「CIFAR-10 FPR95 14.41（较EBO w. Daux降低15.2%）、CIFAR-100 FPR95 33.73（较次优降低33.0%）」

- **CIFAR-10 FPR95**: 14.41，超越次优方法 EBO (w. Daux) 的 16.99，AUROC 达 96.67
- **CIFAR-100 FPR95**: 33.73，大幅领先次优方法 EBO (w. Daux) 的 50.37，AUROC 达 89.76
- **ImageNet-200 AUROC**: 88.10，在报告方法中最优，但 FPR95 41.12 略逊于 MCD (40.07)

## 背景与动机

OOD检测的核心问题是：当部署环境中的测试样本来自训练分布之外时，模型能否可靠地识别并拒绝这些样本。例如，一个仅在CIFAR-10（飞机、汽车等10类）上训练的图像分类器，遇到"熊"或"钟"的图像时，不应盲目猜测，而应输出"不确定"。

现有方法从不同角度解决这一问题。**MSP (Maximum Softmax Probability)** 直接使用分类器输出的最大softmax概率作为置信度，简单但容易对OOD样本过度自信。**EBO (Energy-based Out-of-distribution Detection)** 引入能量函数框架，通过最小化ID样本能量、最大化OOD样本能量来校准模型，但需预训练模型估计均值边际能量分数，并引入两个额外超参数 $m_{in}$、$m_{out}$ 约束能量边界。**OE (Outlier Exposure)** 在训练时引入辅助OOD数据并保持均匀分布，但依赖外部数据的质量和覆盖度。

这些方法的关键局限在于：**EBO (w. Daux)** 的均值能量估计步骤繁琐且引入超参数敏感性；现有分数函数（MSP、S_joint）未能充分利用能量分解结构。作者提出：能否设计一种无需预估计、无需额外超参数的能量损失，同时利用双能量结构在推理时更精细地量化不确定性？本文通过双能量基模型与开放世界不确定性估计，实现了这一目标。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a40f3ec1-4ada-48d6-a412-816131d91ba4/figures/Figure_1.png)
*Figure 1: Figure 1. The dual classifier architecture for our proposed DualEnergy-Based Model and the Dual Energy-Based Model for OODdetection (DEBO). The generated data is sampled from the dis-tribution modeled*



## 核心创新

核心洞察：将联合能量 $E(x,y)$ 显式分解为边际能量 $E(x)$ 与条件能量 $E(x|y)$ 后，边际能量项 $C''_x$ 在优化过程中可自然收敛至零，从而完全规避预训练模型估计均值能量的复杂步骤；基于此分解结构设计的 $S_{dec}$ 分数能更精细地捕捉开放世界中的不确定性。

| 维度 | Baseline (EBO) | 本文 (DEBO) |
|:---|:---|:---|
| 能量估计 | 单能量输出，需预训练估计均值边际能量 | 双能量基结构，联合估计 $E(x,y)$ 与 $E(x)$ |
| 损失设计 | 显式超参数 $m_{in}, m_{out}$ 约束能量边界 | $C''_x$ 自然收敛，零额外超参数 |
| 推理分数 | 能量分数或MSP | $S_{dec}$ 分解式分数，利用能量分解结构 |

## 整体框架



DEBO 的完整数据流如下：

1. **特征提取 (ResNet-18)**：输入图像（ID训练样本或辅助OOD样本）→ 输出特征表示。所有对比方法共享此backbone以保证公平性。

2. **双能量估计**：特征表示同时送入两个分支 → 输出**联合能量** $E(x,y)$（样本-类别对的能量）和**边际能量** $E(x)$（仅关于样本的能量）。这是区别于标准单头分类器的核心结构修改。

3. **DEBO训练**：利用三部分损失 $C_{cse} + C_{en} + C''_x$ 进行端到端训练。$C_{cse}$ 保证ID分类精度，$C_{en}$ 实现ID/OOD能量分离，$C''_x$ 校准边际能量分布。

4. **$S_{dec}$ 分数计算**：测试阶段，基于学习到的联合能量与边际能量，计算分解式OOD检测分数 → 输出最终不确定性估计。

```
Input Image x
    ↓
[ResNet-18 Backbone] ──→ Feature z
    ↓
[Dual Energy Heads] ──┬──→ E(x,y)  (Joint Energy)
                      └──→ E(x)    (Marginal Energy)
    ↓
[Training]  L_DEBO = C_cse + C_en + C''_x
    ↓
[Inference] S_dec(x) = f(E(x), E(x|y))
    ↓
OOD Score
```

## 核心模块与公式推导

### 模块 1: 双能量基损失函数 $L_{DEBO}$（对应框架图：训练阶段）

**直觉**：EBO需要预训练模型估计均值能量并设置超参数边界，这一步骤既繁琐又引入敏感性；通过能量分解，边际能量项可自我校准。

**Baseline 公式 (EBO)**:
$$L_{EBO} = E(x, y_{in}) - E(x, y_{out}) + \text{约束项 with } m_{in}, m_{out}$$
符号: $E(x,y)$ = 联合能量函数; $m_{in}, m_{out}$ = 需预估计的均值能量边界超参数; $y_{in}, y_{out}$ = ID/OOD标签。

**变化点**：EBO 需要先用预训练模型估计 $\mu_{in}, \mu_{out}$，再设置超参数约束；本文发现边际能量分解后，该项可自然优化收敛。

**本文公式（推导）**:
$$\text{Step 1}: \quad p(y|x) = \frac{\exp(-E(x,y))}{\sum_{y'}\exp(-E(x,y'))} \quad \text{（能量-概率对偶，建立基础）}$$
$$\text{Step 2}: \quad E(x) = -\log \sum_y \exp(-E(x,y)) \quad \text{（边际能量定义，从联合能量分解）}$$
$$\text{Step 3}: \quad C''_x \rightarrow 0 \text{ naturally during optimization} \quad \text{（关键：自然收敛，无需预估计）}$$
$$\text{最终}: L_{DEBO} = C_{cse} + C_{en} + C''_x$$
其中 $C_{cse}$ 为ID分类交叉熵，$C_{en} = E(x, y_{in}) - E(x, y_{out})$ 为能量分离项，$C''_x$ 为边际能量校准项。

**对应消融**：去掉 $C_{en}$ 后 CIFAR-10 FPR95 从 14.41 升至 26.56（+12.15）；去掉 $C''_x$ 后 CIFAR-100 FPR95 从 33.73 升至 41.62（+7.89）。

### 模块 2: 分解式检测分数 $S_{dec}$（对应框架图：推理阶段）

**直觉**：传统MSP或简单能量分数未区分"样本本身是否异常"与"给定类别下样本是否异常"；利用双能量结构的分解可更精细量化开放世界不确定性。

**Baseline 公式 (MSP / MCP)**:
$$S_{MCP}(x) = \max_y p(y|x) = \max_y \frac{\exp(f_y(x))}{\sum_j \exp(f_j(x))}$$
符号: $f_y(x)$ = 分类器logit输出; 该分数仅反映条件概率最大值，无法区分边际不确定性。

**变化点**：将联合概率分解为边际与条件部分，设计显式利用 $E(x)$ 与 $E(x|y)$ 关系的分数。

**本文公式（推导）**:
$$\text{Step 1}: \quad S_{joint}(x) = \max_y p(y|x) \text{ based on energy} \quad \text{（中间方案，能量化MCP）}$$
$$\text{Step 2}: \quad p(x,y) = p(x) \cdot p(y|x) \propto \exp(-E(x)) \cdot \exp(-E(x|y)) \quad \text{（联合=边际×条件的能量分解）}$$
$$\text{最终}: S_{dec}(x) = f(E(x), E(x|y)) \quad \text{（显式利用双能量结构的分解式分数）}$$
具体形式见原文 Equation 29；核心是将OOD检测从单一条件概率判断升级为联合-边际-条件的三元不确定性分析。

**对应消融**：使用MCP替代 $S_{dec}$，CIFAR-100 FPR95 从 33.73 升至 48.38（+14.65）；使用 $S_{joint}$ 替代，CIFAR-100 FPR95 升至 48.13（+14.40）。

## 实验与分析


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a40f3ec1-4ada-48d6-a412-816131d91ba4/figures/Table_2.png)
*Table 2: Table 1. Main results comparing competitive OOD detection methods, all trained using a ResNet-18 backbone. Values are presented aspercentages and represent averages across the OOD test datasets descri*



本文在 CIFAR-10、CIFAR-100 和 ImageNet-200 三个数据集上评估OOD检测性能，所有方法统一使用 ResNet-18 backbone 以保证公平比较。核心指标为 FPR95（越低越好，表示95% ID样本被正确识别时OOD样本的误报率）和 AUROC（越高越好）。

**CIFAR-10 结果**：DEBO 取得 FPR95 = 14.41、AUROC = 96.67。相比直接前身 EBO (w. Daux) 的 FPR95 = 16.99，相对降低 15.2%；相比此前最优的 CIDER (AUROC = 95.01)，绝对提升 1.66 点。这一优势在密集类别场景下更为显著——**CIFAR-100 结果**：FPR95 = 33.73，较 EBO (w. Daux) 的 50.37 大幅降低 33.0%（相对），较次优方法 MCD 的 56.02 优势更大；AUROC = 89.76，较 EBO (w. Daux) 的 84.31 提升 5.45 点。ImageNet-200 上 DEBO 的 AUROC = 88.10 为报告方法中最优，但 FPR95 = 41.12 略逊于 MCD 的 40.07，显示在大规模复杂场景下仍有优化空间。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/a40f3ec1-4ada-48d6-a412-816131d91ba4/figures/Table_3.png)
*Table 3: Table 3. The ablative results about different score functions basedon our proposed DEBO training method.*



消融实验验证了各组件的必要性。损失函数层面：仅保留 $C_{cse}$（基础交叉熵）性能最差；$C_{en}$ 对 CIFAR-10 至关重要（移除后 FPR95 +12.15）；$C''_x$ 对 CIFAR-100 贡献突出（移除后 FPR95 +7.89），二者协同实现最优。分数函数层面：$S_{dec}$ 显著优于 MCP 和 $S_{joint}$，在 CIFAR-100 上分别带来 FPR95 -14.65 和 -14.40 的改进，证明能量分解结构在推理时的价值。

**公平性检查**：对比方法包含 MSP、EBO、EBO (w. Daux)、OE、CIDER、UDG、MCD、MIXOE，但缺少近年强效的后处理方法如 ReAct、ASH、DICE，以及基于 ViT 的方法。EBO (w. Daux) 未在 ImageNet-200 上报告（原论文无此实验）。实验未报告标准差或置信区间，且 ID-ACC 值在部分方法上缺失。所有方法使用相同 ResNet-18，参数量公平，但训练/推理的计算开销细节未披露。

## 方法谱系与知识库定位

**方法族**：Energy-Based OOD Detection → **父方法**：EBO (Energy-based Out-of-distribution Detection, NeurIPS 2020)

**修改槽位**：
- **目标函数 (objective)**：EBO 的预估计均值能量+超参数约束 → DEBO 的 $C''_x$ 自然收敛
- **推理策略 (inference_strategy)**：能量分数/MSP → $S_{dec}$ 分解式分数
- **架构 (architecture)**：单能量头 → 双能量基联合-边际估计结构

**直接基线对比**：
- **EBO / EBO (w. Daux)**：直接前身；DEBO 移除预估计步骤和 $m_{in}, m_{out}$ 超参数，改自然收敛损失
- **MSP**：最广泛基线；DEBO 在能量框架下全面超越
- **OE**：同样使用辅助OOD数据；DEBO 不强制均匀分布假设，通过能量分解更灵活利用
- **MCD**：双头结构先驱；DEBO 的双能量头目的不同（能量分解 vs. 分类器差异），且配合全新损失与分数

**后续方向**：(1) 将双能量结构扩展至 ViT/Swin 等更强backbone；(2) 与后处理方法（ReAct、ASH）结合，探索"训练时能量分解 + 推理时激活修剪"的互补性；(3) 将开放世界不确定性估计应用于主动学习或安全关键决策系统。

**标签**：modality: 图像 | paradigm: 能量基学习 | scenario: 分布外检测 | mechanism: 能量分解/双头结构/边际能量自然收敛 | constraint: 需辅助OOD数据训练

