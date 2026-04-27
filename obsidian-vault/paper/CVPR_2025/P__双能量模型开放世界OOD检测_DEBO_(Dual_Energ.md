---
title: Dual Energy-Based Model with Open-World Uncertainty Estimation for Out-of-distribution Detection
type: paper
paper_level: C
venue: CVPR
year: 2025
paper_link: null
aliases:
- 双能量模型开放世界OOD检测
- DEBO (Dual Energ
- DEBO (Dual Energy-Based Model with Open-World Uncertainty Estimation)
acceptance: poster
cited_by: 4
method: DEBO (Dual Energy-Based Model with Open-World Uncertainty Estimation)
---

# Dual Energy-Based Model with Open-World Uncertainty Estimation for Out-of-distribution Detection

**Topics**: [[T__OOD_Detection]], [[T__Classification]] | **Method**: [[M__DEBO]] | **Datasets**: [[D__CIFAR-10]], [[D__CIFAR-100]], [[D__ImageNet-200]]

| 中文题名 | 双能量模型开放世界OOD检测 |
| 英文题名 | Dual Energy-Based Model with Open-World Uncertainty Estimation for Out-of-distribution Detection |
| 会议/期刊 | CVPR 2025 (poster) |
| 链接 | [arXiv](https://doi.org/10.1109/cvpr52734.2025.02396) · [Code] · [Project] |
| 主要任务 | Out-of-Distribution (OOD) Detection / Open-World Uncertainty Estimation |
| 主要 baseline | MSP, EBO, EBO (w. Daux), OE, CIDER, UDG, MCD, MixOE |

> [!abstract] 因为「现有能量方法需预估计边缘能量均值并引入额外超参数，且推理时仅用联合或条件概率无法显式建模开放世界不确定性」，作者在「EBO (Energy-based Regularization)」基础上改了「双能量分解架构 + 自收敛边缘能量损失 C''x + 分解推理分数 Sdec」，在「CIFAR-10/100 和 ImageNet-200 OOD Detection」上取得「CIFAR-10 FPR95 14.41（相比 EBO w. Daux 降低 2.58）、CIFAR-100 FPR95 33.73（相比 EBO w. Daux 降低 16.64）」

- **CIFAR-10 FPR95**: 14.41 vs. EBO (w. Daux) 16.99，AUROC 96.67 vs. CIDER 95.01
- **CIFAR-100 FPR95**: 33.73 vs. EBO (w. Daux) 50.37，AUROC 89.76 vs. EBO (w. Daux) 84.31
- **ImageNet-200 AUROC**: 88.1，为所有对比方法中最高

## 背景与动机

OOD检测的核心问题是：模型在训练时只见过有限类别的分布内（ID）数据，但部署时会遇到来自未知分布的样本。例如，一个只在CIFAR-10的猫、狗等10类上训练的分类器，面对SVHN的数字图像或自然纹理时，需要可靠地识别出"这不是我认识的任何一类"。现有方法主要从两个角度解决此问题：

**MSP (Maximum Softmax Probability)** 取softmax输出的最大概率作为置信度，简单但倾向于对OOD样本过度自信——深度网络的softmax校准本身就很差。**EBO (Energy-based Regularization)** 将分类器输出解释为能量函数，通过最小化ID数据的能量、最大化OOD数据的能量来拉开差距，但推理时仅使用联合能量分数，无法区分"数据本身奇怪"和"模型不确定"这两种不确定性。**EBO (w. Daux)** 进一步引入辅助OOD数据并约束边缘能量的均值，但需要先从预训练模型估计 $m_{in}$、$m_{out}$ 两个均值能量分数，引入两个额外超参数 $\lambda_1, \lambda_2$，流程繁琐且超参数敏感。

上述方法的根本局限在于：**推理时缺乏对不确定性的显式分解**。联合概率 $P(x,y)$ 或条件概率 $P(y|x)$ 都无法同时回答"这个数据点在整个输入空间中有多异常"（边缘不确定性）和"给定这个点，模型预测有多不确定"（条件不确定性）。在开放世界场景中，这两种不确定性来源截然不同——一个ID样本可能因训练不足而条件不确定，一个OOD样本可能因完全陌生而边缘异常。本文提出将联合能量显式分解为边缘与条件分量，并设计自收敛的边缘能量损失项，从而无需预估计即可优化开放世界不确定性估计。

## 核心创新

核心洞察：**将联合能量分布显式分解为边缘能量与条件能量，因为边缘能量 $E(x)$ 和条件能量 $E(y|x)$ 分别对应"数据异常程度"与"模型预测不确定性"两种可解释的开放世界不确定性来源，从而使无需超参数预估计的端到端不确定性建模成为可能。**

| 维度 | Baseline (EBO/EBO w. Daux) | 本文 (DEBO) |
|:---|:---|:---|
| **能量建模对象** | 仅联合能量 $E(x,y)$ | 显式分解：$E(x,y) = E(x) + E(y|x)$，同时建模边缘与条件能量 |
| **边缘能量约束方式** | EBO (w. Daux) 需预估计 $m_{in}, m_{out}$，引入超参数 $\lambda_1, \lambda_2$ | $C''_x$ 损失项自然收敛至零，无需任何预估计或额外超参数 |
| **推理分数设计** | 联合能量分数或MSP条件概率 | $S_{dec}$ 显式组合边缘与条件分量，分离两种不确定性来源 |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/bbba4a88-3d2a-47ea-8c70-66b3c0b742bd/figures/Figure_1.png)
*Figure 1 (architecture): The dual classifier architecture of our proposed Dual Energy-Based Model for Out-of-Distribution detection. Given an in-distribution sample x_in, the Dual Energy-Based Model is employed to learn the joint distribution p(x,y) via two branches: the first branch for estimating the conditional distribution p(y|x) (classification), while the second branch for estimating the marginal distribution p(x) (density). Once well trained, the ID data will be assigned with lower dual energy (higher p(x,y)) than OOD samples.*



DEBO的整体数据流遵循"训练时三重约束，推理时双重分解"的设计：

**输入**：ID训练样本 $x_{in}$（带标签 $y$）+ 辅助OOD样本 $x_{out}$（无标签或伪标签）

**模块1：ID分类头（$C_{cse}$）** — 输入ID样本，输出类别预测与标准交叉熵损失，保证分布内分类精度。

**模块2：能量建模层（$C_{en} + C''_x$）** — 输入ID与OOD样本，输出能量正则化的表示。$C_{en}$ 执行经典的能量最小化/最大化操作；$C''_x$ 是本文核心创新，直接对边缘能量进行约束，使其自然收敛。

**模块3：双分布分解模块** — 将联合能量分布 $E(x,y)$ 在数学上分解为边缘分量 $E(x)$ 和条件分量 $E(y|x)$，为推理时的不确定性分离提供基础。

**模块4：$S_{dec}$ 分数计算** — 输入分解后的边缘与条件能量，输出最终的OOD检测分数。该分数显式结合"数据异常度"与"预测不确定度"，替代传统的MSP或纯联合能量分数。

```
输入 x ──→ [分类头 Ccse] ──→ 类别预测
    │
    └──→ [能量建模 Cen + C''x] ──→ 能量表示
              │
              ↓
         [双分布分解] ──→ E(x) [边缘] + E(y|x) [条件]
              │
              ↓
         [Sdec 计算] ──→ OOD检测分数
```

## 核心模块与公式推导

### 模块1: 自收敛边缘能量损失 $C''_x$（对应框架图：能量建模层）

**直觉**：传统方法需要人工估计边缘能量的均值再施加约束，本质上是用"两步法"解决一个本可端到端优化的问题；$C''_x$ 通过重新设计损失结构，让优化过程本身自动将边缘能量拉向目标值。

**Baseline 公式** (EBO w. Daux):
$$\mathcal{L}_{EBO+D_{aux}} = \mathcal{L}_{EBO} + \lambda_1 (E_{in} - m_{in})^2 + \lambda_2 (E_{out} - m_{out})^2$$
符号: $E_{in}, E_{out}$ = ID/OOD样本的边缘能量；$m_{in}, m_{out}$ = 需预估计的均值能量分数；$\lambda_1, \lambda_2$ = 额外超参数。

**变化点**：EBO (w. Daux) 要求先用预训练模型跑一遍数据估计 $m_{in}, m_{out}$，再作为固定值代入训练，超参数敏感且流程割裂。本文发现：若直接设计损失使相关项在优化过程中自然趋于零，则可完全消除预估计步骤。

**本文公式（推导）**:
$$\text{Step 1}: \mathcal{L}_{DEBO} = C_{cse} + C_{en} + C''_x \quad \text{将边缘能量约束从显式正则改为隐式结构}$$
$$\text{Step 2}: C''_x \text{ 的结构使得其内部第一、二项在梯度下降中相互牵引，自动收敛} \quad \text{无需预设目标值}$$
$$\text{最终}: \mathcal{L}_{DEBO} = C_{cse} + C_{en} + C''_x$$

**对应消融**：去掉 $C''_x$ 后，CIFAR-100 FPR95 从 33.73 升至 41.62（$\Delta$ +7.89），证明其在复杂分布上的关键作用；CIFAR-10 上影响较小（14.41→14.43），说明简单数据集对边缘能量约束的依赖较弱。

### 模块2: 分解推理分数 $S_{dec}$（对应框架图：Sdec分数计算模块）

**直觉**：联合概率 $P(x,y)$ 混淆了"数据是否常见"和"预测是否确定"；将贝叶斯分解 $P(x,y) = P(x)P(y|x)$ 显式写入分数，可分别处理两种不确定性。

**Baseline 公式** (MSP/MCP):
$$S_{MSP} = \max_y P(y|x) = \max_y \frac{P(x,y)}{P(x)}$$
符号: $P(y|x)$ = 条件概率（模型置信度）；$P(x,y)$ = 联合概率；$P(x)$ = 边缘概率（数据密度）。

**变化点**：MSP仅使用条件概率，对OOD检测不足——一个OOD样本可能被模型"自信地"错分类。$S_{joint}$ 虽用联合概率，但仍将两种不确定性捆绑。本文提出显式保留边缘-条件结构，让两者在分数中各司其职。

**本文公式（推导）**:
$$\text{Step 1}: S_{joint} = P(x,y) \propto \exp(-E(x,y)) \quad \text{联合能量分数，作为消融对照}$$
$$\text{Step 2}: E(x,y) = E(x) + E(y|x) \quad \text{能量层面的加性分解（对应概率的乘性分解）}$$
$$\text{最终}: S_{dec} = f\big(\underbrace{E(x)}_{\text{数据异常度}}, \underbrace{E(y|x)}_{\text{预测不确定度}}\big)$$

具体实现上，$S_{dec}$ 将边缘能量（反映 $P(x)$，即数据在输入空间的密度）与条件能量（反映 $P(y|x)$，即模型给定预测的不确定性）以特定方式组合，使OOD样本在两个维度上同时暴露异常。

**对应消融**：Table 3 显示，使用MCP替代 $S_{dec}$ 时CIFAR-100 FPR95从33.73升至48.38（$\Delta$ +14.65）；使用 $S_{joint}$ 替代时升至48.13（$\Delta$ +14.40），证明显式分解优于联合或纯条件方法。

### 模块3: 能量正则化项 $C_{en}$（对应框架图：能量建模层）

**直觉**：经典的能量基OOD方法核心，保持ID能量低、OOD能量高的基本分离。

**Baseline 公式** (EBO):
$$C_{en}: \min E(x_{in}, y), \quad \max E(x_{out}) \text{ (或等价能量边界)}$$

**本文公式**：$C_{en}$ 作为 $\mathcal{L}_{DEBO}$ 的第二项保留，与 $C''_x$ 协同工作——$C_{en}$ 负责联合能量的粗粒度分离，$C''_x$ 负责边缘能量的精细约束。

**对应消融**：去掉 $C_{en}$ 后CIFAR-10 FPR95从14.41暴增至26.56（$\Delta$ +12.15），CIFAR-100从33.73增至58.46，为所有消融中影响最剧烈的单一组件，证明能量正则化仍是基础支柱。

## 实验与分析


![Table 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/bbba4a88-3d2a-47ea-8c70-66b3c0b742bd/figures/Table_2.png)
*Table 2 (ablation): The ablation results about the components of our proposed model.*



本文在CIFAR-10、CIFAR-100和ImageNet-200三个标准OOD检测基准上评估DEBO，使用FPR95（越低越好，表示95%真阳性率时的假阳性率）和AUROC（越高越好）两个核心指标。所有方法统一采用ResNet-18 backbone以保证公平对比。

**核心结果**：在CIFAR-10上，DEBO取得FPR95 14.41、AUROC 96.67，相比直接基线EBO (w. Daux)的FPR95 16.99绝对降低2.58，相比此前最优的CIDER（AUROC 95.01）绝对提升1.66。在更具挑战性的CIFAR-100上，优势大幅扩大：FPR95 33.73 vs. EBO (w. Daux) 50.37，绝对降低16.64（相对33.0%）；AUROC 89.76 vs. EBO (w. Daux) 84.31，绝对提升5.45。ImageNet-200大规模场景下，DEBO AUROC 88.1为所有对比方法最高，但FPR95 41.12略逊于MCD的40.07，显示在极大规模分布偏移下仍有优化空间。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/bbba4a88-3d2a-47ea-8c70-66b3c0b742bd/figures/Table_3.png)
*Table 3 (ablation): An ablation study about different energy function heads. OC is the abbreviation of outlier exposure; ODIN indicates the baseline method of ODIN.*



消融实验进一步验证各组件的必要性。能量正则化项 $C_{en}$ 是最关键组件：去掉后CIFAR-10 FPR95从14.41→26.56（+12.15），CIFAR-100从33.73→58.46。分解分数 $S_{dec}$ 同样关键：替换为MCP导致CIFAR-100 FPR95 +14.65，替换为 $S_{joint}$ 导致+14.40，证明显式边缘-条件分解优于联合或纯条件推理。边缘能量项 $C''_x$ 在CIFAR-100上贡献显著（+7.89），在CIFAR-10上几乎无影响（+0.02），暗示其收益随分布复杂度增加而凸显。

**公平性审视**：对比基线包含MSP、EBO系列、OE、CIDER、UDG、MCD、MixOE等训练时方法，但未包含ReAct、ASH、DICE、GEN、VOS、NPOS、MOS等更强力的后处理方法——这些后hoc方法在ImageNet-scale上往往表现更优。EBO (w. Daux)在ImageNet-200对比中因原论文未报告而被排除，削弱了该基准的基线强度。所有方法使用相同ResNet-18架构，训练数据设置一致，整体对比基本公平，但读者需注意DEBO的优势主要体现在训练时方法范畴内。

## 方法谱系与知识库定位

**方法家族**：Energy-Based Model for OOD Detection → **父方法**：EBO (Energy-based Regularization)

**改动槽位**：
- **架构**：单模型单头/双头 → 显式双能量分解架构（联合/边缘/条件三重分布建模）
- **目标函数**：标准能量损失 / 带预估计均值的辅助损失 → $\mathcal{L}_{DEBO} = C_{cse} + C_{en} + C''_x$，$C''_x$ 自收敛免超参数
- **推理策略**：MSP最大条件概率 / EBO联合能量分数 → $S_{dec}$ 显式分解边缘-条件不确定性
- **训练配方**：保留辅助OOD数据暴露（OE风格），但取消均值预估计步骤
- **数据策划**：标准ID+辅助OOD设置，无特殊数据筛选

**直接基线差异**：
- **vs. EBO**：从联合能量建模扩展到边缘-条件分解；从单一能量分数到 $S_{dec}$
- **vs. EBO (w. Daux)**：$C''_x$ 替代 $(E-m)^2$ 正则，消除预估计与超参数
- **vs. MCD**：同为双头/双分支思想，但MCD用分类器分歧而DEBO用概率分解
- **vs. CIDER**：CIDER用对比学习拉远ID/OOD表示，DEBO用能量分解显式建模不确定性

**后续方向**：(1) 将 $S_{dec}$ 的分解思想扩展至后处理方法（如ReAct+DEBO组合）；(2) 探索 $C''_x$ 自收敛机制在其他需预估计统计量的正则化任务中的通用性；(3) 在更大规模（完整ImageNet-1K）和更细粒度开放世界场景中验证边缘-条件分解的必要性。

**标签**：modality: 图像 / paradigm: 能量模型+概率分解 / scenario: 分布外检测 / mechanism: 边缘-条件能量分解+自收敛优化 / constraint: 需辅助OOD数据，同架构公平对比

