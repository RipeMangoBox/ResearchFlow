---
title: Bayesian Low-rank Adaptation for Large Language Models
type: paper
paper_level: C
venue: ICLR
year: 2024
paper_link: null
aliases:
- 大语言模型的贝叶斯低秩适应
- Laplace-LoRA
acceptance: Poster
cited_by: 111
method: Laplace-LoRA
followups:
- MONA：双分支卷积Adapte_BPSFFV
- MONA：5%参数超越全量微调的_MONA_(Multi-cOmp
- OmniVCus_OmniVCus__Feedforward_Sub
---

# Bayesian Low-rank Adaptation for Large Language Models

**Topics**: [[T__Text_Generation]] | **Method**: [[M__Laplace-LoRA]] | **Datasets**: WG-S, ARC-Challenge, BoolQ, Winogrande-small, ARC-Easy

| 中文题名 | 大语言模型的贝叶斯低秩适应 |
| 英文题名 | Bayesian Low-rank Adaptation for Large Language Models |
| 会议/期刊 | ICLR 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2308.13111) · [Code] · [Project] |
| 主要任务 | 常识推理（Commonsense Reasoning）：Winogrande、ARC-Challenge、ARC-Easy、OpenBookQA、BoolQ |
| 主要 baseline | MAP（标准LoRA微调）、MC Dropout、Checkpoint Ensemble、Temperature Scaling、Deep Ensemble、LLLA |

> [!abstract]
> 因为「标准LoRA微调仅提供点估计，导致模型校准差、不确定性量化不可靠」，作者在「LoRA」基础上改了「引入post-hoc Laplace近似，对LoRA参数建立高斯后验，并采用KFAC Hessian近似与线性化预测实现高效贝叶斯推断」，在「LLaMA-7B/LLaMA2-7B常识推理基准」上取得「ECE从30.8降至6.9，NLL从2.75降至0.66」

- **关键性能 1**：Winogrande-small 上 ECE 6.9 vs MAP 30.8，校准误差降低 77.6%
- **关键性能 2**：Winogrande-small 上 NLL 0.66 vs MAP 2.75，概率似然提升显著
- **关键性能 3**：ARC-Challenge 上 ACC 66.0 vs MAP 64.9，准确率提升 +1.1

## 背景与动机

大型语言模型（LLM）经过LoRA等参数高效微调（PEFT）后，在下游任务上表现出色，但其预测往往过度自信——模型对错误答案给出极高的概率，导致校准（calibration）严重失衡。例如，在多项选择题中，模型可能以99%的置信度选择错误选项，而人类用户无法从输出概率中察觉这种风险。

现有方法如何处理这一问题？**Temperature Scaling**（Guo et al., 2017）在验证集上学习一个全局温度参数，缩放logits以改善校准，但仅调整输出分布的锐度，不改变模型内部的不确定性表示。**MC Dropout**（Gal & Ghahramani, 2016）通过在测试时启用dropout进行多次前向传播来估计不确定性，但需10次推理，计算开销大且对LoRA结构并非最优。**Checkpoint Ensemble**（Chen et al., 2017）集成训练过程中保存的多个检查点，但仅捕获优化轨迹上的点估计，缺乏系统的贝叶斯理论支撑。**Deep Ensemble** 训练多个独立模型，效果较好但成本高昂（本工作仅用3个LoRA模型）。

这些方法的**核心局限**在于：它们要么是对点估计的后处理（Temperature Scaling），要么依赖启发式采样（MC Dropout、Checkpoint Ensemble），均未对LoRA参数建立真正的概率后验分布。特别地，LLM微调数据通常有限（<10k样本），点估计容易过拟合，而贝叶斯方法理论上能自动实现正则化与不确定性量化，但直接应用于全参数不可行。

本文提出 **Laplace-LoRA**：在标准LoRA微调后，对低秩参数进行post-hoc Laplace近似，以极低成本获得贝叶斯预测分布，同时保持甚至提升任务准确率。

## 核心创新

**核心洞察**：LoRA的低秩结构天然适合贝叶斯处理，因为参数数量仅为原模型的0.1-1%，使得在MAP检查点上进行post-hoc Laplace近似计算可行；同时，KFAC（Kronecker-Factored Approximate Curvature）能高效近似LoRA参数的Hessian，而线性化模型预测避免了昂贵的蒙特卡洛采样，从而使大规模LLM的贝叶斯推断首次在LoRA框架内实用化。

| 维度 | Baseline（标准LoRA） | 本文（Laplace-LoRA） |
|:---|:---|:---|
| **推断策略** | MAP点估计：直接使用微调后的权重 $\theta_{\text{MAP}}$ 预测 | 贝叶斯预测分布：对LoRA参数积分，用高斯后验 $q(\theta)$ 加权平均 |
| **Hessian计算** | 无需计算 | KFAC近似：针对低秩矩阵的Kronecker分解，避免显式存储全Hessian |
| **预测方式** | 单次前向传播 $p(y\|x, \theta_{\text{MAP}})$ | 线性化模型：一阶泰勒展开，闭式计算后验预测，无需采样 |
| **不确定性来源** | 无（单一确定性输出） | 参数后验不确定性 + 逐层分解（LA-Early / LA-Last） |
| **额外训练成本** | 标准LoRA微调 | 零额外训练；仅post-hoc在保存的检查点上应用Laplace |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/28f6f604-2ba2-42dd-8094-af98dddf6da6/figures/Figure_1.png)
*Figure 1 (result): Performance of LoRA/MAP methods on in-distribution and out-of-distribution data with 2000 posterior samples for Bayesian methods.*



Laplace-LoRA 的完整数据流分为三个阶段，形成"微调→近似→预测"的流水线：

**阶段一：LoRA Fine-tuning（标准微调）**
- 输入：预训练LLM（LLaMA-7B或LLaMA2-7B）、任务特定数据集（如Winogrande、ARC等）
- 处理：对query、value及output层应用LoRA，秩 $r \ll d,k$，训练至收敛或固定步数
- 输出：MAP检查点 $\theta_{\text{MAP}}$，即低秩矩阵对 $(B, A)$ 的点估计
- 关键操作：每1000梯度步保存一个检查点，供后续post-hoc分析

**阶段二：Laplace Approximation with KFAC（后验近似）**
- 输入：MAP检查点、训练数据（用于曲率估计）
- 处理：在 $\theta_{\text{MAP}}$ 处计算对数后验的Hessian，用KFAC分解为两个Kronecker因子的乘积
- 输出：高斯后验 $q(\theta) = \mathcal{N}(\theta_{\text{MAP}}, \Sigma)$，其中精度矩阵 $\Sigma^{-1}$ 由观测信息（似然Hessian）与先验信息组成

**阶段三：Linearized Prediction（贝叶斯预测）**
- 输入：高斯后验 $q(\theta)$、测试样本 $x$
- 处理：对神经网络在 $\theta_{\text{MAP}}$ 处进行一阶泰勒展开，将贝叶斯模型平均转化为闭式计算
- 输出：预测分布 $p(y|x, \mathcal{D})$，包含均值预测与不确定性估计

```
预训练LLM ──→ [LoRA Fine-tuning] ──→ MAP检查点 θ_MAP
                                              │
                                              ▼
训练数据 ───→ [KFAC Hessian近似] ──→ 高斯后验 q(θ) = N(θ_MAP, Σ)
                                              │
                                              ▼
测试输入 x ─→ [线性化模型预测] ──→ 贝叶斯预测分布 p(y|x, D)
```

## 核心模块与公式推导

### 模块 1: LoRA参数化与MAP基线（对应框架图 阶段一）

**直觉**：仅微调低秩增量而非全权重，将可训练参数降至原模型的0.1-1%，使贝叶斯处理在计算上可行。

**Baseline 公式** (标准LoRA): $$W = W_0 + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}, \quad r \ll \min(d,k)$$
符号: $W_0$ = 预训练冻结权重, $B,A$ = 可训练低秩矩阵, $r$ = 秩（通常 $r \leq 64$）

**变化点**：标准LoRA仅优化 $B,A$ 的点估计，无不确定性；本文将这些低秩参数视为随机变量，建立后验分布。

**本文公式**：参数向量 $\theta = \text{vec}(B, A) \in \mathbb{R}^{(d+k)r}$，后续所有操作仅针对此低维空间。

---

### 模块 2: Laplace近似后验（对应框架图 阶段二）

**直觉**：在MAP点用高斯分布近似真实后验，协方差由Hessian决定；KFAC利用低秩结构的Kronecker分解避免 $O(D^2)$ 存储。

**Baseline 公式** (精确贝叶斯，不可行): $$p(\theta | \mathcal{D}) = \frac{p(\mathcal{D}|\theta)p(\theta)}{\int p(\mathcal{D}|\theta)p(\theta)d\theta}$$
符号: $\mathcal{D}$ = 训练数据, $p(\theta)$ = 先验（通常取高斯）, 分母为边缘似然（intractable for LLM）

**变化点**：精确后验积分在高维参数空间不可计算；Laplace近似在MAP点二阶泰勒展开，得到解析高斯。进一步，全参数Hessian $O(D^2)$ 对LLM不可行，故对LoRA参数的Kronecker结构使用KFAC分解。

**本文公式（推导）**:
$$\text{Step 1}: \log p(\theta | \mathcal{D}) \approx \log p(\theta_{\text{MAP}} | \mathcal{D}) - \frac{1}{2}(\theta - \theta_{\text{MAP}})^\text{top} \Lambda (\theta - \theta_{\text{MAP}}) \quad \text{在MAP点二阶展开，忽略高阶项}$$
$$\text{Step 2}: \Lambda = -\nabla^2_\theta \log p(\mathcal{D}|\theta)|_{\theta_{\text{MAP}}} + \nabla^2_\theta \log p(\theta)|_{\theta_{\text{MAP}}} \quad \text{精度矩阵 = 观测信息 + 先验信息}$$
$$\text{Step 3 (KFAC)}: \nabla^2_\theta \log p(\mathcal{D}|\theta) \approx A \otimes B \quad \text{Kronecker分解，利用低秩矩阵的梯度外积结构}$$
$$\text{最终}: q(\theta) = \mathcal{N}(\theta_{\text{MAP}}, \Lambda^{-1}) = \mathcal{N}(\theta_{\text{MAP}}, \Sigma)$$

**对应消融**：Table 1 / Table 3 显示，仅对最后一层应用Laplace（LLLA）的ECE为11.8，而全层Laplace-LoRA降至6.9，Δ = 4.9。

---

### 模块 3: 线性化预测分布（对应框架图 阶段三）

**直觉**：避免对高斯后验进行昂贵的蒙特卡洛采样，通过一阶展开将积分转化为闭式计算。

**Baseline 公式** (MAP预测): $$p(y|x, \mathcal{D}) \approx p(y|x, \theta_{\text{MAP}}) \quad \text{单次前向，无不确定性量化}$$

**变化点**：贝叶斯模型平均需积分 $p(y|x, \theta)$  over $q(\theta)$，但对神经网络该积分无闭式解；MC采样需数百次前向传播。线性化模型用泰勒展开近似网络输出，使积分可解析。

**本文公式（推导）**:
$$\text{Step 1}: f(x; \theta) \approx f(x; \theta_{\text{MAP}}) + \nabla_\theta f(x; \theta)|_{\theta_{\text{MAP}}}^\text{top} (\theta - \theta_{\text{MAP}}) \quad \text{网络输出的一阶展开}$$
$$\text{Step 2}: \mathbb{E}_q[f(x; \theta)] \approx f(x; \theta_{\text{MAP}}) + \nabla_\theta f^\text{top} \underbrace{\mathbb{E}_q[\theta - \theta_{\text{MAP}}]}_{=0} = f(x; \theta_{\text{MAP}}) \quad \text{均值预测与MAP相同}$$
$$\text{Step 3}: \text{Var}_q[f(x; \theta)] \approx \nabla_\theta f^\text{top} \Sigma \nabla_\theta f \quad \text{预测方差由梯度与后验协方差决定}$$
$$\text{最终}: p(y|x, \mathcal{D}) \approx \int \text{softmax}(f(x; \theta)) q(\theta) d\theta \approx \text{softmax}\left(\frac{\mu_{\text{lin}}}{\sqrt{1 + \pi \sigma^2_{\text{lin}}/8}}\right) \text{ (probit近似)}$$

**对应消融**：Figure 12（LA-Early vs LA-Last）显示，早期层与最后一层对logits标准差的贡献模式不同，验证全层Laplace的必要性。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/28f6f604-2ba2-42dd-8094-af98dddf6da6/figures/Table_1.png)
*Table 1 (comparison): Comparison of different posterior methods applied to the fine-tuned LLaMA-7B models.*



本文在LLaMA-7B与LLaMA2-7B上评估，覆盖六个常识推理基准：Winogrande-small/medium（WG-S/WG-M）、ARC-Challenge/Easy（ARC-C/ARC-E）、OpenBookQA（OBQA）、BoolQ。核心发现来自Table 1与Table 2的对比：Laplace-LoRA在保持与MAP相当准确率的同时，实现了**数量级的校准提升**。以WG-S为例，MAP的ECE高达30.8，经Temperature Scaling后降至12.8，LLLA进一步降至11.8，而Laplace-LoRA达到6.9——不仅优于所有确定性后处理方法，也优于仅对最后一层近似的LLLA。NLL指标同样显著：MAP为2.75，Temperature Scaling与LLLA均为0.68，Laplace-LoRA以0.66略胜一筹。

准确率方面，Laplace-LoRA并非全面超越MAP，而是在关键任务上取得提升：ARC-C从64.9提升至66.0（+1.1），WG-S 66.5略低于MAP 67.0但差距极小（-0.5），其余任务如ARC-E（85.0 vs 85.2）、BoolQ（85.7 vs 85.8）基本持平。这一"准确率持平、校准飞跃"的模式正是贝叶斯方法的核心价值——不牺牲预测能力，但提供可靠的不确定性估计。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/28f6f604-2ba2-42dd-8094-af98dddf6da6/figures/Table_3.png)
*Table 3 (comparison): Comparison of different posterior methods across test and out-of-distribution datasets.*



消融实验（Table 3及Figure 12）聚焦**层-wise贡献**：LLLA（仅最后一层Laplace）ECE 11.8 vs 全层Laplace-LoRA 6.9，Δ = 4.9，证明早期层的贝叶斯处理对校准至关重要。LA-Early与LA-Last的logits标准差分解显示，两层不确定性来源具有不同特征，简单丢弃任何一层都会损失信息。


![Table 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/28f6f604-2ba2-42dd-8094-af98dddf6da6/figures/Table_4.png)
*Table 4 (comparison): Comparison of training times and memory cost of standard LoRA and Bayesian LoRA.*



效率方面，Table 4对比标准LoRA与Bayesian LoRA的训练时间与内存：Laplace-LoRA为**post-hoc方法**，不增加训练成本（标准LoRA微调10,000步，batch size 4），仅在保存的检查点上附加KFAC计算与线性化预测。推理时无需多次前向（vs MC Dropout的10次、Deep Ensemble的3个独立模型），单次线性化即可输出预测分布。

**公平性检查**：本文比较的方法中，Temperature Scaling、MC Dropout、Checkpoint Ensemble均为轻量级基线，Deep Ensemble仅用3个模型（文献常用5-10），存在调优空间。缺失的强基线包括：SWAG/MultiSWAG、变分推断方法、MCMC采样等。此外，实验局限于小数据常识推理（<10k样本），未覆盖生成任务、长上下文或更大规模数据。作者未明确讨论线性化近似的失败模式（如MAP点非局部最优时的近似失效）。

## 方法谱系与知识库定位

**方法家族**：贝叶斯深度学习 → Laplace近似 → 参数高效微调（PEFT）

**父方法**：LoRA（Low-Rank Adaptation, Hu et al., 2022）
- 继承：低秩参数化形式 $\theta = \theta_0 + BA$，冻结预训练权重
- 改变slot：
  - **inference_strategy**：MAP点估计 → Laplace后验 + 线性化贝叶斯预测
  - **architecture**：全层LoRA权重均可被贝叶斯处理（query/value/output），非仅最后一层
  - **training_recipe**：标准微调 + 每1000步保存检查点 → post-hoc Laplace，支持early stopping

**直接基线与差异**：
- **MAP（标准LoRA）**：无不确定性，校准差；本文添加贝叶斯后验
- **LLLA（Last-Layer Laplace）**：仅输出层Laplace，本文扩展至全层
- **MC Dropout**：测试时采样，需多次前向；本文单次线性化，更高效的解析解
- **Temperature Scaling**：仅缩放logits，不改变内部表示；本文从参数层面建模不确定性
- **Deep Ensemble**：训练多个模型；本文单模型贝叶斯平均，成本更低

**后续方向**：
1. **扩展至其他PEFT方法**：将Laplace近似应用于Adapter、Prefix-tuning、IA³等
2. **更紧的变分近似**：探索结构化高斯或正态流替代KFAC，捕获后验多模态
3. **生成任务与长上下文**：当前仅限判别式常识推理，需验证在文本生成、摘要等开放域任务中的校准效果

**标签**：
- modality: 文本（NLP）
- paradigm: 参数高效微调（PEFT）、贝叶斯深度学习
- scenario: 常识推理、模型校准、不确定性量化
- mechanism: Laplace近似、KFAC、线性化模型、低秩适应
- constraint: 计算高效（post-hoc，零额外训练）、小数据场景

## 引用网络

### 后续工作（建立在本文之上）

- [[P__MONA：双分支卷积Adapte_BPSFFV]]: Core algorithmic foundation (LoRA); likely extended/adapted for vision in this w
- [[P__MONA：5%参数超越全量微调的_MONA_(Multi-cOmp]]: Core method source; LoRA is fundamental low-rank adaptation technique likely bui
- [[P__OmniVCus_OmniVCus__Feedforward_Sub]]: LoRA is the standard parameter-efficient fine-tuning method; essential for subje

