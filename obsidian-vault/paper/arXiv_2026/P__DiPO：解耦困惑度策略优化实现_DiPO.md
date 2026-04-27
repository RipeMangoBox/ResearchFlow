---
title: 'DiPO: Disentangled Perplexity Policy Optimization for Fine-grained Exploration-Exploitation Trade-Off'
type: paper
paper_level: B
venue: arXiv
year: 2026
paper_link: https://arxiv.org/abs/2604.13902
aliases:
- DiPO：解耦困惑度策略优化实现细粒度探索利用权衡
- DiPO
- DiPO的核心直觉是：PPL与正确性的关系不是单调的
method: DiPO
paradigm: Reinforcement Learning
---

# DiPO: Disentangled Perplexity Policy Optimization for Fine-grained Exploration-Exploitation Trade-Off

[Paper](https://arxiv.org/abs/2604.13902)

**Topics**: [[T__Math_Reasoning]], [[T__Reinforcement_Learning]] | **Method**: [[M__DiPO]]

> [!tip] 核心洞察
> DiPO的核心直觉是：PPL与正确性的关系不是单调的，而是存在四个象限——真正需要干预的是「正确但高PPL」（CH，需要利用）和「错误但低PPL」（EL，需要探索）这两类反直觉样本。通过动态阈值将样本空间解耦为四象限，再以最小扰动的奖励重分配（仅修改极端组中PPL最大的单个样本）间接引入PPL信号，既避免了直接PPL奖励偏置带来的分布不确定性，又在零梯度的极端组中恢复了有效的训练信号。本质上是一个「精准外科手术式」的插件：在GRPO/DAPO框架上叠加细粒度的样本识别与最小化奖励修正，而非重构训练范式。

| 中文题名 | DiPO：解耦困惑度策略优化实现细粒度探索利用权衡 |
| 英文题名 | DiPO: Disentangled Perplexity Policy Optimization for Fine-grained Exploration-Exploitation Trade-Off |
| 会议/期刊 | arXiv (Cornell University) (预印本) |
| 链接 | [arXiv](https://arxiv.org/abs/2604.13902) · [Code] · [Project] |
| 主要任务 | 基于可验证奖励的强化学习（RLVR）中的探索-利用权衡（EETO），应用于数学推理（AIME24/25）与函数调用（BFCLv3） |
| 主要 baseline | DAPO、GRPO、CDE、DACE、Qwen3-8B-Base |

> [!abstract] 因为「GRPO/DAPO 训练中极端组（全对/全错）优势值为零导致梯度消失，且现有PPL奖励偏置方法因PPL分布不确定性引入噪声」，作者在「DAPO」基础上改了「叠加PSD四象限解耦模块与BRR最小扰动奖励重分配机制」，在「BFCLv3函数调用任务」上取得「F1 86.84 vs DAPO 85.80（+1.04），ACC 78.37 vs 76.94（+1.43）」

- **BFCLv3**: DiPO F1=86.84 / ACC=78.37，较 DAPO（F1=85.80 / ACC=76.94）提升 1.04 / 1.43 个百分点
- **BFCLv3**: 较 Qwen3-8B 基线（F1=67.18）提升 19.66 个百分点
- **AIME24/25**: ACC/mean@8 曲线显示 DiPO 收敛稳定性优于 DAPO（Figure 3，具体数值待补充）

## 背景与动机

在大型语言模型的强化学习训练中，一个隐蔽而致命的问题是：当模型生成的答案全部正确或全部错误时，训练信号会突然消失。以 GRPO（Group Relative Policy Optimization）范式为例，假设一组8个样本全部答对（easy group），组内相对优势全为零，梯度无法传播；同样，全部答错（hard group）时亦然。这种"极端组优势退化"导致模型在训练后期陷入停滞——既无法从错误中探索新策略，也无法从正确中巩固最优解。

现有方法尝试引入困惑度（Perplexity, PPL）作为辅助信号。DACE 对组内所有样本施加统一的 PPL 探索/利用偏置，假设同一组内样本具有同质性；CDE 引入多组加权机制，但未显式区分探索与利用，且超参数繁多。然而，这些方法忽略了一个关键现象：PPL 分布与正确性并非单调相关——部分正确样本反而具有高 PPL（模型"蒙对"了），部分错误样本却具有低 PPL（模型"犹豫"了）。这种交叉使得直接将 PPL 用作奖励偏置会引入显著噪声，破坏验证奖励的稳定性。

更深层的问题在于"粗粒度"处理：现有方法在组级别操作，无法识别组内哪些样本真正需要干预。例如，一个 hard group 中，低 PPL 的错误样本（模型过于自信地错了）才需要鼓励探索，而高 PPL 的错误样本（模型本就犹豫）反而符合预期——但 DACE 会对两者施加相同偏置。

因此，核心挑战归结为：如何在不破坏原始验证奖励分布的前提下，以**细粒度**识别"正确但高 PPL"和"错误但低 PPL"这两类反直觉样本，并实施**最小化干预**？DiPO 正是针对这一缺口设计，通过四象限解耦与双向奖励重分配，在 GRPO/DAPO 框架上实现"外科手术式"的精准修正。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/cc6cc800-0909-477b-a946-4317ffca0a0e/figures/Figure_1.png)
*Figure 1: Figure 1: (a) The proportion of Easy/Normal/Hard groups in each step during the DAPO training.(b) The PPL distribution of correct and error samples in the validation set at 300th steps of DAPOtraining*



## 核心创新

核心洞察：PPL 与正确性的联合分布存在四个象限，其中「正确+高PPL」（CH）和「错误+低PPL」（EL）两类反直觉样本才是真正的干预靶点；因为 PPL 分布的不确定性使得直接奖励偏置会引入噪声，所以通过动态阈值推断和最小方差扰动的奖励重分配间接引入 PPL 信号，从而使在零梯度极端组中恢复有效训练信号成为可能。

| 维度 | Baseline (DACE/CDE/DAPO) | 本文 (DiPO) |
|:---|:---|:---|
| **PPL 使用方式** | 直接作为奖励偏置（加法/加权） | 间接作为样本分类信号，通过 BRR 最小扰动重分配 |
| **干预粒度** | 组级别（group-level），组内同质处理 | 样本级别（sample-level），四象限差异化处理 |
| **阈值设定** | 固定超参数或手动调参 | 动态推断 τ*，最小化分类误差自适应确定 |
| **分布保护** | 未显式控制，PPL 偏置可能破坏验证奖励分布 | 仅修改极端组单样本奖励，保持重分配后方差接近零 |
| **触发范围** | 全样本/全组施加 | 仅零梯度极端组（easy/hard group）触发，normal group 不受影响 |

## 整体框架


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/cc6cc800-0909-477b-a946-4317ffca0a0e/figures/Figure_2.png)
*Figure 2: Figure 2: Illustration of DiPO, consisting of three modules: PPL Queue, Perplexity Space Disentan-gling (PSD), and Bidirectional Reward Reallocation (BRR). Specifically, the PPL Queue caches PPLitems;*



DiPO 作为插件叠加在 GRPO/DAPO 训练范式之上，数据流如下：

**输入**: 一组生成样本及其验证奖励（正确/错误）、各样本的困惑度 PPL

**模块 A: PPL Queue** —— 维护历史 PPL 分布，为动态阈值推断提供统计基础。输入为当前 batch 的 PPL 值，输出为累积 PPL 分布估计。

**模块 B: Perplexity Space Disentangling (PSD)** —— 核心识别模块。输入为样本的（验证奖励, PPL）二元组，通过最小化分类误差推断最优阈值 τ*，将样本空间划分为四象限：CL（正确+低PPL）、CH（正确+高PPL）、EL（错误+低PPL）、EH（错误+高PPL）。输出为每个样本的象限标签，其中 CH 和 EL 为需干预靶点。

**模块 C: Bidirectional Reward Reallocation (BRR)** —— 最小扰动执行模块。输入为极端组的验证奖励分布及 PSD 标签，仅在 easy group（全对）和 hard group（全错）触发：hard group 中将 PPL 最大样本奖励设为 1（鼓励探索），easy group 中将 PPL 最大样本奖励设为 0（鼓励利用）。输出为修正后的奖励，保持组内方差接近零。

**输出**: 修正后的奖励用于 GRPO 优势计算与策略更新

```
[Batch Samples] → [PPL Queue] ──┐
                                ↓
[Verifiable Rewards] → [PSD: τ* inference] → {CL, CH, EL, EH} labels
                                ↓
                        [BRR: conditional reallocation]
                        (only for extreme groups)
                                ↓
                        [Modified Rewards] → [GRPO/DAPO Update]
```

关键设计：PSD 与 BRR 解耦——PSD "识别谁需要干预"，BRR "如何最小代价干预"，两者通过 PPL 阈值 τ* 衔接，形成"诊断-治疗"闭环。

## 核心模块与公式推导

### 模块 1: PPL Queue 与动态阈值推断（对应框架图 PSD 部分）

**直觉**: PPL 与正确性的关系随训练动态变化，固定阈值无法适应模型不同阶段的不确定性特征，需从历史分布中自适应推断。

**Baseline 公式** (DACE/CDE): 直接设定固定阈值 τ₀ 或人工调参
$$r_{\text{ppl}} = f(\text{PPL}; \tau_0, \alpha, \beta)$$
其中 α, β 为多个手工超参数，τ₀ 通常基于启发式设定。

**变化点**: 固定阈值无法捕捉 PPL 分布的动态漂移；且 DACE/CDE 的阈值与验证奖励无关，未联合建模正确性信息。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathcal{D} = \{(y_i, \text{PPL}_i)\}_{i=1}^{N} \quad \text{收集历史样本的正确性与PPL联合分布}$$
$$\text{Step 2}: \quad \tau^* = \text{arg}\min_{\tau} \left[ \epsilon_{\text{CL}}(\tau) + \epsilon_{\text{EH}}(\tau) + \epsilon_{\text{CH}}(\tau) + \epsilon_{\text{EL}}(\tau) \right] \quad \text{最小化四象限分类误差}$$
其中 ε(τ) 表示以 τ 为阈值时的误分类样本比例，即：
- ε_CL(τ): 正确样本中 PPL > τ 被错分为 CH 的比例
- ε_EH(τ): 错误样本中 PPL < τ 被错分为 EL 的比例
- ε_CH(τ), ε_EL(τ) 同理

$$\text{最终}: \quad \text{label}(y, \text{PPL}) = \begin{cases} \text{CL} & \text{if } y=1, \text{PPL} \leq \tau^* \\ \text{CH} & \text{if } y=1, \text{PPL} > \tau^* \\ \text{EL} & \text{if } y=0, \text{PPL} \leq \tau^* \\ \text{EH} & \text{if } y=0, \text{PPL} > \tau^* \end{cases}$$

**对应消融**: 

---

### 模块 2: Bidirectional Reward Reallocation (BRR)（对应框架图 BRR 部分）

**直觉**: 极端组的验证奖励方差为零是"病症根源"，只需最小程度地打破零方差即可恢复梯度，过度干预会引入分布偏移。

**Baseline 公式** (DACE): 
$$r_i^{\text{DACE}} = r_i^{\text{verifiable}} + \alpha \cdot \text{sign}(\text{explore}) \cdot \text{PPL}_i$$
对所有样本施加 PPL 偏置，未区分组内异质性，且 α 为全局超参数。

**变化点**: 直接 PPL 加法偏置导致 (1) 正常组（normal group）被不必要干扰；(2) PPL 分布不确定性使偏置方向可能错误；(3) 全局 α 无法适应不同训练阶段。

**本文公式（推导）**:
$$\text{Step 1}: \quad \mathbb{I}_{\text{trigger}} = \mathbb{1}\left[\text{Var}(r^{\text{verifiable}}) = 0\right] \quad \text{仅方差为零的极端组触发}$$

$$\text{Step 2 (Hard Group)}: \quad i^* = \text{arg}\max_i \text{PPL}_i, \quad r_{i^*}^{\text{new}} = 1 \quad \text{PPL最大样本奖励置1（鼓励探索）}$$
$$\text{Step 2 (Easy Group)}: \quad i^* = \text{arg}\max_i \text{PPL}_i, \quad r_{i^*}^{\text{new}} = 0 \quad \text{PPL最大样本奖励置0（鼓励利用）}$$

$$\text{最终}: \quad r_i^{\text{DiPO}} = \begin{cases} r_i^{\text{verifiable}} & \text{if } \mathbb{I}_{\text{trigger}} = 0 \text{ (normal group)} \\ r_i^{\text{verifiable}} & \text{if } i \neq i^* \text{ (非选中样本)} \\ 1 \text{ or } 0 & \text{if } i = i^* \text{ (选中样本，hard/easy 分别置1/0)} \end{cases}$$

**关键性质**: 对于 n 样本的极端组，原始方差 Var = 0；重分配后仅单个样本改变，新方差为
$$\text{Var}^{\text{new}} = \frac{1}{n}\left(1 - \frac{1}{n}\right)^2 + \frac{n-1}{n}\left(-\frac{1}{n}\right)^2 = \frac{n-1}{n^2} \approx 0 \text{ (for large } n\text{)}$$

**对应消融**: 

## 实验与分析

| Method | BFCLv3 F1 | BFCLv3 ACC | BFCLv3 Precision | AIME24/25 (mean@8) |
|:---|:---|:---|:---|:---|
| Qwen3-8B-Base | 67.18 | — | — | — |
| DAPO | 85.80 | 76.94 | **95.96** |  |
| **DiPO** | **86.84** | **78.37** | 95.69 | 优于 DAPO（Figure 3） |
| Δ (DiPO vs DAPO) | **+1.04** | **+1.43** | -0.27 | — |


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/cc6cc800-0909-477b-a946-4317ffca0a0e/figures/Figure_3.png)
*Figure 3: Figure 4: ACC/mean@8 curves of DiPO andDAPO (raw and smoothed curves) on AIME24and AIME25 with using Qwen3-8B-Base model.*



**核心结论**: BFCLv3 函数调用任务提供了最完整的定量证据。DiPO 在 F1 和 ACC 上均稳定超越 DAPO，绝对提升 1.04 和 1.43 个百分点，相对 Qwen3-8B-Base 提升达 19.66 个百分点，验证了细粒度 PPL 干预的有效性。

**边际代价**: Precision 从 95.96 微降至 95.69（-0.27），说明存在微小的精确率-召回率权衡，但总体 F1 仍净增。

**收敛行为分析** (Figure 3): ACC/mean@8 曲线显示 DiPO 在 AIME24 和 AIME25 上较 DAPO 收敛更稳定，raw 曲线波动更小，smoothed 曲线最终位置更高。但具体最终数值在提供文本中未完整呈现。


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/cc6cc800-0909-477b-a946-4317ffca0a0e/figures/Figure_5.png)
*Figure 5: Figure 5: Entropy curves of maximum-PPL reward and maximum-PPL penalty trained on Qwen3-0.6B model with using DAPO-17K.*



**熵分析** (Figure 5): 在 Qwen3-0.6B 模型上，maximum-PPL reward（探索激励）训练的熵曲线高于 maximum-PPL penalty（探索抑制），定性支持 BRR 的双向设计——对高 PPL 错误样本奖励（保持探索）与对高 PPL 正确样本惩罚（促进利用）产生不同的熵动态。

**PPL 分布演化** (Figure 7，文本提及但未在 figures_available 中): 训练后期 DiPO 的错误样本 PPL 保持较高、正确样本 PPL 保持较低，而 DAPO 两类样本 PPL 均收敛到较低范围，定性验证了 PSD 的解耦效果。

**公平性检查与局限**:
- **Baseline 强度**: DACE 作为直接相关方法未出现在实验对比表中，缺失关键对比；CDE 同样未完整对比
- **计算/数据成本**: 额外维护 PPL Queue 和动态阈值推断，开销较小，但具体 FLOPs 或时间数据未报告
- **适用边界**: 当前验证限于二元验证奖励场景；非二元场景（如 ToolRL 的多步奖励）需额外修改，通用性存疑
- **开放生成任务**: 未经验证，适用性限于 RLVR 场景

## 方法谱系与知识库定位

**方法家族**: 基于 GRPO 的 RLVR 训练增强插件，属于 reward shaping / exploration-exploitation trade-off 分支。

**父方法**: GRPO（Group Relative Policy Optimization）→ DAPO（Dynamic Sampling Policy Optimization，动态采样策略优化）。DiPO 直接叠加于 DAPO 之上，不修改底层策略更新公式。

**改动槽位**: 
- **Objective（目标函数）**: 不变，仍使用 GRPO/DAPO 的 clipped surrogate objective
- **Reward design（奖励设计）**: **核心改动** —— 在验证奖励后叠加 BRR 重分配层
- **Training recipe（训练流程）**: 新增 PPL Queue 累积与 PSD 阈值推断步骤
- **Data curation（数据筛选）**: 不变，但 PSD 提供了新的样本分析维度
- **Inference（推理）**: 无改动，训练时插件

**直接 Baselines 差异**:
| 方法 | 与 DiPO 的核心差异 |
|:---|:---|
| DAPO | DiPO 叠加 PSD+BRR 解决极端组梯度消失；DAPO 本身无 PPL 机制 |
| DACE | DACE 直接 PPL 奖励偏置、组级别处理；DiPO 间接 BRR、样本级别、最小扰动 |
| CDE | CDE 多加权机制、未显式区分探索/利用、超参数多；DiPO 四象限解耦、动态单阈值、双向明确 |

**后续方向**:
1. **多步/连续奖励扩展**: 将 BRR 推广至非二元验证场景（如 ToolRL 的多步工具调用），需重新定义"极端组"概念
2. **与其他探索机制融合**: 结合 entropy bonus、count-based exploration 等，验证 DiPO 的兼容性
3. **理论分析深化**: 当前 BRR 的稳定性声明以理论论证为主，需严格的方差界与收敛性证明

**标签**: 
- **Modality**: 文本生成 / 推理 / 工具调用
- **Paradigm**: 强化学习微调（RL Fine-tuning）
- **Scenario**: 可验证奖励场景（RLVR）
- **Mechanism**: 困惑度（PPL）辅助的奖励重分配
- **Constraint**: 最小扰动原则、零梯度恢复、细粒度样本识别

