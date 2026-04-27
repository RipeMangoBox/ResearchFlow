---
title: A Reliable Cryptographic Framework for Empirical Machine Unlearning Evaluation
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 密码学博弈框架的遗忘质量评估
- SWAP test (Unlea
- SWAP test (Unlearning Quality)
- By modeling unlearning evaluation a
acceptance: Poster
cited_by: 3
method: SWAP test (Unlearning Quality)
modalities:
- Image
paradigm: supervised
---

# A Reliable Cryptographic Framework for Empirical Machine Unlearning Evaluation

**Topics**: [[T__Benchmark_-_Evaluation]] | **Method**: [[M__SWAP_test]] | **Datasets**: [[D__CIFAR-10]] (其他: Multiple architectures, DP budget variation)

> [!tip] 核心洞察
> By modeling unlearning evaluation as a cryptographic game between unlearning algorithms and MIA adversaries, the proposed Unlearning Quality metric (Q) computed via the SWAP test provides provable guarantees that existing metrics fail to satisfy, while remaining practical and efficient.

| 中文题名 | 密码学博弈框架的遗忘质量评估 |
| 英文题名 | A Reliable Cryptographic Framework for Empirical Machine Unlearning Evaluation |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2404.11577) · [Code](未提供) · [Project](未提供) |
| 主要任务 | 机器遗忘评估 (Machine Unlearning Evaluation) |
| 主要 baseline | shadow model-based MIA, correctness-based MIA, confidence-based MIA, modified entropy MIA, RETRAIN, FISHER, FTFINAL, RETRFINAL, NEGGRAD, SALUN, SSD |

> [!abstract]
> 因为「现有机器遗忘评估指标缺乏理论基础且可靠性不足，特别是基于成员推断攻击(MIA)的评估存在多种陷阱」，作者在「Certified data removal (Guo et al., 2020)」基础上改了「引入密码学博弈框架与SWAP test近似」，在「CIFAR-10 with ResNet」上取得「Unlearning Quality (Q) 指标在DP预算变化下保持有效，而IC测试失效」的结果

- **关键性能**: Q指标在DP预算 ϵ 从50降至1时保持稳定负相关，IC测试在部分预算下无法产生有意义结果（Table 4）
- **关键性能**: 不同数据集规模 η 下，RETRAIN/SSD/NONE等方法的相对排名保持一致（Table 1）
- **关键性能**: 不同遗忘比例 α 下，各遗忘算法的相对排序稳定性得到验证（Table 2）

## 背景与动机

机器遗忘（Machine Unlearning）要求模型在删除特定训练数据影响后，仍保持可用性。然而，如何可靠地评估遗忘效果一直是开放问题。现有实践普遍采用临时性指标：例如直接计算MIA攻击成功率、观察遗忘集与保留集的准确率差距，或使用IC（Influence Comparison）测试。这些指标虽直观，但缺乏理论根基——两个不同MIA变体可能给出矛盾结论，且无法区分"真正遗忘"与"模型刚好对该样本不敏感"的情况。

具体而言，shadow model-based MIA通过训练影子模型模拟目标模型行为来推断成员身份；correctness-based MIA和confidence-based MIA则分别基于预测正确性和置信度阈值进行判断；modified entropy MIA利用熵的变化作为信号。这些方法各自为政，没有统一的评估标准。更严重的是，当引入差分隐私（DP）训练时，IC测试在某些预算下完全失效，而传统MIA分数的波动使得跨方法比较变得不可靠。

核心痛点在于：现有指标是"攻击驱动"而非"原理驱动"的——它们测量的是某个特定攻击的成功程度，而非遗忘算法本身的安全保证。这导致评估结果既不可比较，也缺乏可证明的性质。

本文将遗忘评估重新建模为密码学安全博弈，从第一性原理出发定义评估指标，并通过SWAP test实现可计算近似。



## 核心创新

核心洞察：将机器遗忘评估建模为密码学安全博弈，因为密码学中"优势（advantage）"概念天然量化敌手区分两个世界的最大能力，从而使"从博弈中自然诱导出具有可证明保证的评估指标"成为可能。

| 维度 | Baseline (ad-hoc MIA metrics) | 本文 (SWAP test / Unlearning Quality) |
|:---|:---|:---|
| 指标来源 | 特定攻击的成功率或准确率差距 | 密码学博弈中敌手的优势 Adv_A(G) |
| 理论基础 | 无统一理论框架 | 安全归约与可证明保证 |
| 计算方式 | 直接应用现成MIA | SWAP test近似，避免全博弈枚举 |
| 跨方法可比性 | 依赖具体攻击选择，结果矛盾 | 博弈结构固定，指标内在一致 |
| DP兼容性 | IC测试部分预算下失效 | Q与ϵ保持稳定的负相关关系 |

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ab514cc0-f69d-4980-97dc-be86afe17fa7/figures/Figure_1.png)
*Figure 1 (pipeline): An unlearning query delivery game illustrated for our machine unlearning evaluation.*



整体数据流遵循密码学安全博弈的标准结构，但适配到机器遗忘场景：

1. **Initialization（初始化）**：输入原始数据集，按 α = 0.1 的比例参数分割为保留集 R、遗忘集 F 和测试集 T，同时划分目标模型数据集与影子模型数据集。敏感度分布 P_D 默认设为均匀分布 U(D)，表示无先验偏好。

2. **Challenger Phase（挑战者阶段）**：挑战者首先用学习算法 LR 在目标数据上训练模型，然后应用待评估的遗忘算法 UL 得到遗忘后模型。此阶段产出可供敌手检查的模型参数或黑盒访问接口。

3. **Adversary Phase（敌手阶段）**：MIA敌手 A 接收遗忘后模型，尝试推断某个样本是否属于遗忘集 F。敌手的能力上限直接决定评估指标的数值。

4. **SWAP Test Computation（SWAP test计算）**：通过SWAP test近似计算敌手在博弈中的优势，输出最终的 Unlearning Quality Q，避免精确计算优势所需的全博弈枚举。

```
数据集 D ──→ [分割: D_target, D_shadow] ──→ 设置参数 (α, P_D=U(D))
                                              ↓
                    ┌─────────────────────────────────────┐
                    │        Challenger Phase             │
                    │   LR: 训练模型 → UL: 应用遗忘算法   │
                    └─────────────────────────────────────┘
                                              ↓
                    ┌─────────────────────────────────────┐
                    │        Adversary Phase              │
                    │   A: 成员推断攻击 (shadow model MIA) │
                    └─────────────────────────────────────┘
                                              ↓
                    ┌─────────────────────────────────────┐
                    │      SWAP Test Approximation        │
                    │   近似计算 Q = Adv_A(G)              │
                    └─────────────────────────────────────┘
                                              ↓
                                        Unlearning Quality Q
```

## 核心模块与公式推导

### 模块 1: 遗忘样本推断博弈 G = (A, UL, D, P_D, α)（对应框架图 初始化 + 挑战者 + 敌手阶段）

**直觉**: 密码学中所有安全定义都源于博弈，将遗忘评估嵌入此范式可使指标继承可证明性质。

**Baseline 公式** (ad-hoc metrics): 无统一形式，典型如 $$\text{MIA success rate} = \frac{1}{|F|}\sum_{x \in F} \mathbb{1}[A(x) = \text{member}]$$ 或 $$|\text{acc}_{\text{retain}} - \text{acc}_{\text{forget}}|$$

符号: $A$ = MIA敌手, $UL$ = 遗忘算法, $D$ = 数据集, $P_D$ = 敏感度分布, $\alpha$ = 遗忘比例参数

**变化点**: Baseline指标直接测量某个攻击的成功率，但该成功率同时受攻击强度和遗忘算法影响，无法分离；本文改为测量"最优敌手在规范博弈中的优势"，将评估对象从"攻击"转向"遗忘算法本身的安全保证"。

**本文公式（推导）**:
$$\text{Step 1}: G = (A, UL, D, P_D, \alpha) \quad \text{定义五元组博弈结构，其中 } P_D = U(D) \text{ 为均匀敏感度分布}$$
$$\text{Step 2}: s = (R, F, T) \in \mathcal{S}_\alpha, \quad O_s(0) = U(F), \quad O_s(1) = U(T) \quad \text{建立分割与预言机，敌手需区分遗忘集与测试集}$$
$$\text{最终}: Q = \text{Adv}_A(G) = |\Pr[A \text{ wins} | b=1] - \Pr[A \text{ wins} | b=0]| \text{ 或等价形式}$$

**对应消融**: Table 1 显示不同数据集规模 η 下，基于该博弈的Q保持相对排名一致。

---

### 模块 2: SWAP test 近似（对应框架图 SWAP Test Computation）

**直觉**: 精确计算敌手优势需要枚举所有可能的分割和模型训练，计算不可行；SWAP test源自量子信息中的状态区分技术，可通过统计方法高效近似。

**Baseline 公式** (exact advantage): $$\text{Adv}_A^{\text{exact}} = \sup_{A} \left| \mathbb{E}_{s \sim \mathcal{S}_\alpha}[\cdots] - \mathbb{E}_{s \sim \mathcal{S}_\alpha}[\cdots] \right|$$ 需要全枚举，计算复杂度随数据集指数增长

符号: $\hat{Q}$ = SWAP test近似值, $\rho_0, \rho_1$ = 对应两个假设的模型状态分布

**变化点**: 精确优势计算涉及对分割空间 $\mathcal{S}_\alpha$ 的积分和模型训练分布的期望，本文用SWAP test结合Monte Carlo采样和额外近似（Section 3.4-3.5）将其转化为可计算形式。

**本文公式（推导）**:
$$\text{Step 1}: \text{SWAP test原始形式} \propto \text{tr}(\rho_0 \rho_1) \quad \text{量子态重叠度量，本文适配到经典概率分布}$$
$$\text{Step 2}: \text{加入shadow model近似} \quad \text{用影子模型估计敌手优势，避免训练大量目标模型}$$
$$\text{Step 3}: \text{重归一化与有偏校正} \quad \text{保证 } \hat{Q} \in [0,1] \text{ 且与精确Q的偏差可控}$$
$$\text{最终}: \hat{Q}_{\text{SWAP}} = f(\text{shadow model outputs}, \text{sample pairs}) \approx Q$$

**对应消融**: Table 6-7 显示不同模型架构和数据集下SWAP test近似保持排名一致性，验证近似质量。

---

### 模块 3: DP蕴含的认证删除保证（理论验证模块）

**直觉**: 若评估指标合理，则应与已知理论结果相容——差分隐私训练自动提供认证删除保证。

**Baseline 公式** (Guo et al. [2020] certified removal): $$(\epsilon, \delta)\text{-DP training} \Rightarrow (\epsilon', \delta')\text{-certified removal for exact unlearning}$$

符号: $\epsilon, \delta$ = DP参数, $UL = \text{NONE}$ = 不应用任何遗忘算法

**变化点**: 本文证明即使不修改遗忘算法（UL=NONE），DP训练本身即蕴含认证删除；这验证了Q指标应能检测到此性质——当ϵ减小时，Q应相应降低（遗忘效果增强）。

**本文公式（推导）**:
$$\text{Step 1}: (\epsilon, \delta)\text{-DP} \Rightarrow (\epsilon, \delta)\text{-certified removal for } UL = \text{NONE} \quad \text{理论保证}$$
$$\text{Step 2}: \text{预测：} Q \text{ 应与 } \epsilon \text{ 负相关，即 } \epsilon \text{downarrow} \Rightarrow Q \text{downarrow} \quad \text{指标一致性检验}$$
$$\text{最终验证}: \text{Table 3 显示 } Q \text{ 随 } \epsilon \text{ 从 } \infty \text{ 降至 } 50 \text{ 而递减，理论预测成立}$$

**对应消融**: Table 8 在线性模型中验证优势随ϵ减小的变化趋势，与理论预测一致。

## 实验与分析


![Table 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ab514cc0-f69d-4980-97dc-be86afe17fa7/figures/Table_1.png)
*Table 1 (comparison): Unlearning Quality score (higher score is better, in percentage). The relative ranking of different unlearning methods with identical training and testing data remains consistent.*



本文在CIFAR-10图像分类任务上使用ResNet架构进行验证，核心发现围绕评估指标的可靠性而非追求SOTA准确率。Table 1展示了不同数据集规模η下的Unlearning Quality得分：RETRAIN（从头重训练的黄金标准）始终获得最高Q值，SSD（Selective Synaptic Dampening）作为近似遗忘的SOTA方法位列第二，而NONE（不遗忘）最差——这一相对排名在不同η下保持稳定，证明Q具有尺度一致性，不受数据集大小扭曲。


![Table 4](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ab514cc0-f69d-4980-97dc-be86afe17fa7/figures/Table_4.png)
*Table 4 (comparison): Comparison between IC score and MIA score under different DP budgets. See Table 3 for comparison details.*



Table 3和Table 4进一步检验DP预算变化下的指标行为。当隐私预算ϵ从∞降至50（δ=10^-5固定），Q呈现稳定的负相关趋势，即更强的隐私保证对应更低的Q值（更好的遗忘效果）。相比之下，IC测试在部分预算设置下完全失效（Table 4A标注"Fails to produce meaningful results"），而传统MIA分数的波动幅度大于Q。这一对比直接验证了核心主张：Q在现有指标失效的场景下仍保持可靠。


![Table 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/ab514cc0-f69d-4980-97dc-be86afe17fa7/figures/Table_3.png)
*Table 3 (comparison): Unlearning Quality score DP Budgets.*



消融实验覆盖多个维度：α变化（Table 2）验证遗忘比例参数不影响方法间相对排序；模型架构变化（Table 6）和数据集变化（Table 7）确认SWAP test近似的跨设置稳定性。所有消融均支持"相对排名一致性"而非绝对数值比较，这与博弈论指标的内在性质一致。

公平性检查：实验仅覆盖CIFAR-10与ResNet，未验证NLP或图数据；仅使用黑盒MIA敌手（作者承认文献中缺乏非弱MIA方法）；SWAP test作为近似，其与大规模设置的精确误差界未完全刻画。此外，影子模型训练需要额外数据分割，可能减少目标模型可用数据量。

## 方法谱系与知识库定位

方法家族：机器遗忘评估（Unlearning Quality / SWAP test）

父方法：Certified data removal (Guo et al., 2020) —— 本文继承其DP-认证删除的理论基础，但将评估对象从"算法是否满足DP"转向"博弈诱导的通用质量指标"。

改变槽位：
- **objective**: 从DP-based认证目标 → 密码学博弈诱导的Q指标
- **architecture**: 添加形式化博弈结构 G = (A, UL, D, P_D, α)
- **inference_strategy**: 从直接MIA应用 → SWAP test近似计算
- **data_curation**: 引入α控制的分割空间 S_α 与敏感度分布 P_D

直接baseline差异：
- **shadow model-based MIA** [3,7,13]: 本文将其作为博弈中的敌手组件而非直接指标
- **IC test / MIA score** [2,5]: 本文证明其在DP预算变化下失效，Q取而代之
- **SALUN, SSD, FISHER** [16,17,19]: 作为遗忘算法被评估对象，非评估方法本身

后续方向：
1. 扩展至白盒MIA敌手与非图像模态（NLP、图数据），验证指标普适性
2. 精确刻画SWAP test近似的误差界，开发更紧的近似算法
3. 将博弈框架扩展至联邦学习等分布式遗忘场景

标签：#image #supervised #evaluation_benchmark #cryptographic_game #membership_inference #differential_privacy #adversarial_evaluation #provable_guarantees

## 引用网络

### 直接 baseline（本文基于）

- SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image Classification and Generation _(ICLR 2024, 实验对比, 未深度分析)_: Recent unlearning method (2024). Likely included in experimental comparison as a

