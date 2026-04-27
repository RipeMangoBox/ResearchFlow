---
title: 'DKDR: Dynamic Knowledge Distillation for Reliability in Federated Learning'
type: paper
paper_level: A
venue: NeurIPS
year: 2025
paper_link: null
aliases:
- 联邦学习动态知识蒸馏可靠性方法
- DKDR (Dynamic Kn
- DKDR (Dynamic Knowledge Distillation for Reliability)
- DKDR dynamically assigns weights to
acceptance: Poster
method: DKDR (Dynamic Knowledge Distillation for Reliability)
modalities:
- Image
paradigm: supervised
---

# DKDR: Dynamic Knowledge Distillation for Reliability in Federated Learning

**Topics**: [[T__Federated_Learning]], [[T__Classification]] | **Method**: [[M__DKDR]] | **Datasets**: [[D__CIFAR-100]]

> [!tip] 核心洞察
> DKDR dynamically assigns weights to forward and reverse KLD based on knowledge discrepancies and uses knowledge decoupling to identify domain experts, enabling reliable distillation in heterogeneous federated learning.

| 中文题名 | 联邦学习动态知识蒸馏可靠性方法 |
| 英文题名 | DKDR: Dynamic Knowledge Distillation for Reliability in Federated Learning |
| 会议/期刊 | NeurIPS 2025 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2505.xxxxx) · [Code](待补充) · [Project](待补充) |
| 主要任务 | Federated Learning, Image Classification |
| 主要 baseline | FedAvg, FedProx, FedDyn, Scaffold, FedProto, MOON, FedNTD, FedDf |

> [!abstract] 因为「联邦学习中知识蒸馏存在 pathway unreliability（前向KLD在多峰分布下产生偏差）和 teacher unreliability（全局模型在跨域场景下对某些域失效）」，作者在「FedMD 的标准前向KL蒸馏」基础上改了「动态双向KL蒸馏（DKD）+ SVD域专家识别（KDP）」，在「Cifar-100、Office31、Office Home」上取得「Cifar-100 ζ=0.5 时 53.18 vs FedAvg 48.57（+4.61），Office31 50.57 vs FedAvg 34.47（+16.1）」

- **Cifar-100 (ζ=0.3)**: DKDR 51.84 vs Scaffold 50.33 (+1.51)，为最强基线
- **Office31 多域**: DKDR 50.57，去掉 KDP 后暴跌至 36.83 (-13.74)
- **超参数稳定性**: µ=0.5、c=1.25 在跨数据集场景保持稳定

## 背景与动机

联邦学习（Federated Learning, FL）允许多个客户端在不共享原始数据的前提下协同训练模型，但数据异质性（Non-IID）始终是核心瓶颈。想象一个医疗场景：不同医院的CT图像因设备品牌、扫描参数差异呈现显著域偏移——此时全局模型在某些医院的数据上可能表现极差，而传统蒸馏方法仍强制所有客户端向同一个"平均化"的全局教师学习，导致知识传递不可靠。

现有方法如何应对？**FedAvg** 直接均匀聚合客户端更新，在异质数据下模型发散严重；**FedProx** 加入近端正则项约束本地更新，但对强异质性缓解有限；**Scaffold** 引入控制变量修正本地漂移，是现有最强基线之一但仍基于均匀聚合；**FedNTD** 等蒸馏方法采用标准前向KL散度（Forward KLD）进行知识传递，即强制学生分布拟合教师分布。然而，这些方法共同忽略了两个关键问题：


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/fa120b55-fd53-4bef-930a-68c7efc7a72e/figures/fig_001.png)
*Figure: Problem illustration. (a) Typical Distillation Methods exist two unreliability problems as follows. (b) Pathway Unreliability: in multi-domain scenarios, the aggregated global model shows a significantly higher*



**Pathway Unreliability（路径不可靠）**：当全局教师模型输出为多峰分布时，Forward KLD 会导致学生模型产生模式坍塌（mode collapse），只拟合其中一个峰值而忽略其他重要模式。**Teacher Unreliability（教师不可靠）**：在跨域场景中，全局模型经过多轮聚合后，对某些特定域的表示能力反而下降，此时仍用该全局模型作为所有客户端的教师，会引入负迁移。

本文提出 DKDR，通过动态双向蒸馏和域专家识别机制，系统性解决上述两种不可靠性。

## 核心创新

核心洞察：知识蒸馏的可靠性取决于"何时信任教师、何时信任学生"，因为前向KLD和反向KLD各有其失效边界——前向KLD在多峰教师下模式坍塌，反向KLD在单峰教师下过度平滑——从而使动态权衡两者、并引入域特定专家替代不可靠全局教师成为可能。

| 维度 | Baseline (FedMD / 标准KD) | 本文 (DKDR) |
|:---|:---|:---|
| **蒸馏目标** | 仅 Forward KLD: KL[Z\|\|Zw] | 动态加权: α·KL[Z\|\|Zw] + (1-α)·KL[Zw\|\|Z] |
| **教师选择** | 单一全局教师服务所有客户端 | KDP 识别域专家，客户端匹配最近域专家 |
| **聚合方式** | FedAvg 式均匀聚合 | 域感知匹配，基于SVD过滤后的梯度子空间 |
| **可靠性机制** | 无 | 知识差异度量动态调控α + 域专家兜底 |

两个组件缺一不可：DKD 解决 pathway 层面的蒸馏方向选择问题，KDP 解决 teacher 层面的教师质量保障问题。

## 整体框架


![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/fa120b55-fd53-4bef-930a-68c7efc7a72e/figures/fig_002.png)
*Figure: Empirical Analysis. The dashed line represents distillation*



DKDR 的整体数据流如下，包含四个核心阶段：

1. **本地训练（Local Training）**：各客户端基于本地数据训练，输出模型更新（非新组件，标准FL流程）。

2. **KDP 域专家识别（SVD-based Domain Clustering）**：收集客户端梯度矩阵 G，进行SVD分解后截断保留前 r 个奇异值得到 G_filtered = U_r Σ_r V_r^T，基于过滤后的低维子空间聚类识别域特定专家模型。该模块替代了"单一全局教师"，输出域专家-客户端匹配关系。

3. **DKD 动态蒸馏（Dynamic Knowledge Distillation）**：每个客户端接收匹配域专家的输出 Zw（或全局教师 Z），计算知识差异并动态生成权重 α，组合前向KL和反向KL得到蒸馏损失。该模块替代了标准前向KLD损失。

4. **域感知聚合（Domain-aware Aggregation）**：将更新后的客户端模型按域归属聚合，同时维护全局模型和多个域专家。该模块替代了 FedAvg 的均匀聚合。

```
本地数据 → [本地训练] → 梯度更新 G
                              ↓
                    [KDP: SVD过滤 + 域聚类] → 域专家 {E_1, ..., E_k}
                              ↓
                    [DKD: 动态α计算] → α·KL[Z||Zw] + (1-α)·KL[Zw||Z]
                              ↓
                    [域感知聚合] → 更新全局模型 + 更新域专家
```

关键设计：KDP 的 r=0.1 为经验最优，过小（如0.05）会导致域信号收敛到共同低维子空间而聚类失败。

## 核心模块与公式推导

### 模块 1: Dynamic Knowledge Distillation (DKD)（对应框架图右上蒸馏模块）

**直觉**: 前向KLD强制学生覆盖教师所有模式，当教师多峰时学生坍缩到某一峰；反向KLD强制教师覆盖学生模式，当教师单峰时学生过度平滑。需要根据"教师-学生知识差异"动态切换主导方向。

**Baseline 公式** (标准知识蒸馏 [13]):
$$\mathcal{L}_{KD} = KL[Z \| Z_w] = \sum_i Z(i) \log \frac{Z(i)}{Z_w(i)}$$
符号: $Z$ = 学生输出分布, $Z_w$ = 教师输出分布, 温度参数已吸收。

**变化点**: 单一前向KLD无法处理多峰教师分布；反向KLD单独使用又会在单峰场景失效。需要自适应机制判断当前教师-学生对的"可靠性状态"。

**本文公式（推导）**:
$$\text{Step 1}: \mathcal{L}_{bidirectional} = KL[Z \| Z_w] + KL[Z_w \| Z] \quad \text{对称KL同时考虑双向视角}$$
$$\text{Step 2}: \mathcal{L}_{DKD} = \alpha \cdot KL[Z \| Z_w] + (1-\alpha) \cdot KL[Z_w \| Z] \quad \text{引入自适应权重α平衡两者}$$
$$\text{Step 3}: \alpha = f(\text{knowledge discrepancy}; \mu, c) \quad \text{基于知识差异度量动态调整，µ为平衡点，c控制过渡陡度}$$
$$\text{最终}: \mathcal{L}_{DKD} = \alpha(\mathcal{D}) \cdot KL[Z \| Z_w] + (1-\alpha(\mathcal{D})) \cdot KL[Z_w \| Z]$$

其中知识差异度量 $\mathcal{D}$ 基于教师与学生分布的熵/散度差异。理论保证：当 µ < 0.5 时退化为前向KLD，µ > 0.5 时退化为反向KLD，µ=0.5 为动态平衡点。

**对应消融**: Table 6 显示仅前向KLD (FKD) 准确率 47.52 vs DKD 48.79 (-1.95)，仅反向KLD (RKD) 48.03 vs DKD 48.79 (-0.76)。

### 模块 2: Knowledge Decoupling with Proximity (KDP)（对应框架图左上域识别模块）

**直觉**: 跨域场景中全局教师对某些域是"负资产"，应从历史梯度中挖掘"哪些客户端属于同一域"，为其匹配专属专家。

**Baseline**: 无对应基线，传统FL直接使用全局模型作为教师。

**本文公式**:
$$G_{filtered} = U_r \Sigma_r V_r^T$$
符号: $G$ = 客户端梯度矩阵（客户端×参数维度）, $U, \Sigma, V^T$ = SVD分解, $r$ = 保留奇异值比例（实验最优 r=0.1）。

**变化点**: 原始梯度包含大量噪声和任务无关信息，直接聚类无法区分域结构。通过低秩近似过滤，保留最能区分域归属的子空间。

**推导过程**:
$$\text{Step 1}: G \text{xrightarrow}{SVD} U\Sigma V^T \quad \text{完整分解}$$
$$\text{Step 2}: \Sigma_r = \text{diag}(\sigma_1, ..., \sigma_{\lfloor rn \rfloor}), \quad U_r, V_r \text{对应截断}} \quad \text{保留主要方差方向$$
$$\text{Step 3}: G_{filtered} = U_r \Sigma_r V_r^T \approx G \quad \text{低秩近似，过滤噪声}$$
$$\text{Step 4}: \text{DBSCAN/聚类 on rows of } G_{filtered} \rightarrow \text{域标签} \rightarrow \text{专家分配}$$

**对应消融**: Table 5 显示去掉 KDP（仅用DKD）Office31 准确率从 50.57 降至 36.83 (-13.74)，为所有组件中最大降幅；仅KDP无DKD为 48.59 (-1.98)。

### 模块 3: 动态权重函数 α(D)（DKD 的控制核心）

**直觉**: 需要一个可学习的、有理论边界保证的切换函数，避免硬阈值的不稳定性。

**Baseline**: α ≡ 1（纯前向KLD）或 α ≡ 0（纯反向KLD）。

**本文设计**（具体形式基于文本推断）:
$$\alpha = \sigma\left(c \cdot (\mathcal{D} - \mu)\right)$$
其中 $\sigma$ 为 sigmoid 函数, $\mathcal{D}$ = 知识差异度量（如教师熵 - 学生熵或JS散度）, $c$ = 1.25 控制斜率, $\mu$ = 0.5 为平衡点。

**性质**: 当 $\mathcal{D} = \mu$ 时 α = 0.5（完全对称）；$\mathcal{D} \ll \mu$ 时 α → 1（信任教师，前向主导）；$\mathcal{D} \gg \mu$ 时 α → 0（不信任教师，反向主导）。

**对应消融**: Figure 5 显示 µ=0.5, c=1.25 在 Office31 和 Cifar-100 上均为稳定最优，偏离此值性能下降。

## 实验与分析



本文在单域基准 Cifar-100（Dirichlet 分布 ζ∈{0.1, 0.3, 0.5} 控制异质性）和多域基准 Office31、Office Home 上评估 DKDR。
![Figure](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/fa120b55-fd53-4bef-930a-68c7efc7a72e/figures/fig_003.png)
*Figure: Architecture illustration of DKDR. DKDR consists of two core components: ❶The top right*

 展示了训练收敛曲线。

核心数值如下：在 **Cifar-100 ζ=0.5**（强异质性）上，DKDR 达到 **53.18**，超越最强基线 Scaffold 的 51.76（+1.42），较 FedAvg 48.57 提升 **+4.61**。在 **ζ=0.3** 上，DKDR **51.84** vs Scaffold 50.33（+1.51），优势随异质性增强而扩大。多域场景差距更为显著：**Office31** 上 DKDR **50.57**，而 FedAvg 仅 34.47，差距达 **+16.1**——这揭示了传统方法在跨域蒸馏中的系统性失效。



消融实验（Table 5/6 及 Figure 6）揭示了组件贡献的极端不对称性：移除 **KDP**（知识解耦）导致 Office31 从 50.57 暴跌至 **36.83（-13.74）**，说明域专家识别是多域场景的决定性因素；移除 **DKD**（仅保留KDP）降至 48.59（-1.98），影响相对温和。在蒸馏方向消融中，纯前向KLD 47.52 vs DKD 48.79（-1.95），纯反向KLD 48.03 vs DKD 48.79（-0.76），验证了动态加权的必要性。



超参数敏感性（Figure 5）显示 **µ=0.5, c=1.25** 在 Office31 和 Cifar-100 上均为最优且稳定；SVD过滤强度（Figure 6）显示 **r=0.1** 最优，r=0.05 时聚类精度骤降（域信号收敛到共同低维子空间），r=0.2 时保留过多噪声。

公平性检查：基线覆盖较全面（FedAvg/FedProx/FedDyn/Scaffold/FedProto/MOON/FedNTD/FedDf），但缺少引文中提及的 FedDC、FedX 等更强基线的直接对比；未量化 KDP 中 SVD 和多专家维护的额外计算/通信开销；结果缺乏标准差或置信区间；敏感性分析仅覆盖两个数据集。

## 方法谱系与知识库定位

DKDR 属于 **Federated Learning + Knowledge Distillation** 方法族，直接继承自 **FedMD** [26]（异构联邦学习 via 模型蒸馏），但进行了结构性重构而非插件式改进。

**改变的 slots**:
- **Objective**: 前向KLD → 动态双向KLD（DKD）
- **Training recipe**: 单一全局教师 → SVD域专家识别+近邻匹配（KDP）
- **Credit assignment**: 均匀聚合 → 域感知专家-客户端匹配

**直接基线差异**:
- **FedMD [26]**: 同样用蒸馏处理异构FL，但固定前向KLD+单一教师，无动态机制和域专家
- **FedDyn [2]**: 动态正则化但非动态蒸馏，无知识差异感知的双向KL
- **Scaffold [19]**: 控制变量修正本地漂移，无蒸馏机制，均匀聚合
- **FedNTD [25]**: 保留全局知识的蒸馏，但仍固定前向KLD+单一教师

**后续方向**:
1. **模态扩展**: 当前仅限图像分类，向文本/音频/多模态FL蒸馏扩展
2. **计算优化**: KDP的SVD开销与多专家通信成本需量化削减，如用随机SVD或增量更新
3. **理论深化**: 动态权重α的收敛保证、域专家数量的自适应确定（当前r固定）

**标签**: #modality:image #paradigm:supervised #scenario:federated_heterogeneous #mechanism:knowledge_distillation + dynamic_weighting + SVD_clustering #constraint:privacy_preserving

