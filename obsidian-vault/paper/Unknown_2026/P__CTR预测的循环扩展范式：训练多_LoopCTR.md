---
title: 'LoopCTR: Unlocking the Loop Scaling Power for Click-Through Rate Prediction'
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.19550
aliases:
- CTR预测的循环扩展范式：训练多循环推理零循环
- LoopCTR
- 标准扩展范式将「更多计算」与「更多参数」强绑定
method: LoopCTR
---

# LoopCTR: Unlocking the Loop Scaling Power for Click-Through Rate Prediction

[Paper](https://arxiv.org/abs/2604.19550)

**Topics**: [[T__Compression]] (其他: Recommender System) | **Method**: [[M__LoopCTR]]

> [!tip] 核心洞察
> 标准扩展范式将「更多计算」与「更多参数」强绑定，而 LoopCTR 的核心洞察是：训练时的多次循环计算可以通过过程监督被「压缩」进共享参数，使推理时无需执行循环即可复现多循环带来的表示质量提升。这本质上是一种训练-推理不对称设计——训练时用计算换表示质量，推理时用参数共享换部署效率。Hyper-Connected Residuals 和 MoE 解决了单层共享参数表达能力不足的问题，使循环复用真正有效；过程监督解决了推理效率问题，使多循环训练收益可以在零循环推理时被复用。

| 中文题名 | CTR预测的循环扩展范式：训练多循环推理零循环 |
| 英文题名 | LoopCTR: Unlocking the Loop Scaling Power for Click-Through Rate Prediction |
| 会议/期刊 | 2026 arXiv预印本 |
| 链接 | [arXiv](https://arxiv.org/abs/2604.19550) · Code · Project |
| 主要任务 | Click-Through Rate (CTR) 预测 |
| 主要 baseline | HSTU, StackCTR, 标准Transformer深度堆叠模型 |

> [!abstract]
> 因为「深度堆叠Transformer的参数量与计算量同步增长导致推理延迟过高（HSTU达775ms无法部署）」，作者在「参数共享的循环复用结构」基础上改了「三明治架构 + Hyper-Connected Residuals + MoE + 过程监督训练策略」，在「Amazon/TaobaoAds/KuaiVideo/InHouse四个数据集」上取得「LoopCTR(0/3)零循环推理即超越所有baseline，InHouse延迟仅9.26ms（比HSTU低84×）」

- **效率突破**: LoopCTR(0/3) 在 InHouse 上仅 9.26ms 延迟、13.38M FLOPs，比 HSTU 少 160× FLOPs、低 84× 延迟
- **精度突破**: LoopCTR(0/3) 零循环推理即超越所有 baseline，所有改进 p<0.05 统计显著
- **参数效率**: LoopCTR(3/3) 与 StackCTR(3) iso-FLOPs 下仅 1.27M 活跃参数 vs. 1.95M，同等计算量省 35% 参数

## 背景与动机

CTR预测是推荐系统的核心任务，但现有模型的扩展方式正陷入一场「效率危机」。想象一个典型的工业推荐场景：用户刷新信息流，系统需在毫秒级内返回个性化排序结果。然而，当前基于Transformer的CTR模型（如HSTU）为了追求更高精度，普遍采用「堆叠更多参数」的扩展范式——更深、更宽、更长的序列。这种范式的致命缺陷在于计算量与参数量强耦合：HSTU在InHouse数据集上推理延迟高达775ms，远超工业部署的毫秒级约束，实质上无法直接上线。

现有方法如何应对？**深度堆叠**（如标准Transformer、StackCTR）通过增加独立参数层提升容量，但每新增一层即增加等量推理开销；**序列扩展**（如加长用户行为序列）通过更丰富的输入信号提升精度，但注意力复杂度随序列长度平方增长；**参数共享循环结构**（早期探索）尝试复用同一层减少参数量，但标准Transformer block的固定计算流限制了多次迭代的表示精炼能力，单层表达上限迅速饱和。

这些方案的共同短板可归结为两点：**表达能力瓶颈**——固定结构的循环复用无法动态融合不同深度的表示，循环收益快速递减；**推理效率瓶颈**——即便参数共享，多循环推理的延迟仍与循环次数成正比，低延迟场景下不可接受。此外，CTR数据高度稀疏，深度堆叠带来的额外参数还加剧了过拟合风险。

LoopCTR的核心命题由此浮现：能否实现「训练时用计算换表示质量，推理时完全跳过计算」的不对称设计？本文通过三明治架构、超连接残差、MoE稀疏扩展与过程监督训练，首次使「train-multi-loop, infer-zero-loop」成为现实。
![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/11dafb2c-08af-4e63-af1b-cee234f0d0d6/figures/Figure_1.png)
*Figure 1: Figure 1: Architecture of LoopCTR. Left: the sandwich design consisting of an Entry Block(heterogeneous feature projection + grouped self-attention), a Loop Block (prefix attention withshared paramete*



## 核心创新

**核心洞察**：训练时的多次循环计算可以通过过程监督被「压缩」进共享参数，因为 Hyper-Connected Residuals 打破了固定计算流、MoE 在不增计算的前提下扩展了单层容量，从而使「零循环推理复现多循环表示质量」成为可能。

| 维度 | Baseline（深度堆叠/标准循环） | 本文 LoopCTR |
|:---|:---|:---|
| 扩展方式 | 堆叠异构层，参数量∝计算量 | 循环复用共享参数，计算量与参数量解耦 |
| 残差连接 | 固定相邻层连接，信息流受限 | Hyper-Connected Residuals，跨循环深度动态融合 |
| 单层容量 | 单一投影矩阵，表达能力固定 | MoE 稀疏激活，k/E 专家扩展容量而不增计算 |
| 监督位置 | 仅最终输出监督，循环收益难沉淀 | 每轮循环过程监督，多轮增益「蒸馏」进共享参数 |
| 推理策略 | 循环次数=延迟，无法兼得精度与速度 | train-multi-loop(L=3), infer-zero-loop，Loop Block 完全省略 |

## 整体框架



LoopCTR 采用清晰的三明治架构，数据流如下：

**输入** → **Entry Block** → **Loop Block × L（训练时循环，推理时可省略）** → **Exit Block** → **输出预测分数**

各模块职责：
- **Entry Block**：负责异构特征投影（heterogeneous feature projection）与分组自注意力（grouped self-attention），将原始稀疏特征编码为初始隐状态表示，为后续循环迭代准备「原料」。
- **Loop Block**：核心计算单元，参数在 L 次循环中完全共享。内部集成 Hyper-Connected Residuals 与 MoE 投影层，每次循环接收前一轮输出（及跨层历史状态）进行迭代精炼。训练时 L=3，推理时 L=0 即完全跳过此模块。
- **Exit Block**：接收 Loop Block 最终输出（或 Entry Block 直接输出，当推理零循环时），完成分数预测。

关键设计：过程监督分支从 Loop Block 的**每一次循环迭代输出**引出辅助损失，而非仅在 Exit Block 末端计算。这使得 L=3 训练时的三轮迭代优化信号全部作用于同一组共享参数，为零循环推理时的参数「记忆」多轮增益提供训练机制。

```
训练前向:  x → Entry → [Loop]¹ → [Loop]² → [Loop]³ → Exit → ŷ
                    ↓      ↓       ↓
                  loss₁  loss₂   loss₃   (过程监督)
推理前向(零循环): x → Entry ───────────────→ Exit → ŷ
                         (Loop Block 物理省略)
```

## 核心模块与公式推导

### 模块 1: Hyper-Connected Residuals（对应框架图 Loop Block 内部）

**直觉**: 标准残差连接的固定相邻层叠加，使循环复用时信息流僵化；超连接允许任意循环深度间的动态融合，打破单层表达上限。

**Baseline 公式** (标准Transformer残差):
$$\mathbf{h}^{(l)} = \text{LayerNorm}\left(\mathbf{h}^{(l-1)} + \text{FNN}\left(\text{LayerNorm}\left(\mathbf{h}^{(l-1)} + \text{Attn}(\mathbf{h}^{(l-1)})\right)\right)\right)$$
符号: $\mathbf{h}^{(l)}$ = 第 $l$ 层输出, Attn = 自注意力, FNN = 前馈网络

**变化点**: 标准残差仅依赖 $\mathbf{h}^{(l-1)}$，循环加深时梯度路径单一、表示精炼能力饱和。Hyper-Connected Residuals 引入跨循环深度的可学习聚合权重。

**本文公式（推导）**:
$$\text{Step 1}: \quad \tilde{\mathbf{h}}^{(l)} = \sum_{i=0}^{l-1} \alpha_{i}^{(l)} \cdot \mathbf{h}^{(i)} \quad \text{（聚合所有历史循环状态，权重 } \alpha \text{ 可学习）}$$
$$\text{Step 2}: \quad \mathbf{h}^{(l)} = \text{LayerNorm}\left(\tilde{\mathbf{h}}^{(l)} + \text{FNN}\left(\text{LayerNorm}\left(\tilde{\mathbf{h}}^{(l)} + \text{Attn}(\tilde{\mathbf{h}}^{(l)})\right)\right)\right) \quad \text{（标准子层计算，但基于融合后的输入）}$$
$$\text{最终}: \quad \mathbf{h}^{(l)} = \text{LoopBlock}\left(\{\mathbf{h}^{(i)}\}_{i=0}^{l-1}; \theta_{\text{shared}}\right)$$

**对应消融**: Figure 3 显示移除 Hyper-Connected Residuals 后性能下降（具体 Δ 数值待补充，Amazon/KuaiVideo 双数据集验证）。

---

### 模块 2: Mixture-of-Experts (MoE) 投影层（对应框架图 Loop Block 内部 FNN）

**直觉**: 单层共享参数的容量有限，MoE 通过稀疏路由在不增加单次推理计算量的前提下扩展有效参数量。

**Baseline 公式** (标准FFN投影):
$$\mathbf{o} = \sigma(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2, \quad \mathbf{W}_1 \in \mathbb{R}^{d \times d_{ff}}, \mathbf{W}_2 \in \mathbb{R}^{d_{ff} \times d}$$
符号: $\mathbf{x}$ = 输入, $\mathbf{W}$ = 投影矩阵, $d$ = 隐藏维度, $d_{ff}$ = 前馈扩展维度

**变化点**: 单一矩阵 $\mathbf{W}$ 容量固定。MoE 将矩阵分解为 $E$ 个专家，每个 token 经路由网络 $g(\cdot)$ 仅激活 top-$k$ 个专家，实现「宽参数、窄计算」。

**本文公式（推导）**:
$$\text{Step 1}: \quad g(\mathbf{x}) = \text{Softmax}(\mathbf{x}\mathbf{W}_g), \quad \mathbf{W}_g \in \mathbb{R}^{d \times E} \quad \text{（路由网络产生专家分布）}$$
$$\text{Step 2}: \quad \mathcal{T} = \text{TopK}(g(\mathbf{x}), k), \quad |\mathcal{T}| = k \ll E \quad \text{（仅保留 top-k 专家索引，保证稀疏性）}$$
$$\text{Step 3}: \quad \mathbf{o} = \sum_{e \in \mathcal{T}} g(\mathbf{x})_e \cdot \sigma(\mathbf{x}\mathbf{W}_1^{(e)} + \mathbf{b}_1^{(e)})\mathbf{W}_2^{(e)} + \mathbf{b}_2^{(e)} \quad \text{（加权聚合激活专家的输出）}$$
$$\text{最终}: \quad \text{MoE-FFN}(\mathbf{x}) = \sum_{e=1}^{E} g(\mathbf{x})_e \cdot \mathbb{1}_{[e \in \mathcal{T}]} \cdot \text{FFN}^{(e)}(\mathbf{x})$$

额外参数量: $(E-1)\cdot(2d^2 + 2d \cdot d_{ff})$，但**激活参数量**保持与标准FFN同级（仅 $k/E$ 比例的专家参与计算）。

**对应消融**: Figure 3 显示移除 MoE 后性能下降（具体 Δ 数值待补充）。

---

### 模块 3: 过程监督训练目标（对应框架图每轮循环分支）

**直觉**: 仅在最终输出监督，多轮循环的迭代收益难以沉淀到共享参数；对每轮循环输出施加监督，强制共享参数在每一深度都具备独立预测能力，使零循环时参数已「记住」迭代精华。

**Baseline 公式** (标准最终监督):
$$\mathcal{L}_{\text{base}} = \text{BCE}(\hat{y}^{(L)}, y), \quad \hat{y}^{(L)} = \text{Exit}(\mathbf{h}^{(L)})$$
符号: $\hat{y}^{(L)}$ = 第 $L$ 轮循环后预测, $y$ = 标签, BCE = 二元交叉熵

**变化点**: 仅监督 $\hat{y}^{(L)}$ 导致中间循环深度的优化信号间接且微弱。过程监督将多轮训练收益「蒸馏」进共享参数。

**本文公式（推导）**:
$$\text{Step 1}: \quad \hat{y}^{(l)} = \text{Exit}_l(\mathbf{h}^{(l)}), \quad l = 1, 2, \ldots, L \quad \text{（每轮循环输出经独立Exit头预测）}$$
$$\text{Step 2}: \quad \mathcal{L}_{\text{proc}}^{(l)} = \text{BCE}(\hat{y}^{(l)}, y), \quad \forall l \in \{1,\ldots,L\} \quad \text{（每轮均计算监督损失）}$$
$$\text{Step 3}: \quad \mathcal{L}_{\text{final}} = \text{BCE}(\hat{y}^{(L)}, y) + \lambda \sum_{l=1}^{L-1} \mathcal{L}_{\text{proc}}^{(l)} \quad \text{（最终损失 = 最终预测损失 + 过程损失加权和）}$$
$$\text{最终}: \quad \mathcal{L}_{\text{LoopCTR}} = \mathcal{L}_{\text{final}}$$

关键：推理时 Exit Block 直接接收 Entry Block 输出（$\mathbf{h}^{(0)}$ 经简单投影），无需执行任何 Loop Block，但参数已通过过程监督内化了 $L=3$ 训练的迭代精炼模式。

**对应消融**: 过程监督的完整消融数据在提供文本中未完整呈现，但 Figure 2 的 scaling 曲线间接支持：训练循环数 $L$ 增加时，同推理循环下的性能持续提升，暗示过程监督有效传递了多轮收益。

## 实验与分析

**主实验结果** (Table 2 核心对比，数值基于提供文本整理):

| Method | Amazon AUC | TaobaoAds AUC | KuaiVideo AUC | InHouse AUC | 推理循环/延迟 |
|:---|:---|:---|:---|:---|:---|
| 最强 Baseline | — | — | — | — | — |
| LoopCTR(0/3) | **超越所有baseline** | **超越所有baseline** | **超越所有baseline** | **超越所有baseline** | 0循环 / 9.26ms |
| LoopCTR(3/3) | — | — | — | — | 3循环 / 同FLOPs |
| HSTU | — | — | — | — | 775ms |

> 注：Table 2 完整数值未在提供文本中给出，「超越所有baseline」为原文声明，所有改进 p<0.05 统计显著。


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/11dafb2c-08af-4e63-af1b-cee234f0d0d6/figures/Figure_2.png)
*Figure 2: Figure 2: Loop scaling analysis across four datasets. Top row: AUC under different training loopcounts L (colored lines) and inference loop counts i (x-axis). The gray dashed line marks the L=0baselin*



**Scaling 分析** (Figure 2): 训练循环数 $L$ 与推理循环数的组合实验揭示关键规律。固定训练 $L=3$，推理从 0 到 3 循环，AUC 随推理循环增加而提升，但 **LoopCTR(0/3) 已超越所有 baseline**——这是核心主张的最直接证据。同时，同推理循环下，训练 $L$ 更大的模型 consistently 更优，证明过程监督成功将多轮收益「存储」进共享参数。

**效率对比** (Table 6): LoopCTR(0/3) 在 InHouse 上仅需 **13.38M FLOPs** 和 **9.26ms** 延迟，对比 HSTU 的 **~2140M FLOPs** 和 **775ms**，实现 **160× FLOPs 缩减** 与 **84× 延迟降低**。LoopCTR(3/3) 与 StackCTR(3) 严格 iso-FLOPs 对齐下，活跃参数 **1.27M vs. 1.95M**，参数效率提升 **35%**。


![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/11dafb2c-08af-4e63-af1b-cee234f0d0d6/figures/Figure_3.png)
*Figure 3: Figure 3: Ablation study on Amazon (top) and KuaiVideo (bottom). Each variant removes onecomponent from the full LoopCTR(3/3) model (red bar). The red dashed line marks the full modelperformance. HCR:*



**消融实验** (Figure 3): Amazon（上）与 KuaiVideo（下）双数据集上，从完整 LoopCTR(3/3)（红色 bar）逐一移除组件。Hyper-Connected Residuals 与 MoE 的移除均导致性能下降，验证二者对打破单层表达上限的必要性。具体 ΔAUC 数值待补充。

**Loss Landscape** (Figure 4/5): 滤波归一化随机方向法可视化显示，随训练循环数 $L=1,2,3$ 增加，损失曲面更平缓（暖色区域扩展），指示优化稳定性提升与泛化改善，为循环扩展的有效性提供几何视角证据。

**公平性检查**: 
- Baselines 包含 HSTU、StackCTR 等，但工业级十亿参数模型（RankMixer、Zenith、TokenMixer-Large）仅在 Related Work 提及，未纳入实验，竞争强度可能低估。
- InHouse 为内部数据集，采样策略与标签定义未完全公开，独立复现受限。
- MoE 稀疏路由的硬件效率受专家负载均衡影响，实际吞吐可能偏离理论线性预测。

## 方法谱系与知识库定位

**方法家族**: Transformer-based CTR 预测 → 参数高效扩展范式 → 训练-推理不对称设计

**Parent Method**: 参数共享的循环/递归神经网络结构（早期在 CV/NLP 中的 weight tying 探索），但此前未在 CTR 领域成功解决「表达能力+推理效率」双重瓶颈。

**改动槽位**:
- **架构**: 三明治结构（Entry-Loop-Exit）+ Hyper-Connected Residuals 跨层融合 + MoE 稀疏扩展
- **目标**: 过程监督（per-loop auxiliary losses），将多轮优化信号蒸馏进共享参数
- **训练配方**: train-multi-loop (L=3) 配合逐轮退出头
- **推理**: infer-zero-loop，Loop Block 物理省略
- **数据**: 标准 CTR 数据集，无特殊 curation

**直接 Baselines 与差异**:
- **HSTU**: 深度堆叠独立层，参数量∝计算量；LoopCTR 解耦二者，零循环省略计算
- **StackCTR**: 同深度异构层堆叠；LoopCTR 同层共享参数 + iso-FLOPs 下省 35% 参数
- **标准 Transformer 循环复用**: 单层容量不足、无过程监督；LoopCTR 以 Hyper-Connected Residuals + MoE + 过程监督三管齐下使其有效

**后续方向**:
1. **自适应推理**: Oracle 分析揭示 0.02–0.04 AUC 的样本自适应循环深度潜力，需实现动态退出机制
2. **更大规模验证**: 与 RankMixer、Zenith 等十亿参数工业模型的直接对比
3. **MoE 负载均衡优化**: 解决专家激活不均导致的硬件效率损失，逼近理论稀疏加速

**知识库标签**: `modality:tabular` / `paradigm:loop_scaling` / `scenario:recommendation_CTR` / `mechanism:parameter_sharing+process_supervision+sparse_MoE` / `constraint:low_latency_inference`

