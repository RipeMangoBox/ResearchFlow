---
title: 'OnlineTAS: An Online Baseline for Temporal Action Segmentation'
type: paper
paper_level: C
venue: NeurIPS
year: 2024
paper_link: null
aliases:
- 在线时序动作分割的CFA注意力基线
- OnlineTAS
acceptance: Poster
cited_by: 6
code_url: https://comp.nus.edu.sg/~dinggd/projects/online/online.html
method: OnlineTAS
---

# OnlineTAS: An Online Baseline for Temporal Action Segmentation

[Code](https://comp.nus.edu.sg/~dinggd/projects/online/online.html)

**Topics**: [[T__Segmentation]] | **Method**: [[M__OnlineTAS]] | **Datasets**: [[D__50Salads]], [[D__Breakfast]]

| 中文题名 | 在线时序动作分割的CFA注意力基线 |
| 英文题名 | OnlineTAS: An Online Baseline for Temporal Action Segmentation |
| 会议/期刊 | NeurIPS 2024 (Poster) |
| 链接 | [arXiv](https://arxiv.org/abs/2411.01122) · [Code](https://comp.nus.edu.sg/~dinggd/projects/online/online.html) · [Project](https://comp.nus.edu.sg/~dinggd/projects/online/online.html) |
| 主要任务 | Online Temporal Action Segmentation (在线时序动作分割) |
| 主要 baseline | Causal TCN (自研), LSTR, MV-TAS (在线); MS-TCN, ASFormer, DiffAct (离线) |

> [!abstract] 因为「在线时序动作分割缺乏强基线，现有在线方法(LSTR/MV-TAS)性能远低于离线方法(MS-TCN/ASFormer/DiffAct)」，作者在「Causal TCN」基础上改了「增加CFA跨帧注意力模块、GRU记忆与自适应显式记忆，并支持半在线推理模式」，在「Breakfast/50Salads」上取得「Breakfast Acc 57.4% (semi-online), 50Salads Acc 80.9% (semi-online), 相比Causal TCN提升+5.7% Acc」

- **Breakfast**: OnlineTAS semi-online Acc 57.4%, 相比自研Causal TCN (55.3%) 提升 +2.1%; 后处理使Edit从19.6%提升至56.0% (+36.4%)
- **50Salads**: semi-online Acc 80.9%, 相比Causal TCN (75.2%) 提升 +5.7%; Edit从19.6%提升至28.8% (+9.2%)
- **推理速度**: 使用I3D特征时4.2ms (238.1 FPS)，完整pipeline含原始输入时29.5ms (33.8 FPS) on Nvidia A40

## 背景与动机

时序动作分割(Temporal Action Segmentation, TAS)要求对未修剪视频的每一帧预测动作类别，是细粒度视频理解的核心任务。现有方法大多假设可以访问未来帧，属于**离线设置**——例如MS-TCN通过多阶段时间卷积利用双向上下文，ASFormer借助Transformer的全局自注意力建模长程依赖，DiffAct则用扩散模型生成平滑的分割结果。这些方法在Breakfast、50Salads等数据集上取得了优异性能，但无法满足实时应用需求。

**在线场景**的痛点在于：系统必须在观察到当前帧后立即输出预测，不能等待未来信息。现有在线方法如LSTR（为在线动作检测设计）和MV-TAS（多视角TAS）在分割任务上表现极差——LSTR在Breakfast上Acc仅24.2%，F1@10仅5.5%，与离线方法差距悬殊。这种性能鸿沟使得在线TAS几乎无法实用：一个实时监控或机器人辅助系统若频繁误分动作边界，将导致后续决策连锁错误。

问题的根源在于**时序上下文聚合的困境**：纯因果卷积的感受野有限，难以捕捉跨动作的长程依赖；而直接套用检测任务的模型又缺乏对分割特有的时序连续性建模。更关键的是，领域缺乏一个**公平且强力的在线基线**——现有在线方法要么架构过时，要么任务设定不同，导致研究者无法准确评估在线算法的真实进展。

本文提出OnlineTAS，核心思想是在保持因果性的前提下，通过跨帧注意力机制(CFA)高效聚合历史上下文，并引入GRU记忆与自适应显式记忆双路机制弥补纯卷积的时序建模不足，同时提供逐帧在线与片段级半在线两种推理模式，为在线TAS建立第一个系统性强基线。
![Figure 3](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/23fe258c-70b2-47aa-9b4f-2e8ed6b8d19c/figures/Figure_3.png)
*Figure 3 (qualitative): Visualization of segmentation results in few actions: ... 50Salads [19].*



## 核心创新

**核心洞察**：在线时序动作分割的性能瓶颈不在于特征提取器本身，而在于**历史上下文的聚合方式**——因为纯因果卷积的局部感受野无法建立帧与帧之间的显式长程关联，从而使跨帧注意力(CFA)配合双路记忆机制(GRU内部状态+自适应显式记忆)成为在严格因果约束下逼近离线性能的关键。

| 维度 | Baseline (Causal TCN) | 本文 (OnlineTAS) |
|:---|:---|:---|
| 时序建模 | 单层因果卷积，局部感受野 | GRU隐状态 + 自适应显式记忆库，全局可检索 |
| 帧间交互 | 无显式跨帧关联，仅堆叠卷积层 | CFA模块：2层Transformer decoder + 2层Swin cross-attention |
| 推理模式 | 仅逐帧 (L=1) | 在线(L=1) + 半在线(clip-wise, L=T)，非重叠片段δ=w |
| 输出优化 | 原始预测，过分割严重 | 置信度阈值过滤 + 最小长度约束，去除短片段 |

与直接改造离线Transformer为因果版本不同，本文CFA模块专门针对**当前片段特征与历史记忆**的交互设计：Swin attention降低计算复杂度，GRU保证流式更新效率，显式记忆则保留可检索的关键历史信息。三者的组合使得模型在4.2ms延迟内实现了对长程依赖的有效利用。

## 整体框架


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/23fe258c-70b2-47aa-9b4f-2e8ed6b8d19c/figures/Figure_1.png)
*Figure 1 (pipeline): Construction Pipeline for the CFA module. CFA takes in input features F_1, and outputs the aggregated logits y_t with temporal information captured via an adaptive query A_t and memory M_t. L_tA and L_tA^read are the cross-entropy losses for CFA and reading operation.*



OnlineTAS的完整数据流如下，采用六阶段级联结构：

**输入**: 原始RGB视频帧或预计算I3D特征 → **输出**: 每帧动作类别预测

1. **Feature Extraction (I3D)**: 输入原始帧或预计算特征，输出clip级特征向量。使用标准I3D网络提取时空表征，作为后续模块的输入。

2. **Causal TCN Backbone**: 输入clip特征，输出初始帧级预测。保持因果性，仅使用当前及历史信息，作为整个系统的编码器基础。

3. **GRU Memory**: 输入TCN特征，输出内部隐状态$h_t$。以循环方式累积时序上下文，每帧更新，保证流式推理的恒定内存开销。

4. **Adaptive Explicit Memory**: 输入历史clip特征，输出外部记忆库$\mathcal{M}$。选择性存储关键历史片段表征，供CFA模块显式检索，弥补GRU可能的信息遗忘。

5. **CFA (Cross-Frame Attention) Module**: 输入当前clip特征$F_t$、GRU隐状态$h_t$、显式记忆$\mathcal{M}$，输出增强特征$\hat{F}_t$。核心创新：通过2层Transformer decoder（8头）和2层Swin self-/cross-attention（4头）实现当前片段与历史记忆的多尺度交互。

6. **Post-processing**: 输入原始预测，输出精炼分割结果。应用置信度阈值$\theta$过滤低置信度预测，并以最小长度因子$\sigma$约束去除过短片段，缓解过分割问题。

```
Raw Frames / I3D Features
    ↓
[Feature Extraction] ──→ clip features
    ↓
[Causal TCN Backbone] ──→ initial predictions + TCN features
    ↓                    ↓
[GRU Memory] ←───────── TCN features (recurrent update)
    ↓
[Adaptive Explicit Memory] ←── historical clip features (selective storage)
    ↓
[CFA Module] ←────────── current features + GRU state + explicit memory
    ↓
enhanced features → classifier → raw predictions
    ↓
[Post-processing] ──→ θ threshold + σ min-length → final segmentation
```


![Figure 2](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/23fe258c-70b2-47aa-9b4f-2e8ed6b8d19c/figures/Figure_2.png)
*Figure 2 (pipeline): Online inference pipeline: (a) Online Inference (T = t). (b) Future online Inference (T = t+1). The solid line represents observed data, while the dashed line denotes future data. T and C_t are the mask and prediction of perceived.*

对应Figure 2的在线推理流程：(a) 时刻T=t的在线推理，实线表示已观测数据；(b) 时刻T=t+1的未来在线推理，展示记忆更新与CFA的流式计算。

## 核心模块与公式推导

### 模块 1: 推理模式定义（框架图输入端）

**直觉**: 在线与半在线的灵活切换是平衡延迟与精度的关键，需要明确定义输入长度$L$的两种配置。

**Baseline 公式** (Causal TCN): 
$$L = 1 \quad \text{(逐帧处理，每帧独立前向传播)}$$

符号: $L$ = 输入clip长度，决定每次推理覆盖的帧数。

**变化点**: 纯逐帧推理(L=1)缺乏片段级上下文，而离线方法的$L=T$（整段视频）违反因果约束。本文提出**半在线模式**：积累$w$帧形成非重叠clip（步长$\delta=w$），在延迟与精度间取得平衡。

**本文公式**:
$$\text{Online mode: } L = 1, \quad \delta = 1 \quad \text{(逐帧，最小延迟)}$$
$$\text{Semi-online mode: } L = w, \quad \delta = w \quad \text{(clip级，非重叠，效率优先)}$$

**对应消融**: Section 4.3显示semi-online在Breakfast上Acc 57.4% vs online 56.7%，Edit 19.6% vs 19.3%，片段级聚合对精度有稳定提升。

---

### 模块 2: CFA 跨帧注意力机制（框架图核心）

**直觉**: 当前片段需要与历史建立显式关联，但标准self-attention的$O(T^2)$复杂度不适合在线流式场景；Swin attention的局部窗口+移位设计可降低计算，而cross-attention实现查询-记忆交互。

**Baseline 公式** (标准Transformer decoder cross-attention):
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
其中$Q$=当前帧查询，$K,V$=历史键值（需存储全部历史，$O(T)$内存）。

符号: $Q \in \mathbb{R}^{L \times d}$ = 当前clip查询, $K,V \in \mathbb{R}^{M \times d}$ = 记忆键值, $d_k$ = 头维度, $M$ = 记忆长度。

**变化点**: 
- (1) 纯cross-attention缺乏当前片段内部的细粒度建模；
- (2) 全局attention随视频长度线性增长，在线不可行；
- (3) 单一记忆源（仅GRU或仅显式记忆）各有局限。

**本文公式（推导）**:

$$\text{Step 1: 自注意力增强当前片段表征}$$
$$\hat{F}_t^{\text{self}} = \text{Swin-Self-Attn}(F_t) + F_t \quad \text{(残差连接，2层，4头，局部窗口)}$$
$$\text{加入了 Swin self-attention 以在可控复杂度内建立片段内帧间关联}$$

$$\text{Step 2: 双路记忆准备}$$
$$h_t = \text{GRU}(\text{Pool}(F_t), h_{t-1}) \quad \text{(GRU状态更新，压缩时序)}$$
$$\mathcal{M}_t = \text{Update}(\mathcal{M}_{t-1}, \text{Pool}(F_t)) \quad \text{(显式记忆库，选择性追加/替换)}$$
$$\text{重归一化以保证记忆容量有界，避免无限增长}$$

$$\text{Step 3: 跨帧交叉注意力}$$
$$\hat{F}_t^{\text{cross}} = \text{Swin-Cross-Attn}(\hat{F}_t^{\text{self}}, [h_t; \mathcal{M}_t]) + \hat{F}_t^{\text{self}}$$
$$\text{其中记忆键值 } K,V \text{ 来自GRU状态与显式记忆的拼接/融合}$$
$$\text{最终: } y_t = \text{MLP}(\text{LayerNorm}(\hat{F}_t^{\text{cross}}))$$

**对应消融**: Table 8/9显示移除GRU后Edit从27.1降至19.6 (-7.1)，移除自适应记忆（仅用伪记忆/当前clip）Edit降至22.3 (-4.8)，验证双路记忆的互补必要性；CFA交互轮数I=1 vs I=2差异仅1.7 Edit，说明核心收益来自模块存在而非深度堆叠。

---

### 模块 3: 后处理精炼（框架图输出端）

**直觉**: 在线模型的逐帧预测天然存在抖动，需利用分割任务的时序连续性先验进行约束，但后处理必须与基线公平对比。

**Baseline 公式**: 
$$\hat{y}_t = \text{arg}\max_c y_t^{(c)} \quad \text{(硬阈值，无平滑)}$$

**变化点**: 在线模型的过分割（over-segmentation）比离线更严重，因为缺乏未来信息修正当前判断。引入**置信度过滤+最小长度约束**的两阶段后处理。

**本文公式**:
$$\text{Step 1: 置信度阈值过滤}$$
$$\tilde{y}_t = \begin{cases} \hat{y}_t & \text{if } \max_c y_t^{(c)} > \theta \\ \text{背景/保持} & \text{otherwise} \end{cases}$$

$$\text{Step 2: 最小长度约束}$$
$$\text{对于每段连续预测 } [t_s, t_e], \text{ 若 } (t_e - t_s + 1) < \sigma \cdot L_{\text{avg}}, \text{ 则合并至邻段}$$
$$\text{其中 } \sigma \text{ 为最小长度因子，} L_{\text{avg}} \text{ 为数据集平均动作长度}$$

**对应消融**: Breakfast上后处理使Edit从19.6%提升至56.0% (+36.4%)，F1@10从16.8%提升至38.0% (+21.2%)，但作者未明确报告该后处理是否同样应用于所有基线（公平性存疑，见Section 7）。

## 实验与分析



本文在Breakfast、50Salads、GTEA三个标准TAS数据集上评估，核心指标为帧精度Acc、编辑距离Edit、以及F1@{10,25,50}（容忍边界偏差10%/25%/50%的片段F1）。关键发现在于：OnlineTAS显著缩小了在线与离线方法的差距，同时保持实时推理能力。

在**Breakfast**数据集上，OnlineTAS online模式取得Acc 56.7%、Edit 19.3%，semi-online提升至Acc 57.4%、Edit 19.6%。相比自研Causal TCN baseline（Acc 55.3%、Edit 18.7%），提升幅度为+1.4% Acc与+0.6% Edit。然而，后处理的作用极为显著：应用后处理后Edit跃升至56.0%（+36.4%），F1@10达38.0%（+21.2%）。这一后处理增益甚至超过了模型本身的设计改进，提示读者需关注后处理的公平应用问题。与离线方法对比，MS-TCN（Acc 69.3%）、ASFormer（73.5%）、DiffAct（75.1%）仍有巨大领先，但这属于任务设定差异（离线可用未来信息）而非方法缺陷。

在**50Salads**数据集上，semi-online模式达到Acc 80.9%、Edit 28.8%，较Causal TCN（Acc 75.2%、Edit 19.6%）提升+5.7% Acc与+9.2% Edit。F1指标同样全面领先：F1@10 43.0 vs 26.8（+16.2），F1@25 41.1 vs 24.4（+16.7），F1@50 34.7 vs 19.6（+15.1）。这是本文最完整的性能优势展示，说明在细粒度动作（17类沙拉制备）上，跨帧记忆聚合的收益更为显著。





消融实验（Table 8）验证了各组件贡献：移除GRU导致Edit从27.1降至19.6（-7.1），代价最大；移除自适应显式记忆（仅用当前clip/伪记忆）Edit降至22.3（-4.8）；单独保留GRU去除CFA仍有适度提升但不及完整模型。CFA交互深度I=2 vs I=1差异微小（+1.7 Edit），表明模块存在性比层数更重要。

**公平性检验**：本文存在若干方法论局限。首先，主要对比的在线基线LSTR（Acc 24.2%）和MV-TAS（Acc 41.6%）性能极弱，可能夸大相对改进幅度；更强的在线方法未纳入比较。其次，后处理（threshold θ + min-length σ）是否统一应用于所有基线不明确——若仅用于本文方法则对比不公平。第三，离线基线MS-TCN/ASFormer/DiffAct因使用未来信息而严格不可比，但主表呈现方式可能误导读者对"在线vs离线"差距的认知。最后，评估仅限烹饪视频（Breakfast/50Salads/GTEA），领域多样性不足。推理速度方面，I3D特征输入时4.2ms（238.1 FPS）满足实时，但完整原始输入pipeline为29.5ms（33.8 FPS），实际部署需预计算特征。

## 方法谱系与知识库定位

**方法族**: 在线时序动作分割 (Online Temporal Action Segmentation)

**父方法**: Causal TCN —— 本文明确声明"Our baseline is a single-stage causal TCN"，在其上扩展CFA模块、双路记忆机制及半在线推理模式。

**改动槽位**:
- **Architecture**: Causal TCN → +CFA (2-layer Transformer decoder + 2-layer Swin attention) + GRU memory + adaptive explicit memory
- **Inference strategy**: L=1逐帧 → 支持online (L=1) 与 semi-online (L=w, δ=w) 双模式
- **Data pipeline**: 重叠或单帧输入 → 非重叠clip (δ=w) 提升效率
- **Training recipe**: 标准TCN训练 → 端到端50 epochs, lr=5e-4

**直接基线对比**:
| 基线 | 本文差异 |
|:---|:---|
| Causal TCN (自研) | +CFA +GRU +显式记忆 +后处理，Acc +5.7% (50Salads) |
| LSTR | 专为检测设计，本文针对分割优化，Acc 24.2% → 56.7% |
| MV-TAS | 仅多视角，本文单视角+记忆机制，Acc 41.6% → 57.4% |
| MS-TCN/ASFormer/DiffAct | 离线→在线，牺牲未来信息换取实时性，性能差距为任务设定所致 |

**后续方向**:
1. **更强在线基线**: 将ASFormer等离线SOTA改造为因果版本，建立更公平的在线基准
2. **记忆机制优化**: 自适应显式记忆的更新/淘汰策略（如基于不确定性的主动遗忘）
3. **跨领域验证**: 从烹饪视频扩展至体育、工业装配等动作边界更模糊的场景

**知识库标签**:
- **Modality**: Video (RGB, I3D features)
- **Paradigm**: Online / Semi-online inference
- **Scenario**: Real-time temporal action segmentation
- **Mechanism**: Cross-frame attention, Dual memory (GRU implicit + explicit), Causal convolution
- **Constraint**: No future context, low latency (4.2ms with features)

