---
title: (1D) Ordered Tokens Enable Efficient Test-Time Search
type: paper
paper_level: B
venue: Unknown
year: 2026
paper_link: https://arxiv.org/abs/2604.15453
aliases:
- 1D有序Token赋能高效推理时搜索
- OTEETS
- 1D coarse-to-fine有序token的核心优势在于：生成过
code_url: https://github.com/EPFL-VILAB/search-over-tokens
modalities:
- Image
paradigm: Reinforcement Learning
---

# (1D) Ordered Tokens Enable Efficient Test-Time Search

[Paper](https://arxiv.org/abs/2604.15453) | [Code](https://github.com/EPFL-VILAB/search-over-tokens)

**Topics**: [[T__Agent]], [[T__Image_Generation]], [[T__Reasoning]], [[T__Benchmark_-_Evaluation]]

> [!tip] 核心洞察
> 1D coarse-to-fine有序token的核心优势在于：生成过程的中间状态（部分前缀）已编码全局语义信息，verifier可对其进行可靠评估，从而使beam search能够在生成早期有效剪枝劣质候选。2D grid token的中间状态仅包含局部像素信息，语义信号弱或具有误导性，导致beam search失效。换言之，token的排列顺序决定了中间状态的语义密度，而语义密度决定了verifier引导的有效性，进而决定了推理时搜索的可扩展性。这是一个关于表示结构与搜索效率之间关系的基本洞察。

| 中文题名 | 1D有序Token赋能高效推理时搜索 |
| 英文题名 | (1D) Ordered Tokens Enable Efficient Test-Time Search |
| 会议/期刊 | 2026 arXiv预印本 |
| 链接 | [arXiv](https://arxiv.org/abs/2604.15453) · [Code](https://github.com/EPFL-VILAB/search-over-tokens) · [Project](待补充) |
| 主要任务 | 图像自回归生成中的推理时搜索（test-time search）、token结构对搜索效率的影响分析、无需训练的文本到图像生成 |
| 主要 baseline | 2D grid tokenizer（光栅扫描顺序）、FlexTok（1D有序tokenizer）、Janus（2D grid AR模型） |

> [!abstract] 因为「2D grid token的中间生成状态缺乏全局语义，verifier无法可靠评估导致beam search失效」，作者在「FlexTok 1D有序tokenizer」基础上进行了「系统性的token结构与搜索算法交互分析」，在「COCO/GenEval」上取得「beam search对1D token显著提升（具体数值待补充），ensemble verifier Overall 67 vs. base 57」

- **关键性能1**: 1D有序token + beam search在test-time scaling上显著优于2D grid token，CLIPScore随NFE（number of function evaluations）快速提升（Figure 6）
- **关键性能2**: 8种verifier ensemble在GenEval上达到Overall 67，较无搜索基线57提升10分，接近oracle上限76
- **关键性能3**: 无需AR模型训练，直接搜索FlexTok token空间即可实现zero-shot文本到图像生成（Figure 4）

## 背景与动机

自回归（AR）图像生成模型在推理时可以通过搜索算法（如beam search、best-of-N）分配更多计算以提升输出质量，但这一策略的实际效果严重依赖于token结构的语义可解释性。考虑一个具体场景：当AR模型生成了图像的前50% token时，我们能否判断这张半成品图像是否匹配文本提示？对于传统2D grid tokenizer（如光栅扫描顺序的VQ-VAE），答案是「很难」——这些token按空间位置排列，前缀仅包含图像左上角的局部像素块，缺乏全局语义信息，verifier难以给出可靠评分。

现有方法从不同角度处理这一问题：
- **2D grid AR模型**（如Janus、Parti）：采用光栅扫描顺序的自回归生成，配合image-text verifier进行best-of-N重排序，但中间状态的局部性限制了beam search等需要逐步评估的算法；
- **1D有序tokenizer**（如FlexTok、Semanticist、Infinity）：采用coarse-to-fine结构，早期token编码全局语义，但已有工作聚焦于训练效率或表示质量，未系统研究其在推理时搜索中的独特优势；
- **Test-time scaling方法**（如ScalingAR、GridAR）：专为2D grid结构设计搜索策略，通过改进verifier或搜索算法缓解中间状态评估困难，但未从根本上改变token结构的局限性。

这些方法的共同短板在于：**token结构本身决定了中间状态的语义密度**。2D grid token的中间状态语义信号弱或具有误导性，导致beam search等依赖中间状态评分的算法几乎失效——这是一个关于表示结构与搜索效率的基本瓶颈。本文的核心动机正是：系统量化token结构对推理时搜索可扩展性的决定性影响，并探索1D有序token结构能否解锁传统方法无法实现的新能力。


![Figure 1](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/136707c3-bef1-439f-a99c-d670ee5bea93/figures/Figure_1.png)
*Figure 1: Figure 1. (a) Intermediate readouts. 1D ordered tokens provide a coarse-to-fine structure with interpretable readouts amenable totest-time search. For the prompt ‘a potted plant and a donut”, tokens p*



## 核心创新

核心洞察：1D coarse-to-fine有序token的部分前缀已编码全局语义信息，因为verifier可对中间状态进行可靠评估，从而使beam search能够在生成早期有效剪枝劣质候选，实现推理时搜索的高效扩展。

与 baseline 的差异：

| 维度 | Baseline（2D grid token） | 本文（1D ordered token） |
|:---|:---|:---|
| **Token结构** | 光栅扫描顺序，空间局部性 | Coarse-to-fine层次，语义全局性 |
| **中间状态语义** | 仅含局部像素块，全局信息缺失 | 早期token编码全局结构，可解释性强 |
| **Beam search效果** | 几乎失效，verifier信号误导 | 显著提升，CLIPScore随NFE单调增长 |
| **训练需求** | 需预训练AR模型提供先验 | 可直接搜索token空间，无需AR训练 |
| **核心贡献类型** | 新训练方法/模型架构 | 系统性分析工作，揭示结构-效率关系 |

## 整体框架


![Figure 5](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/136707c3-bef1-439f-a99c-d670ee5bea93/figures/Figure_5.png)
*Figure 5: Figure 5. Overview of the Search-over-Tokens (SoTo) evaluation framework. The framework studies test-time scaling behavior ofimage tokenizers when combined with autoregressive generation and search. (*



本文提出**Search-over-Tokens (SoTo)** 评估框架，系统研究token结构对推理时搜索可扩展性的影响。数据流如下：

**输入**: 文本提示 + 图像tokenizer（1D有序FlexTok 或 2D grid对照基线）

→ **模块1: Tokenizer编码**（输入：图像/文本；输出：token序列）
- FlexTok将图像编码为1D有序token序列，按coarse-to-fine排列，首个token捕获全局语义（Figure 3展示首个token词汇表的可视化）
- 对照基线采用相同数据/架构/训练计算量，仅将token结构改为2D grid

→ **模块2: 搜索算法**（输入：部分token前缀；输出：扩展后的候选序列）
- Best-of-N：独立采样N个完整序列，verifier选最优
- Beam search：每步保留top-k候选，基于中间状态评分逐步扩展
- Lookahead search：前瞻多步评估当前决策质量

→ **模块3: Verifier评估**（输入：解码后的中间/最终图像；输出：标量奖励）
- 8种verifier：Likelihood、CLIPScore、AestheticScore、CycleReward、HPSv2、ImageReward、Grounded SAM、PickScore
- 支持单verifier或ensemble（平均排名聚合）

→ **模块4: 解码与指标计算**（输入：最终token序列；输出：生成图像 + GenEval/COCO指标）
- Flow-based detokenizer将token解码为图像
- 在COCO 300-image子集和GenEval基准上评估Overall、CLIPScore等

核心控制变量设计：1D vs. 2D的唯一差异是token排列顺序，数据分布、模型架构、训练计算量完全匹配，从而隔离token结构的因果效应。

```
[Text Prompt] ──┐
                ▼
[Image] → [Tokenizer: FlexTok 1D / 2D Grid] → [Token Sequence]
                                                    │
                    ┌───────────────────────────────┼───────────────────────────────┐
                    ▼                               ▼                               ▼
              [Best-of-N]                     [Beam Search]                  [Lookahead Search]
                    │                               │                               │
                    └───────────────────────────────┼───────────────────────────────┘
                                                    ▼
                                          [Verifier: 8种选项]
                                                    │
                                                    ▼
                                          [Flow Detokenizer]
                                                    │
                                                    ▼
                                          [Image + Metrics]
```

## 核心模块与公式推导

### 模块 1: 1D有序Tokenizer与中间状态语义编码（对应框架图 Tokenizer模块）

**直觉**: 将图像表示为按重要性排序的token序列，使前缀即具备可解释的全局语义，为verifier提供可靠的中间评估信号。

**Baseline公式**（2D grid tokenizer, 如VQ-VAE with raster scan）：
$$z = \text{Encode}(x) \in \mathbb{R}^{h \times w \times d}, \quad \text{flatten}(z) = [z_{1,1}, z_{1,2}, ..., z_{h,w}] \in \mathbb{R}^{hw \times d}$$
符号: $x$ = 输入图像, $z_{i,j}$ = 空间位置$(i,j)$的离散token, flatten按光栅扫描顺序展开。

**变化点**: 2D grid的flatten顺序仅反映空间邻域，前缀$[z_{1,1}, ..., z_{k}]$对应图像左上角局部区域，缺乏全局语义。verifier对局部像素块的评估无法预测最终图像质量，导致beam search的剪枝决策失效。

**本文公式（FlexTok 1D ordered tokens）**:
$$\text{Step 1}: \quad z = \text{FlexTok-Encode}(x) = [t_1, t_2, ..., t_L] \quad \text{其中 } t_i \in \mathcal{V}_i \text{（层级化词汇表）}$$
$$\text{Step 2}: \quad p(t_i | t_{<i}) = \text{AR}_\theta(t_i | t_1, ..., t_{i-1}) \quad \text{自回归建模，但搜索时可跳过AR训练}$$
$$\text{最终}: \quad \hat{x} = \text{Flow-Decode}([t_1, ..., t_k]) \text{ 对任意前缀 } k \leq L \text{ 产生可解释的中间图像}$$

关键设计：$t_1$专门编码全局语义（Figure 3验证），$t_2, t_3$逐步细化细节。前缀$[t_1, ..., t_k]$的解码结果具有coarse-to-fine结构，verifier可据此评估生成方向是否正确。

**对应消融**: Figure 6显示，移除有序结构（改用2D grid）后，beam search的test-time scaling曲线近乎平坦，而1D有序token的CLIPScore随NFE显著提升。

---

### 模块 2: 搜索算法与Verifier交互（对应框架图 Search+Verifier模块）

**直觉**: 不同搜索算法对中间状态评估的依赖程度不同，token结构的影响具有算法特异性。

**Baseline公式**（标准best-of-N with final-state verifier）：
$$S_\text{BoN} = \text{arg}\max_{i \in [N]} V(\text{Decode}(z^{(i)}_{1:L})), \quad z^{(i)} \sim p_\theta(\cdot | c)$$
符号: $c$ = 文本条件, $z^{(i)}_{1:L}$ = 第$i$个完整采样序列, $V$ = verifier, 仅评估最终状态。

**变化点**: Best-of-N不依赖中间状态，因此token结构对其影响有限。但beam search需要每步评估部分前缀：
$$S_\text{beam}^{(k)} = \text{TopK}_{z_{1:k}} \left\{ V(\text{Decode}(z_{1:k})) \right\}$$
对于2D grid，$V(\text{Decode}(z_{1:k}))$噪声大、信号弱，导致错误剪枝；对于1D有序token，$V(\text{Decode}(z_{1:k}))$与最终质量高度相关，剪枝有效。

**本文公式（Beam search with intermediate verification）**:
$$\text{Step 1}: \quad \text{初始化 } \mathcal{B}_0 = \{\emptyset\}, \text{ beam width } = K$$
$$\text{Step 2}: \quad \mathcal{C}_k = \text{bigcup}_{z_{1:k-1} \in \mathcal{B}_{k-1}} \{[z_{1:k-1}, v] : v \in \mathcal{V}_k\} \quad \text{（扩展候选）}$$
$$\text{Step 3}: \quad \mathcal{B}_k = \text{TopK}_{z_{1:k} \in \mathcal{C}_k} \left\{ V(\text{Flow-Decode}(z_{1:k})) \right\} \quad \text{（verifier评估中间状态）}$$
$$\text{最终}: \quad z^* = \text{arg}\max_{z_{1:L} \in \mathcal{B}_L} V(\text{Flow-Decode}(z_{1:L}))$$

**关键洞察公式**（test-time scaling效率）:
$$\eta_\text{search} = \frac{\partial \mathbb{E}[V(x_\text{final})]}{\partial \text{NFE}} \propto \rho(V(\hat{x}_{1:k}), V(x_\text{final}))$$
即搜索效率与「中间状态verifier评分」和「最终状态verifier评分」的相关性正相关。1D有序token通过提升$\rho$值，从根本上改善scaling效率。

**对应消融**: Figure 6显示，lookahead search在两种结构下表现相近（均依赖最终或近最终状态评估），而beam search的1D vs. 2D差距最大，验证上述相关性机制。

---

### 模块 3: 无需AR训练的纯搜索生成（对应框架图 Training-free分支）

**直觉**: 若token空间本身具有强语义结构，可直接搜索最优token组合，无需学习AR先验$p_\theta$。

**Baseline公式**（标准AR生成）：
$$z^* = \text{arg}\max_{z} p_\theta(z|c) \cdot V(\text{Decode}(z)) \approx \text{采样自 } p_\theta(\cdot|c) \text{ 再用 } V \text{ 重排序}$$

**变化点**: 训练AR模型$p_\theta$成本高昂，且2D grid的$p_\theta$无法为搜索提供有效先验。1D有序token的全局语义结构使得「无模型」搜索成为可能。

**本文公式（Direct search over token space）**:
$$\text{Step 1}: \quad t_1^* = \text{arg}\max_{t_1 \in \mathcal{V}_1} V(\text{Flow-Decode}([t_1])) \quad \text{（首个token决定全局语义）}$$
$$\text{Step 2}: \quad t_k^* = \text{arg}\max_{t_k \in \mathcal{V}_k} V(\text{Flow-Decode}([t_1^*, ..., t_{k-1}^*, t_k])) \quad \text{（贪心或beam扩展）}$$
$$\text{最终}: \quad x^* = \text{Flow-Decode}([t_1^*, ..., t_L^*])$$

此过程完全无需AR模型，仅依赖：(1) FlexTok的编码-解码器；(2) image-text verifier $V$。Figure 4展示了该方法的实际生成效果。

**对应消融**: —— 需对比有/无AR先验的搜索效率差异。

## 实验与分析

主实验结果（COCO/GenEval上的test-time scaling对比）：

| Method | Token结构 | 搜索算法 | GenEval Overall | CLIPScore趋势 | 备注 |
|:---|:---|:---|:---|:---|:---|
| 无搜索基线 | 1D有序 (FlexTok) | — | 57 | — | 贪心解码 |
| Best-of-N | 1D有序 (FlexTok) | best-of-N |  | 平缓提升 | 与2D相近 |
| Beam search | 1D有序 (FlexTok) | beam search | **显著提升** | **随NFE快速上升** | Figure 6核心结果 |
| Lookahead | 1D有序 (FlexTok) | lookahead |  | 中等提升 | 与2D相近 |
| Beam search | 2D grid (对照) | beam search | 边际改善 | 近乎平坦 | 结构决定效率 |
| Beam search | 2D grid (Janus) | beam search |  | 对比验证 | Figure 7 |


![Figure 6](https://researchflow.oss-cn-shanghai.aliyuncs.com/papers/136707c3-bef1-439f-a99c-d670ee5bea93/figures/Figure_6.png)
*Figure 6: Figure 6. Test-time scaling across token structures. We compare inference-time search algorithms on two tokenizers: 1D ordered tokens(FlexTok) and a controlled 2D grid tokenizer. While best-of-N and l*



核心发现分析：
- **支持核心claim的数据**：Figure 6显示，在相同NFE预算下，FlexTok (1D) + beam search的CLIPScore显著高于2D grid对照，且曲线斜率更大，证明1D有序token实现了更高效的test-time scaling。这一优势在beam search中最为突出，因为该算法最依赖中间状态的可靠评估。
- **边际/中性结果**：Best-of-N和lookahead search在两种结构下表现相近（Figure 6），因为这些算法主要评估最终或近最终状态，对中间语义结构的敏感度较低。这说明token结构的优势具有**算法特异性**，并非普适所有搜索策略。
- **Verifier消融**（Figure 5, Figure 11/Table, Figure 15）：8种verifier中，ImageReward和HPSv2是最强单一verifier；ensemble采用平均排名聚合达到Overall 67，较基线57提升10分，但仍低于oracle上限76。Grounded SAM因规则型设计不易被过度优化，具有抗verifier hacking特性。大多数verifier间呈正相关，优化ImageReward对Aesthetic Score的迁移效果最强。



公平性检查与局限：
- **Baseline强度**：未与专为2D grid设计的TTS方法（如ScalingAR、GridAR）直接定量对比，1D优势幅度可能存在高估；Janus对比（Figure 7）为不同架构，非严格对照。
- **计算成本**：Flow-based detokenizer需多步去噪，搜索中反复解码中间token导致推理成本高，实际部署受限。
- **Verifier瓶颈**：oracle与ensemble差距9分，verifier质量是主要瓶颈；足够大的搜索预算下存在verifier hacking风险（Figure 16得分轨迹显示部分退化）。
- **失败案例**：早期前缀重建质量可能下降，影响verifier信号；泛化性仅在Semanticist和Infinity上初步验证，文本/视频模态未覆盖。

## 方法谱系与知识库定位

**方法家族**: 自回归图像生成 → 离散表示学习 → 推理时计算优化（test-time scaling / search）

**Parent method**: FlexTok (Bachmann et al., 2024) — 本文以其为1D有序token的代表，但核心贡献是**分析token结构的影响**而非改进FlexTok本身。

**改变的slot**: 
- **representation_change**: 将2D grid token重新排列为1D coarse-to-fine有序结构（已有工作），但首次系统量化该变化对test-time search的因果效应
- **training_recipe**: 提出无需AR训练的纯搜索生成范式（新能力）
- **evaluation_protocol**: 建立SoTo受控评估框架，隔离token结构变量

**Direct baselines与差异**: 
- **Janus (Wu et al., 2024a)**: 2D grid AR模型，本文Figure 7对比显示其beam search scaling效率低于FlexTok；差异源于token结构而非架构
- **ScalingAR / GridAR**: 专为2D grid设计TTS方法，本文未直接对比，是主要局限
- **VQ-VAE / MaskGIT**: 传统离散表示，缺乏有序语义结构，无法支持中间状态评估

**Follow-up方向**:
1. **高效detokenizer**: 替换flow-based解码器为单步或少量步数解码，降低搜索推理成本
2. **跨模态验证**: 将1D有序token结构扩展至视频、3D、文本生成，检验coarse-to-fine假设的普适性
3. **自适应verifier**: 学习动态权重或课程式verifier，缩小ensemble与oracle的9分差距，抑制verifier hacking

**知识库标签**: 
- modality: image
- paradigm: autoregressive_generation, test_time_search, training_free_generation
- scenario: text_to_image, inference_time_compute_scaling
- mechanism: coarse_to_fine_tokenization, intermediate_state_verification, beam_search_with_structured_representation
- constraint: discrete_representation, verifier_quality_bottleneck, high_inference_cost

